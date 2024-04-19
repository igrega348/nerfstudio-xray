"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal, Optional, Type

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from jaxtyping import Float, Shaped
from nerfstudio.data.datamanagers.base_datamanager import (DataManager,
                                                           DataManagerConfig,
                                                           VanillaDataManager)
from nerfstudio.data.datamanagers.full_images_datamanager import \
    FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import \
    ParallelDataManager
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (VanillaPipeline,
                                                VanillaPipelineConfig)
from nerfstudio.utils import profiler
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn)
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from method_template.template_datamanager import TemplateDataManagerConfig
from method_template.template_model import TemplateModel, TemplateModelConfig


@dataclass
class TemplatePipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: TemplatePipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = TemplateDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = TemplateModelConfig()
    """specifies the model config"""
    volumetric_training: bool = False
    """specifies if the training is volumetric or from projections"""


class TemplatePipeline(VanillaPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: TemplatePipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                TemplateModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])
    
    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    raise NotImplementedError("Saving images is not implemented yet")

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        # evaluate volumetric loss on a 100x100x100 grid
        pos = torch.linspace(0, 1, 100, device=self.device)
        pos = torch.stack(torch.meshgrid(pos, pos, pos, indexing='ij'), dim=-1).reshape(-1, 3)
        self.train()
        with torch.no_grad():
            self._model.field.volumetric_training = True
            model_outputs = self._model.field.forward(pos)
            self._model.field.volumetric_training = False
            pred_density = model_outputs[FieldHeadNames.DENSITY].squeeze()
        density = self.datamanager.object.t_density(pos)
        density_loss = torch.nn.functional.mse_loss(pred_density, density)
        metrics_dict['volumetric_loss'] = density_loss.item()
        self.train()
        return metrics_dict

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.config.volumetric_training:
            # sample positions
            pos = torch.rand((self.config.datamanager.train_num_rays_per_batch*32, 3), device=self.device)
            density = self.datamanager.object.t_density(pos)
            model_outputs = self._model.forward(pos)  # train distributed data parallel model if world_size > 1
            loss_dict = {'loss': torch.nn.functional.mse_loss(model_outputs[FieldHeadNames.DENSITY], density)}
            metrics_dict = {}
        else:
            ray_bundle, batch = self.datamanager.next_train(step)
            model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        
        if step == 500:
            for z in np.arange(0.0, 1.0, 0.02):
                # self.eval_along_plane(plane='xy', distance=z, fn=f'C:/Users/ig348/Documents/nerfstudio/outputs/sphere_render/method-template/slices/out_{z:.3f}.png', engine='matplotlib')
                self.eval_along_plane(plane='xy', distance=z, fn=f'C:/temp/nerfstudio/outputs/sphere_render/method-template/slices/out_{z:.3f}.png')
            # _,_ = self.eval_along_line()
            # self.eval_along_plane(plane='xy')

        return model_outputs, loss_dict, metrics_dict
    
    def eval_along_line(self):
        z = torch.linspace(0, 1, 500, device=self.device)
        y = 0.5*torch.ones_like(z)
        x = 0.5*torch.ones_like(z)
        pos = torch.stack([x, y, z], dim=-1)
        with torch.no_grad():
            self._model.field.volumetric_training = True
            model_outputs = self._model.field.forward(pos)
            self._model.field.volumetric_training = False
        density = self.datamanager.object.t_density(pos)
        pred_density = model_outputs[FieldHeadNames.DENSITY]
        plt.plot(z.cpu().numpy(), pred_density.cpu().numpy())
        plt.savefig('C:/temp/nerfstudio/outputs/sphere_render/method-template/out.png')
        plt.close()
        return pred_density, density
    
    def eval_along_plane(self, plane='xy', distance=0.5, fn=None, engine='cv'):
        a = torch.linspace(0, 1, 500, device=self.device)
        b = torch.linspace(0, 1, 500, device=self.device)
        A,B = torch.meshgrid(a,b, indexing='ij')
        C = distance*torch.ones_like(A)
        if plane == 'xy':
            pos = torch.stack([A, B, C], dim=-1)
        elif plane == 'yz':
            pos = torch.stack([C, A, B], dim=-1)
        elif plane == 'xz':
            pos = torch.stack([A, C, B], dim=-1)
        with torch.no_grad():
            self._model.field.volumetric_training = True
            model_outputs = self._model.field.forward(pos)
            self._model.field.volumetric_training = False
        pred_density = model_outputs[FieldHeadNames.DENSITY]
        if engine=='matplotlib':
            plt.imshow(pred_density.cpu().numpy(), extent=[0,1,0,1], origin='lower', cmap='gray')
            if fn is not None:
                plt.savefig(fn)
            plt.close()
        elif engine=='cv':
            pred_density = pred_density.cpu().numpy()
            # pred_density = (pred_density - pred_density.min())/(pred_density.max() - pred_density.min())
            # pred_density is between 0 and 1 anyways
            pred_density = (pred_density*255).astype(np.uint8)
            if fn is not None:
                cv.imwrite(fn, pred_density)