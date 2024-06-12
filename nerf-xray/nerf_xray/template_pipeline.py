"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Type, Union

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
from nerfstudio.utils.rich_utils import CONSOLE
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerf_xray.template_datamanager import TemplateDataManagerConfig
from nerf_xray.template_model import TemplateModel, TemplateModelConfig


@dataclass
class TemplatePipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: TemplatePipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = TemplateDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = TemplateModelConfig()
    """specifies the model config"""
    volumetric_supervision: bool = False
    """specifies if the training gets volumetric supervision"""
    volumetric_supervision_start_step: int = 100
    """start providing volumetric supervision at this step"""
    load_density_ckpt: Optional[Path] = None
    """specifies the path to the density field to load"""


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

        if config.load_density_ckpt is not None:
            self.load_density_field(config.load_density_ckpt)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                TemplateModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

    def load_density_field(self, ckpt_path: Path) -> None:
        """Load the checkpoint from the given path

        Args:
            ckpt_path: path to the checkpoint
        """
        assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist"
        loaded_state = torch.load(ckpt_path, map_location="cpu")
        loaded_state = loaded_state['pipeline']
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        state = {key: value for key, value in state.items() if 'model.field' in key}
        self.load_state_dict(state)
        CONSOLE.print(f"Done loading density field from checkpoint from {ckpt_path}")

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict: Dict
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        if self.datamanager.object is not None:
            metrics_dict.update(self.calculate_density_loss())
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        # evaluate along a few lines
        # self.eval_along_lines(b=[0.5,0.75,0.22,0.21,0.68], c=[0.43,0.79,0.2,0.75,0.3], line='x', fn=f'C:/Users/ig348/Documents/nerfstudio/outputs/balls/method-template/line_{step:04d}.png')
        self.train()
        return model_outputs, loss_dict, metrics_dict
    
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
            TimeRemainingColumn(),
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
        if self.datamanager.object is not None:
            # evaluate volumetric loss on a 100x100x100 grid
            metrics_dict.update(self.calculate_density_loss())
        self.train()
        return metrics_dict
    
    def calculate_density_loss(self, sampling: str = 'random'):
        if sampling=='grid':
            pos = torch.linspace(-1, 1, 200, device=self.device) # scene box goes between -1 and 1 
            pos = torch.stack(torch.meshgrid(pos, pos, pos, indexing='ij'), dim=-1).reshape(-1, 3)
        elif sampling=='random':
            pos = 2*torch.rand((self.config.datamanager.train_num_rays_per_batch*32, 3), device=self.device) - 1.0
        pred_density = self._model.field.get_density_from_pos(pos).squeeze() # remove nograd here so can use in training
        density = self.datamanager.object.density(pos).squeeze() # density between -1 and 1
        
        x = density
        y = pred_density

        density_loss = torch.nn.functional.mse_loss(y, x)

        density_n = (x - x.min()) / (x.max() - x.min())
        pred_dens_n = (y - y.min()) / (y.max() - y.min())
        scaled_density_loss = torch.nn.functional.mse_loss(pred_dens_n, density_n)
        
        mux = x.mean()
        muy = y.mean()
        dx = x-mux
        dy = y-muy
        normed_correlation = torch.sum(dx*dy) / torch.sqrt(dx.pow(2).sum() * dy.pow(2).sum())
        return {
            'volumetric_loss': density_loss, 
            'scaled_volumetric_loss': scaled_density_loss,
            'normed_correlation': normed_correlation
            }

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        
        if self.config.volumetric_supervision and step>self.config.volumetric_supervision_start_step:
            # provide supervision to visual training. Use cross-corelation loss
            density_loss = self.calculate_density_loss(sampling='random')
            loss_dict['volumetric_loss'] = -0.01*density_loss['normed_correlation']

        return model_outputs, loss_dict, metrics_dict
    
    def eval_along_lines(self, b: Sequence[float], c: Sequence[float], line='x', fn=None, engine='matplotlib'):
        a = torch.linspace(0, 1, 500, device=self.device)
        bc = torch.ones_like(a)
        pred_densities = []
        true_densities = []
        with torch.no_grad():
            for _b, _c in zip(b,c):
                if line == 'x':
                    pos = torch.stack([a, bc*_b, bc*_c], dim=-1)
                elif line == 'y':
                    pos = torch.stack([bc*_b, a, bc*_c], dim=-1)
                elif line == 'z':
                    pos = torch.stack([bc*_b, bc*_c, a], dim=-1)
                pred_density = self._model.field.get_density_from_pos(pos)
                pred_densities.append(pred_density.cpu().numpy())
                true_density = self.datamanager.object.t_density(pos)
                true_densities.append(true_density.cpu().numpy())
        a = a.cpu().numpy()
        if engine=='matplotlib':
            plt.figure(figsize=(6,6))
            for i in range(len(b)):
                label = f'y={b[i]}, z={c[i]}' if line == 'x' else f'x={b[i]}, z={c[i]}' if line == 'y' else f'x={b[i]}, y={c[i]}'
                plt.plot(a, pred_densities[i], label=label, ls='-', color=f'C{i}', alpha=0.7)
                plt.plot(a, true_densities[i], ls='--', color=f'C{i}', alpha=0.7)
            plt.legend()
            if fn is not None:
                plt.savefig(fn)
            plt.close()
        else:
            raise NotImplementedError("Only matplotlib supported for now")
    
    def eval_along_plane(self, plane='xy', distance=0.5, fn=None, engine='cv', resolution=500):
        a = torch.linspace(-1, 1, resolution, device=self.device) # scene box will map to 0-1
        b = torch.linspace(-1, 1, resolution, device=self.device) # scene box will map to 0-1
        A,B = torch.meshgrid(a,b, indexing='ij')
        C = distance*torch.ones_like(A)
        if plane == 'xy':
            pos = torch.stack([A, B, C], dim=-1)
        elif plane == 'yz':
            pos = torch.stack([C, A, B], dim=-1)
        elif plane == 'xz':
            pos = torch.stack([A, C, B], dim=-1)
        with torch.no_grad():
            pred_density = self._model.field.get_density_from_pos(pos)
        if engine=='matplotlib':
            plt.figure(figsize=(6,6))
            plt.imshow(pred_density.cpu().numpy(), extent=[0,1,0,1], origin='lower', cmap='gray', vmin=0, vmax=1)
            if fn is not None:
                plt.savefig(fn)
            plt.close()
        elif engine in ['cv', 'opencv']:
            pred_density = pred_density.cpu().numpy()
            # pred_density = (pred_density - pred_density.min())/(pred_density.max() - pred_density.min())
            # pred_density is between 0 and 1 anyways
            pred_density = (pred_density*255).astype(np.uint8)
            if fn is not None:
                if isinstance(fn, Path):
                    fn = fn.as_posix()
                cv.imwrite(fn, pred_density)
        else:
            raise ValueError(f"Invalid engine {engine}")