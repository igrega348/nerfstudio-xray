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

from nerf_xray.xray_datamanager import XrayDataManagerConfig
from nerf_xray.canonical_model import CanonicalModel, CanonicalModelConfig



@dataclass
class CanonicalPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: CanonicalPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=lambda: XrayDataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=lambda: CanonicalModelConfig)
    """specifies the model config"""
    volumetric_supervision: bool = False
    """specifies if the training gets volumetric supervision"""
    volumetric_supervision_start_step: int = 100
    """start providing volumetric supervision at this step"""
    volumetric_supervision_coefficient: float = 0.005
    """coefficient for the volumetric supervision loss"""
    load_density_ckpt: Optional[Path] = None
    """specifies the path to the density field to load"""
    flat_field_penalty: float = 0.01
    """penalty to increase flat field"""


class CanonicalPipeline(VanillaPipeline):
    """Canonical Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: CanonicalPipelineConfig,
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
                CanonicalModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        assert batch['image'].ndim==2
        batch['image'] = batch['image'][...,[0]] # [..., 1]
        model_outputs = self.model(ray_bundle)
        metrics_dict: Dict
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        if self.datamanager.object is not None:
            metrics_dict.update(self.calculate_density_loss())
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict
    
    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False, which=None, **kwargs
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
    
    def calculate_density_loss(self, sampling: str = 'random', resolution: Optional[int] = None) -> Dict[str, Any]:
        if sampling=='grid':
            if resolution is None:
                resolution = 250
            pos = torch.linspace(-1, 1, resolution, device=self.device) # scene box goes between -1 and 1 
            pos = torch.stack(torch.meshgrid(pos, pos, pos, indexing='ij'), dim=-1).reshape(-1, 3)
        elif sampling=='random':
            if resolution is None:
                resolution = self.config.datamanager.train_num_rays_per_batch*32
            pos = 2*torch.rand((resolution, 3), device=self.device) - 1.0
        
        object = self.datamanager.object
        pred_density = self._model.field.get_density_from_pos(pos).squeeze()

        density = object.density(pos).squeeze() # density between -1 and 1
        
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

    def get_eval_density_loss(
        self,
        sampling: Literal['random', 'grid'] = 'random',
        time: Optional[float] = None,
        target=None,
        npoints: Optional[int] = None,
        extent: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate density loss for CanonicalPipeline (single-field model).

        Args:
            sampling: 'grid' or 'random' point sampling strategy.
            time: time parameter (unused for canonical model, kept for API compatibility).
            target: an Object whose .density() provides ground-truth density.
                    Falls back to self.datamanager.object if None.
            npoints: number of sample points (per axis for grid, total for random).
            extent: axis-aligned bounding box as ((x0,x1),(y0,y1),(z0,z1)).
                    Defaults to ((-1,1),(-1,1),(-1,1)).
            batch_size: max points per forward pass (to fit in GPU memory).
        """
        if extent is None:
            extent = ((-1, 1), (-1, 1), (-1, 1))

        if sampling == 'grid':
            if npoints is None:
                npoints = 200 ** 3
            n_per_axis = int(round(npoints ** (1 / 3)))
            lsp = torch.linspace(0, 1, n_per_axis, device=self.device)
            pos = torch.stack(
                torch.meshgrid(
                    extent[0][0] + lsp * (extent[0][1] - extent[0][0]),
                    extent[1][0] + lsp * (extent[1][1] - extent[1][0]),
                    extent[2][0] + lsp * (extent[2][1] - extent[2][0]),
                    indexing='ij',
                ),
                dim=-1,
            ).reshape(-1, 3)
        elif sampling == 'random':
            if npoints is None:
                npoints = self.config.datamanager.train_num_rays_per_batch * 32
            pos = torch.rand((npoints, 3), device=self.device)
            pos[:, 0] = extent[0][0] + pos[:, 0] * (extent[0][1] - extent[0][0])
            pos[:, 1] = extent[1][0] + pos[:, 1] * (extent[1][1] - extent[1][0])
            pos[:, 2] = extent[2][0] + pos[:, 2] * (extent[2][1] - extent[2][0])
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling!r}")

        if batch_size is None:
            batch_size = 1 << 50

        pred_density_chunks = []
        for i in range(0, pos.shape[0], batch_size):
            pos_batch = pos[i:i + batch_size]
            if self._model.deformation_field is not None:
                chunk = self._model.field.get_density_from_pos(
                    pos_batch,
                    deformation_field=self._model.deformation_field,
                    time=time,
                ).squeeze()
            else:
                chunk = self._model.field.get_density_from_pos(pos_batch).squeeze()
            pred_density_chunks.append(chunk)
        pred_density = torch.cat(pred_density_chunks, dim=0)

        if target is None:
            obj = self.datamanager.object
        else:
            obj = target

        # Convert pos from NeRF coordinate space to original world coordinate space
        # before querying obj.density, which uses world-space coordinates from object.json.
        # dataparser_transform T (3x4) maps world→NeRF: p_nerf = R @ p_world  (column-vec convention)
        # For row-vector tensors (N,3): p_nerf = p_world @ R.T
        # Inverse: p_world = p_nerf @ R  (since R is orthogonal: R^{-1} = R^T, row-vec: p_world = p_nerf @ R)
        try:
            T = self.datamanager.train_dataparser_outputs.dataparser_transform  # (3, 4)
            R = T[:3, :3].to(pos)  # (3, 3)
            world_pos = pos @ R  # (N, 3)
        except Exception:
            world_pos = pos

        density = obj.density(world_pos).squeeze()

        x = density
        y = pred_density

        density_loss = torch.nn.functional.mse_loss(y, x)

        density_n = (x - x.min()) / (x.max() - x.min() + 1e-8)
        pred_dens_n = (y - y.min()) / (y.max() - y.min() + 1e-8)
        scaled_density_loss = torch.nn.functional.mse_loss(pred_dens_n, density_n)

        mux = x.mean()
        muy = y.mean()
        dx = x - mux
        dy = y - muy
        normed_correlation = torch.sum(dx * dy) / torch.sqrt(dx.pow(2).sum() * dy.pow(2).sum() + 1e-8)

        return {
            'volumetric_loss': density_loss,
            'scaled_volumetric_loss': scaled_density_loss,
            'normed_correlation': normed_correlation,
        }

    def get_flat_field_penalty(self):
        return -self.config.flat_field_penalty*self.model.flat_field

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        assert batch['image'].ndim==2
        batch['image'] = batch['image'][...,[0]] # [..., 1]
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        loss_dict['flat_field_loss'] = self.get_flat_field_penalty()
        
        if self.config.volumetric_supervision and step>self.config.volumetric_supervision_start_step:
            # provide supervision to visual training. Use cross-corelation loss
            assert self.datamanager.object is not None
            density_loss = self.calculate_density_loss(sampling='random')
            loss_dict['volumetric_loss'] = self.config.volumetric_supervision_coefficient*(1-density_loss['normed_correlation'])

        return model_outputs, loss_dict, metrics_dict
    
    def eval_along_plane(
        self, 
        target: Literal['field', 'datamanager', 'both'],
        plane='xy', 
        distance=0.0, 
        fn=None, 
        engine='cv', 
        resolution=500,
        rhomax=1.0,
        time=0.0,
        which=None
    ):
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
        if target in ['field', 'both']:
            with torch.no_grad():
                if self.model.deformation_field is not None:
                    pred_density = self._model.field.get_density_from_pos(pos, deformation_field=self._model.deformation_field, time=time).squeeze()
                else:
                    pred_density = self._model.field.get_density_from_pos(pos).squeeze()
                pred_density = pred_density.cpu().numpy() / rhomax
        if target in ['datamanager', 'both']:
            pos_shape = pos.shape
            assert pos_shape[-1] == 3
            obj_density = self.datamanager.object.density(pos.view(-1,3)).view(pos_shape[:-1])
            max_density = self.datamanager.object.max_density
            obj_density = obj_density.cpu().numpy() / max_density
        if target == 'both':
            density = np.concatenate([obj_density, pred_density], axis=1)
        elif target == 'field':
            density = pred_density
        elif target == 'datamanager':
            density = obj_density

        if engine=='matplotlib':
            plt.figure(figsize=(6,6) if target!='both' else (12,6))
            plt.imshow(
                density, 
                extent=[-1,1,-1,1] if plane=='xy' else [-1,3,-1,1], 
                origin='lower', cmap='gray', vmin=0, vmax=1
            )
            if fn is not None:
                plt.savefig(fn)
            plt.close()
        elif engine in ['cv', 'opencv']:
            density = np.clip(density, 0, 1)
            density = (density*255).astype(np.uint8)
            if fn is not None:
                if isinstance(fn, Path):
                    fn = fn.as_posix()
                cv.imwrite(fn, density)
        elif engine=='numpy':
            return density
        else:
            raise ValueError(f"Invalid engine {engine}")