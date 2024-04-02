"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch
import torch.distributed as dist
from jaxtyping import Float, Shaped
from nerfstudio.data.datamanagers.base_datamanager import (DataManager,
                                                           DataManagerConfig)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (VanillaPipeline,
                                                VanillaPipelineConfig)
from nerfstudio.utils import profiler
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

    def true_density(self, pos: Shaped[Tensor, "*bs 3"]) -> Shaped[Tensor, "*bs 1"]:
        pos = pos - 0.5
        r = torch.sqrt(torch.sum(pos**2, dim=-1, keepdim=True))
        density = r.new_zeros(r.size())
        btm_mask = torch.all(pos>-0.25, dim=-1, keepdim=True)
        top_mask = torch.all(pos<0.25, dim=-1, keepdim=True)
        mask = btm_mask & top_mask & (r<0.25)
        density[mask] = 1.0
        return density

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        # ray_bundle, batch = self.datamanager.next_train(step)
        # model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        # metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        # loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        # sample positions
        pos = torch.rand((self.config.datamanager.train_num_rays_per_batch, 3), device=self.device)
        density = self.true_density(pos)
        model_outputs = self._model(pos)  # train distributed data parallel model if world_size > 1
        loss_dict = {'loss': torch.nn.functional.mse_loss(model_outputs, density)}

        # return model_outputs, loss_dict, metrics_dict
        return model_outputs, loss_dict, None