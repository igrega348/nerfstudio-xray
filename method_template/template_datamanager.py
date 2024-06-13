"""
Template DataManager
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple, Type, Union

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)
from nerfstudio.utils.rich_utils import CONSOLE

from .objects import Object


@dataclass
class TemplateDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: TemplateDataManager)
    train_split_fraction: float = 1.0


class TemplateDataManager(VanillaDataManager):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: TemplateDataManagerConfig

    def __init__(
        self,
        config: TemplateDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self.object = None
        if config.data is not None:
            folder = config.data.parent if config.data.suffix=='.json' else config.data
            try:
                yaml_files = list(folder.glob("*.yaml"))
                assert len(yaml_files) == 1, f"Expected 1 yaml file, got {len(yaml_files)}"
                self.object = Object.from_yaml(yaml_files[0])
            except AssertionError:
                self.object = None
                string = "Did not find a yaml file in the data folder. Volumetric loss cannot be computed."
                CONSOLE.print(f"[bold yellow]{string}")

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch
