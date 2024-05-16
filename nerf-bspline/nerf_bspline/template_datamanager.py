"""
Template DataManager
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, Generic, Literal, Sequence, Tuple, Type, Union

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)
from typing_extensions import TypeVar

from .objects import Object
from .template_dataset import TemplateDataset


@dataclass
class TemplateDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: TemplateDataManager)
    train_split_fraction: float = 1.0
    time_proposal_steps: int = -1


TDataset = TypeVar("TDataset", bound=TemplateDataset, default=TemplateDataset)

class TemplateDataManager(VanillaDataManager, Generic[TDataset]):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: TemplateDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

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
        self.timestamp_sampler = TimestampSampler(
            self.config.time_proposal_steps, 
            np.unique(self.train_dataset.metadata['image_timestamps'])
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
                print("Did not find a yaml file in the data folder. Volumetric loss cannot be computed.")

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        return TemplateDataset

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader) 
        assert isinstance(image_batch, dict)
        # pick just one timestamp for the batch
        timestamp = self.timestamp_sampler.sample_timestep(step)
        chosen_idx = [i for i, idx in enumerate(image_batch['image_idx']) if self.train_dataset.metadata['image_timestamps'][idx]==timestamp]
        image_batch = {k: v[chosen_idx] for k, v in image_batch.items()}
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

class TimestampSampler:
    def __init__(self, time_proposal_steps: int, uq_timesteps: Sequence[int]) -> None:
        self.time_proposal_steps = time_proposal_steps
        self.uq_timesteps = uq_timesteps
    
    def sample_timestep(self, step: int) -> int:
        t = np.linspace(0, 1, len(self.uq_timesteps))
        if step >= self.time_proposal_steps:
            return np.random.choice(self.uq_timesteps, 1)[0]
        
        L = 10*(1-step/self.time_proposal_steps)

        p = np.exp(-L*t)
        return np.random.choice(self.uq_timesteps, 1, p=p/p.sum())[0]