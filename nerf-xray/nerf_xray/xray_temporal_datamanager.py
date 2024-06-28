"""
Template DataManager
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import (Dict, Generic, Literal, Optional, Sequence, Tuple, Type,
                    Union)

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE
from typing_extensions import TypeVar
from nerfstudio.data.datasets.base_dataset import InputDataset

from .objects import Object
from .xray_temporal_dataloaders import CacheDataloader


@dataclass
class XrayTemporalDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: XrayTemporalDataManager)
    init_volume_grid_file: Optional[Path] = None
    """load initial volume grid into object"""
    final_volume_grid_file: Optional[Path] = None
    """load final volume grid into object"""
    time_proposal_steps: Optional[int] = None
    """Until this time prefer early timestamps"""


TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)

class XrayTemporalDataManager(VanillaDataManager, Generic[TDataset]):
    """Xray Temporal DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: XrayTemporalDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(
        self,
        config: XrayTemporalDataManagerConfig,
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
        self.final_object = None
        if config.data is not None:
            folder = config.data.parent if config.data.suffix=='.json' else config.data
            if config.init_volume_grid_file is not None:
                assert config.init_volume_grid_file.exists(), f"Volume grid file {config.init_volume_grid_file} does not exist."
                self.object = Object.from_file(config.init_volume_grid_file)
            else:
                try:
                    yaml_files = list(folder.glob("*.yaml"))
                    assert len(yaml_files) == 1, f"Expected 1 yaml file, got {len(yaml_files)}"
                    self.object = Object.from_yaml(yaml_files[0])
                except AssertionError:
                    self.object = None
                    print("Did not find a yaml file in the data folder. Volumetric loss cannot be computed.")
            if config.final_volume_grid_file is not None:
                assert config.final_volume_grid_file.exists(), f"Volume grid file {config.final_volume_grid_file} does not exist."
                self.final_object = Object.from_file(config.final_volume_grid_file)

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        return InputDataset
    
    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=2,
            num_times_to_repeat_images=0,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        timestamp = self.timestamp_sampler.sample_timestep(step)
        indices_to_sample_from = [idx for idx, t in enumerate(self.train_dataset.metadata['image_timestamps']) if t==timestamp]
        self.train_image_dataloader.indices_to_sample_from = indices_to_sample_from

        image_batch = next(self.iter_train_image_dataloader) 
        assert isinstance(image_batch, dict)
        # pick just one timestamp for the batch
        # t_max = step / self.config.time_proposal_steps if self.config.time_proposal_steps else 1e10
        # if t_max < 1:
        # chosen_idx = [i for i, idx in enumerate(image_batch['image_idx']) if self.train_dataset.metadata['image_timestamps'][idx]==timestamp]
        # image_batch = {k: v[chosen_idx] for k, v in image_batch.items()}
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