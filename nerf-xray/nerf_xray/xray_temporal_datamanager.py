"""
Template DataManager
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import (Dict, Generic, Literal, Optional, Sequence, Tuple, Type,
                    Union, Iterable)

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE
from typing_extensions import TypeVar
from nerfstudio.data.datasets.base_dataset import InputDataset

from .objects import Object
from .xray_temporal_dataloaders import CacheDataloader, FixedIndicesEvalDataloader, RandIndicesEvalDataloader


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
    time_proposal_steps: Optional[int] = 0
    """Until this time prefer early timestamps"""
    max_images_per_timestamp: Optional[int] = 1<<26
    """Maximum number of images per timestamp"""
    save_sampling_locations: bool = False
    """save sampling locations"""
    visualize_sampling_locations: bool = False
    """visualize sampling locations"""


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
            self.config.max_images_per_timestamp
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
            num_images_to_sample_from=-1, # temporary hack for using just 1 timestamp
            num_times_to_repeat_images=-1, # temporary hack for using just 1 timestamp
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
        indices_to_sample_from = self.timestamp_sampler.choose_indices(self.train_dataset.metadata['image_timestamps'], step)
        self.train_image_dataloader.indices_to_sample_from = indices_to_sample_from

        image_batch = next(self.iter_train_image_dataloader) 
        assert isinstance(image_batch, dict)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        
        # Save sampling locations if needed
        if self.config.save_sampling_locations:
            self._save_sampling_locations(ray_indices, step)
            
        # Visualize sampling locations if needed
        if self.config.visualize_sampling_locations:
            self._visualize_sampling_locations(ray_indices, image_batch, step)
            
        return ray_bundle, batch

    def _save_sampling_locations(self, indices: torch.Tensor, step: int):
        """Save sampled pixel locations to a file.
        
        Args:
            indices: Tensor of shape (N, 3) containing (image_idx, y, x) coordinates
            step: Current training step
        """
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path("sampling_locations")
        output_dir.mkdir(exist_ok=True)
        
        # Get unique image indices
        unique_image_indices = torch.unique(indices[:, 0])
        
        # Create a dictionary to store data for each image
        sampling_data = {}
        
        for img_idx in unique_image_indices:
            # Get pixels sampled from this image
            img_indices = indices[indices[:, 0] == img_idx]
            
            # Get the image filename from the dataset
            img_filename = self.train_dataset.image_filenames[int(img_idx)]
            
            # Store data for this image
            sampling_data[int(img_idx)] = {
                'filename': img_filename,
                'indices': img_indices.cpu().numpy()
            }
        
        # Save the data
        np.save(output_dir / f"sampling_locations_step_{step}.npy", sampling_data)

    def _visualize_sampling_locations(self, indices: torch.Tensor, image_batch: Dict, step: int):
        """Visualize sampled pixel locations on the images.
        
        Args:
            indices: Tensor of shape (N, 3) containing (image_idx, y, x) coordinates
            image_batch: Batch of images
            step: Current training step
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path("sampling_visualizations")
        output_dir.mkdir(exist_ok=True)
        
        # Get unique image indices
        unique_image_indices = torch.unique(indices[:, 0])
        
        for img_idx in unique_image_indices:
            # Get pixels sampled from this image
            img_indices = indices[indices[:, 0] == img_idx]
            
            # Get the image
            img = image_batch["image"][int(img_idx)].cpu().numpy()
            
            # Create figure
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            
            # Plot sampled points
            plt.scatter(img_indices[:, 2], img_indices[:, 1], c='red', s=1, alpha=0.5)
            
            # Save plot
            plt.savefig(output_dir / f"sampling_step_{step}_image_{int(img_idx)}.png")
            plt.close()

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
    
    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        indices_to_sample_from = self.timestamp_sampler.choose_indices(self.eval_dataset.metadata['image_timestamps'], step)
        self.eval_image_dataloader.indices_to_sample_from = indices_to_sample_from

        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        indices_to_sample_from = self.timestamp_sampler.choose_indices(self.eval_dataset.metadata['image_timestamps'], step)
        self.fixed_indices_eval_dataloader.image_indices = indices_to_sample_from
        for camera, batch in self.fixed_indices_eval_dataloader:
        # for camera, batch in self.eval_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
        raise ValueError("No more eval images")

class TimestampSampler:
    def __init__(self, time_proposal_steps: int, max_images_per_timestamp: int, max_unique_timestamps: int = 5) -> None:
        self.time_proposal_steps = time_proposal_steps
        self.max_images_per_timestamp = max_images_per_timestamp
        self.max_unique_timestamps = max_unique_timestamps
    
    def choose_indices(self, image_timestamps: Iterable, step: int) -> int:
        max_timestamp = max(image_timestamps)
        min_timestamp = min(image_timestamps)
        if self.time_proposal_steps > 0:
            cutoff_timestamp = min(max_timestamp, min_timestamp + step/self.time_proposal_steps*(max_timestamp-min_timestamp))
        else:
            cutoff_timestamp = max_timestamp
        indices = np.arange(len(image_timestamps))
        np.random.shuffle(indices)
        indices_to_sample_from = []
        num_per_timestamp = {}
        for idx in indices:
            t = image_timestamps[idx]
            if len(num_per_timestamp)>=self.max_unique_timestamps: break
            if t <= cutoff_timestamp and num_per_timestamp.get(t, 0) < self.max_images_per_timestamp:
                indices_to_sample_from.append(idx)
                num_per_timestamp[t] = num_per_timestamp.get(t, 0) + 1
        return indices_to_sample_from