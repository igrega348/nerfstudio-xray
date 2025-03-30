"""
Template DataManager
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple, Type, Union, Optional

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)
from nerfstudio.utils.rich_utils import CONSOLE

from .objects import Object


@dataclass
class XrayDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: XrayDataManager)
    volume_grid_file: Optional[Path] = None
    """load volume grid into object"""
    save_sampling_locations: bool = False
    """save sampling locations"""
    visualize_sampling_locations: bool = False
    """visualize sampling locations"""
    save_sampling_locations_every_n_steps: int = 10
    """save sampling locations every n steps"""
    visualize_sampling_locations_every_n_steps: int = 100
    """visualize sampling locations every n steps"""

class XrayDataManager(VanillaDataManager):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: XrayDataManagerConfig

    def __init__(
        self,
        config: XrayDataManagerConfig,
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
            if config.volume_grid_file is not None:
                assert config.volume_grid_file.exists(), f"Volume grid file {config.volume_grid_file} does not exist."
                self.object = Object.from_file(config.volume_grid_file)
            else:
                try:
                    yaml_files = list(folder.glob("*.yaml"))
                    assert len(yaml_files) == 1, f"Expected 1 yaml file, got {len(yaml_files)}"
                    self.object = Object.from_yaml(yaml_files[0])
                except AssertionError:
                    self.object = None
                    print("Did not find a yaml file in the data folder. Volumetric loss cannot be computed.")

        if self.config.save_sampling_locations:
            self.sampling_locations_parent = Path('./').absolute()
            print(f"Saving sampling locations to {self.sampling_locations_parent}")

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        
        # Save sampling locations if needed
        if self.config.save_sampling_locations and step % self.config.save_sampling_locations_every_n_steps == 0:
            self._save_sampling_locations(ray_indices, step)
            
        # Visualize sampling locations if needed
        if self.config.visualize_sampling_locations and step % self.config.visualize_sampling_locations_every_n_steps == 0:
            self._visualize_sampling_locations(ray_indices, image_batch, step)
            
        return ray_bundle, batch

    def _save_sampling_locations(self, indices: torch.Tensor, step: int):
        """Save sampled pixel locations to a file.
        
        Args:
            indices: Tensor of shape (N, 3) containing (image_idx, y, x) coordinates
            step: Current training step
        """
        import numpy as np
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = self.sampling_locations_parent / "sampling_locations"
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
            sampling_data['image_'+str(int(img_idx))] = {
                'filename': str(img_filename),
                'indices': img_indices.cpu().numpy()
            }
        
        # Save the data in compressed npz format
        np.savez_compressed(output_dir / f"sampling_locations_step_{step}.npz", **sampling_data)

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
        output_dir = self.sampling_locations_parent / "sampling_visualizations"
        output_dir.mkdir(exist_ok=True)
        
        # Get unique image indices
        unique_image_indices = torch.unique(indices[:, 0])
        
        for img_idx in unique_image_indices:
            # Get pixels sampled from this image
            img_indices = indices[indices[:, 0] == img_idx]
            
            # Get the image
            img = image_batch["image"][int(img_idx)].cpu().numpy()
            
            # Create figure
            plt.figure(figsize=(5.12, 5.12), dpi=100)  # 5.12 inches at 100 DPI = 512x512 pixels
            plt.imshow(img)
            
            # Plot sampled points
            plt.scatter(img_indices[:, 2], img_indices[:, 1], c='red', s=2, alpha=0.5)
            
            # Remove padding and save plot
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(output_dir / f"sampling_step_{step}_image_{int(img_idx)}.png", 
                       bbox_inches='tight', 
                       pad_inches=0,
                       dpi=100)
            plt.close()
