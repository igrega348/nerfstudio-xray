"""
X-ray renderer for rendering attenuation along rays.
"""
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from jaxtyping import Float, Int
from nerfstudio.utils import colors
from nerfstudio.cameras.rays import RaySamples
from torch import Tensor, nn

BackgroundColor = Union[Literal["random", "last_sample", "black", "white", "trainable"], Float[Tensor, "3"], Float[Tensor, "*bs 3"]]
BACKGROUND_COLOR_OVERRIDE: Optional[Float[Tensor, "3"]] = None

class AttenuationRenderer(nn.Module):
    """X-ray rendering.

    Args:
        background_color: Background color as RGB. Uses white if None.
    """

    def __init__(
        self, 
        background_color: BackgroundColor = "white",
        background_trainable: bool = False,
    ) -> None:
        super().__init__()
        if background_trainable:
            assert not isinstance(background_color, str)
            background_color: BackgroundColor = torch.tensor(background_color)
            self.background_color = nn.Parameter(background_color, requires_grad=True)
        else:
            if isinstance(background_color, str):
                self.background_color: BackgroundColor = background_color
            else:
                self.background_color: BackgroundColor = torch.tensor(background_color)

    @classmethod
    def forward(
        cls,
        densities: Float[Tensor, "*batch num_samples 1"],
        ray_samples: RaySamples,
        background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*batch 1"]:

        delta_density = ray_samples.deltas * densities
        acc = torch.sum(delta_density, dim=-2)            
        attenuation = torch.exp(-acc) 
        attenuation = torch.nan_to_num(attenuation)
        return attenuation
    
    @classmethod
    def get_background_color(
        cls, background_color: BackgroundColor, shape: Tuple[int, ...], device: torch.device
    ) -> Union[Float[Tensor, "3"], Float[Tensor, "*bs 3"]]:
        """Returns the RGB background color for a specified background color.
        Note:
            This function CANNOT be called for background_color being either "last_sample" or "random".

        Args:
            background_color: The background color specification. If a string is provided, it must be a valid color name.
            shape: Shape of the output tensor.
            device: Device on which to create the tensor.

        Returns:
            Background color as RGB.
        """
        assert background_color not in {"last_sample", "random"}
        assert shape[-1] == 3, "Background color must be RGB."
        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if isinstance(background_color, str) and background_color in colors.COLORS_DICT:
            background_color = colors.COLORS_DICT[background_color]
        assert isinstance(background_color, Tensor)

        # Ensure correct shape
        return background_color.expand(shape).to(device)
    
    def blend_background(
        self,
        image: Tensor,
        background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Blends the background color into the image if image is RGBA.
        Otherwise no blending is performed (we assume opacity of 1).

        Args:
            image: RGB/RGBA per pixel.
            opacity: Alpha opacity per pixel.
            background_color: Background color.

        Returns:
            Blended RGB.
        """
        if image.size(-1) < 4:
            return image

        rgb, opacity = image[..., :3], image[..., 3:]
        if background_color is None:
            background_color = self.background_color
            if background_color in {"last_sample", "random"}:
                background_color = "black"
        background_color = self.get_background_color(background_color, shape=rgb.shape, device=rgb.device)
        assert isinstance(background_color, torch.Tensor)
        return rgb * opacity + background_color.to(rgb.device) * (1 - opacity)
    
    def blend_background_for_loss_computation(
        self,
        pred_image: Tensor,
        pred_accumulation: Tensor,
        gt_image: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Blends a background color into the ground truth and predicted image for
        loss computation.

        Args:
            gt_image: The ground truth image.
            pred_image: The predicted RGB values (without background blending).
            pred_accumulation: The predicted opacity/ accumulation.
        Returns:
            A tuple of the predicted and ground truth RGB values.
        """
        background_color = self.background_color
        if background_color == "last_sample":
            background_color = "black"  # No background blending for GT
        elif background_color == "random":
            background_color = torch.rand_like(pred_image)
            pred_image = pred_image + background_color * (1.0 - pred_accumulation)
        gt_image = self.blend_background(gt_image, background_color=background_color)
        return pred_image, gt_image