"""
X-ray renderer for rendering attenuation along rays.
"""
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from jaxtyping import Float, Int
from nerfstudio.cameras.rays import RaySamples
from torch import Tensor, nn


class AttenuationRenderer(nn.Module):

    @classmethod
    def forward(
        cls,
        densities: Float[Tensor, "*batch num_samples 1"],
        ray_samples: RaySamples,
    ) -> Float[Tensor, "*batch 1"]:

        delta_density = ray_samples.deltas * densities
        acc = torch.sum(delta_density, dim=-2)
        attenuation = torch.exp(-acc) 
        attenuation = torch.nan_to_num(attenuation)
        return attenuation