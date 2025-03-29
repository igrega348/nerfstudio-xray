from typing import Callable, Dict, Iterable, Optional, Tuple, Union, List, Type, Literal
from dataclasses import dataclass, field
from abc import abstractmethod
from math import ceil

import numpy as np
import torch
from torch import Tensor

from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.configs.base_config import InstantiateConfig
from nerf_xray.deformation_fields import BsplineTemporalDeformationField3d, BsplineTemporalDeformationField3dConfig, BSplineField1d

@dataclass
class FieldMixerConfig(InstantiateConfig):
    """Configuration for deformation field instantiation"""

    _target: Type = field(default_factory=lambda: ConstantMixer)
    """target class to instantiate"""

class FieldMixer(torch.nn.Module):
    """Field mixer abstract class"""

    config: FieldMixerConfig

    @abstractmethod
    def __init__(
        self,
        config: FieldMixerConfig,
        **kwargs,
    ) -> None:
        """Initialize the field mixer

        Args:
            config: configuration for the deformation field
        """
    
    @abstractmethod
    def get_mixing_coefficient(self, positions: Tensor, times: Tensor) -> Tensor:
        """Get the mixing coefficient

        Args:
            positions: positions of the points
            times: times of the points
        """

@dataclass 
class ConstantMixerConfig(FieldMixerConfig):
    """Configuration for constant field mixer instantiation"""

    _target: Type = field(default_factory=lambda: ConstantMixer)
    """target class to instantiate"""
    alpha: float = 0.5
    """Alpha value for the constant field mixer"""

class ConstantMixer(torch.nn.Module):
    """Constant field mixer"""

    config: ConstantMixerConfig

    def __init__(self, config: ConstantMixerConfig) -> None:
        super().__init__()
        self.config = config
        self.register_parameter('alpha', torch.nn.parameter.Parameter(torch.tensor(config.alpha)))
    
    def get_mixing_coefficient(self, positions: Tensor, times: Tensor) -> Tensor:
        return self.alpha
   
@dataclass
class SpatioTemporalMixerConfig(FieldMixerConfig):
    """Configuration for deformation field instantiation"""

    _target: Type = field(default_factory=lambda: SpatioTemporalMixer)
    """target class to instantiate"""
    support_outside: bool = True
    """Support outside the control points"""
    support_range: Optional[List[Tuple[float,float]]] = None
    """Support range for the deformation field"""
    num_control_points: Optional[Tuple[int,int,int]] = None
    """Number of control points in each dimension"""
    weight_nn_width: int = 16
    """Width of the neural network for the weights"""
    weight_nn_bias: bool = True
    """Whether to use bias in the neural network for the weights"""
    weight_nn_init_gain: float = 1
    """Initialization gain for the weights of the final layer"""
    displacement_method: Literal['neighborhood','matrix'] = 'matrix'
    """Whether to use neighborhood calculation of bsplines or assemble full matrix""" 
    num_components: int = 1
    """Number of components in the deformation field"""

class SpatioTemporalMixer(FieldMixer):

    config: SpatioTemporalMixerConfig

    def __init__(
            self, 
            config: SpatioTemporalMixerConfig,
        ) -> None:
        super().__init__(config)
        self.config = config
        
        # Create the B-spline temporal deformation field
        deformation_config = BsplineTemporalDeformationField3dConfig(
            support_outside=config.support_outside,
            support_range=config.support_range,
            num_control_points=config.num_control_points,
            weight_nn_width=config.weight_nn_width,
            weight_nn_bias=config.weight_nn_bias,
            weight_nn_gain=config.weight_nn_init_gain,
            displacement_method=config.displacement_method,
            num_components=config.num_components
        )
        self.deformation_field = deformation_config.setup()
    
    def get_mixing_coefficient(self, positions: Tensor, times: Union[Tensor, float]) -> Tensor:
        """Get the mixing coefficient using the neural network.
        
        Args:
            positions: positions of shape [ray, nsamples, 3]
            times: times of shape [ray, nsamples, 1] or float
            
        Returns:
            Tensor: mixing coefficients of shape [ray, nsamples, 1]
        """
        if isinstance(times, float):
            times = torch.ones_like(positions[...,0:1]) * times
        alpha = self.deformation_field.displacement(positions, times)
        return torch.sigmoid(alpha)
    
@dataclass
class TemporalMixerConfig(FieldMixerConfig):
    """Configuration for temporal field mixer instantiation"""

    _target: Type = field(default_factory=lambda: TemporalMixer)
    """target class to instantiate"""
    num_control_points: int = 10
    """Number of control points"""

class TemporalMixer(FieldMixer):
    """Temporal mixer"""

    config: TemporalMixerConfig

    def __init__(self, config: TemporalMixerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.mixing_field = BSplineField1d(
            torch.nn.parameter.Parameter(torch.zeros(10)), 
            support_outside=True, 
            support_range=(0,1) # time range
        )
    
    def get_mixing_coefficient(self, positions: Tensor, times: Tensor) -> Tensor:
        alpha = self.mixing_field(times)
        return torch.sigmoid(alpha)