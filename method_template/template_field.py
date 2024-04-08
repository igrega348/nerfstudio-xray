"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Dict, Literal, Optional, Tuple, Union

import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field  # for custom Field
from nerfstudio.fields.base_field import get_normalized_directions
from nerfstudio.fields.nerfacto_field import \
    NerfactoField  # for subclassing NerfactoField
from torch import Tensor


class TemplateNerfField(NerfactoField):
    """Template Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    aabb: Tensor

    # def __init__(
    #     self,
    #     aabb: Tensor,
    #     num_images: int,
    # ) -> None:
    #     super().__init__(aabb=aabb, num_images=num_images)

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        average_init_density: float = 1.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        volumetric_training: bool = False,
    ) -> None:
        super().__init__(
            aabb=aabb,
            num_images=num_images,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            geo_feat_dim=geo_feat_dim,
            num_levels=num_levels,
            base_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            num_layers_color=num_layers_color,
            num_layers_transient=num_layers_transient,
            features_per_level=features_per_level,
            hidden_dim_color=hidden_dim_color,
            hidden_dim_transient=hidden_dim_transient,
            appearance_embedding_dim=appearance_embedding_dim,
            transient_embedding_dim=transient_embedding_dim,
            use_transient_embedding=use_transient_embedding,
            use_semantics=use_semantics,
            num_semantic_classes=num_semantic_classes,
            pass_semantic_gradients=pass_semantic_gradients,
            use_pred_normals=use_pred_normals,
            use_average_appearance_embedding=use_average_appearance_embedding,
            spatial_distortion=spatial_distortion,
            average_init_density=average_init_density,
            implementation=implementation,
        )
        # REMOVE RGB HEAD
        del self.mlp_head
        self.volumetric_training = volumetric_training

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # h = torch.cat(
        #     [
        #         d,
        #         density_embedding.view(-1, self.geo_feat_dim),
        #     ]
        #     + (
        #         [embedded_appearance.view(-1, self.appearance_embedding_dim)] if embedded_appearance is not None else []
        #     ),
        #     dim=-1,
        # )
        # rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        rgb = torch.zeros((*outputs_shape, 3)).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
    
    def get_density(self, ray_samples: Union[RaySamples, Tensor]) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.training and self.volumetric_training:
            positions: Tensor
            positions = ray_samples
            h_to_shape = list(positions.shape[:-1])
        else:
            if self.spatial_distortion is not None:
                positions = ray_samples.frustums.get_positions()
                positions = self.spatial_distortion(positions)
                positions = (positions + 2.0) / 4.0
            else:
                positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
            h_to_shape = ray_samples.frustums.shape
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*h_to_shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        # Offset this to more negative
        density_before_activation = density_before_activation - 2.0
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        # density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        # try sigmoid activation
        density = self.average_init_density * torch.sigmoid(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    
    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if self.training and self.volumetric_training:
            density, density_embedding = self.get_density(ray_samples)
            field_outputs = {}
            field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

            return field_outputs
        else:
            if compute_normals:
                with torch.enable_grad():
                    density, density_embedding = self.get_density(ray_samples)
            else:
                density, density_embedding = self.get_density(ray_samples)

            field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
            field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

            if compute_normals:
                with torch.enable_grad():
                    normals = self.get_normals()
                field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
            return field_outputs