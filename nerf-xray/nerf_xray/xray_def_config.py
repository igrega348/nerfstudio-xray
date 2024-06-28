"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import \
    NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import (AdamOptimizerConfig,
                                          RAdamOptimizerConfig)
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerf_xray.xray_temporal_datamanager import XrayTemporalDataManagerConfig
from nerf_xray.template_dataparser import TemplateDataParserConfig
from nerf_xray.template_model import TemplateModelConfig
from nerf_xray.template_pipeline import TemplatePipelineConfig

nerf_xray = MethodSpecification(
    config=TrainerConfig(
        method_name="nerf_def_xray", 
        steps_per_eval_batch=10,
        steps_per_eval_all_images=10000,
        steps_per_eval_image=100,
        steps_per_save=5000,
        max_num_iterations=1001,
        mixed_precision=True,
        pipeline=TemplatePipelineConfig(
            datamanager=XrayTemporalDataManagerConfig(
                dataparser=TemplateDataParserConfig(
                    auto_scale_poses=False,
                    center_method='none',
                    downscale_factors={'train': 1, 'val': 1, 'test': 1},
                    eval_mode='filename+modulo',
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=TemplateModelConfig(
                use_appearance_embedding=False,
                background_color='white',
                flat_field_value=0.02,
                flat_field_trainable=False,
                eval_num_rays_per_chunk=1 << 15,
                disable_scene_contraction=True,
                train_density_field=False,
                train_deformation_field=True,
            ),
            volumetric_supervision=False,
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "flat_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=5000),
                # "optimizer": AdamOptimizerConfig(lr=1e-11, eps=1e-15),
                # "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-12, max_steps=5000),
            },
        },
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15, 
            camera_frustum_scale=0.5,
            quit_on_train_completion=True,
        ),
        vis="tensorboard",
    ),
    description="Nerfstudio method template.",
)
