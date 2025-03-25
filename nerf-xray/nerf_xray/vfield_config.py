"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import \
    NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import (AdamOptimizerConfig,
                                          RAdamOptimizerConfig,
                                          AdamWOptimizerConfig)
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerf_xray.xray_temporal_datamanager import XrayTemporalDataManagerConfig
from nerf_xray.template_dataparser import TemplateDataParserConfig
from nerf_xray.vfield_model import VfieldModelConfig
from nerf_xray.vfield_pipeline import VfieldPipelineConfig
from nerf_xray.deformation_fields import BsplineTemporalIntegratedVelocityField3dConfig, DeformationFieldConfig
from nerf_xray.utils import ColdRestartLinearDecaySchedulerConfig

xray_vfield = MethodSpecification(
    config=TrainerConfig(
        method_name="xray_vfield", 
        steps_per_eval_batch=10,
        steps_per_eval_all_images=1000000,
        steps_per_eval_image=100,
        steps_per_save=5000,
        max_num_iterations=501,
        mixed_precision=True,
        load_scheduler=False,
        load_optimizer=False,
        pipeline=VfieldPipelineConfig(
            datamanager=XrayTemporalDataManagerConfig(
                dataparser=TemplateDataParserConfig(
                    auto_scale_poses=False,
                    center_method='none',
                    downscale_factors={'train': 1, 'val': 8, 'test': 8},
                    eval_mode='filename+modulo',
                    includes_time=True,
                ),
                train_num_rays_per_batch=512,
                eval_num_rays_per_batch=512,
                max_images_per_timestamp=3,
                time_proposal_steps=500,
            ),
            model=VfieldModelConfig(
                use_appearance_embedding=False,
                background_color='white',
                flat_field_value=0.02,
                flat_field_trainable=True,
                eval_num_rays_per_chunk=512,
                num_nerf_samples_per_ray=1024,
                disable_scene_contraction=True,
                train_density_field=False,
                train_deformation_field=True,
                deformation_field=BsplineTemporalIntegratedVelocityField3dConfig(
                    support_range=[(-1,1),(-1,1),(-1,1)],
                    num_control_points=(4,4,4),
                    timedelta=0.05,
                ),
                field_weighing=DeformationFieldConfig(),
                train_field_weighing=False,
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
                "optimizer": AdamWOptimizerConfig(lr=1e-4, eps=1e-15, weight_decay=1e-8),
                "scheduler": ColdRestartLinearDecaySchedulerConfig(warmup_steps=50, lr_final=3e-5, max_steps=5000),
            },
            "field_weighing": {
                "optimizer": AdamWOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=3e-4, max_steps=10000),
            },
            # "fields": { 
            #     "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            # },
            "flat_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=50000),
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
