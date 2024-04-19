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

from method_template.template_datamanager import TemplateDataManagerConfig
from method_template.template_model import TemplateModelConfig
from method_template.template_pipeline import TemplatePipelineConfig

_volumetric_training = False
method_template = MethodSpecification(
    config=TrainerConfig(
        method_name="method-template",  # TODO: rename to your own model
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=510,
        mixed_precision=True,
        pipeline=TemplatePipelineConfig(
            datamanager=TemplateDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    # # scale_factor=2.0,
                    auto_scale_poses=False,
                    center_method='none'
                    # scene_scale=1.5,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=TemplateModelConfig(
                background_color='white',
                eval_num_rays_per_chunk=1 << 15,
                volumetric_training=_volumetric_training,
                disable_scene_contraction=True,
            ),
            volumetric_training=_volumetric_training,
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
            "camera_opt": {
                # "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                # "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
                "optimizer": AdamOptimizerConfig(lr=1e-11, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-12, max_steps=5000),
            },
        },
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15, 
            camera_frustum_scale=0.5,
            quit_on_train_completion=True,
        ),
        vis="viewer",
    ),
    description="Nerfstudio method template.",
)
