# nerfstudio-xray

Nerfstudio methods for X-ray tomographic reconstruction, including static canonical volumes, 4D deformation fields, and spatio-temporal field mixing.

## Methods

| Entry point | Class | Use |
|---|---|---|
| `nerf_xray` | `CanonicalPipelineConfig` | Single-timestep X-ray NeRF (canonical volume) |
| `xray_vfield` | `VfieldPipelineConfig` | 4D velocity field trained from canonical endpoints |
| `spatiotemporal_mix` | `VfieldPipelineConfig` | Spatio-temporal mixing of forward/backward canonical fields |

## Dataparsers

| Entry point | Use |
|---|---|
| `multi-camera-dataparser` | Multi-angle X-ray data; supports `time` field for 4D |
| `xray-dataparser` | Single-camera X-ray data |

## Installation

```bash
# From repo root — install nerfstudio first, then this package:
pip install -e neural_xray/nerfstudio
pip install -e neural_xray/nerfstudio-xray/nerf-xray
```

## Training

### Canonical (static) volume

```bash
python nerfstudio/nerfstudio/scripts/train.py nerf_xray \
  --data path/to/transforms_00.json \
  --output_dir outputs/ \
  --pipeline.volumetric_supervision False \
  --pipeline.model.flat_field_trainable False \
  --max-num-iterations 2001 \
  --timestamp canonical_F \
  multi-camera-dataparser
```

### Velocity field (4D deformation)

First combine forward and backward canonical checkpoints:

```bash
python nerfstudio-xray/nerf-xray/nerf_xray/combine_forward_backward_checkpoints.py \
  --fwd_ckpt outputs/.../canonical_F/nerfstudio_models/step-000002000.ckpt \
  --bwd_ckpt outputs/.../canonical_B/nerfstudio_models/step-000002000.ckpt \
  --out_fn   outputs/.../xray_vfield/vel_6/nerfstudio_models/step-000002000.ckpt
```

Then train the velocity field:

```bash
python nerfstudio/nerfstudio/scripts/train.py xray_vfield \
  --data path/to/transforms_00_to_20.json \
  --output_dir outputs/ \
  --load-checkpoint outputs/.../vel_6/nerfstudio_models/step-000002000.ckpt \
  --pipeline.model.deformation_field.num_control_points 6 6 6 \
  --pipeline.model.deformation_field.timedelta 0.1 \
  --pipeline.model.deformation_field.displacement_method matrix \
  --max-num-iterations 2000 \
  --timestamp vel_6 \
  multi-camera-dataparser
```

Refine to higher resolution:

```bash
python nerfstudio-xray/nerf-xray/nerf_xray/refine_vfield.py \
  --load-config outputs/.../xray_vfield/vel_6/config.yml \
  --new-resolution 12 \
  --new-nn-width 20 \
  --out-path outputs/.../xray_vfield/vel_12/nerfstudio_models/step-000004000.ckpt
```

### Spatio-temporal mixing

Trains a learned mixing field on top of frozen forward/backward canonical volumes. The mixer learns where and when to blend the two canonical fields, producing smooth time-varying reconstructions without a full deformation field.

Requires a combined checkpoint from `combine_forward_backward_checkpoints.py` as the starting point (same as `xray_vfield`). Only the mixer weights are trained; canonical density fields are frozen.

```bash
python nerfstudio/nerfstudio/scripts/train.py spatiotemporal_mix \
  --data path/to/transforms_00_to_20.json \
  --output_dir outputs/ \
  --load-checkpoint outputs/.../vel_6/nerfstudio_models/step-000002000.ckpt \
  --max-num-iterations 501 \
  --timestamp stmix \
  multi-camera-dataparser
```

## Volumetric supervision

When `--pipeline.volumetric_supervision True`, the pipeline computes a loss between the NeRF density field and a rasterised version of a known YAML volume. Pass the YAML via `--pipeline.datamanager.volume_grid_file`. Use `--pipeline.volumetric_supervision_coefficient 1e-4`; too high causes the network to ignore X-ray data.

## `multi-camera-dataparser` key options

| Option | Effect |
|---|---|
| `includes_time=True` | Pass the `time` field from JSON to the model (required for `xray_vfield`) |
| `auto_scale_poses=False` | Do not normalise camera positions (X-ray geometry is metric) |
| `center_method='none'` | Do not recentre the scene |
| `eval_mode='filename+modulo'` | Split by `train_*` / `eval_*` filename prefix |
| `orientation_method='none'` | Do not auto-orient poses (default; preserves world frame) |
