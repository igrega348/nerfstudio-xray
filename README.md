# nerfstudio-xray

Nerfstudio methods for X-ray tomographic reconstruction, including static canonical volumes, 4D deformation fields, and spatio-temporal field mixing.

## Methods

| Entry point | Class | Use |
|---|---|---|
| `nerf_xray` | `CanonicalPipelineConfig` | Single-timestep X-ray NeRF (canonical volume) |
| `xray_vfield` | `VfieldPipelineConfig` | 4D velocity field trained from canonical endpoints |
| `spatiotemporal_mix` | `VfieldPipelineConfig` | Learned spatio-temporal blending of two frozen canonical fields |
| `nerf_def_xray` | `XrayDefConfig` | X-ray NeRF with a direct deformation field (experimental; no training example provided) |

## Dataparsers

| Entry point | Use |
|---|---|
| `multi-camera-dataparser` | Multi-angle X-ray data; supports `time` field for 4D methods |
| `xray-dataparser` | Single-camera X-ray data |

## Installation

All commands below assume the repo root as working directory.

```bash
pip install -e neural_xray/nerfstudio
pip install -e neural_xray/nerfstudio-xray/nerf-xray
```

## Training

### Canonical (static) volume

```bash
python neural_xray/nerfstudio/nerfstudio/scripts/train.py nerf_xray \
  --data path/to/transforms_00.json \
  --output_dir outputs/ \
  --pipeline.datamanager.volume_grid_file path/to/volume_00.yaml \
  --pipeline.volumetric_supervision False \
  --pipeline.model.flat_field_trainable False \
  --pipeline.datamanager.train_num_rays_per_batch 2048 \
  --max-num-iterations 2001 \
  --timestamp canonical_F \
  multi-camera-dataparser
```

Omit `--pipeline.datamanager.volume_grid_file` if no ground-truth volume is available. Set `--pipeline.volumetric_supervision True` to enable the volumetric loss (requires the file).

### Velocity field (4D deformation)

First combine forward and backward canonical checkpoints:

```bash
python neural_xray/nerfstudio-xray/nerf-xray/nerf_xray/combine_forward_backward_checkpoints.py \
  --fwd_ckpt outputs/.../canonical_F/nerfstudio_models/step-000002000.ckpt \
  --bwd_ckpt outputs/.../canonical_B/nerfstudio_models/step-000002000.ckpt \
  --out_fn   outputs/.../xray_vfield/vel_6/nerfstudio_models/step-000002000.ckpt
```

Then train the velocity field (note: code default is `num_control_points=(4,4,4)` and `timedelta=0.05`; the values below are recommended overrides):

```bash
python neural_xray/nerfstudio/nerfstudio/scripts/train.py xray_vfield \
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

`xray_vfield` sets `includes_time=True` in its registered config, so no CLI override is needed.

To add volumetric supervision, use the temporal datamanager flags:
- `--pipeline.datamanager.init_volume_grid_file path/to/volume_00.yaml`
- `--pipeline.datamanager.final_volume_grid_file path/to/volume_20.yaml`

Refine to higher resolution:

```bash
python neural_xray/nerfstudio-xray/nerf-xray/nerf_xray/refine_vfield.py \
  --load-config outputs/.../xray_vfield/vel_6/config.yml \
  --new-resolution 12 \
  --new-nn-width 20 \
  --out-path outputs/.../xray_vfield/vel_12/nerfstudio_models/step-000004000.ckpt
```

### Spatio-temporal mixing

Trains a learned mixer that blends two frozen canonical density fields by position and time. No velocity field or deformation is trained — `train_density_field` and `train_deformation_field` are both `False`; only the `field_weighing` module is updated.

Requires a combined canonical checkpoint (same `combine_forward_backward_checkpoints.py` step as `xray_vfield`):

```bash
python neural_xray/nerfstudio/nerfstudio/scripts/train.py spatiotemporal_mix \
  --data path/to/transforms_00_to_20.json \
  --output_dir outputs/ \
  --load-checkpoint outputs/.../combined/nerfstudio_models/step-000002000.ckpt \
  --max-num-iterations 501 \
  --timestamp stmix \
  multi-camera-dataparser
```

`spatiotemporal_mix` sets `includes_time=True` in its registered config, so no CLI override is needed.

## Data format

### transforms.json

Standard nerfstudio JSON. Each frame requires:

```json
{
  "camera_angle_x": 0.698,
  "fl_x": 686.87, "fl_y": 686.87,
  "w": 500, "h": 500,
  "cx": 250, "cy": 250,
  "frames": [
    {
      "file_path": "images_00/train_00.png",
      "time": 0.0,
      "transform_matrix": [[...], [...], [...], [0,0,0,1]]
    }
  ]
}
```

`transform_matrix` is a 4×4 camera-to-world matrix: `M[:3,:3]` is the rotation, `M[:3,3]` is the camera origin. The `time` field (normalised float in `[0,1]`) is required when `includes_time=True`; it is ignored otherwise.

### Image file naming and train/eval split

With the default `eval_mode='filename+modulo'`, the split is inferred from the filename stem:

- `train_NN.png` — training frame; `NN` is a zero-padded integer
- `eval_NN.png` — evaluation frame

Both prefixes must be present. The dataparser raises an error if a file has neither. The "modulo" part refers to the `imin`/`imax`/`istep` options that further filter which training frames are included (e.g. to sub-sample timesteps). Note: `indices` is an alternative to the range filter — if set, it overrides `imin`/`imax`/`istep` entirely.

### Image downscaling

If `downscale_factors` is not set, the dataparser picks the largest power-of-2 factor such that the max image dimension stays at or below 2000 px. When a factor `D` is chosen, it looks for pre-downscaled images in `<parent>_D/` next to the originals (e.g. `images_00_2/` for 2× downscale of `images_00/`). These folders must be pre-generated; the dataparser does not resize on the fly.

Recommended: set `--downscale-factors.val 2 --downscale-factors.test 2` and leave train at `1` to speed up eval without affecting training.

### Volumetric supervision

When `--pipeline.volumetric_supervision True`, the pipeline adds a loss between the NeRF's predicted density and a reference volume loaded from `--pipeline.datamanager.volume_grid_file`.

Accepted file formats:

| Format | Notes |
|---|---|
| `.yaml` / `.json` | Scene primitives (sphere/cylinder/object_collection). Coordinates in world units. |
| `.npy` / `.npz` | Voxel grid. For `.npz`, array key must be `"vol"`. |

**Voxel grid axis convention:** `.npy` / `.npz` files must be stored with shape `(Nx, Ny, Nz)` — axis 0 = x, axis 1 = y, axis 2 = z. On load, axes 0 and 2 are swapped internally before passing to PyTorch `grid_sample`. Raw binary (`.raw`) files follow the opposite convention: `(Nz, Ny, Nx)` — z-major, x-fastest — which `raw_to_npy.py` handles by reshaping as `(Nz, Ny, Nx)` then swapping to `(Nx, Ny, Nz)` before saving the `.npz`.

**Supervision coefficient:** the code default is `0.005`; for typical use set `--pipeline.volumetric_supervision_coefficient 1e-4` explicitly. The loss is `coefficient * (1 - normed_correlation)` between predicted and reference density, so the effective scale depends on the normalisation — not a raw MSE. Too high causes the network to ignore projection data.

For `xray_vfield`, pass separate files for the two endpoints:
- `--pipeline.datamanager.init_volume_grid_file` — volume at t=0
- `--pipeline.datamanager.final_volume_grid_file` — volume at t=1

## `multi-camera-dataparser` key options

The values below are what the registered method entry points (`nerf_xray`, `xray_vfield`, `spatiotemporal_mix`) set. The class-level defaults for `auto_scale_poses` and `center_method` are `True` and `'poses'` respectively — if you construct `multi-camera-dataparser` outside a registered method, you must override these explicitly or camera positions will be silently normalised/recentred, which breaks X-ray geometry.

| Option | Value in method configs | Effect |
|---|---|---|
| `includes_time` | `False` (`True` in `xray_vfield` / `spatiotemporal_mix`) | Pass the `time` field from JSON to the model |
| `auto_scale_poses` | `False` | Do not normalise camera positions (X-ray geometry is metric) |
| `center_method` | `'none'` | Do not recentre the scene |
| `orientation_method` | `'none'` | Do not auto-orient poses (preserves world frame) |
| `eval_mode` | `'filename+modulo'` | Split by `train_*` / `eval_*` filename prefix |
| `downscale_factors` | `{'val': 8, 'test': 8}` | Per-split downscale; override with e.g. `--downscale-factors.val 2` |
| `imin` / `imax` / `istep` | `0 / ∞ / 1` | Filter training frame indices (mutually exclusive with `indices`) |
| `indices` | `None` | Explicit list of training frame indices; overrides `imin`/`imax`/`istep` |
