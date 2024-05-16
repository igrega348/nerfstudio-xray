# %%
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from deformation_fields import AffineTemporalDeformationField, BsplineTemporalDeformationField3d
# %%
ckpt = Path('../../../nerfstudio/outputs/bspline/nerf_bspline/2024-05-14_230731/nerfstudio_models/step-000001999.ckpt')
# %%
ckpt_data = torch.load(ckpt)
pipeline_data = ckpt_data['pipeline']
# %%
for key in pipeline_data.keys():
    if 'deformation' in key:
        print(key)
        break
# %%
df = AffineTemporalDeformationField(pipeline_data[key])
# %%
A = df.A.cpu().numpy()
with np.printoptions(precision=2, suppress=True):
    print(np.cumsum(A, axis=0))
# %%
plt.plot(np.cumsum(A, axis=0)[:,2,2])
# %%
for f in Path('../../../nerfstudio/outputs/bspline/nerf_bspline').glob('deformation_field*.pt'):
    step = int(f.stem.split('_')[-1])
    ckpt_data = torch.load(f)
    plt.plot(np.cumsum(ckpt_data['A'].cpu().numpy(), axis=0)[:,2,2], label=f'{step}')
plt.legend()
plt.show()

# %% Bspline field
for f in Path('../../../nerfstudio/outputs/bspline/nerf_bspline').glob('deformation_field*.pt'):
    step = int(f.stem.split('_')[-1])
    ckpt_data = torch.load(f)
    if 'phi_x' not in ckpt_data:
        continue
    phi_x = ckpt_data['phi_x'].cpu()
    # df = BsplineTemporalDeformationField3d(phi_x)
    # plt.plot(np.cumsum(ckpt_data['A'].cpu().numpy(), axis=0)[:,2,2], label=f'{step}')
    plt.plot(torch.cumsum(phi_x[:,2,1,1,1], dim=0), label=f'{step}')
plt.legend()
plt.ylabel('phi_x')
plt.xlabel('time')
plt.show()
# %% NN Bspline field
fig, ax = plt.subplots(1,3,sharey=True, figsize=(12,3))
df = BsplineTemporalDeformationField3d(phi_x=None, support_outside=True, num_control_points=(4,4,4))
for f in Path('../../../nerfstudio/outputs/bspline/nerf_bspline').glob('deformation_field*.pt'):
    step = int(f.stem.split('_')[-1])
    ckpt_data = torch.load(f)
    if 'weight_nn.W.0.weight' not in ckpt_data:
        continue
    df.load_state_dict(ckpt_data)
    t = torch.linspace(0, 10, 100)    
    positions = torch.tensor([0,0,0.5]).view(1,1,3)
    times = torch.tensor([0,1,2,3,4,5,6,7,8,9,10])
    us = []
    for t in times:
        with torch.no_grad():
            x = df(positions.clone(), t)
        u = x - positions
        us.append(u.cpu().squeeze().numpy())
    ax[0].plot(times, np.array(us)[:,0], label=f'{step}')
    ax[1].plot(times, np.array(us)[:,1], label=f'{step}')
    ax[2].plot(times, np.array(us)[:,2], label=f'{step}')
for s,a in zip('xyz',ax):
    a.legend(title=f'$u_{s}$')
fig.tight_layout()
plt.show()
# %%
