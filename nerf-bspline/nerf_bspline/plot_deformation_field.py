# %%
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import tyro

from deformation_fields import AffineTemporalDeformationField, BSplineField3d, BsplineTemporalDeformationField3d, BSplineField1d
# %% Test support argument in field
phi = torch.tensor([0,0,0,1,0,0,0])
df = BSplineField1d(phi, support_range=(-0.8,1), support_outside=True)
t = torch.linspace(-2, 2, 100)
y = df.displacement(t)
plt.plot(t, y)
# %% Test in 3d
phi = torch.zeros(1,5,5,5)
phi[0,2,2,2] = 1
df = BSplineField3d(phi, support_range=[(0,1),(0,1),(0,1)], support_outside=True)
z = torch.linspace(-4, 4, 100)
for a in [0.1, 0.5, 0.99]:
    x = torch.ones_like(z)*a
    y = torch.ones_like(z)*a
    u = df.displacement(x,y,z,0)
    plt.plot(z, u)
# %%
ckpt = Path('./neural_xray/outputs/kel_def/nerf_bspline/2024-05-20_170844/nerfstudio_models/step-000000999.ckpt')
# %%
ckpt_data = torch.load(ckpt)
pipeline_data = ckpt_data['pipeline']
# %%
sel_dict = {}
for key in pipeline_data.keys():
    if 'deformation' in key:
        print(key)
        k = key.split('deformation_field.')[-1]
        sel_dict[k] = pipeline_data[key]
# %% NN Bspline field
df = BsplineTemporalDeformationField3d(phi_x=None, support_outside=True, num_control_points=(4,4,4))
df.load_state_dict(sel_dict)
# %%
fig, ax = plt.subplots(1,3,sharey=True, figsize=(12,3))

z = torch.linspace(0,1,100)
x = torch.zeros_like(z)
y = torch.zeros_like(z)
positions = torch.stack([x,y,z], dim=1)
times = torch.linspace(0,1,11).view(-1,1)
for t in times:
    with torch.no_grad():
        x = df(positions.clone(), t)
    u = x - positions
    u = u.cpu().squeeze().numpy()
    ax[0].plot(z.numpy(), u[:,0], label=f'{t.item():.2f}')
    ax[1].plot(z.numpy(), u[:,1], label=f'{t.item():.2f}')
    ax[2].plot(z.numpy(), u[:,2], label=f'{t.item():.2f}')
for s,a in zip('xyz',ax):
    a.legend(title=f'$u_{s}$')
fig.tight_layout()
plt.show()
# %%
