# %%
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import tyro

from nerf_bspline.deformation_fields import BsplineTemporalDeformationField3d, BsplineTemporalDeformationField1d, NeuralPhiX
# %%
ckpt = Path('./neural_xray/outputs/kel_tens/nerf_bspline/2024-05-20_223703/nerfstudio_models')
ckpt = next(iter(ckpt.glob('*ckpt')))
ckpt_data = torch.load(ckpt)
pipeline_data = ckpt_data['pipeline']
sel_dict = {}
for key in pipeline_data.keys():
    if 'deformation' in key:
        print(key)
        k = key.split('deformation_field.')[-1]
        sel_dict[k] = pipeline_data[key]
# NN Bspline field
df = BsplineTemporalDeformationField3d(phi_x=None, support_range=[(0,1),(0,1),(0,1)], support_outside=True, num_control_points=(8,8,8))
df.load_state_dict(sel_dict)
# %%
fig, ax = plt.subplots(1,3,sharey=True, figsize=(12,3))

z = torch.linspace(0,1,100)
x = 0.5*torch.ones_like(z)
y = 0.5*torch.ones_like(z)
positions = torch.stack([x,y,z], dim=1)
times = torch.linspace(0,1,5).view(-1,1)
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
ckpt = Path('./neural_xray/outputs/kel_tens/nerf_bspline/2024-05-20_212036/nerfstudio_models')
ckpt = next(iter(ckpt.glob('*ckpt')))
ckpt_data = torch.load(ckpt)
pipeline_data = ckpt_data['pipeline']
sel_dict = {}
for key in pipeline_data.keys():
    if 'deformation' in key:
        print(key)
        k = key.split('deformation_field.')[-1]
        sel_dict[k] = pipeline_data[key]
# NN Bspline field
df = BsplineTemporalDeformationField1d(phi_x=None, support_range=(0,1), support_outside=True, num_control_points=4)
df.load_state_dict(sel_dict)
# %%
fig, ax = plt.subplots(1,3,sharey=True, figsize=(12,3))

z = torch.linspace(0,1,100)
x = 0.5*torch.ones_like(z)
y = 0.5*torch.ones_like(z)
positions = torch.stack([x,y,z], dim=1)
times = torch.linspace(0,1,5).view(-1,1)
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