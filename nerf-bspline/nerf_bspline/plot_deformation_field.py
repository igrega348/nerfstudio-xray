# %%
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import tyro

from nerf_bspline.deformation_fields import BsplineTemporalDeformationField3d, BsplineTemporalDeformationField1d, NeuralPhiX
# %%
folds = [
    'A_rich_start',
    'B_rich_start_end',
    'C_volsup_t0',
    'D_volsup'
]
fd = folds[3]
ckpt = Path(f'./neural_xray/outputs_fields/f2_2/nerf_bspline/{fd}/nerfstudio_models')
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
df = BsplineTemporalDeformationField3d(phi_x=None, support_range=[(0,1),(0,1),(0,1)], support_outside=True, num_control_points=(6,6,6))
df.load_state_dict(sel_dict)
# %% Field 1
z_norm = lambda z: z*2 - 1
defz = lambda pos,t: pos[:,2] - 0.1*t / (1 + torch.exp(-(pos[:,2]-0.1*t)/0.3))
# defz = lambda z,t: z-0.1*t / (1+torch.exp(-(z-0.1*t)/0.3))
z_back = lambda z: z/2 + 0.5
# %% Field 2
pos_norm = lambda pos: pos*2 - 1
defz = lambda pos,t: pos[:,2] - 0.1*t / (1 + torch.exp(-(pos[:,2]-0.1*t-pos[:,0]*pos[:,1])/0.3))
z_back = lambda z: z/2 + 0.5
# %%
err = 0
N = 10000
for t in torch.linspace(0,1,5).view(-1,1):
    positions = torch.rand(N,3)
    with torch.no_grad():
        x = df(positions.clone(), t)
    z = positions[:,2]
    z_def = z_back(defz(pos_norm(positions), t))
    uz = (z_def - z)
    ureal = torch.zeros(N, 3)
    ureal[:,2] = uz
    u = x - positions
    loss = torch.nn.functional.mse_loss(u, ureal)
    err += loss.item()
print(np.mean(err))
# %%
fig, ax = plt.subplots(1,3,sharey=True, figsize=(12,3))

z = torch.linspace(0,1,100)
x = 0.25*torch.ones_like(z)
y = 0.25*torch.ones_like(z)
positions = torch.stack([x,y,z], dim=1)
times = torch.linspace(0,1,5).view(-1,1)
xs = [0.25,0,-0.25]
for i,t in enumerate(times):
    z = torch.linspace(0,1,100)
    y = 0.25*torch.ones_like(z)
    for j in range(3):
        x = xs[j]*torch.ones_like(z)
        positions = torch.stack([x,y,z], dim=1)
        with torch.no_grad():
            x = df(positions.clone(), t)
        z_def = z_back(defz(pos_norm(positions), t))
        uz = (z_def - z).cpu().squeeze().numpy()
        u = x - positions
        u = u.cpu().squeeze().numpy()
        # ax[0].plot(z.numpy(), u[:,0], label=f'{t.item():.2f}')
        # ax[1].plot(z.numpy(), u[:,1], label=f'{t.item():.2f}')
        ax[j].plot(z.numpy(), u[:,2], label=f'{t.item():.2f}', color=f'C{i}')
        ax[j].plot(z.numpy(), uz, ls='--', color=f'C{i}')
    
for s,a in zip('xyz',ax):
    a.set_title(f'$u_z$')
    a.legend(title=f'$t$')
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