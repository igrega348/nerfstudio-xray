# %%
import torch
from deformation_fields import BSplineField3d
# %%
phi_x = torch.randn(3,4,4,4)
df = BSplineField3d(phi_x=phi_x, support_outside=True)

# %%
x = torch.randn(2,100)
y = torch.randn(2,100)
z = torch.randn(2,100)
dx = df.displacement(x,y,z,0)
dy = df.displacement(x,y,z,1)
dz = df.displacement(x,y,z,2)

# %%
u = df.vectorized_displacement(x,y,z)

# %%
assert torch.allclose(u[...,0], dx) and torch.allclose(u[...,1], dy) and torch.allclose(u[...,2], dz)