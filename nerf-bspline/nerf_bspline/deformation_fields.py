from typing import Callable, Dict, Iterable, Optional, Tuple, Union, List

import numpy as np
import torch
from torch import Tensor


class NeuralPhiX(torch.nn.Module):
    def __init__(self, num_control_points: int = 4, depth: int = 3, width: int = 10):
        super().__init__()
        self.W = torch.nn.Sequential(torch.nn.Linear(1, width), torch.nn.SELU())
        for _ in range(depth-1):
            self.W.append(torch.nn.Linear(width, width))
            self.W.append(torch.nn.SELU())
        lin = torch.nn.Linear(width, num_control_points, bias=False)
        torch.nn.init.xavier_uniform_(lin.weight, gain=1e-1)
        self.W.append(lin)

    def forward(self, x):
        return self.W(x)
    
class BSplineField1d(torch.nn.Module):
    """1D B-spline field used for prototyping. Differentiable field now
    """
    @staticmethod
    def bspline(u, i):
        if i == 0:
            return (1 - u)**3 / 6
        elif i == 1:
            return (3*u**3 - 6*u**2 + 4) / 6
        elif i == 2:
            return (-3*u**3 + 3*u**2 + 3*u + 1) / 6
        elif i == 3:
            return u**3 / 6
        
    def __init__(
            self, 
            phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None, 
            support_outside: bool = False, 
            support_range: Optional[Tuple[float,float]] = None,
            num_control_points: Optional[int] = None
        ) -> None:
        super().__init__()
        assert phi_x is not None or num_control_points is not None
        if phi_x is not None:
            assert phi_x.ndim == 1
            assert phi_x.shape[0] > 3
            num_control_points = phi_x.shape[0]
        self.phi_x = phi_x
        if support_range is None:
            # provide support over -1 to 1
            self.dx = 2/(num_control_points-3)
            self.origin = -1 - self.dx
        else:
            support_min, support_max = support_range
            self.dx = (support_max - support_min) / (num_control_points-3)
            self.origin = support_min - self.dx
        self.support_outside = support_outside

    def displacement(self, _t: Tensor, phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None) -> Tensor:
        # # support on -1 to 1
        # _t = 0.5 * (1+_t)*(self.phi_x.shape[0]-3) * self.dx
        # xor for phi_x
        assert (phi_x is None) != (self.phi_x is None)
        if phi_x is None:
            phi_x = self.phi_x
        assert _t.ndim == 1
        t = _t - self.origin - self.dx
        x = _t.new_zeros(t.shape)
        indices = torch.floor(t/self.dx).long() 
        if not self.support_outside:
            invalid = (indices < 0) | (indices >= len(phi_x)-3)
            valid = ~invalid
            x[invalid] = torch.nan
        else:
            valid = _t.new_ones(t.shape, dtype=torch.bool)

        for i in range(4):
            inds_loc = indices[valid] + i
            if self.support_outside:
                inds_loc = torch.clamp(inds_loc, 0, len(phi_x)-1) # support outside the domain
            x[valid] += self.bspline(t[valid]/self.dx - indices[valid], i) * phi_x[inds_loc]

        return x
    
class BsplineDeformationField(torch.nn.Module):
    def __init__(self, phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None, support_outside: bool = False, num_control_points: Optional[int] = None) -> None:
        super().__init__()
        self.bspline_field = BSplineField1d(phi_x, support_outside, num_control_points)
    
    def forward(self, x: Tensor) -> Tensor:
        x[...,2] += self.bspline_field.displacement(x[...,2].view(-1)).view(x.shape[:-1])
        return x
    
class BSplineField3d(torch.nn.Module):
    """Cubic B-spline 3d field.
    """
    
    @staticmethod
    def bspline(u: torch.Tensor, i: int) -> Tensor:
        """B-spline functions.

        Using lru_cache to speed up the computation if 
        the same u and i are used multiple times.

        Args:
            u (torch.Tensor): coordinate in domain
            i (int): index of the B-spline. One of {0,1,2,3}

        Returns:
            torch.Tensor: B-spline weight
        """
        if i == 0:
            return (1 - u)**3 / 6
        elif i == 1:
            return (3*u**3 - 6*u**2 + 4) / 6
        elif i == 2:
            return (-3*u**3 + 3*u**2 + 3*u + 1) / 6
        elif i == 3:
            return u**3 / 6
        else:
            raise ValueError(f"Invalid B-spline index {i}")

    def __init__(
            self, 
            phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None,
            support_outside: bool = False,
            support_range: Optional[List[Tuple[float,float]]] = None,
            num_control_points: Optional[Tuple[int,int,int]] = None
    ) -> None:
        """Set up the B-spline field.

        Args:
            phi_x (Union[torch.Tensor, np.ndarray]): degrees of freedom 
                of the B-spline field in order [dim, nx, ny, nz]
            support_outside (bool, optional): whether to provide support
                for locations outside the control points. Defaults to False.
        """
        super().__init__()
        assert phi_x is not None or num_control_points is not None
        if phi_x is not None:
            assert phi_x.ndim == 4
            assert phi_x.shape[1] > 3 and phi_x.shape[2] > 3 and phi_x.shape[3] > 3
            num_control_points = phi_x.shape[1:]
        self.phi_x = phi_x
        nx,ny,nz = num_control_points
        self.grid_size = np.array([nx, ny, nz])
        if support_range is None:
            # provide support for range -1 to 1 along each dimension
            self.spacing = 2 / (self.grid_size - 3)
            self.origin = -1 - self.spacing
        else:
            assert len(support_range) == 3 and all(len(r)==2 for r in support_range)
            support_min = np.array([r[0] for r in support_range])
            support_max = np.array([r[1] for r in support_range])
            self.spacing = (support_max - support_min) / (self.grid_size - 3)
            self.origin = support_min - self.spacing
        self.support_outside = support_outside

    def __repr__(self) -> str:
        f = self
        return f"BSplineField(phi_x={f.grid_size}, origin={f.origin}, spacing={f.spacing})\nfull support on {f.origin + f.spacing} to {f.origin + f.spacing*(f.grid_size-2)}\n"

    def displacement(
            self, x: Tensor, y: Tensor, z: Tensor, i: int, phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None
    ) -> torch.Tensor:
        """Displacement at points x,y,z in the direction i.

        We implement support for locations beyond control points.

        Args:
            x (torch.Tensor): x-coordinates. Can be 1d or meshgrid.
            y (torch.Tensor): y-coordinates. -"-
            z (torch.Tensor): z-coordinates. -"-
            i (int): index of the displacement direction. (x=0, y=1, z=2)

        Returns:
            torch.Tensor: displacement
        """
        if phi_x is None:
            phi_x = self.phi_x
        dx, dy, dz = self.spacing
        u = (x - self.origin[0] - dx)/dx
        v = (y - self.origin[1] - dy)/dy
        w = (z - self.origin[2] - dz)/dz
        ix = torch.floor(u).long()
        iy = torch.floor(v).long()
        iz = torch.floor(w).long()
        if not self.support_outside:
            ix_nan = (ix < 0) | (ix >= self.grid_size[0]-3)
            iy_nan = (iy < 0) | (iy >= self.grid_size[1]-3)
            iz_nan = (iz < 0) | (iz >= self.grid_size[2]-3)
        u = u - ix
        v = v - iy
        w = w - iz
        T = torch.zeros_like(x, dtype=torch.float32)
        for l in range(4):
            ix_loc = torch.clamp(ix + l, 0, self.grid_size[0]-1)
            for m in range(4):
                iy_loc = torch.clamp(iy + m, 0, self.grid_size[1]-1)
                for n in range(4):
                    iz_loc = torch.clamp(iz + n, 0, self.grid_size[2]-1)
                    T += self.bspline(u, l) * self.bspline(v, m) * self.bspline(w, n) * phi_x[i, ix_loc, iy_loc, iz_loc]
        if not self.support_outside:
            T[ix_nan | iy_nan | iz_nan] = torch.nan
        return T
    
class BsplineDeformationField3d(torch.nn.Module):
    def __init__(
            self, 
            phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None, 
            support_outside: bool = False, 
            support_range: Optional[List[Tuple[float,float]]] = None,
            num_control_points: Optional[Tuple[int,int,int]] = None
        ) -> None:
        super().__init__()
        self.bspline_field = BSplineField3d(phi_x, support_outside, support_range, num_control_points)
    
    def forward(self, x: Tensor) -> Tensor:
        # x [ray, nsamples, 3]
        x0, x1, x2 = x[...,0].view(-1), x[...,1].view(-1), x[...,2].view(-1)
        x[...,0] += self.bspline_field.displacement(x0, x1, x2, 0).view(x.shape[:-1])
        x[...,1] += self.bspline_field.displacement(x0, x1, x2, 1).view(x.shape[:-1])
        x[...,2] += self.bspline_field.displacement(x0, x1, x2, 2).view(x.shape[:-1])
        return x
    
class BsplineTemporalDeformationField3d(torch.nn.Module):
    def __init__(
            self, 
            phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]]=None, 
            support_outside: bool = False, 
            support_range: Optional[List[Tuple[float,float]]]=None,
            num_control_points: Optional[Tuple[int,int,int]]=None
        ) -> None:
        super().__init__()
        if phi_x is not None:
            assert phi_x.ndim == 5 # [ntimes, 3, nx, ny, nz]
            num_control_points = phi_x.shape[2:]
            self.phi_x = phi_x
        else:
            assert num_control_points is not None
            self.phi_x = None
            self.weight_nn = NeuralPhiX(3*np.prod(num_control_points), 3, 16)
        self.bspline_field = BSplineField3d(support_outside=support_outside, support_range=support_range, num_control_points=num_control_points)

    def forward(self, positions: Tensor, times: Tensor) -> Tensor:
        # positions, times of shape [ray, nsamples, 3]
        x0, x1, x2 = positions[...,0].view(-1), positions[...,1].view(-1), positions[...,2].view(-1)
        uq_times = torch.unique(times)
        assert len(uq_times)==1
        if self.phi_x is None:
            phi = self.weight_nn(uq_times[0].view(-1,1)).view(3, *self.bspline_field.grid_size)
        else:
            phi = self.phi_x[:uq_times[0]+1].sum(dim=0)
        out = positions.clone()
        out[...,0] += self.bspline_field.displacement(x0, x1, x2, 0, phi_x=phi).view(positions.shape[:-1])
        out[...,1] += self.bspline_field.displacement(x0, x1, x2, 1, phi_x=phi).view(positions.shape[:-1])
        out[...,2] += self.bspline_field.displacement(x0, x1, x2, 2, phi_x=phi).view(positions.shape[:-1])
        return out
    
class BsplineTemporalDeformationField1d(torch.nn.Module):
    def __init__(
            self, 
            phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]]=None, 
            support_outside: bool = False, 
            support_range: Optional[Tuple[float,float]]=None,
            num_control_points: Optional[int]=None
        ) -> None:
        super().__init__()
        if phi_x is not None:
            assert phi_x.ndim == 2 # [ntimes, nz]
            num_control_points = phi_x.shape[-1]
            self.phi_x = phi_x
        else:
            assert num_control_points is not None
            self.phi_x = None
            self.weight_nn = NeuralPhiX(num_control_points, 3, 16)
        self.bspline_field = BSplineField1d(support_outside=support_outside, support_range=support_range, num_control_points=num_control_points)

    def forward(self, positions: Tensor, times: Tensor) -> Tensor:
        # positions, times of shape [ray, nsamples, 3]
        x0, x1, x2 = positions[...,0].view(-1), positions[...,1].view(-1), positions[...,2].view(-1)
        uq_times = torch.unique(times)
        assert len(uq_times)==1
        if self.phi_x is None:
            phi = self.weight_nn(uq_times[0].view(-1,1)).view(-1)
        else:
            phi = self.phi_x[:uq_times[0]+1].sum(dim=0)
        out = positions.clone()
        out[...,2] += self.bspline_field.displacement(x2, phi_x=phi).view(positions.shape[:-1])
        return out

class IdentityDeformationField(torch.nn.Module):
    def forward(self, x: Tensor, times: Optional[Tensor]) -> Tensor:
        return x
    
class AffineTemporalDeformationField(torch.nn.Module):
    def __init__(self, A: Union[Tensor, torch.nn.parameter.Parameter]) -> None:
        super().__init__()
        self.A = A
    
    def forward(self, x: Tensor, times: Optional[Tensor]) -> Tensor:
        uq_times = torch.unique(times)
        assert len(uq_times)==1
        try:
            _A = self.A[:uq_times[0]+1].sum(dim=0)
            return x + x @ _A.T
        except TypeError:
            return x
    
class ComposedDeformationField(torch.nn.Module):
    def __init__(self, deformation_fields: Iterable[Callable]) -> None:
        super().__init__()
        self.deformation_fields = deformation_fields
    
    def forward(self, x: Tensor, times: Optional[Tensor]) -> Tensor:
        for field in self.deformation_fields:
            x = field(x, times)
        return x