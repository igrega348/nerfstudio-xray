from typing import Callable, Union, Iterable, Tuple, Optional, Dict
import torch
from torch import Tensor

class NeuralPhiX(torch.nn.Module):
    def __init__(self, num_control_points=4, depth=3, width=10):
        super().__init__()
        self.W = torch.nn.Sequential(torch.nn.Linear(1, width), torch.nn.SELU())
        for _ in range(depth-1):
            self.W.append(torch.nn.Linear(width, width))
            self.W.append(torch.nn.SELU())
        self.W.append(torch.nn.Linear(width, num_control_points))
        

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
        
    def __init__(self, phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None, support_outside: bool = False, num_control_points: Optional[int] = None) -> None:
        super().__init__()
        assert phi_x is not None or num_control_points is not None
        if phi_x is not None:
            assert phi_x.ndim == 1
            assert phi_x.shape[0] > 3
            num_control_points = phi_x.shape[0]
        self.phi_x = phi_x
        # provide support over -1 to 1
        self.dx = 2/(num_control_points-3)
        self.origin = -1 - self.dx
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