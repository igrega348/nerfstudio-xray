from abc import abstractmethod
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import torch
import yaml


class Object:
    max_density: float = 1.0

    @abstractmethod
    def density(self, pos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def t_density(self, pos: torch.Tensor) -> torch.Tensor:
        return self.density(self.transform_pos(pos))

    def transform_pos(self, pos: torch.Tensor) -> torch.Tensor:
        # Default input Object coordinates are -1 to 1
        # Default nerfstudio is 0 to 1
        return pos*2 - 1
    
    @staticmethod
    def from_file(path: Union[str, Path]) -> "Object":
        path = Path(path)
        if path.suffix == ".yaml":
            return Object.from_yaml(path)
        if path.suffix in [".npy",".npz"]:
            return VoxelGrid.from_file(path)
        raise ValueError(f"Unknown file type: {path.suffix}")

    @staticmethod
    def from_yaml(path: Path) -> "Object":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return Object.from_dict(d)
    
    @staticmethod
    def from_dict(d: dict) -> "Object":
        if d["type"] == "sphere":
            return Sphere(torch.tensor(d["center"]), d["radius"], d["rho"])
        if d["type"] == "cube":
            return Cube(torch.tensor(d["center"]), d["side"], d["rho"])
        if d["type"] == "cylinder":
            return Cylinder(torch.tensor(d["p0"]), torch.tensor(d["p1"]), d["radius"], d["rho"])
        if d["type"] == "object_collection":
            return ObjectCollection(Object.from_dict(o) for o in d["objects"])
        if d['type'] == 'unit_cell':
            return UnitCell(
                struts=Object.from_dict(d['struts']), 
                min_lims=torch.tensor([d['xmin'], d['ymin'], d['zmin']]), 
                max_lims=torch.tensor([d['xmax'], d['ymax'], d['zmax']])
            )
        if d['type'] == 'tessellated_obj_coll':
            return TessellatedObjColl(
                uc=Object.from_dict(d['uc']),
                min_lims=torch.tensor([d['xmin'], d['ymin'], d['zmin']]),
                max_lims=torch.tensor([d['xmax'], d['ymax'], d['zmax']])
            )
        if d['type'] == 'box':
            return Box(d['center'], d['sides'], d['rho'])
        if d['type'] == 'parallelepiped':
            return Parallelepiped(
                torch.tensor(d['origin']),
                torch.tensor(d['v1']),
                torch.tensor(d['v2']),
                torch.tensor(d['v3']),
                d['rho']
            )
        raise ValueError(f"Unknown object type: {d['type']}")

class ObjectCollection(Object):
    def __init__(self, objects: Iterable[Object]):
        self.objects = list(objects)
        self.max_density = max(obj.max_density for obj in self.objects)

    def density(self, pos: torch.Tensor) -> torch.Tensor:
        # sum densities of all objects clipped between 0 and 1
        rho = torch.stack([obj.density(pos) for obj in self.objects], dim=-1)
        return torch.clamp(torch.sum(rho, dim=-1), 0, 1)

    def __iter__(self):
        return iter(self.objects)
    
    def __len__(self):
        return len(self.objects)
    
class Sphere(Object):
    def __init__(self, center: torch.Tensor, radius: float, rho: float):
        self.radius = radius
        self.center = center
        self.rho = rho
        self.max_density = rho

    def density(self, pos: torch.Tensor):
        r2 = torch.sum((pos - self.center.to(pos.device))**2, dim=-1)
        rho = pos.new_zeros(r2.size())
        mask_inside = r2 < self.radius**2
        rho[mask_inside] = self.rho
        return rho
    
class Cube(Object):
    def __init__(self, center: torch.Tensor, side: float, rho: float):
        self.side = side
        self.center = center
        self.rho = rho
        self.max_density = rho

    def density(self, pos: torch.Tensor):
        half_side = self.side / 2
        min_corner = self.center.to(pos.device) - half_side
        max_corner = self.center.to(pos.device) + half_side
        mask_inside = torch.all(pos >= min_corner, dim=-1) & torch.all(pos <= max_corner, dim=-1)
        rho = pos.new_zeros(mask_inside.size())
        rho[mask_inside] = self.rho
        return rho
    
class Cylinder(Object):
    def __init__(self, p0: torch.Tensor, p1: torch.Tensor, radius: float, rho: float):
        self.radius = radius
        self.p0 = p0
        self.p1 = p1
        self.rho = rho
        self.max_density = rho

    def density(self, pos: torch.Tensor):
        p0 = self.p0.to(pos.device)
        p1 = self.p1.to(pos.device)
        v = p1 - p0
        v = v / torch.norm(v)
        p = pos
        a = torch.sum((p - p0) * v, dim=-1)
        b = torch.sum((p - p1) * v, dim=-1)
        mask_inside = (a > 0) & (b < 0)
        d = torch.norm((p - p0) - a[..., None] * v, dim=-1)
        rho = pos.new_zeros(d.size())
        rho[mask_inside & (d < self.radius)] = self.rho
        return rho
    
class UnitCell(Object):
    def __init__(self, struts: ObjectCollection, min_lims: torch.Tensor, max_lims: torch.Tensor):
        self.struts = struts
        self.min_lims = min_lims.reshape(1,3)
        self.max_lims = max_lims.reshape(1,3)
        self.max_density = struts.max_density

    def density(self, pos: torch.Tensor):
        assert pos.ndim == 2 and pos.size(1) == 3, f"Expected (N, 3) tensor, got {pos.size()}"
        rho = pos.new_zeros(pos.size(0))
        mask = torch.all(pos >= self.min_lims.to(pos), dim=-1) & torch.all(pos <= self.max_lims.to(pos), dim=-1)
        rho[mask] = self.struts.density(pos[mask])
        return rho
    
class TessellatedObjColl(Object):
    """
    Represents a tessellated object collection with a unit cell and bounding box.
    """
    def __init__(self, uc: UnitCell, min_lims: torch.Tensor, max_lims: torch.Tensor):
        self.uc = uc  # UnitCell object
        self.min_lims = min_lims.reshape(1,3)  # Minimum limits of the bounding box
        self.max_lims = max_lims.reshape(1,3)  # Maximum limits of the bounding box
        self.max_density = uc.max_density

    def remap(self, pos: torch.Tensor):
        dd = (self.uc.max_lims - self.uc.min_lims).to(pos) # (1,3)
        pos = pos - torch.floor((pos - self.uc.min_lims.to(pos)) / dd) * dd 
        return pos

    def density(self, pos: torch.Tensor):
        assert pos.ndim == 2 and pos.size(1) == 3, f"Expected (N, 3) tensor, got {pos.size()}"
        rho = pos.new_zeros(pos.size(0))
        mask = torch.all(pos >= self.min_lims.to(pos), dim=-1) & torch.all(pos <= self.max_lims.to(pos), dim=-1)

        pos_remapped = self.remap(pos[mask])
        # Remap point to unit cell space
        rho[mask] = self.uc.density(pos_remapped)
        return rho
    
class Box(Object):
    def __init__(self, center: List[float], sides: List[float], rho: float):
        self.center = center
        self.sides = sides
        self.rho = rho
        self.max_density = rho
        
    def density(self, pos: torch.Tensor):
        x = torch.abs(pos[:, 0] - self.center[0])
        y = torch.abs(pos[:, 1] - self.center[1])
        z = torch.abs(pos[:, 2] - self.center[2])
        mask_inside = (x < 0.5*self.sides[0]) & (y < 0.5*self.sides[1]) & (z < 0.5*self.sides[2])
        rho = pos.new_zeros(mask_inside.size())
        rho[mask_inside] = self.rho
        return rho

class Parallelepiped(Object):
    def __init__(self, origin: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor, rho: float):
        self.origin = origin
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.rho = rho
        self.cube = Cube(torch.tensor([0.5,0.5,0.5]), 1, rho)
        self.inv = torch.inverse(torch.stack([v1, v2, v3], dim=1))
        self.max_density = rho

    def density(self, pos: torch.Tensor):
        pos = pos - self.origin.to(pos)
        pos = torch.einsum('ij,...j->...i', self.inv.to(pos), pos)
        return self.cube.density(pos)

class VoxelGrid(Object):
    def __init__(self, rho: torch.Tensor):
        self.rho = rho
        self.max_density = rho.max().item()

    @staticmethod
    def from_file(path: Union[str, Path]) -> "VoxelGrid":
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path) as data:
                vol = np.swapaxes(data["vol"], 0, 2)
                rho = torch.tensor(vol, dtype=torch.float32)
        else:
            assert path.suffix == ".npy", f"Expected .npy file, got {path.suffix}"
            vol = np.load(path).swapaxes(0, 2)
            rho = torch.tensor(vol, dtype=torch.float32)
        return VoxelGrid(rho)

    def density(self, pos: torch.Tensor):
        # expect pos in -1 to 1 range
        # use grid_sample
        if pos.ndim==2:
            _pos = pos.view(1,1,1,-1,3)
        elif pos.ndim==3:
            _pos = pos.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {pos.ndim}D")
        rho = torch.nn.functional.grid_sample(self.rho.unsqueeze(0).unsqueeze(0).to(pos), _pos, align_corners=True)
        return rho.reshape(*pos.shape[:-1], 1)