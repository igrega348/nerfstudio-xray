from abc import abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml


class Object:
    @abstractmethod
    def density(self, pos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def t_density(self, pos: torch.Tensor) -> torch.Tensor:
        return self.density(self.transform_pos(pos))

    def transform_pos(self, pos: torch.Tensor) -> torch.Tensor:
        # Default input Object coordinates are -1 to 1
        # Default nerfstudio is 0 to 1
        return (pos + 1) / 2

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
        raise ValueError(f"Unknown object type: {d['type']}")

class ObjectCollection(Object):
    def __init__(self, objects: Iterable[Object]):
        self.objects = list(objects)

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