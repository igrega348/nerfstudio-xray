from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml


class Object:
    @staticmethod
    def from_yaml(d: dict):
        if d["type"] == "sphere":
            return Sphere(torch.tensor(d["center"]), d["radius"], d["rho"])
        if d["type"] == "cube":
            return Cube(torch.tensor(d["center"]), d["side"], d["rho"])
        if d["type"] == "cylinder":
            return Cylinder(torch.tensor(d["p0"]), torch.tensor(d["p1"]), d["radius"], d["rho"])
        if d["type"] == "object_collection":
            return ObjectCollection(Object.from_yaml(o) for o in d["objects"])
        raise ValueError(f"Unknown object type: {d['type']}")

class ObjectCollection:
    def __init__(self, objects: Iterable[Object]):
        self.objects = list(objects)

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
        r2 = torch.sum((pos - self.center)**2, dim=-1)
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
        min_corner = self.center - half_side
        max_corner = self.center + half_side
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
        v = self.p1 - self.p0
        v = v / torch.norm(v)
        p0 = self.p0
        p1 = self.p1
        p = pos
        a = torch.sum((p - p0) * v, dim=-1)
        b = torch.sum((p - p1) * v, dim=-1)
        mask_inside = (a > 0) & (b < 0)
        d = torch.norm((p - p0) - a[..., None] * v, dim=-1)
        rho = pos.new_zeros(d.size())
        rho[mask_inside & (d < self.radius)] = self.rho
        return rho