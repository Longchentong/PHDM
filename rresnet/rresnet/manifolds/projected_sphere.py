import math

import torch

from .manifold import Manifold
from ..utils import EPS


class ProjectedSphere(Manifold):
    def __init__(self, c=None):
        super().__init__()
        self.c = c
        self.max_norm = 1e6
        self.max_angle = math.pi / 2 - 1e-4

    def _curvature(self, ref):
        if self.c is None:
            return torch.tensor(1.0, dtype=ref.dtype, device=ref.device)
        if torch.is_tensor(self.c):
            return self.c.to(dtype=ref.dtype, device=ref.device)
        return torch.tensor(self.c, dtype=ref.dtype, device=ref.device)

    def lambda_x(self, x, keepdim=False):
        c = self._curvature(x)
        return 2 / (1 + c * x.pow(2).sum(dim=-1, keepdim=keepdim)).clamp_min(EPS[x.dtype])

    def _scaled_tan(self, x, ref):
        c = self._curvature(ref).clamp_min(EPS[ref.dtype])
        sqrt_c = torch.sqrt(c)
        return torch.tan(torch.clamp(sqrt_c * x, min=-self.max_angle, max=self.max_angle)) / sqrt_c

    def _scaled_atan(self, x, ref):
        c = self._curvature(ref).clamp_min(EPS[ref.dtype])
        sqrt_c = torch.sqrt(c)
        return torch.atan(sqrt_c * x) / sqrt_c

    def inner(self, x, u, v, keepdim=False):
        return self.lambda_x(x, keepdim=True) ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def proju(self, x, u):
        return u

    def projx(self, x):
        norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
        clipped = x * (self.max_norm / norm)
        return torch.where((norm > self.max_norm).expand_as(x), clipped, x)

    def mobius_addition(self, x, y):
        c = self._curvature(x)
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 - 2 * c * xy - c * y2) * x + (1 + c * x2) * y
        denom = 1 - 2 * c * xy + c.pow(2) * x2 * y2
        return num / denom.clamp_min(EPS[x.dtype])

    def exp(self, x, u):
        u_norm = u.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
        angle = 0.5 * self.lambda_x(x, keepdim=True) * u_norm
        second_term = self._scaled_tan(angle, x) * u / u_norm
        return self.projx(self.mobius_addition(x, second_term))

    def log(self, x, y):
        sub = self.mobius_addition(-x, y)
        sub_norm = sub.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
        lam = self.lambda_x(x, keepdim=True)
        return 2 / lam * self._scaled_atan(sub_norm, x) * sub / sub_norm

    def rand(self, *shape, out=None):
        return self.projx(torch.randn(*shape))

    def randvec(self, x, norm=1):
        y = torch.randn_like(x)
        return y * norm / y.norm(dim=-1, keepdim=True).clamp_min(EPS[y.dtype])

    def __str__(self):
        return "Projected Sphere"
