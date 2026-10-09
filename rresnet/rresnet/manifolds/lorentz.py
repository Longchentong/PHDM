import torch

from .manifold import Manifold
from ..utils import EPS, arcosh


class Lorentz(Manifold):
    def __init__(self, c=None):
        super().__init__()
        self.c = c
        self.min_norm = 1e-15
        self.max_norm = 1e6
        self.max_scale = 50.0
        self.max_tangent_norm = 10.0
        self.max_spatial_norm = 50.0

    def _curvature(self, ref):
        if self.c is None:
            return torch.tensor(1.0, dtype=ref.dtype, device=ref.device)
        if torch.is_tensor(self.c):
            return self.c.to(dtype=ref.dtype, device=ref.device)
        return torch.tensor(self.c, dtype=ref.dtype, device=ref.device)

    def _radius_sq(self, ref):
        return self._curvature(ref).clamp_min(EPS[ref.dtype]).reciprocal()

    def _radius(self, ref):
        return self._radius_sq(ref).sqrt()

    def minkowski_dot(self, x, y, keepdim=False):
        prod = (x * y).sum(dim=-1, keepdim=keepdim)
        time = 2 * x[..., :1] * y[..., :1] if keepdim else 2 * x[..., 0] * y[..., 0]
        return prod - time

    def inner(self, x, u, v, keepdim=False):
        return self.minkowski_dot(u, v, keepdim=keepdim)

    def proju(self, x, u):
        return u + self.minkowski_dot(x, u, keepdim=True) * x / self._radius_sq(x)

    def projx(self, x):
        radius_sq = self._radius_sq(x)
        spatial = x[..., 1:]
        spatial_norm = spatial.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
        max_spatial = x.new_tensor(self.max_spatial_norm)
        spatial = torch.where(
            (spatial_norm > max_spatial).expand_as(spatial),
            spatial * (max_spatial / spatial_norm),
            spatial,
        )
        spatial_sqnorm = spatial.pow(2).sum(dim=-1, keepdim=True)
        time = torch.sqrt((radius_sq + spatial_sqnorm).clamp_min(EPS[x.dtype]))
        return torch.cat([time, spatial], dim=-1)

    def exp(self, x, u):
        radius = self._radius(x)
        u = self.proju(x, u)
        u_norm = self.norm(x, u, keepdim=True).clamp_min(EPS[x.dtype]).clamp_max(self.max_norm)
        step_scale = torch.clamp(x.new_tensor(self.max_tangent_norm) / u_norm, max=1.0)
        u = u * step_scale
        u_norm = u_norm * step_scale
        scaled = (u_norm / radius).clamp_max(self.max_scale)
        ret = torch.cosh(scaled) * x + radius * torch.sinh(scaled) * u / u_norm
        return torch.where((u_norm > EPS[x.dtype]).expand_as(x), self.projx(ret), x)

    def log(self, x, y):
        radius_sq = self._radius_sq(x)
        alpha = self.minkowski_dot(x, y, keepdim=True) / radius_sq
        alpha = torch.clamp(alpha, max=-(1.0 + EPS[x.dtype]))
        u = y + alpha * x
        u = self.proju(x, u)
        u_norm = self.norm(x, u, keepdim=True).clamp_min(EPS[x.dtype])
        dist = self._radius(x) * arcosh(-alpha)
        ret = dist * u / u_norm
        return torch.where((u_norm > EPS[x.dtype]).expand_as(x), ret, torch.zeros_like(x))

    def rand(self, *shape, out=None):
        return self.projx(torch.randn(*shape))

    def randvec(self, x, norm=1):
        y = self.proju(x, torch.randn_like(x))
        return y * norm / self.norm(x, y, keepdim=True).clamp_min(EPS[y.dtype])

    def __str__(self):
        return "Hyperbolic Space (Lorentz)"
