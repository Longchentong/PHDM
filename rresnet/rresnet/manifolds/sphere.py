import torch
from .manifold import Manifold
from ..utils import EPS


class Sphere(Manifold):
    def __init__(self, c=None):
        super().__init__()
        self.c = c

    def _curvature(self, ref):
        if self.c is None:
            return torch.tensor(1.0, dtype=ref.dtype, device=ref.device)
        if torch.is_tensor(self.c):
            return self.c.to(dtype=ref.dtype, device=ref.device)
        return torch.tensor(self.c, dtype=ref.dtype, device=ref.device)

    def _radius(self, ref):
        return self._curvature(ref).clamp_min(EPS[ref.dtype]).rsqrt()

    def inner(self, x, u, v, keepdim=False):
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def proju(self, x, u):
        radius = self._radius(x)
        return u - (x * u).sum(dim=-1, keepdim=True) * x / radius.pow(2)

    def projx(self, x):
        radius = self._radius(x)
        norm = x.norm(dim=-1, keepdim=True)
        safe = radius * x / norm.clamp_min(EPS[x.dtype])
        default = torch.zeros_like(x)
        default[..., 0] = radius.squeeze(-1) if radius.ndim == x.ndim else radius
        return torch.where((norm > EPS[x.dtype]).expand_as(x), safe, default)

    def exp(self, x, u):
        radius = self._radius(x)
        u = self.proju(x, u)
        u_norm = u.norm(dim=-1, keepdim=True)
        scaled = u_norm / radius
        ret = torch.cos(scaled) * x + radius * torch.sin(scaled) * u / u_norm.clamp_min(EPS[x.dtype])
        return torch.where((u_norm > EPS[x.dtype]).expand_as(x), self.projx(ret), x)

    def log(self, x, y):
        radius = self._radius(x)
        inner = ((x * y).sum(dim=-1, keepdim=True) / radius.pow(2)).clamp(-1 + EPS[x.dtype], 1 - EPS[x.dtype])
        u = self.proju(x, y - inner * x)
        u_norm = u.norm(dim=-1, keepdim=True)
        dist = radius * torch.acos(inner)
        ret = dist * u / u_norm.clamp_min(EPS[x.dtype])
        return torch.where((u_norm > EPS[x.dtype]).expand_as(x), ret, torch.zeros_like(x))

    def rand(self, *shape, out=None):
        return self.projx(torch.randn(*shape))

    def randvec(self, x, norm=1):
        y = self.proju(x, torch.randn_like(x))
        return y * norm / y.norm(dim=-1, keepdim=True).clamp_min(EPS[y.dtype])

    def __str__(self):
        return "Sphere"
