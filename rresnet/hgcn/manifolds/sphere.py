"""Sphere manifold."""

import torch

from manifolds.base import Manifold


class Sphere(Manifold):
    """Unit sphere embedded in Euclidean space."""

    def __init__(self):
        super(Sphere, self).__init__()
        self.name = 'Sphere'
        self.min_norm = 1e-15

    def _curvature(self, ref, c):
        if c is None:
            return torch.tensor(1.0, dtype=ref.dtype, device=ref.device)
        if torch.is_tensor(c):
            return c.to(dtype=ref.dtype, device=ref.device)
        return torch.tensor(c, dtype=ref.dtype, device=ref.device)

    def _radius(self, ref, c):
        return self._curvature(ref, c).clamp_min(self.min_norm).rsqrt()

    def _origin(self, ref, c):
        origin = torch.zeros_like(ref)
        origin[..., 0] = self._radius(ref, c)
        return origin

    def normalize(self, p):
        return self.proj(p, c=None)

    def sqdist(self, p1, p2, c):
        radius = self._radius(p1, c)
        inner = (p1 * p2).sum(dim=-1) / radius.pow(2)
        inner = inner.clamp(-1 + self.eps, 1 - self.eps)
        return radius.pow(2) * torch.acos(inner).pow(2)

    def egrad2rgrad(self, p, dp, c):
        return self.proj_tan(dp, p, c)

    def proj(self, p, c):
        radius = self._radius(p, c)
        norm = p.norm(dim=-1, keepdim=True)
        safe = radius * p / norm.clamp_min(self.min_norm)
        default = self._origin(p, c)
        return torch.where((norm > self.min_norm).expand_as(p), safe, default)

    def proj_tan(self, u, p, c):
        radius = self._radius(p, c)
        return u - (u * p).sum(dim=-1, keepdim=True) * p / radius.pow(2)

    def proj_tan0(self, u, c):
        vals = u.clone()
        vals[..., 0] = 0
        return vals

    def expmap(self, u, p, c):
        radius = self._radius(p, c)
        u = self.proj_tan(u, p, c)
        u_norm = u.norm(dim=-1, keepdim=True)
        scaled = u_norm / radius
        direction = u / u_norm.clamp_min(self.min_norm)
        ret = torch.cos(scaled) * p + radius * torch.sin(scaled) * direction
        return torch.where((u_norm > self.min_norm).expand_as(p), self.proj(ret, c), self.proj(p, c))

    def logmap(self, p1, p2, c):
        radius = self._radius(p1, c)
        inner = (p1 * p2).sum(dim=-1, keepdim=True) / radius.pow(2)
        inner = inner.clamp(-1 + self.eps, 1 - self.eps)
        tangent = p2 - inner * p1
        tangent = self.proj_tan(tangent, p1, c)
        tangent_norm = tangent.norm(dim=-1, keepdim=True)
        theta = torch.acos(inner)
        ret = radius * theta * tangent / tangent_norm.clamp_min(self.min_norm)
        return torch.where((tangent_norm > self.min_norm).expand_as(p1), ret, torch.zeros_like(p1))

    def expmap0(self, u, c):
        origin = self._origin(u, c)
        return self.expmap(self.proj_tan0(u, c), origin, c)

    def logmap0(self, p, c):
        origin = self._origin(p, c)
        return self.logmap(origin, p, c)

    def mobius_add(self, x, y, c, dim=-1):
        u = self.logmap0(y, c)
        transported = self.ptransp0(x, u, c)
        return self.expmap(transported, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        mu = self.proj_tan0(mu, c)
        return self.expmap0(mu, c)

    def init_weights(self, w, c, irange=1e-5):
        w.data.normal_(0, 1)
        w.data.copy_(self.proj(w.data, c))
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, v, c):
        radius = self._radius(x, c)
        denom = (radius.pow(2) + (x * y).sum(dim=-1, keepdim=True)).clamp_min(self.min_norm)
        transported = v - ((v * y).sum(dim=-1, keepdim=True) / denom) * (x + y)
        return self.proj_tan(transported, y, c)

    def ptransp0(self, x, v, c):
        origin = self._origin(x, c)
        return self.ptransp(origin, x, self.proj_tan0(v, c), c)
