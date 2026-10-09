"""Projected sphere manifold via stereographic coordinates."""

import math

import torch

from manifolds.base import Manifold


class ProjectedSphere(Manifold):
    """
    Positive-curvature stereographic model of the sphere.

    Coordinates live in R^n and represent points on S_K via stereographic
    projection. Curvature parameter c = K > 0 and sphere radius is 1 / sqrt(c).
    """

    def __init__(self):
        super(ProjectedSphere, self).__init__()
        self.name = 'ProjectedSphere'
        self.min_norm = 1e-15
        self.max_norm = 1e6
        self.max_angle = math.pi / 2 - 1e-4

    def _radius(self, c):
        return c.clamp_min(self.min_norm).rsqrt()

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. + c * x_sqnorm).clamp_min(self.min_norm)

    def _scaled_tan(self, x, c):
        sqrt_c = c.sqrt()
        return torch.tan(torch.clamp(sqrt_c * x, min=-self.max_angle, max=self.max_angle)) / sqrt_c

    def _scaled_atan(self, x, c):
        sqrt_c = c.sqrt()
        return torch.atan(sqrt_c * x) / sqrt_c

    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        sub = self.mobius_add(-p1, p2, c, dim=-1)
        dist_c = torch.atan(sqrt_c * sub.norm(dim=-1, p=2, keepdim=False))
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        clipped = x * (self.max_norm / norm)
        return torch.where((norm > self.max_norm).expand_as(x), clipped, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p, c)
        second_term = self._scaled_tan(lam * u_norm / 2, c) * u / u_norm
        return self.proj(self.mobius_add(p, second_term, c), c)

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        return 2 / lam * self._scaled_atan(sub_norm, c) * sub / sub_norm

    def expmap0(self, u, c):
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = self._scaled_tan(u_norm, c) * u / u_norm
        return self.proj(gamma_1, c)

    def logmap0(self, p, c):
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = self._scaled_atan(p_norm, c) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 - 2 * c * xy - c * y2) * x + (1 + c * x2) * y
        denom = 1 - 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        angle = mx_norm / x_norm * self._scaled_atan(x_norm, c)
        res_c = self._scaled_tan(angle, c) * mx / mx_norm
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        return torch.where(cond, res_0, res_c)

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim=-1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 - c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 + c * uw
        d = 1 - 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_sphere(self, x, c):
        radius = self._radius(c)
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        denom = (radius.pow(2) + sqnorm).clamp_min(self.min_norm)
        first = radius * (radius.pow(2) - sqnorm) / denom
        spatial = 2 * radius.pow(2) * x / denom
        return torch.cat([first, spatial], dim=1)
