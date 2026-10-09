import torch
from .manifold import Manifold
from ..utils import EPS, tanh, artanh


class Poincare(Manifold):
    def __init__(self, c=None):
        super().__init__()
        self.c = c
        self.edge_eps = 1e-3

    def _curvature(self, ref):
        if self.c is None:
            return torch.tensor(1.0, dtype=ref.dtype, device=ref.device)
        if torch.is_tensor(self.c):
            return self.c.to(dtype=ref.dtype, device=ref.device)
        return torch.tensor(float(self.c), dtype=ref.dtype, device=ref.device)

    def lambda_x(self, x, keepdim=False):
        c = self._curvature(x)
        denominator = 1 - c * x.pow(2).sum(dim=-1, keepdim=keepdim)
        return 2 / denominator.clamp_min(EPS[x.dtype])

    def inner(self, x, u, v, keepdim=False):
        return self.lambda_x(x, keepdim=True) ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def proju(self, x, u):
        return u

    def projx(self, x, inplace=False):
        norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS[x.dtype])
        maxnorm = (1 - self.edge_eps) / self._curvature(x).clamp_min(EPS[x.dtype]).sqrt()
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def exp(self, x, u):
        c = self._curvature(x).clamp_min(EPS[x.dtype])
        sqrt_c = c.sqrt()
        u_norm = u.norm(dim=-1, keepdim=True).clamp_min(min=EPS[x.dtype])
        scaled = 0.5 * sqrt_c * self.lambda_x(x, keepdim=True) * u_norm
        second_term = tanh(scaled) * u / (sqrt_c * u_norm)
        gamma_1 = self.mobius_addition(x, second_term)
        return self.projx(gamma_1)

    def exp0(self, u):
        return self.exp(torch.zeros_like(u), u)

    def mobius_addition(self, x, y):
        c = self._curvature(x)
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c.pow(2) * x2 * y2
        return num / denom.clamp_min(EPS[x.dtype])

    def log(self, x, y):
        c = self._curvature(x).clamp_min(EPS[x.dtype])
        sqrt_c = c.sqrt()
        sub = self.mobius_addition(-x, y)
        sub_norm = sub.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
        lam = self.lambda_x(x, keepdim=True)
        return 2 / (sqrt_c * lam) * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def log0(self, y):
        return self.log(torch.zeros_like(y), y)

    def rand(self, *shape, out=None):
        x = torch.randn(*shape)
        return self.projx(x)

    def randvec(self, x, norm=1):
        y = torch.rand(x.shape)
        return y * norm / y.norm(dim=-1, keepdim=True)

    def __str__(self):
        return "Hyperbolic Space (Poincare)"
