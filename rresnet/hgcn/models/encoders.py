"""PHDM input maps and RResNet encoders for Cora link prediction."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from typing import Any, Dict, Literal, Optional, Tuple, Union
import manifolds
import utils.math_utils as pmath
from rresnet import RResNet, ProjVecField, GraphProjVecField, SphereLocalGraphProjVecField, FeatureMapVecFieldSimpleGraphBase
import rresnet.manifolds.hyperbolic as hyperbolic
import rresnet.manifolds.lorentz as lorentz
import rresnet.manifolds.projected_sphere as projected_sphere
import rresnet.manifolds.sphere as sphere
from rresnet.nn.hyperbolic_feat import distance_to_horosphere


def _diagnostics_to_floats(values):
    result = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        result[key] = float(value)
    return result


class PHDM_Poincare(nn.Module):
    """
    从欧式 R^n 映射到庞加莱球 B^m_c 的全连接层（HNN++ 的 Poincaré FC，v 用欧式 linear）。

    输入:  x ∈ R^n
    步骤:
        1) 欧式 logit: v = W x + b，v ∈ R^m
        2) 视 v 为“到超平面的有符号距离”，按 HNN++ 的公式映射到庞加莱球:
               w_k = (1/√c) * sinh(√c * v_k)
               y   = w / (1 + sqrt(1 + c * ||w||^2))
           保证对每个 k 都有:
               d_c(y, H̄^c_{e(k),0}) = v_k(x)
           其中 d_c 是庞加莱球上“点到超平面”的距离公式 (56)。

    输出: y ∈ B^m_c (形状: (..., out_features))
    """

    def __init__(self, manifold=None, in_features=1, out_features=1,
                 c=None, eps=1e-9, bias=True, max_scale=15.0, input_scale=1.0):
        super().__init__()

        assert in_features >= 1
        assert out_features >= 1

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.max_scale = float(max_scale)
        self.input_scale = float(input_scale)
        if self.input_scale <= 0:
            raise ValueError("input_scale must be positive.")
        self._last_diagnostics = {}

        # 曲率幅值 c > 0；优先从 manifold 里读，读不到就用显式 c
        if c is not None:
            self.c = float(c)
        elif manifold is not None:
            if hasattr(manifold, "c"):
                self.c = float(manifold.c)
            elif hasattr(manifold, "k"):
                # 兼容你之前的 k<0 写法
                k = float(manifold.k)
                self.c = -k if k < 0 else k
            else:
                raise ValueError("manifold 里找不到 c 或 k，请手动传 c>0")
        else:
            raise ValueError("需要传 manifold 或 c（曲率幅值 c>0）")

        # 欧式 Linear: v = W x + b
        # self.fc = nn.Linear(in_features, out_features, bias=bias)
        # 和之前一样的初始化
        # nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
        # if bias:
            # nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        x : (..., in_features)  欧式输入
        return y : (..., out_features)  庞加莱球上的点
        """
        # 1) 欧式 logit
        # v = self.fc(x)  # (..., m)
        v = self.input_scale * x
        # 2) v -> w -> y  按 HNN++ 式 (7)
        # c = x.new_tensor(self.c)
        c = x.new_tensor(self.c)
        sqrt_c = torch.sqrt(c)

        # w_k = (1/√c) * sinh(√c * v_k)
        raw_scaled = sqrt_c * v
        scaled = raw_scaled.clamp(min=-self.max_scale, max=self.max_scale)
        w = torch.sinh(scaled) / sqrt_c          # (..., m)

        w_sq = (w ** 2).sum(dim=-1, keepdim=True)    # (..., 1)

        # y = w / (1 + sqrt(1 + c * ||w||^2))
        denom = 1.0 + torch.sqrt(1.0 + c * w_sq + self.eps)  # (..., 1)
        y = w / denom                                # (..., m)

        clipped = torch.abs(raw_scaled) > self.max_scale
        self._last_diagnostics = {
            'sinh_clip_fraction': clipped.float().mean().detach(),
            'max_abs_scaled_logit': torch.abs(raw_scaled).max().detach(),
            'input_scale': self.input_scale,
        }

        # y 理论上已经在 B^m_c 里，可以视需要再做一次 proj
        return y

    def get_diagnostics(self):
        return _diagnostics_to_floats(self._last_diagnostics)


class PHDM_Lorentz(nn.Module):
    """
    Mapping from Euclidean Space R^n to Lorentz Manifold L^m.

    This layer interprets the output of a standard Euclidean linear layer as
    'logits' (v), and then maps these logits to the Lorentz manifold using
    the coordinate construction formulas from Theorem 1 of the paper.

    Transformation:
    1. v = xW^T + b  (Standard Euclidean Linear)
    2. y_s = (1/√(-K)) * sinh(√(-K) * v)  (Map logits to spatial coords)
    3. y_t = √(1/(-K) + ||y_s||²)       (Recover time coord)

    Parameters:
    -----------
    manifold : CustomLorentz
        Lorentz manifold instance with curvature K < 0
    in_features : int
        Input dimension in Euclidean space (n)
    out_features : int
        Output dimension of the spatial component on Lorentz manifold (m).
        The final output tensor shape will be (..., m+1).
    """

    def __init__(self, manifold, in_features, out_features, bias=True, eps=1e-9,
                 max_scale=10.0, input_scale=1.0, init_scale=1.0):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.max_scale = float(max_scale)
        self.input_scale = float(input_scale)
        if self.input_scale <= 0:
            raise ValueError("input_scale must be positive.")
        self.init_scale = float(init_scale)
        if self.init_scale <= 0:
            raise ValueError("init_scale must be positive.")
        self._last_diagnostics = {}
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.linear.weight)
        with torch.no_grad():
            self.linear.weight.mul_(self.init_scale)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, x):
        """
        x : (..., in_features)  输入点在欧式空间 R^n
        returns y : (..., out_features + 1) 输出点在 Lorentz 流形 L^m
        """
        c = self.manifold._curvature(x).clamp_min(x.new_tensor(self.eps))
        sqrt_c = torch.sqrt(c)
        radius = c.reciprocal().sqrt()

        v = self.input_scale * self.linear(x)
        raw_scaled = sqrt_c * v
        scaled = torch.clamp(raw_scaled, min=-self.max_scale, max=self.max_scale)
        y_s = torch.sinh(scaled) / sqrt_c
        y_t = torch.sqrt(radius.pow(2) + (y_s ** 2).sum(dim=-1, keepdim=True) + self.eps)
        y = torch.cat([y_t, y_s], dim=-1)
        self._last_diagnostics = {
            'sinh_clip_fraction': (torch.abs(raw_scaled) > self.max_scale).float().mean().detach(),
            'max_abs_scaled_logit': torch.abs(raw_scaled).max().detach(),
            'input_scale': self.input_scale,
        }
        return self.manifold.projx(y)

    def get_diagnostics(self):
        return _diagnostics_to_floats(self._last_diagnostics)


def build_hyperbolic_input_layers(args, manifold):
    proj_type = getattr(args, 'proj_type', 'original')
    linear = nn.Linear(args.feat_dim, args.dim)
    if proj_type == 'original':
        return linear, None
    if proj_type == 'poincare_fc':
        init_scale = float(getattr(args, 'phdm_init_scale', 1.0))
        if init_scale <= 0:
            raise ValueError("phdm_init_scale must be positive.")
        with torch.no_grad():
            linear.weight.mul_(init_scale)
        return linear, PHDM_Poincare(
            c=args.c if args.c is not None else 1.0,
            in_features=args.dim,
            out_features=args.dim,
            bias=bool(args.bias),
            input_scale=float(getattr(args, 'phdm_scale', 1.0)),
        )
    raise ValueError(
        f"Unsupported proj_type={proj_type!r} for hyperbolic encoders. "
        "Use one of: original, poincare_fc."
    )


def apply_hyperbolic_input(linear, input_map, manifold, x):
    x = linear(x)
    return manifold.projx(x) if input_map is None else input_map(x)


def build_lorentz_input_layer(args, manifold):
    proj_type = getattr(args, 'proj_type', 'original')
    if proj_type == 'original':
        return nn.Linear(args.feat_dim, args.dim, bias=bool(args.bias))
    if proj_type == 'lorentz_fc':
        if args.dim < 2:
            raise ValueError("Lorentz RResNet encoders with lorentz_fc require args.dim >= 2.")
        return PHDM_Lorentz(
            manifold=manifold,
            in_features=args.feat_dim,
            out_features=args.dim - 1,
            bias=bool(args.bias),
            input_scale=float(getattr(args, 'phdm_scale', 1.0)),
            init_scale=float(getattr(args, 'phdm_init_scale', 1.0)),
        )
    raise ValueError(
        f"Unsupported proj_type={proj_type!r} for Lorentz RResNet encoders. "
        "Use one of: original, lorentz_fc."
    )


def apply_lorentz_input(linear, manifold, proj_type, x):
    x = linear(x)
    return manifold.projx(x) if proj_type == 'original' else x


def build_projected_spherical_input_layer(args, manifold):
    proj_type = getattr(args, 'proj_type', 'original')
    if proj_type == 'original':
        return nn.Linear(args.feat_dim, args.dim, bias=bool(args.bias))
    if proj_type == 'psphere_fc':
        return PHDM_ProjectSphere(
            in_features=args.feat_dim,
            out_features=args.dim,
            curvature=manifold.c,
            bias=bool(args.bias),
            mode="exact",
            infeasible="scale",
            principal_branch="clip",
            input_scale=float(getattr(args, 'phdm_scale', 1.0)),
            init_scale=float(getattr(args, 'phdm_init_scale', 1.0)),
        )
    raise ValueError(
        f"Unsupported proj_type={proj_type!r} for projected spherical encoders. "
        "Use one of: original, psphere_fc."
    )


def apply_projected_spherical_input(linear, manifold, proj_type, x):
    x = linear(x)
    return manifold.projx(x) if proj_type == 'original' else x


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output


class RRNetHyperbolic(Encoder):
    """
    Riemannian ResNet over hyperbolic space with an embedded vector field
    """

    def __init__(self, c, args):
        super(RRNetHyperbolic, self).__init__(c)
        self.manifold = hyperbolic.Poincare(c)
        self.linear, self.input_map = build_hyperbolic_input_layers(args, self.manifold)

        acts = {
            'relu': nn.ReLU,
            'lrelu': nn.LeakyReLU,
            None: nn.Identity,
        }

        self.act = acts.get(args.act, nn.Identity)

        self.rresnet = RResNet(
            self.manifold, [ProjVecField(
                self.manifold, args.dim, args.hdim, args.dim, args.num_layers, act=self.act) for _ in range(args.num_blocks)]
        )

    def encode(self, x, adj):
        act = self.act(negative_slope=0.5) if self.act is nn.LeakyReLU else self.act()
        x = apply_hyperbolic_input(self.linear, self.input_map, self.manifold, x)
        return act(self.rresnet(x))


class RRNetGraphHyperbolic(Encoder):
    """
    Riemannian ResNet with graph information
    """

    def __init__(self, c, args, learnable_horospheres=True):
        super(RRNetGraphHyperbolic, self).__init__(c)

        self.manifold = hyperbolic.Poincare(c)
        self.linear, self.input_map = build_hyperbolic_input_layers(args, self.manifold)

        self.learnable_horospheres = learnable_horospheres

        acts = {
            'relu': nn.ReLU,
            None: nn.Identity,
            'lrelu': nn.LeakyReLU,
        }
        self.act = acts.get(args.act, nn.Identity)

        self.num_horospheres = args.num_horospheres

        self.w = nn.Parameter(torch.randn(args.num_blocks, self.num_horospheres,
                              args.dim, device=args.device), requires_grad=learnable_horospheres)

        self.b = nn.Parameter(torch.randn(args.num_blocks, self.num_horospheres,
                              device=args.device), requires_grad=learnable_horospheres)

        f = [lambda x, i=i: distance_to_horosphere(
            x, self.w[i], self.b[i]) for i in range(args.num_blocks)]

        self.rresnet = RResNet(
            self.manifold, [FeatureMapVecFieldSimpleGraphBase(
                f[i], args.hdim, self.num_horospheres, self.manifold) for i in range(args.num_blocks)]
        )

    def encode(self, x, adj):
        x = apply_hyperbolic_input(self.linear, self.input_map, self.manifold, x)
        x = self.rresnet(x, adj)
        act = self.act(negative_slope=0.5) if self.act is nn.LeakyReLU else self.act()
        return act(x)


class RRNetHyperbolicLorentz(Encoder):
    """
    Riemannian ResNet over the Lorentz model of hyperbolic space.
    """

    def __init__(self, c, args):
        super(RRNetHyperbolicLorentz, self).__init__(c)
        self.manifold = lorentz.Lorentz(c)
        self.proj_type = getattr(args, 'proj_type', 'original')
        self.linear = build_lorentz_input_layer(args, self.manifold)

        acts = {
            'relu': nn.ReLU,
            'lrelu': nn.LeakyReLU,
            None: nn.Identity,
        }
        self.act = acts.get(args.act, nn.Identity)

        self.rresnet = RResNet(
            self.manifold, [ProjVecField(
                self.manifold, args.dim, args.hdim, args.dim, args.num_layers, act=self.act) for _ in range(args.num_blocks)]
        )

    def encode(self, x, adj):
        x = apply_lorentz_input(self.linear, self.manifold, self.proj_type, x)
        return self.rresnet(x)


class RRNetGraphHyperbolicLorentz(Encoder):
    """
    Graph-aware Riemannian ResNet over the Lorentz model of hyperbolic space.
    """

    def __init__(self, c, args):
        super(RRNetGraphHyperbolicLorentz, self).__init__(c)
        if getattr(args, 'graph_aggregation_mode', 'ambient') != 'ambient':
            raise ValueError("RRNetGraphHyperbolicLorentz currently supports graph_aggregation_mode='ambient' only.")

        self.manifold = lorentz.Lorentz(c)
        self.proj_type = getattr(args, 'proj_type', 'original')
        self.linear = build_lorentz_input_layer(args, self.manifold)

        acts = {
            'relu': nn.ReLU,
            'lrelu': nn.LeakyReLU,
            None: nn.Identity,
        }
        self.act = acts.get(args.act, nn.Identity)

        self.rresnet = RResNet(
            self.manifold, [GraphProjVecField(
                self.manifold, args.dim, args.hdim, args.dim, args.num_layers, act=self.act, adj_power=args.adj_power, mix_mode=args.graph_mix_mode) for _ in range(args.num_blocks)]
        )

    def encode(self, x, adj):
        x = apply_lorentz_input(self.linear, self.manifold, self.proj_type, x)
        return self.rresnet(x, adj)


class RRNetProjectedSpherical(Encoder):
    """
    Riemannian ResNet over the projected sphere model.
    """

    def __init__(self, c, args):
        super(RRNetProjectedSpherical, self).__init__(c)
        self.manifold = projected_sphere.ProjectedSphere(c)
        self.proj_type = getattr(args, 'proj_type', 'original')
        self.linear = build_projected_spherical_input_layer(args, self.manifold)

        acts = {
            'relu': nn.ReLU,
            'lrelu': nn.LeakyReLU,
            None: nn.Identity,
        }
        self.act = acts.get(args.act, nn.Identity)

        self.rresnet = RResNet(
            self.manifold, [ProjVecField(
                self.manifold, args.dim, args.hdim, args.dim, args.num_layers, act=self.act) for _ in range(args.num_blocks)]
        )

    def encode(self, x, adj):
        act = self.act(negative_slope=0.5) if self.act is nn.LeakyReLU else self.act()
        x = apply_projected_spherical_input(self.linear, self.manifold, self.proj_type, x)
        return act(self.rresnet(x))


class RRNetGraphProjectedSpherical(Encoder):
    """
    Graph-aware Riemannian ResNet over the projected sphere model.
    """

    def __init__(self, c, args):
        super(RRNetGraphProjectedSpherical, self).__init__(c)
        if getattr(args, 'graph_aggregation_mode', 'ambient') != 'ambient':
            raise ValueError("RRNetGraphProjectedSpherical currently supports graph_aggregation_mode='ambient' only.")

        self.manifold = projected_sphere.ProjectedSphere(c)
        self.proj_type = getattr(args, 'proj_type', 'original')
        self.linear = build_projected_spherical_input_layer(args, self.manifold)

        acts = {
            'relu': nn.ReLU,
            'lrelu': nn.LeakyReLU,
            None: nn.Identity,
        }
        self.act = acts.get(args.act, nn.Identity)

        self.rresnet = RResNet(
            self.manifold, [GraphProjVecField(
                self.manifold, args.dim, args.hdim, args.dim, args.num_layers, act=self.act, adj_power=args.adj_power, mix_mode=args.graph_mix_mode) for _ in range(args.num_blocks)]
        )

    def encode(self, x, adj):
        act = self.act(negative_slope=0.5) if self.act is nn.LeakyReLU else self.act()
        x = apply_projected_spherical_input(self.linear, self.manifold, self.proj_type, x)
        return act(self.rresnet(x, adj))


class PHDM_Sphere(nn.Module):
    """
    Map Euclidean features x in R^n to a point y on the hypersphere S_K^m, K > 0.

    Output shape:
        (..., out_features + 1)

    Convention:
        y[..., 0]  -> first/axis coordinate (the "pole axis")
        y[..., 1:] -> spatial coordinates on the sphere chart

    Two modes are supported:

    1) mode="exact"
       Exact point-to-hyperplane spherical decoder:
           v      = Wx + b
           y_s,k  = sin(sqrt(K) * v_k) / sqrt(K)
           y_0    = +/- sqrt(1/K - ||y_s||^2)

       This preserves the coordinate-wise spherical signed-distance interpretation
       only when ||y_s||^2 <= 1/K. If this is violated, you can:
           - infeasible="raise" : raise an error
           - infeasible="scale" : radially rescale y_s back into the feasible ball
                                  (engineering fix; no longer exact)

    2) mode="expmap"
       Always-valid exponential-map decoder from the north pole:
           v      = Wx + b
           y      = exp_{mu0}(v)

       This is globally valid on the sphere but does not keep the exact
       point-to-hyperplane equality per coordinate.

    Parameters
    ----------
    in_features : int
        Euclidean input dimension.
    out_features : int
        Spatial output dimension m. Final output has dimension m+1.
    manifold : optional
        An object with one of attributes: .k, .curvature, or .c
        representing positive spherical curvature K > 0.
    curvature : optional
        Positive spherical curvature K > 0. Used if manifold is None.
    bias : bool
        Whether to use bias in the Euclidean linear layer.
    mode : {"exact", "expmap"}
        Decoder choice.
    infeasible : {"raise", "scale"}
        Used only when mode="exact".
    hemisphere : {"north", "south"}
        Only used when mode="exact". Chooses the sign of y_0.
        "north" gives +sqrt(...), "south" gives -sqrt(...).
    eps : float
        Numerical stability constant.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        manifold: Optional[Any] = None,
        curvature: Optional[Union[float, Tensor]] = None,
        bias: bool = True,
        mode: Literal["exact", "expmap"] = "exact",
        infeasible: Literal["raise", "scale"] = "raise",
        principal_branch: Literal["raise", "clip"] = "raise",
        hemisphere: Literal["north", "south"] = "north",
        input_scale: float = 1.0,
        init_scale: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if manifold is None and curvature is None:
            raise ValueError("Provide either `manifold` or `curvature`.")
        if mode not in {"exact", "expmap"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if infeasible not in {"raise", "scale"}:
            raise ValueError(f"Unsupported infeasible mode: {infeasible}")
        if principal_branch not in {"raise", "clip"}:
            raise ValueError(f"Unsupported principal_branch mode: {principal_branch}")
        if hemisphere not in {"north", "south"}:
            raise ValueError(f"Unsupported hemisphere: {hemisphere}")

        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.mode = mode
        self.infeasible = infeasible
        self.principal_branch = principal_branch
        self.hemisphere = hemisphere
        self.input_scale = float(input_scale)
        if self.input_scale <= 0:
            raise ValueError("input_scale must be positive.")
        self.init_scale = float(init_scale)
        if self.init_scale <= 0:
            raise ValueError("init_scale must be positive.")
        self.eps = float(eps)
        self._last_diagnostics = {}

        if curvature is None:
            self._curvature_buffer = None
        else:
            self.register_buffer(
                "_curvature_buffer",
                torch.tensor(float(curvature), dtype=torch.get_default_dtype()),
            )

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.linear.weight)
        with torch.no_grad():
            self.linear.weight.mul_(self.init_scale)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mode={self.mode}, infeasible={self.infeasible}, "
            f"principal_branch={self.principal_branch}, hemisphere={self.hemisphere}, "
            f"input_scale={self.input_scale:g}, init_scale={self.init_scale:g}"
        )

    def _get_positive_curvature(self, ref: Tensor) -> Tensor:
        if self.manifold is not None:
            if hasattr(self.manifold, "k"):
                K = self.manifold.k
            elif hasattr(self.manifold, "curvature"):
                K = self.manifold.curvature
            elif hasattr(self.manifold, "c"):
                K = self.manifold.c
            else:
                raise AttributeError(
                    "manifold must expose one of: .k, .curvature, or .c"
                )
        else:
            K = self._curvature_buffer

        if K is None:
            raise ValueError("Curvature is not available.")

        if isinstance(K, Tensor):
            K_t = K.to(dtype=ref.dtype, device=ref.device)
        else:
            K_t = torch.tensor(float(K), dtype=ref.dtype, device=ref.device)

        if torch.any(K_t <= 0):
            raise ValueError(
                "PHDM_Sphere expects spherical curvature K > 0."
            )
        return K_t

    def _decode_exact(self, v: Tensor, K: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Exact spherical point-to-hyperplane decoder:
            y_s = sin(sqrt(K) * v) / sqrt(K)
            y_0 = +/- sqrt(1/K - ||y_s||^2)
        """
        sqrt_K = torch.sqrt(K)
        R = 1.0 / sqrt_K

        theta = sqrt_K * v
        limit = (math.pi / 2.0) - 1e-6
        principal_ok = torch.abs(theta) <= limit
        clipped_mask = torch.zeros_like(theta, dtype=torch.bool)
        theta_used = theta
        if not bool(principal_ok.all()):
            if self.principal_branch == "raise":
                max_abs = torch.abs(theta).max().item()
                raise RuntimeError(
                    "Exact spherical decoder left the principal arcsin branch: "
                    f"max |theta| = {max_abs:.6f} > pi/2. "
                    "Reduce logits or set principal_branch='clip'."
                )
            clipped_mask = ~principal_ok
            theta_used = theta.clamp(min=-limit, max=limit)

        # Spatial coordinates from exact coordinate-wise decoder
        y_s = torch.sin(theta_used) / sqrt_K  # (..., m)
        y_s_sq_norm = (y_s * y_s).sum(dim=-1, keepdim=True)  # (..., 1)
        raw_y_s_sq_norm = y_s_sq_norm
        radius_sq = R * R
        safe_radius_sq = radius_sq * (1.0 - 1e-6)

        feasible_mask = y_s_sq_norm <= (radius_sq + self.eps)
        projected_mask = torch.zeros_like(feasible_mask, dtype=torch.bool)

        needs_scale = y_s_sq_norm > safe_radius_sq
        if bool(needs_scale.any()):
            if self.infeasible == "raise" and not bool(feasible_mask.all()):
                # helpful diagnostic
                overshoot = torch.sqrt(
                    torch.clamp(y_s_sq_norm / torch.clamp(radius_sq, min=self.eps), min=0.0)
                ).max().item()
                raise RuntimeError(
                    "Exact spherical decoder is infeasible for at least one sample: "
                    "||y_s||^2 > 1/K. This is intrinsic to the sphere constraint. "
                    f"Max overshoot factor: {overshoot:.6f}. "
                    "Try smaller logits, `infeasible='scale'`, or `mode='expmap'`."
                )

            # Engineering fallback: radial projection back into feasible region.
            # This keeps the output on the sphere but no longer preserves the exact
            # point-to-hyperplane equalities.
            if self.infeasible == "scale":
                projected_mask = needs_scale
            else:
                needs_scale = torch.zeros_like(needs_scale)
            y_s_norm = torch.sqrt(torch.clamp(y_s_sq_norm, min=self.eps))
            max_radius = R * math.sqrt(1.0 - 1e-6)
            scale = torch.clamp(max_radius / y_s_norm, max=1.0)
            y_s = torch.where(needs_scale.expand_as(y_s), y_s * scale, y_s)
            y_s_sq_norm = (y_s * y_s).sum(dim=-1, keepdim=True)

        y0_sq = torch.clamp(radius_sq - y_s_sq_norm, min=0.0)
        sign = 1.0 if self.hemisphere == "north" else -1.0
        y0 = sign * torch.sqrt(y0_sq)

        y = torch.cat([y0, y_s], dim=-1)

        info = {
            "v": v,
            "theta": theta,
            "theta_used": theta_used,
            "principal_ok": principal_ok,
            "clipped_mask": clipped_mask,
            "feasible_mask": feasible_mask,
            "projected_mask": projected_mask,
            "overshoot": torch.sqrt(
                torch.clamp(raw_y_s_sq_norm / torch.clamp(radius_sq, min=self.eps), min=0.0)
            ),
        }
        return y, info

    def _decode_expmap(self, v: Tensor, K: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Globally valid map via the hypersphere exponential map at the north pole mu0.

        If v in R^m is interpreted as a tangent vector at mu0 = (1/sqrt(K), 0, ..., 0),
        then
            y_0 = (1/sqrt(K)) * cos(sqrt(K) * ||v||)
            y_s = (1/sqrt(K)) * sin(sqrt(K) * ||v||) * v / ||v||
        """
        sqrt_K = torch.sqrt(K)
        R = 1.0 / sqrt_K

        v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)  # (..., 1)
        safe_norm = torch.clamp(v_norm, min=self.eps)
        angle = sqrt_K * v_norm

        coef = torch.sin(angle) / (sqrt_K * safe_norm)
        coef = torch.where(v_norm > self.eps, coef, torch.ones_like(v_norm))

        y_s = coef * v
        y0 = R * torch.cos(angle)

        y = torch.cat([y0, y_s], dim=-1)
        info = {
            "v": v,
            "angle": angle,
            "feasible_mask": torch.ones_like(v_norm, dtype=torch.bool),
            "projected_mask": torch.zeros_like(v_norm, dtype=torch.bool),
        }
        return y, info

    def forward(
        self,
        x: Tensor,
        *,
        return_info: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Parameters
        ----------
        x : Tensor
            Shape (..., in_features), Euclidean input.
        return_info : bool
            If True, also return diagnostic info.

        Returns
        -------
        y : Tensor
            Shape (..., out_features + 1), point on S_K^{out_features}.
        """
        K = self._get_positive_curvature(x)
        v = self.input_scale * self.linear(x)  # (..., m)

        if self.mode == "exact":
            y, info = self._decode_exact(v, K)
        elif self.mode == "expmap":
            y, info = self._decode_expmap(v, K)
        else:
            raise RuntimeError(f"Unexpected mode: {self.mode}")

        if self.mode == "exact":
            self._last_diagnostics = {
                'principal_clip_fraction': info['clipped_mask'].float().mean().detach(),
                'joint_scale_fraction': info['projected_mask'].float().mean().detach(),
                'max_abs_theta': torch.abs(info['theta']).max().detach(),
                'max_overshoot': info['overshoot'].max().detach(),
                'input_scale': v.new_tensor(self.input_scale),
            }
        else:
            self._last_diagnostics = {
                'principal_clip_fraction': v.new_tensor(0.0),
                'joint_scale_fraction': v.new_tensor(0.0),
                'max_abs_theta': info.get('angle', v.new_tensor(0.0)).max().detach(),
                'max_overshoot': v.new_tensor(1.0),
                'input_scale': v.new_tensor(self.input_scale),
            }

        if return_info:
            return y, info
        return y

    def get_diagnostics(self):
        return _diagnostics_to_floats(self._last_diagnostics)


class SphereManifold(sphere.Sphere):
    """Backward-compatible wrapper around the project's sphere manifold."""

    def __init__(self, k: float) -> None:
        if k <= 0:
            raise ValueError("Sphere curvature must satisfy k > 0.")
        super().__init__(k)
        self.k = float(k)


def build_spherical_input_layer(args, manifold):
    proj_type = getattr(args, 'proj_type', 'original')
    if proj_type == 'original':
        return nn.Linear(args.feat_dim, args.dim, bias=bool(args.bias))
    if proj_type == 'sphere_exact':
        return PHDM_Sphere(
            args.feat_dim,
            args.dim - 1,
            manifold=manifold,
            bias=bool(args.bias),
            mode="exact",
            infeasible="scale",
            principal_branch="clip",
            input_scale=float(getattr(args, 'phdm_scale', 1.0)),
            init_scale=float(getattr(args, 'phdm_init_scale', 1.0)),
        )
    if proj_type == 'sphere_expmap':
        return PHDM_Sphere(
            args.feat_dim,
            args.dim - 1,
            manifold=manifold,
            bias=bool(args.bias),
            mode="expmap",
            input_scale=float(getattr(args, 'phdm_scale', 1.0)),
            init_scale=float(getattr(args, 'phdm_init_scale', 1.0)),
        )
    raise ValueError(
        f"Unsupported proj_type={proj_type!r} for spherical encoders. "
        "Use one of: original, sphere_exact, sphere_expmap."
    )


def apply_spherical_input(linear, manifold, proj_type, x):
    x = linear(x)
    return manifold.projx(x) if proj_type == 'original' else x


class RRNetSpherical(Encoder):
    """
    Riemannian ResNet over spherical space with an embedded vector field
    """

    def __init__(self, c, args):
        super(RRNetSpherical, self).__init__(c)
        self.rr_manifold = sphere.Sphere(c)
        self.manifold = self.rr_manifold
        self.proj_type = getattr(args, 'proj_type', 'original')
        self.linear = build_spherical_input_layer(args, self.rr_manifold)

        acts = {
            'relu': nn.ReLU,
            'lrelu': nn.LeakyReLU,
            None: nn.Identity,
        }
        self.act = acts.get(args.act, nn.Identity)

        self.rresnet = RResNet(
            self.rr_manifold, [ProjVecField(
                self.rr_manifold, args.dim, args.hdim, args.dim, args.num_layers, act=self.act) for _ in range(args.num_blocks)]
        )

    def encode(self, x, adj):
        x = apply_spherical_input(self.linear, self.rr_manifold, self.proj_type, x)
        x = self.rresnet(x)
        act = self.act(negative_slope=0.5) if self.act is nn.LeakyReLU else self.act()
        return act(x)


class RRNetGraphSpherical(Encoder):
    """
    Riemannian ResNet over spherical space with graph information and an embedded vector field
    """

    def __init__(self, c, args):
        super(RRNetGraphSpherical, self).__init__(c)
        self.rr_manifold = sphere.Sphere(c)
        self.manifold = self.rr_manifold
        self.proj_type = getattr(args, 'proj_type', 'original')
        self.linear = build_spherical_input_layer(args, self.rr_manifold)

        acts = {
            'relu': nn.ReLU,
            'lrelu': nn.LeakyReLU,
            None: nn.Identity,
        }
        self.act = acts.get(args.act, nn.Identity)

        if args.graph_aggregation_mode == 'sphere_local':
            vector_fields = [SphereLocalGraphProjVecField(
                self.rr_manifold, args.dim, args.hdim, args.dim, args.num_layers, act=self.act, neighbor_scale=args.graph_neighbor_scale, use_neighbors=True
            ) for _ in range(args.num_blocks)]
        elif args.graph_aggregation_mode == 'self_only':
            vector_fields = [SphereLocalGraphProjVecField(
                self.rr_manifold, args.dim, args.hdim, args.dim, args.num_layers, act=self.act, neighbor_scale=0.0, use_neighbors=False
            ) for _ in range(args.num_blocks)]
        else:
            vector_fields = [GraphProjVecField(
                self.rr_manifold, args.dim, args.hdim, args.dim, args.num_layers, act=self.act, adj_power=args.adj_power, mix_mode=args.graph_mix_mode
            ) for _ in range(args.num_blocks)]

        self.rresnet = RResNet(self.rr_manifold, vector_fields)

    def encode(self, x, adj):
        x = apply_spherical_input(self.linear, self.rr_manifold, self.proj_type, x)
        x = self.rresnet(x, adj)
        act = self.act(negative_slope=0.5) if self.act is nn.LeakyReLU else self.act()
        return act(x)


class PHDM_ProjectSphere(nn.Module):
    """
    Euclidean R^n -> projected sphere D_K^m (positive curvature K > 0).

    Output lives in projected-sphere coordinates y in D_K^m ~= R^m,
    not in ambient sphere coordinates.

    Supported modes
    ----------------
    1) mode="exact"
       Hyperplane-exact decoder in projected sphere coordinates.

       Let:
           v = Wx + b
           theta = 2 * sqrt(K) * v     if euclidean_compatible=True
           theta =     sqrt(K) * v     if euclidean_compatible=False
           s = sin(theta)
           u = ||s||^2

       Then the north-branch exact solution is:
           y = s / ( sqrt(K) * (1 + sqrt(1 - u)) )

       The south branch uses:
           y = s / ( sqrt(K) * (1 - sqrt(1 - u)) )

       Important:
       - exactness requires every theta_i to stay in the principal arcsin branch
         (-pi/2, pi/2)
       - and requires u <= 1

    2) mode="expmap"
       Native projected-sphere exp-map at the origin:
           y = tan(sqrt(K) * ||v||) / (sqrt(K) * ||v||) * v

       This is Euclidean-compatible near K -> 0, but it has poles at
           sqrt(K) * ||v|| = pi/2 + k*pi

    Parameters
    ----------
    in_features : int
    out_features : int
    manifold : optional object with .k or .curvature or .c
    curvature : optional positive curvature K > 0
    bias : bool
    mode : "exact" or "expmap"
    euclidean_compatible : bool
        Only used in exact mode. If True, exact mode uses 0.5 * signed distance = v,
        so the layer recovers the Euclidean FC limit as K -> 0.
    branch : "north" or "south"
        Exact mode has two branches. North is the practical default.
    infeasible : "raise" or "scale"
        What to do if ||sin(theta)||^2 > 1 in exact mode.
    principal_branch : "raise" or "clip"
        What to do if theta leaves the principal arcsin interval in exact mode.
    eps : float
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        manifold: Optional[Any] = None,
        curvature: Optional[Union[float, Tensor]] = None,
        bias: bool = True,
        mode: Literal["exact", "expmap"] = "exact",
        euclidean_compatible: bool = True,
        branch: Literal["north", "south"] = "north",
        infeasible: Literal["raise", "scale"] = "raise",
        principal_branch: Literal["raise", "clip"] = "raise",
        input_scale: float = 1.0,
        init_scale: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if manifold is None and curvature is None:
            raise ValueError("Provide either `manifold` or `curvature`.")
        if mode not in {"exact", "expmap"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if branch not in {"north", "south"}:
            raise ValueError(f"Unsupported branch: {branch}")
        if infeasible not in {"raise", "scale"}:
            raise ValueError(f"Unsupported infeasible mode: {infeasible}")
        if principal_branch not in {"raise", "clip"}:
            raise ValueError(f"Unsupported principal_branch mode: {principal_branch}")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.manifold = manifold
        self.mode = mode
        self.euclidean_compatible = bool(euclidean_compatible)
        self.branch = branch
        self.infeasible = infeasible
        self.principal_branch = principal_branch
        self.input_scale = float(input_scale)
        if self.input_scale <= 0:
            raise ValueError("input_scale must be positive.")
        self.init_scale = float(init_scale)
        if self.init_scale <= 0:
            raise ValueError("init_scale must be positive.")
        self.eps = float(eps)
        self._last_diagnostics = {}

        if curvature is None:
            self._curvature_buffer = None
        else:
            self.register_buffer(
                "_curvature_buffer",
                torch.tensor(float(curvature), dtype=torch.get_default_dtype()),
            )

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.linear.weight)
        with torch.no_grad():
            self.linear.weight.mul_(self.init_scale)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mode={self.mode}, euclidean_compatible={self.euclidean_compatible}, "
            f"branch={self.branch}, infeasible={self.infeasible}, "
            f"principal_branch={self.principal_branch}, input_scale={self.input_scale:g}, "
            f"init_scale={self.init_scale:g}"
        )

    def _get_positive_curvature(self, ref: Tensor) -> Tensor:
        if self.manifold is not None:
            if hasattr(self.manifold, "k"):
                K = self.manifold.k
            elif hasattr(self.manifold, "curvature"):
                K = self.manifold.curvature
            elif hasattr(self.manifold, "c"):
                K = self.manifold.c
            else:
                raise AttributeError(
                    "manifold must expose one of: .k, .curvature, .c"
                )
        else:
            K = self._curvature_buffer

        if K is None:
            raise ValueError("Curvature is not available.")

        if isinstance(K, Tensor):
            K_t = K.to(dtype=ref.dtype, device=ref.device)
        else:
            K_t = torch.tensor(float(K), dtype=ref.dtype, device=ref.device)

        if torch.any(K_t <= 0):
            raise ValueError(
                "PHDM_ProjectSphere expects positive curvature K > 0."
            )
        return K_t

    @staticmethod
    def inverse_stereographic(y: Tensor, K: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Projected sphere D_K^m -> ambient sphere S_K^m.

        Returns z with shape (..., m+1):
            z[..., 0]  : pole-axis coordinate
            z[..., 1:] : spatial coordinates
        """
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        denom = 1.0 + K * y_sq

        z0 = (1.0 - K * y_sq) / (torch.sqrt(K) * torch.clamp(denom, min=eps))
        zs = 2.0 * y / torch.clamp(denom, min=eps)
        return torch.cat([z0, zs], dim=-1)

    @staticmethod
    def signed_distance_to_coordinate_hyperplanes(
        y: Tensor,
        K: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Vector of signed distances from y in D_K^m to the coordinate hyperplanes {y_k = 0}:

            d_k = (1/sqrt(K)) * asin( 2*sqrt(K)*y_k / (1 + K||y||^2) )
        """
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        arg = 2.0 * torch.sqrt(K) * y / (1.0 + K * y_sq)
        arg = torch.clamp(arg, min=-1.0 + eps, max=1.0 - eps)
        return torch.asin(arg) / torch.sqrt(K)

    def _decode_exact(
        self,
        v: Tensor,
        K: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        sqrt_K = torch.sqrt(K)

        # Euclidean-compatible convention:
        #   v = 0.5 * d_signed
        # Raw-distance convention:
        #   v = d_signed
        scale = 2.0 if self.euclidean_compatible else 1.0
        theta = scale * sqrt_K * v

        # Condition 1: principal arcsin branch
        limit = (math.pi / 2.0) - 1e-6
        principal_ok = torch.abs(theta) <= limit
        theta_used = theta
        clipped_mask = torch.zeros_like(theta, dtype=torch.bool)

        if not bool(principal_ok.all()):
            if self.principal_branch == "raise":
                max_abs = torch.abs(theta).max().item()
                raise RuntimeError(
                    "Exact projected-sphere decoder left the principal arcsin branch: "
                    f"max |theta| = {max_abs:.6f} > pi/2. "
                    "Reduce logits or set principal_branch='clip'."
                )
            clipped_mask = ~principal_ok
            theta_used = theta.clamp(min=-limit, max=limit)

        s = torch.sin(theta_used)
        u = (s * s).sum(dim=-1, keepdim=True)

        # Condition 2: joint spherical feasibility
        feasible_mask = u <= (1.0 + self.eps)
        projected_mask = torch.zeros_like(feasible_mask, dtype=torch.bool)

        s_used = s
        u_used = u

        needs_scale = u > (1.0 - 1e-6)
        if bool(needs_scale.any()):
            if self.infeasible == "raise" and not bool(feasible_mask.all()):
                max_u = u.max().item()
                raise RuntimeError(
                    "Exact projected-sphere decoder is infeasible: ||sin(theta)||^2 > 1 "
                    f"for at least one sample (max u = {max_u:.6f}). "
                    "Reduce logits or set infeasible='scale'."
                )

            if self.infeasible == "scale":
                projected_mask = needs_scale
            else:
                needs_scale = torch.zeros_like(needs_scale)
            s_norm = torch.sqrt(torch.clamp(u, min=self.eps))
            max_norm = math.sqrt(1.0 - 1e-6)
            factor = torch.clamp(
                max_norm / torch.clamp(s_norm, min=self.eps),
                max=1.0,
            )
            s_used = torch.where(needs_scale.expand_as(s), s * factor, s)
            u_used = (s_used * s_used).sum(dim=-1, keepdim=True)

        q = torch.sqrt(torch.clamp(1.0 - u_used, min=0.0))

        if self.branch == "north":
            denom = 1.0 + q
            ambient_sign = 1.0
        else:
            denom = 1.0 - q
            ambient_sign = -1.0
            if bool((denom <= self.eps).any()):
                raise RuntimeError(
                    "South branch hit the stereographic pole "
                    "(denominator too close to zero)."
                )

        y = s_used / (sqrt_K * torch.clamp(denom, min=self.eps))

        # Corresponding ambient sphere point
        z0 = ambient_sign * q / sqrt_K
        zs = s_used / sqrt_K
        z = torch.cat([z0, zs], dim=-1)

        # Recovered logits from the resulting projected-sphere point
        d_coord = self.signed_distance_to_coordinate_hyperplanes(
            y,
            K,
            eps=self.eps,
        )
        recovered_v = 0.5 * d_coord if self.euclidean_compatible else d_coord

        info = {
            "v": v,
            "theta": theta,
            "theta_used": theta_used,
            "principal_ok": principal_ok,
            "clipped_mask": clipped_mask,
            "s": s,
            "s_used": s_used,
            "u": u,
            "u_used": u_used,
            "feasible_mask": feasible_mask,
            "projected_mask": projected_mask,
            "ambient_point": z,
            "recovered_v": recovered_v,
        }
        return y, info

    def _decode_expmap(
        self,
        v: Tensor,
        K: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Projected-sphere exp map at the origin:
            y = tan(sqrt(K) * ||v||) / (sqrt(K) * ||v||) * v
        """
        sqrt_K = torch.sqrt(K)
        v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        safe_norm = torch.clamp(v_norm, min=self.eps)
        angle = sqrt_K * v_norm

        coef = torch.tan(angle) / (sqrt_K * safe_norm)
        coef = torch.where(v_norm > self.eps, coef, torch.ones_like(v_norm))
        y = coef * v

        z = self.inverse_stereographic(y, K, eps=self.eps)
        info = {
            "v": v,
            "angle": angle,
            "ambient_point": z,
        }
        return y, info

    def forward(
        self,
        x: Tensor,
        *,
        return_info: bool = False,
        return_ambient: bool = False,
    ):
        """
        x : (..., in_features)

        Returns
        -------
        y : (..., out_features)
            Projected-sphere coordinates in D_K^m

        Optional:
        ---------
        return_ambient=True
            also returns the corresponding ambient sphere point in S_K^m
            with shape (..., out_features + 1)

        return_info=True
            also returns diagnostic tensors
        """
        K = self._get_positive_curvature(x)
        v = self.input_scale * self.linear(x)

        if self.mode == "exact":
            y, info = self._decode_exact(v, K)
        elif self.mode == "expmap":
            y, info = self._decode_expmap(v, K)
        else:
            raise RuntimeError(f"Unexpected mode: {self.mode}")

        if self.mode == "exact":
            self._last_diagnostics = {
                'principal_clip_fraction': info['clipped_mask'].float().mean().detach(),
                'joint_scale_fraction': info['projected_mask'].float().mean().detach(),
                'max_abs_theta': torch.abs(info['theta']).max().detach(),
                'max_overshoot': torch.sqrt(torch.clamp(info['u'], min=0.0)).max().detach(),
                'euclidean_compatible': v.new_tensor(float(self.euclidean_compatible)),
                'input_scale': v.new_tensor(self.input_scale),
            }
        else:
            self._last_diagnostics = {
                'principal_clip_fraction': v.new_tensor(0.0),
                'joint_scale_fraction': v.new_tensor(0.0),
                'max_abs_theta': info['angle'].max().detach(),
                'max_overshoot': v.new_tensor(1.0),
                'euclidean_compatible': v.new_tensor(float(self.euclidean_compatible)),
                'input_scale': v.new_tensor(self.input_scale),
            }

        if return_ambient and return_info:
            return y, info["ambient_point"], info
        if return_ambient:
            return y, info["ambient_point"]
        if return_info:
            return y, info
        return y

    def get_diagnostics(self):
        return _diagnostics_to_floats(self._last_diagnostics)
