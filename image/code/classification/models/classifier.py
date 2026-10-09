import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.geoopt.manifolds.stereographic import PoincareBall
from lib.lorentz.layers import LorentzMLR
from lib.lorentz.manifold import CustomLorentz
from lib.models.resnet import resnet18
from lib.poincare.layers import UnidirectionalPoincareMLR


RESNET_MODEL = {"euclidean": {18: resnet18}}


class ResNetClassifier(nn.Module):
    """Euclidean ResNet-18 followed by a PHDM manifold classifier."""

    def __init__(
        self,
        num_layers: int,
        enc_type: str = "euclidean",
        dec_type: str = "lorentz",
        enc_kwargs=None,
        dec_kwargs=None,
    ):
        super().__init__()
        enc_kwargs = dict(enc_kwargs or {})
        dec_kwargs = dict(dec_kwargs or {})

        self.enc_type = enc_type
        self.dec_type = dec_type
        self.clip_r = dec_kwargs["clip_r"]
        self.phdm_input_scale = float(os.environ.get("HYPERCV_PHDM_INPUT_SCALE", "0.1"))
        if enc_type != "euclidean" or num_layers != 18:
            raise ValueError("PHDM image configurations use the Euclidean ResNet-18 encoder.")

        self.encoder = RESNET_MODEL[enc_type][num_layers](remove_linear=True, **enc_kwargs)
        self.enc_manifold = self.encoder.manifold

        embed_dim = dec_kwargs["embed_dim"] * self.encoder.block.expansion
        num_classes = dec_kwargs["num_classes"]
        decoder_type = dec_kwargs["type"]
        self.dec_manifold = None
        self.projector = None

        if dec_type == "lorentz":
            self.dec_manifold = CustomLorentz(
                k=dec_kwargs["k"], learnable=dec_kwargs["learn_k"]
            )
            self.decoder = LorentzMLR(self.dec_manifold, embed_dim + 1, num_classes)
            self.projector = PHDM_Lorentz(
                self.dec_manifold, embed_dim, embed_dim
            )
        elif dec_type == "poincare":
            self.dec_manifold = PoincareBall(
                c=dec_kwargs["k"], learnable=dec_kwargs["learn_k"]
            )
            self.decoder = UnidirectionalPoincareMLR(
                embed_dim, num_classes, True, self.dec_manifold
            )
            self.projector = PHDM_Poincare(
                self.dec_manifold, embed_dim, embed_dim
            )
        else:
            raise ValueError(f"Unsupported decoder manifold: {dec_type}")

        if decoder_type != "mlr":
            raise ValueError(f"Unsupported decoder type: {decoder_type}")

    @staticmethod
    def _scale_to_radius(x, radius, maximum_scale=1.0):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-15)
        scale = torch.minimum(
            torch.full_like(norm, maximum_scale),
            torch.as_tensor(radius, dtype=x.dtype, device=x.device) / norm,
        )
        return scale * x

    def check_manifold(self, x):
        if not torch.isfinite(x).all():
            raise FloatingPointError("Non-finite Euclidean features before PHDM map")
        x = self._scale_to_radius(x, self.clip_r, self.phdm_input_scale)
        return self.projector(x)

    def embed(self, x):
        return self.check_manifold(self.encoder(x))

    def forward(self, x):
        return self.decoder(self.embed(x))


class PHDM_Lorentz(nn.Module):
    """PHDM map from Euclidean logits to a Lorentz hyperboloid.

    The Euclidean ResNet already produces the logits, so this module has no
    extra affine parameters.
    """

    def __init__(self, manifold, in_features, out_features, bias=True, eps=1e-9):
        super().__init__()
        if in_features != out_features:
            raise ValueError("The parameter-free image projector preserves dimension")
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

    def forward(self, x):
        curvature = self.manifold.k.to(dtype=x.dtype, device=x.device).clamp_min(self.eps)
        sqrt_curvature = torch.sqrt(curvature)
        spatial = torch.sinh(sqrt_curvature * x) / sqrt_curvature
        time = torch.sqrt(
            curvature.reciprocal()
            + spatial.square().sum(dim=-1, keepdim=True)
            + self.eps
        )
        return torch.cat((time, spatial), dim=-1)


class PHDM_Poincare(nn.Module):
    """PHDM closed-form map from Euclidean logits to the Poincare ball."""

    def __init__(
        self,
        manifold=None,
        in_features=1,
        out_features=1,
        c=None,
        eps=1e-9,
        bias=True,
    ):
        super().__init__()
        if in_features != out_features:
            raise ValueError("The parameter-free image projector preserves dimension")
        if manifold is None and c is None:
            raise ValueError("Provide a Poincare manifold or a positive curvature magnitude")
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        if c is None:
            self.register_buffer("explicit_c", None)
        else:
            self.register_buffer("explicit_c", torch.as_tensor(float(c)))

    def _curvature(self, x):
        curvature = self.explicit_c if self.explicit_c is not None else self.manifold.c
        return curvature.to(dtype=x.dtype, device=x.device).clamp_min(self.eps)

    def forward(self, x):
        curvature = self._curvature(x)
        sqrt_curvature = torch.sqrt(curvature)
        coordinates = torch.sinh(sqrt_curvature * x) / sqrt_curvature
        squared_norm = coordinates.square().sum(dim=-1, keepdim=True)
        denominator = 1.0 + torch.sqrt(1.0 + curvature * squared_norm + self.eps)
        return coordinates / denominator
