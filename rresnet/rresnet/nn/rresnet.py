import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import create_network


class ProjVecField(nn.Module):
    def __init__(self, manifold, in_dim, hidden_dim, out_dim, n_hidden, act=nn.Tanh):
        super().__init__()
        self.func = create_network(in_dim, hidden_dim, out_dim, n_hidden, act=act)
        self.manifold = manifold

    def forward(self, x):
        v = self.func(x)
        return self.manifold.proju(x, v)


class GraphProjVecField(nn.Module):
    def __init__(self, manifold, in_dim, hidden_dim, out_dim, n_hidden, act=nn.Tanh, adj_power=2, mix_mode='sum'):
        super().__init__()
        func_in_dim = in_dim if mix_mode in ('sum', 'gate') else 2 * in_dim
        self.func = create_network(func_in_dim, hidden_dim, out_dim, n_hidden, act=act)
        self.manifold = manifold
        self.adj_power = adj_power
        self.mix_mode = mix_mode
        if mix_mode == 'gate':
            self.gate = nn.Linear(2 * in_dim, in_dim)

    def forward(self, x, adj):
        agg = x
        for _ in range(self.adj_power):
            agg = torch.spmm(adj, agg) if adj.is_sparse else torch.mm(adj, agg)
        if self.mix_mode == 'concat':
            feat = torch.cat([x, agg], dim=-1)
        elif self.mix_mode == 'gate':
            gate = torch.sigmoid(self.gate(torch.cat([x, agg], dim=-1)))
            feat = x + gate * agg
        else:
            feat = x + agg
        v = self.func(feat)
        return self.manifold.proju(x, v)


class SphereLocalGraphProjVecField(nn.Module):
    def __init__(self, manifold, in_dim, hidden_dim, out_dim, n_hidden, act=nn.Tanh, neighbor_scale=1.0, use_neighbors=True):
        super().__init__()
        self.self_func = create_network(in_dim, hidden_dim, out_dim, n_hidden, act=act)
        self.neighbor_func = create_network(in_dim, hidden_dim, out_dim, n_hidden, act=act)
        self.manifold = manifold
        self.neighbor_scale = neighbor_scale
        self.use_neighbors = use_neighbors

    def _aggregate_sparse(self, x, adj):
        adj = adj.coalesce()
        row, col = adj.indices()
        weight = adj.values().unsqueeze(-1)
        logs = self.manifold.log(x[row], x[col])
        out = torch.zeros_like(x)
        out.index_add_(0, row, weight * logs)
        return out

    def _aggregate_dense(self, x, adj):
        logs = torch.stack([self.manifold.log(x[i].expand_as(x), x) for i in range(x.size(0))], dim=0)
        return (adj.unsqueeze(-1) * logs).sum(dim=1)

    def forward(self, x, adj):
        self_term = self.self_func(x)
        total = self_term
        if self.use_neighbors and self.neighbor_scale != 0:
            tangent_summary = self._aggregate_sparse(x, adj) if adj.is_sparse else self._aggregate_dense(x, adj)
            neighbor_term = self.neighbor_func(tangent_summary)
            total = total + self.neighbor_scale * neighbor_term
        return self.manifold.proju(x, total)


class FeatureMapVecFieldSimple(nn.Module):
    def __init__(self, fi, interm_dim, feat_dim, manifold):
        super().__init__()
        self.fi = fi
        self.interm_dim = interm_dim
        self.feat_dim = feat_dim

        self.interm = nn.Linear(feat_dim, interm_dim)
        self.bn1 = nn.BatchNorm1d(interm_dim)
        self.coeffs = nn.Linear(interm_dim, feat_dim)
        self.manifold = manifold

    def forward(self, x):
        feat = self.fi(x)
        interm = self.bn1(F.relu(self.interm(feat)))
        coeffs = self.coeffs(interm)  # bs x d
        new_vecs = torch.autograd.grad(
            self.fi(x), x, grad_outputs=coeffs, create_graph=True)
        ret = self.manifold.proju(x, new_vecs[0])
        return ret


class FeatureMapVecFieldSimpleGraphBase(nn.Module):
    def __init__(self, fi, interm_dim, feat_dim, manifold, adj_power=2):
        super().__init__()
        self.fi = fi
        self.interm_dim = interm_dim
        self.feat_dim = feat_dim
        self.adj_power = adj_power

        self.interm = nn.Linear(feat_dim, interm_dim)
        self.bn1 = nn.BatchNorm1d(interm_dim)
        self.coeffs = nn.Linear(interm_dim, feat_dim)
        self.manifold = manifold

    def forward(self, x, adj):
        outer_grad_enabled = torch.is_grad_enabled()
        # This vector field needs a feature-map derivative during evaluation too.
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            feat = self.fi(x)
            for _ in range(self.adj_power):
                feat = torch.spmm(adj, feat) if adj.is_sparse else torch.mm(adj, feat)
            interm = self.bn1(F.leaky_relu(self.interm(feat), negative_slope=0.5))
            coeffs = self.coeffs(interm)  # bs x d
            new_vecs = torch.autograd.grad(
                self.fi(x), x, grad_outputs=coeffs, create_graph=outer_grad_enabled)
            ret = self.manifold.proju(x, new_vecs[0])
        return ret if outer_grad_enabled else ret.detach()


class RResNet(nn.Module):
    def __init__(self, manifold, vector_fields):
        """
        manifold (Manifold): Input Manifold
        vector_fields (nn.Module list): List of modules that map M -> TM
        """
        super().__init__()

        self.manifold = manifold
        self.vector_fields = nn.ModuleList(vector_fields)

    def forward(self, x, adj=None):
        for v_f in self.vector_fields:
            vf = v_f(x) if adj is None else v_f(x, adj)
            x = self.manifold.exp(x, vf)
            x = self.manifold.projx(x)

        return x
