import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)
        return x


class GraphConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, args):
        super(GraphConv, self).__init__()
        self.dropout = args.gnn_dropout
        self.num_layers = args.gnn_num_layers
        self.dropout = args.gnn_dropout
        self.use_bn = args.gnn_use_bn
        self.use_residual = args.gnn_use_residual
        self.use_weight = args.gnn_use_weight
        self.use_init = args.gnn_use_init
        self.use_act = args.gnn_use_act

        # num_layers = args., dropout = 0.5, use_bn = True, use_residual = True,
        # use_weight = True, use_init = False, use_act = True
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(self.num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, self.use_weight, self.use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = self.dropout
        self.activation = F.relu
        self.use_bn = self.use_bn
        self.use_residual = self.use_residual
        self.use_act = self.use_act
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x
