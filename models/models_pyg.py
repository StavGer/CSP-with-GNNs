import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, MessagePassing

class SAGECONV_edges(MessagePassing):

    def __init__(self,
                 in_channels_feats,
                 in_channels_edges,
                 out_channels,
                 agg_type='max',
                 root_weight = True):

        super(SAGECONV_edges, self).__init__()

        self.in_channels_feats = in_channels_feats
        self.in_channels_edges = in_channels_edges
        self.out_channels = out_channels
        self.aggr = agg_type
        self.root_weight = root_weight
        self.lin1 = Linear(in_channels_edges + in_channels_feats, out_channels, bias=True)
        self.lin2 = Linear(in_channels_feats, out_channels, bias=True)


    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.lin2(x) + out

    def message(self, x_j, edge_attr):
        return self.lin1(torch.cat([x_j, edge_attr], dim=-1))

class SAGE_with_EdgeConv(torch.nn.Module):

    def __init__(self, num_in_channels_feats: int, num_in_channels_edges: int, hidden_dim: int,
                 num_out_channels: int, dropout):

        super(SAGE_with_EdgeConv, self).__init__()

        self.num_in_channels_feats = num_in_channels_feats
        self.num_in_channels_edges = num_in_channels_edges
        self.hidden_dim = hidden_dim
        self.num_out_channels = num_out_channels
        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGECONV_edges(self.num_in_channels_feats, self.num_in_channels_edges, self.hidden_dim))
        # output layer
        self.layers.append(SAGECONV_edges(self.hidden_dim, self.num_in_channels_edges, self.num_out_channels))


    def forward(self, data):
        x = data.x
        #adj = SparseTensor(row = data.edge_index[0], col = data.edge_index[1], value = data.edge_attr)
        for i, layer in enumerate(self.layers):
            if i!=0:
                x = self.dropout(x)
            x = layer(x, data.edge_index, data.edge_attr)
            if i!=1:
                x = F.relu(x)
        return x
