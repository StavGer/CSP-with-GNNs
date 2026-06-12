import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  MLP, MessagePassing
from torch_geometric.nn.dense.linear import Linear as PyGLinear


class BondEncoderLinear(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super(BondEncoderLinear, self).__init__()
        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, edge_attr):
        return self.mlp(edge_attr)



class GINEConv(MessagePassing):
    def __init__(self, bond_encoder, in_channels, out_channels, **kwargs):

        super(GINEConv, self).__init__(aggr="add")

        self.mlp = MLP(
            channel_list=[in_channels, out_channels, out_channels],
            act="gelu",
        )

        self.eps = torch.nn.Parameter(torch.Tensor([0.0]))

        if bond_encoder is not None:
            self.bond_encoder = torch.nn.Sequential(
                bond_encoder, PyGLinear(-1, in_channels)
            )
        else:
            self.bond_encoder = None

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        if edge_weight is not None and edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None]

        if edge_attr is not None:
            edge_embedding = (
                self.bond_encoder(edge_attr) if edge_attr is not None else None
            )

        out = self.mlp(
            (1 + self.eps) * x
            + self.propagate(
                edge_index,
                x=x,
                edge_attr=edge_embedding if edge_attr is not None else None,
                edge_weight=edge_weight if edge_weight is not None else None,
            )
        )
        return out

    def message(self, x_j, edge_attr, edge_weight):
        m = F.gelu(x_j + edge_attr) if edge_attr is not None else x_j
        return m * edge_weight if edge_weight is not None else m

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        # Reset parameters of MLP
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Reset parameters of BatchNorm
        self.bn.reset_parameters()

        # Reset parameters of bond_encoder if it exists
        if self.bond_encoder is not None:
            for layer in self.bond_encoder:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # Reset the epsilon parameter
        torch.nn.init.constant_(self.eps, 0.0)
    

class GINModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        bond_encoder=None,
        device=None,
        **kwargs,
    ):
        super(GINModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            hidden_channels = out_channels
        self.convs.append(GINEConv(bond_encoder, in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers - 1):
            if _ == num_layers - 2:
                self.convs.append(
                    GINEConv(bond_encoder, hidden_channels, out_channels, **kwargs)
                )
            else:
                self.convs.append(GINEConv(bond_encoder, hidden_channels, hidden_channels, **kwargs))

        self.device = device

    def reset_parameters(self):
        # Reset parameters of each GINEConv layer
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, data, edge_weight=None):
        x, edge_index = (
            data.x.to(self.device),
            data.edge_index.to(self.device),
        )
        if data.edge_attr is not None:
            edge_attr = data.edge_attr.to(self.device)
        else:
            edge_attr = None
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=self.device)
        else:
            edge_weight.to(self.device)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr, edge_weight=edge_weight)
            if i != len(self.convs) - 1:
                x = F.relu(x)

        return x

    

