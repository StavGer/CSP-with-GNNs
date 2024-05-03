import dgl
import torch
import random
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from time import time
import torch.nn.functional as F
from models.models_dgl import GNNConv
import matplotlib.pyplot as plt
import torch.nn as nn

from utils import generate_graph, crystal_coordinates, gaussian_kernel, interactions_to_edge_features
from potts_utils import loss_func
from lp_to_bqm import BQM, qubo_to_torch
from dgl.nn.pytorch import SAGEConv, GraphConv, GATConv, GINConv

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.set_printoptions(linewidth=180)
torch.set_printoptions(linewidth=180)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Perform Hyperparameter Search using Hyperopt"
    )

    # Instance
    parser.add_argument("--instances-dir", type=Path, help="path to instance")

    # Graph Encoder
    parser.add_argument("--Graph-Encoder", type=str, default="GraphSAGE")

    # LR scheduler
    parser.add_argument("--LR Scheduler", type=str, default="None")

    # Optim
    parser.add_argument(
        "--epochs",
        default=1e5,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    return parser


def set_seed(seed):
    """
    Sets random seeds for training.

    :param seed: Integer used for seed.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')

t_script_start = time()

# ## PROBLEM SETUP ##
# file_to_parse = 'Y2Ti2O7G8.lp'
file_to_parse = 'SrTiO3G4.lp'

graph_type = 'complete'

# GNN hypers
graph_encoder = 'GraphSAGE'
number_epochs = int(1e5) # max epoch number
PROB_THRESHOLD = 0.5

leq_weight = 1  # penalty weight for atom or void per position constraints
eq_weight = 100  # penalty weight for stoichiometry constraints

# Early stopping to allow NN to train to near-completion
tol = 1e-3         # loss must change by more than tol, or trigger

Gumbel_sinkhorn = False  # if False Potts model, if True G.S. Potts
with_void = True

# Set up problem to solve
bqm_model = BQM(Potts=True, Gumbel_sinkhorn=Gumbel_sinkhorn)

struct_info = file_to_parse.split('G')
struct_name = struct_info[0]
g = int(struct_info[1][0])  # cell discretization
bqm_model.parse_lp("instances/" + file_to_parse)

# pre-constrained Q matrix
Q, num_positions, atoms, stoich_const = qubo_to_torch(
    bqm_model, eq_inf=eq_weight, leq_inf=leq_weight, with_void=with_void, Gumbel_sinkhorn=Gumbel_sinkhorn,
    torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE
)
pos = crystal_coordinates(struct_name, g, num_positions) # atoms' candidate 3d coordinates within the unit cell #except for SrO

n_atoms = len(atoms)
num_classes = (n_atoms + 1) if with_void else n_atoms  # number of node classes
n = num_positions  # number of graph nodes = number of crystal's positions
print("Crystal Structure Prediction of " + file_to_parse.split('.')[0] + " with " + str(num_positions)
      + " positions" + " and " + str(atoms) + " atoms.")

# Constructs a complete graph
lr_scheduler_type = "No"  # learning rate scheduler
cutoff_dist = 3 if graph_type == "cutoff" else 0
nx_graph = generate_graph(
    n=n, d=None, p=None, pos=pos, cutoff_dist=cutoff_dist, graph_type=graph_type
)

# get DGL graph from networkx graph, load onto device
graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
graph_dgl = graph_dgl.to(TORCH_DEVICE)

distvec = pdist(pos)
square_dist = squareform(distvec)  # square distance matrix between positions

node_dist = list(square_dist.reshape(square_dist.shape[0]*square_dist.shape[1]))  # flatten node distances
node_dist = [i for i in node_dist if i != 0]  # remove 0s
if cutoff_dist > 0:
    node_dist = [i for i in node_dist if i < cutoff_dist]  # remove distances of nodes with distance greater than cutoff
node_dist = torch.tensor(node_dist)
gaussian_dist = gaussian_kernel(node_dist, sigma=1).to(TORCH_DEVICE)

pos = torch.tensor((pos-np.min(pos))/(np.max(pos)-np.min(pos))).to(TORCH_DEVICE)
graph_dgl.ndata['pos'] = torch.tensor(pos).type(TORCH_DTYPE)  # add 3D coordinates as nodes' features

# graph_dgl.edata["edge_weights"] = gaussian_dist.type(TORCH_DTYPE)
# if graph_encoder == "GAT":
#     graph_dgl = dgl.add_self_loop(graph_dgl)  # required for GAT

patience = np.inf
agg_type = 'mean'
hypers = {
    # Params to search over
    # 'dim_embedding': 32,
    'hidden_dim': 128,
    'dropout': 0.10,
    'weight_decay': 0.000,
    'lr': 5e-5,
    'temperature': 1,
    'scaling': 1,
    # possibly experiment with layer_agg_type
    # Fixed params - GNN
    'number_classes': num_classes,
    'prob_threshold': PROB_THRESHOLD,
    'number_epochs': number_epochs,
    'tolerance': tol,
    'patience': patience,  # turn off patience (temporarily)
    'model': graph_encoder,
    'layer_agg_type': agg_type,
    # Fixed params - problem
    'scheduler': lr_scheduler_type,
    'leq_weight': leq_weight,
    # 'scaling' : 1.1,
    'eq_weight': eq_weight
}

# Parse hyperparameters
gnn_hypers = hypers
opt_keys = ['lr', 'weight_decay']
opt_params = {k: gnn_hypers.pop(k) for k in opt_keys}

lr_scheduler_type = gnn_hypers.pop('scheduler')

print(f'Weight decay: {opt_params["weight_decay"]}')
print(f'Learning rate: {opt_params["lr"]}')
print(f'Temperature Scaling of logits: {gnn_hypers["temperature"]}')

seed_value = 0
print('Running GNN...')

random.seed(seed_value)  # seed python RNG
np.random.seed(seed_value)  # seed global NumPy RNG
torch.manual_seed(seed_value)  # seed torch RNG
gnn_hypers["seed"] = seed_value

# Initialize with different strategy
def weights_init(m):
    if isinstance(m, nn.Linear):
        print('updating linear layer init')
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


# Define simple fully connected net
class TestNet(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(TestNet, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hidden_size))
        # output layer
        self.layers.append(nn.Linear(hidden_size, num_classes))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = F.dropout(h, p=0.2)
            h = F.leaky_relu_(h, negative_slope=1e-2)

        return h


# Define GNN GraphConv object
class GNNConv(nn.Module):

    def __init__(self, g, in_feats, hidden_size, num_classes, dropout):
        """
        Initialize the model object. Establishes model architecture and relevant hypers (`dropout`, `num_classes`, `agg_type`)

        :param g: Input graph object
        :type g: dgl.DGLHeteroGraph
        :param in_feats: Size (number of nodes) of input layer
        :type in_feats: int
        :param hidden_size: Size of hidden layer
        :type hidden_size: int
        :param num_classes: Size of output layer (one node per class)
        :type num_classes: int
        :param dropout: Dropout fraction, between two convolutional layers
        :type dropout: float
        """

        super(GNNConv, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, hidden_size))
        # output layer
        self.layers.append(GraphConv(hidden_size, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):

        h = features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
            h = self.dropout(h)
            h = F.leaky_relu_(h, negative_slope=1e-2)

        return h


# net = TestNet(in_feats=8, hidden_size=32, num_classes=4)
net = GNNConv(graph_dgl, in_feats=num_positions, hidden_size=Q.shape[0], num_classes=num_classes, dropout=0.2)
net = net.type(TORCH_DTYPE).to(TORCH_DEVICE)
# embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)

net.apply(weights_init)

# set up Adam optimizer
# print('Building ADAM-W optimizer...')
# optimizer = torch.optim.AdamW(net.parameters(), **opt_params)
optimizer = torch.optim.Adam(net.parameters(), lr=opt_params['lr'], weight_decay=opt_params['weight_decay'])

# model inputs will be one-hot encoded node IDs
n_var = num_positions * (n_atoms + 1)
inputs = torch.nn.functional.one_hot(torch.arange(num_positions))
inputs = inputs.float()

# For tracking hyperparameters in results object
gnn_hypers.update(opt_params)

# Ensure RNG seeds are reset each training run
print(f'Function run_gnn_training(): Setting seed to {seed_value}')
set_seed(seed_value)

best_bitstring = torch.zeros((graph_dgl.number_of_nodes(), stoich_const.shape[0])).type(Q.dtype).to(graph_dgl.device)
bitstring = torch.zeros((graph_dgl.number_of_nodes(), stoich_const.shape[0])).type(Q.dtype).to(graph_dgl.device)

# Early stopping to allow NN to train to near-completion
prev_loss = 1.  # initial loss value (arbitrary)
cnt = 0  # track number times early stopping is triggered
best_loss = 1e8

# Training logic
bitstring_o = bitstring
probs_save = []
bitstring_save = []
loss_track = []

number_epochs = 500000

t_gnn_start = time()

# TODO - temp
# Q_test = Q.triu(diagonal=1).T + torch.diag(Q.diag()) + Q.triu(diagonal=1)
Q_test = Q
for epoch in range(number_epochs):
    # get soft prob assignments
    logits = net(inputs)
    # apply softmax for normalization
    probs = F.softmax(logits, dim=1)

    loss = loss_func(Q_test, probs, bqm_model.offset, stoich_const)
    loss_track.append(loss.item())
    bitstring_n = (probs.detach() >= hypers['prob_threshold']) * 1

    # # TEST - test sample solution
    # pr = torch.zeros(size=(8, 4))
    # pr[0, 2] = 1.  # Ti at position 0
    # pr[1, 0] = 1.  #  O at position 1
    # pr[2, 0] = 1.  #  O at position 2
    # pr[3, 3] = 1.  # Null
    # pr[4, 0] = 1.  #  O at position 4
    # pr[5, 3] = 1.  # Null
    # pr[6, 3] = 1.  # Null
    # pr[7, 1] = 1.  # Sr at position 7
    #
    # loss_func(Q_test, pr, bqm_model.offset, stoich_const)

    probs_save.append(probs.detach())
    bitstring_save.append(bitstring_n.detach())

    hard_loss = loss_func(Q_test, bitstring_n.float(), bqm_model.offset, stoich_const)

    # Early stopping check
    # If loss increases or change in loss is too small, trigger
    if (abs(loss - prev_loss) <= hypers['tolerance']) | (loss >= best_loss):
        cnt += 1
    else:
        cnt = 0

    if loss < best_loss:
        best_loss = loss
        best_bitstring = bitstring_n
        best_loss_epoch = epoch
        best_probs = probs

    prev_loss = loss

    if cnt >= patience:
        print(f'Stopping early on epoch {epoch}. Patience count: {cnt}')
        break
    # run optimization with backpropagation
    optimizer.zero_grad()  # clear gradient for step
    loss.backward()  # calculate gradient through compute graph
    optimizer.step()  # take step, update weights
    # tracking: print intermediate loss at regular interval
    if epoch % 100 == 0:
        print('Epoch %d | Total Soft Loss: %.5f' % (epoch, loss.item()))
        print('Epoch %d | Total Hard Loss: %.5f' % (epoch, hard_loss.item()))
        # if epoch % 1000 == 0:
            # TODO - gradient curves showing some terminal nodes contain no information,
            #  which leads to signal loss and zero gradients throughout the layers. Why??
            # layer = net.layers[1]
            # fig, ax = plt.subplots(1, 1)
            # im = ax.imshow(layer.weight.grad, cmap='viridis')
            # # axarr[1].imshow(layer.weight.grad == 0)
            # plt.colorbar(im)
            # plt.title(f'Last layer gradients, epoch {epoch}')
            # plt.show()
            # for l_id, layer in enumerate(net.layers):
            #     w_grad = layer.weight.grad.mean(axis=1)
            #     print(f'Layer {l_id} gradient: {w_grad}')

    bitstring_o = bitstring_n

final_hard_loss = loss_func(Q_test, bitstring_n.float(), bqm_model.offset, stoich_const)
best_hard_loss = loss_func(Q_test, best_bitstring.float(), bqm_model.offset, stoich_const)

t_gnn = round(time() - t_gnn_start, 3)
t_epoch = round(t_gnn / number_epochs, 3)
t_script = round(time() - t_script_start, 3)

# NOTE: Optimal loss on SrTiO3G2 = -158.7579
#       Optimal loss on SrTiO3G4 = -158.7579  (same problem, more voids)
print(f'Final Hard loss: {final_hard_loss.item()}')
print(f'Best Hard loss: {best_hard_loss.item()} (found on epoch {best_loss_epoch})')
print(f'Script took: {t_script}s . Model training took: {round(t_gnn, 2)}s (per epoch: {t_epoch}s)')

# DEBUGGING
plt.figure()
plt.plot(loss_track)
plt.title('Loss curve')
plt.show()

# import pandas as pd
# import itertools
# atoms = ['O', 'Sr', 'Ti', 'NULL']
# positions = range(0, 8)
# cols = [f'{y}_{x}' for x in positions for y in atoms]
# pd.DataFrame(Q.numpy(), columns=cols, index=cols).to_csv('Q_mat.csv')
