import dgl
import torch
import random
import os
import yaml
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import wandb
from time import time


# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DEVICE = 'cpu'
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')

from utils import generate_graph, get_gnn, run_gnn_training, qubo_dict_to_torch, loss_func
from lp_to_bqm import BQM, qubo_to_torch


# Graph hypers
d = 3
p = None
graph_type = 'complete'
graph_encoder = 'GraphSAGE'
# NN learning hypers #
number_epochs = int(1e5)
learning_rate = 1e-3
PROB_THRESHOLD = 0.5
sparsity_threshold = 0.5
leq_weight = 200
eq_weight = 200
scheduler_bool = False

# Early stopping to allow NN to train to near-completion
tol = 1e-4         # loss must change by more than tol, or trigger
patience = 100    # number early stopping triggers before breaking loop

bqm_model = BQM()
bqm_model.parse_lp("SrO.lp")

# print(bqm_model.constraints_dict)
# print(bqm_model.variables)
# print(bqm_model.quadratic)
Q, Qunc, C1, C2, variables = qubo_to_torch(bqm_model, eq_inf=eq_weight, leq_infinity=leq_weight) # pre-constrained Q matrix

means = Q.mean(dim = 1, keepdim = True)
stds = Q.std(dim = 1, keepdim = True)
#normalize Q matrix
normQ = (Q - means) / stds
#finding Q elements with low contributions
ind = torch.where(abs(normQ) < sparsity_threshold)

# Problem size (e.g. graph size)
n = Q.shape[0] # number of nodes

# Establish dim_embedding and hidden_dim values
# dim_embedding = int(np.sqrt(n))    # e.g. 10
# hidden_dim = int(dim_embedding/2)  # e.g. 5
dim_embedding = 32   # e.g. 10
hidden_dim =  32

# Constructs a random d-regular or p-probabilistic graph
nx_graph = generate_graph(n=n, d=d, p=p, graph_type=graph_type)
A = nx.adjacency_matrix(nx_graph).todense()
#sparsify adjacency matrix w.r.t. Q elements' contributions
A[ind] = 0
G = nx.from_numpy_matrix(A)
# Networkx does not enforce node order by default
nx_temp = nx.relabel.convert_node_labels_to_integers(G)
# Need to pull nx graph into OrderedGraph so training will work properly
nx_graph = nx.OrderedGraph()
nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
nx_graph.add_edges_from(nx_temp.edges)
# get DGL graph from networkx graph, load onto device
graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
graph_dgl = graph_dgl.to(TORCH_DEVICE)

# weight_decay = 1e-1
weight_decay = 0
# Establish pytorch GNN + optimizer
opt_params = {'lr': learning_rate, 'weight_decay': weight_decay}
gnn_hypers = {
    'dim_embedding': dim_embedding,
    'hidden_dim': hidden_dim,
    'dropout': 0.0,
    'number_classes': 1,
    'prob_threshold': PROB_THRESHOLD,
    'number_epochs': number_epochs,
    'tolerance': tol,
    'patience': patience
}
for seed_value in range(5, 8):
    random.seed(seed_value)        # seed python RNG
    np.random.seed(seed_value)     # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    run_hypers = {'lr': learning_rate,
                  'weight_decay': weight_decay,
                  'scheduler': scheduler_bool,
                  'seed': seed_value,
                  'sparsity threshold': sparsity_threshold,
                  'graph encoder': graph_encoder,
                  'leq_weight': leq_weight,
                  'eq_weight': eq_weight}
    run_hypers.update(gnn_hypers)

    wandb.init(
        # set the wandb project where this run will be logged
        project="CSP with GNNs",

        # track hyperparameters and run metadata
        config=run_hypers
    )
    # wandb.init(mode="disabled")
    net, embed, optimizer, scheduler = get_gnn(n, graph_encoder, gnn_hypers, opt_params, scheduler_bool, TORCH_DEVICE, TORCH_DTYPE)

    # For tracking hyperparameters in results object
    gnn_hypers.update(opt_params)

    print('Running GNN...')
    gnn_start = time()

    _, epoch, final_bitstring, best_bitstring = run_gnn_training(
        Qunc, C1, C2, bqm_model.offset, graph_dgl, net, embed, optimizer, scheduler, gnn_hypers['number_epochs'],
        gnn_hypers['tolerance'], gnn_hypers['patience'], gnn_hypers['prob_threshold'])

    gnn_time = time() - gnn_start


    final_loss = loss_func(final_bitstring.float(), Qunc, C1, C2, bqm_model.offset)
    final_bitstring_str = ','.join([str(x) for x in final_bitstring])
    wandb.run.summary["final_bitstring"] = final_bitstring
    wandb.finish(exit_code = None, quiet = None)
