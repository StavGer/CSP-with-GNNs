import dgl
import torch
import random
import numpy as np
import pickle
import networkx as nx
import functools
# import wandb
from datetime import datetime

from copy import copy
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll import scope
from time import time

# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')

from utils import generate_graph, get_gnn, run_gnn_training, loss_func
from lp_to_bqm import BQM, qubo_to_torch

# Graph hypers
d = 3
p = None
graph_type = 'complete'

# GNN hypers
graph_encoder = 'GraphSAGE'
number_epochs = int(1e4)
PROB_THRESHOLD = 0.5

# Problem parameters/hypers
sparsity_threshold = 0
leq_weight = 200
eq_weight = 200
scheduler_bool = False

# Early stopping to allow NN to train to near-completion
tol = 1e-4         # loss must change by more than tol, or trigger

# Set up problem to solve
bqm_model = BQM(Potts = False)
bqm_model.parse_lp("instances/LiMgAlPO_G8.lp")
with_void = False
# pre-constrained Q matrix
Q, Qunc, C1, C2, variables, n_atoms = qubo_to_torch(
    bqm_model, eq_inf=eq_weight, leq_infinity=leq_weight, with_void = with_void,
    torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE
)

num_positions = int(len(variables)/n_atoms)
num_variables = (n_atoms + 1)*num_positions # with void
num_variables = n_atoms*num_positions # without void
print(num_variables, "Variables in the binary program")

n = num_variables # number of nodes
# Constructs a random d-regular or p-probabilistic graph
nx_graph = generate_graph(n=n, d=d, p=p, Q = Q, sparsity_threshold=0, graph_type=graph_type)

# get DGL graph from networkx graph, load onto device
graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
graph_dgl = graph_dgl.to(TORCH_DEVICE)


seed_value = 0
random.seed(seed_value)        # seed python RNG
np.random.seed(seed_value)     # seed global NumPy RNG
torch.manual_seed(seed_value)  # seed torch RNG


def model_step(hypers, Qunc, C1, C2, bqm_model, graph_dgl, torch_device, torch_dtype):
    # Parse hyperparameters
    run_hypers = copy(hypers)
    gnn_hypers = hypers

    opt_keys = ['lr', 'weight_decay']
    opt_params = {k: gnn_hypers.pop(k) for k in opt_keys}

    scheduler_bool = gnn_hypers.pop('scheduler')
    graph_encoder = gnn_hypers.pop('graph encoder')

    print(f'Weight decay: {opt_params["weight_decay"]}')
    print(f'Learning rate: {opt_params["lr"]}')
    print(f'Temperature Scaling of logits: {gnn_hypers["temperature"]}')


    net, embed, optimizer, scheduler = get_gnn(
        Qunc.shape[0], graph_encoder, gnn_hypers, opt_params, scheduler_bool,
        torch_device, torch_dtype
    )

    # For tracking hyperparameters in results object
    gnn_hypers.update(opt_params)

    print('Running GNN...')
    gnn_start = time()

    _, epoch, final_bitstring, best_bitstring = run_gnn_training(
        Qunc, C1, C2, bqm_model.offset, graph_dgl, net, embed, optimizer, scheduler, gnn_hypers['number_epochs'],
        gnn_hypers['tolerance'], gnn_hypers['patience'], gnn_hypers['prob_threshold'])

    gnn_time = time() - gnn_start

    final_loss = loss_func(final_bitstring.float(), Qunc, C1, C2, bqm_model.offset)
    final_bitstring_str = ','.join([str(x.item()) for x in final_bitstring])
    best_bitstring_str = ','.join([str(x.item()) for x in best_bitstring])
    print(f'Step took: {round(gnn_time, 2)}s')
    print(f'Final bitstring: {final_bitstring_str}')
    print(f'Best bitstring: {best_bitstring_str}')


    return {'loss': final_loss[0], 'final_bitstring' : final_bitstring_str, 'best_bitstring' : best_bitstring_str, 'status': STATUS_OK}


search_space = {
    # Params to search over
    'dim_embedding': scope.int(hp.uniform('dim_embedding', 16, 64)),
    'hidden_dim': scope.int(hp.uniform('hidden_dim', 16, 64)),
    'dropout': scope.float(hp.uniform('dropout', 0.0, 0.5)),
    'weight_decay': scope.float(hp.loguniform('weight_decay', -5, -1)),
    'lr': scope.float(hp.loguniform('lr', -5, -2)),
    'temperature': scope.float(hp.uniform('temperature', 1.0, 3.0)),
    'patience': scope.int(hp.uniform('patience', 20, 200)),
    # Fixed params - GNN
    'number_classes': 1,
    'prob_threshold': PROB_THRESHOLD,
    'number_epochs': number_epochs,
    'tolerance': tol,
    # 'patience': 100,  # turn off patience (temporarily)
    # Fixed params - problem
    'scheduler': scheduler_bool,
    'seed': seed_value,
    'sparsity threshold': sparsity_threshold,
    'graph encoder': graph_encoder,
    'leq_weight': leq_weight,
    'eq_weight': eq_weight
}

obj_func = functools.partial(
    model_step, Qunc=Qunc, C1=C1, C2=C2, bqm_model=bqm_model, graph_dgl=graph_dgl,
    torch_device=TORCH_DEVICE, torch_dtype=TORCH_DTYPE
)

# do hpo
best = fmin(
    obj_func,
    space=search_space,
    algo=tpe.suggest,
    max_evals=100
)

