import dgl
import torch
import random
import numpy as np
import pickle
import networkx as nx
import wandb
import matplotlib
from time import time
import joblib


print(torch.__version__)
print(dgl.__version__)
print(nx.__version__)
print(matplotlib._get_version())
SEED_VALUE = 0
random.seed(SEED_VALUE)        # seed python RNG
np.random.seed(SEED_VALUE)     # seed global NumPy RNG
torch.manual_seed(SEED_VALUE)  # seed torch RNG
# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')

from utils import generate_graph
from potts_utils import get_gnn, run_gnn_training, loss_func
from lp_to_bqm import BQM, qubo_to_torch


Gumbel_sinkhorn = True
file_to_parse = 'SrTiO3G2.lp'
file_to_write = file_to_parse.split('.')[0]
trials_history = joblib.load('trials_file_GS_' + str(Gumbel_sinkhorn) + '_' + file_to_write)

losses = []
for i in range(len(trials_history)):
    losses.append(trials_history.results[i]['loss'])
desc_losses_ind = np.argsort(losses)


bqm_model = BQM(Potts = True, Gumbel_sinkhorn = Gumbel_sinkhorn)
bqm_model.parse_lp("instances/" + file_to_parse)

# Graph hypers
d = 3
p = None
graph_type = 'sparse'
graph_encoder = 'GraphSAGE'
# NN learning hypers #
number_epochs = int(1e5)
PROB_THRESHOLD = 0.5
sparsity_threshold = 0.0
leq_weight = 200
eq_weight = 200
scheduler_bool = False

with_void = True # include void for each position
Q, elements, n_atoms, stoich_const = qubo_to_torch(bqm_model, eq_inf=eq_weight, leq_infinity=leq_weight,
                                                   with_void = with_void,  torch_dtype=TORCH_DTYPE, Gumbel_sinkhorn=Gumbel_sinkhorn,
                                                   torch_device=TORCH_DEVICE) # pre-constrained Q matrix

num_classes = (n_atoms + 1) if with_void else n_atoms
num_positions = int(len(elements)/n_atoms)
print(num_positions, "Variables in the MultiClass Classification Problem")
# Problem size (e.g. graph size)
n = num_positions # number of nodes

# Sample hyperparameters
if TORCH_DEVICE.type == 'cpu':  # example with CPU
    hypers = {
        'model': 'GraphConv',   # set either with 'GraphConv' or 'GraphSAGE'. It cannot take other input
        'dim_embedding': 64,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'hidden_dim': 64,
        'seed': SEED_VALUE,
        'temperature' : 1
    }
else:                           # example with GPU
    hypers = {
        'model': 'GraphSAGE',
        'dim_embedding': int(trials_history.argmin['dim_embedding']),
        'dropout': trials_history.argmin['dropout'],
        'learning_rate': trials_history.argmin['lr'],
        'hidden_dim': int(trials_history.argmin["hidden_dim"]),
        'seed': SEED_VALUE,
        'temperature': trials_history.argmin["temperature"],
    }
# Default meta parameters
solver_hypers = {
    'tolerance': 1e-3,           # Loss must change by more than tolerance, or add towards patience count
    'number_epochs': int(5e4),   # Max number training steps
    'patience': 500,             # Number early stopping triggers before breaking loop
    'layer_agg_type': 'mean',    # How aggregate neighbors sampled within graphSAGE
    'number_classes': num_classes
}

# Combine into a single set
hypers.update(solver_hypers)


# Constructs a random d-regular or p-probabilistic graph
nx_graph = generate_graph(n=n, d=d, p=p, Q=Q, sparsity_threshold= sparsity_threshold, graph_type=graph_type)


# get DGL graph from networkx graph, load onto device
graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
graph_dgl = graph_dgl.to(TORCH_DEVICE)

# Establish pytorch GNN + optimizer
# Retrieve known optimizer hypers
opt_hypers = {
    'lr': hypers.get('learning_rate'),
    'weight_decay' : trials_history.argmin["weight_decay"]
}

gnn_hypers = {
    'dim_embedding': hypers['dim_embedding'],
    'hidden_dim': hypers['hidden_dim'],
    'dropout': hypers['dropout'],
    'number_classes': num_classes,
    'prob_threshold': PROB_THRESHOLD,
    'number_epochs': solver_hypers['number_epochs'],
    'tolerance': solver_hypers['tolerance'],
    'patience': solver_hypers['patience']
}

avg_best_hard_loss = 0
min_energy_found = 0
for seed_value in range(10, 20):
    random.seed(seed_value)        # seed python RNG
    np.random.seed(seed_value)     # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG

    hypers["seed"] = seed_value
    net, embed, optimizer, scheduler = get_gnn(graph_dgl, nx_graph.number_of_nodes(), hypers, opt_hypers, scheduler_bool, TORCH_DEVICE, TORCH_DTYPE)



    print('Running GNN...')
    gnn_start = time()

    probs, epoch, final_bitstring, best_bitstring = run_gnn_training(
        Q, bqm_model.offset, stoich_const, Gumbel_sinkhorn, graph_dgl, net, embed, optimizer, scheduler, hypers["temperature"], hypers['number_epochs'],
        hypers['patience'], hypers['tolerance'], seed = seed_value)

    gnn_time = time() - gnn_start


    final_hard_loss = loss_func(Q, final_bitstring.float(), bqm_model.offset)
    best_hard_loss = loss_func(Q, best_bitstring.float(), bqm_model.offset)
    if round(best_hard_loss.item(), 3) == bqm_model.minimum_energy:
        min_energy_found += 1
    avg_best_hard_loss += best_hard_loss
    print('Training took: ' + str(round(gnn_time, 2)) + 'secs')
    print('Final Hard loss: ', final_hard_loss.item())
    print('Best Hard loss: ', best_hard_loss.item())
    print('Final bitstring: ', final_bitstring)
    print('Best bitstring: ', best_bitstring)
print('Minimum energy: ', bqm_model.minimum_energy)
print('Average Best Hard loss: ', avg_best_hard_loss.item()/10)
print('The method has reached the ground state ' + str(min_energy_found) + ' times out of 10.')

