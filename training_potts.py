import dgl
import torch
import random
import numpy as np
import networkx as nx
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


Gumbel_sinkhorn = True # if False Potts model, if True G.S. Potts
file_to_parse = 'SrO.lp'
file_to_write = file_to_parse.split('.')[0]
filename = file_to_write + '_Test'
temp_scaling = True
scaling_str = "_dynscal" if temp_scaling else "_no_scaling"
trials_history = joblib.load('trials_file_GS_' + str(Gumbel_sinkhorn) + '_' + file_to_write + scaling_str) # read

#In case we want to inspect more models rather than just the best one
# losses = []
# for i in range(len(trials_history)):
#     losses.append(trials_history.results[i]['loss'])
# desc_losses_ind = np.argsort(losses)

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
leq_weight = 200 # penalty weight for atom or void per position constraints for Potts model
eq_weight = 200 # penalty weight for stoichiometry constraints for Potts model
scheduler_bool = False # learning rate scheduler

with_void = True # include void for each position
Q, num_positions, atoms, stoich_const = qubo_to_torch(
    bqm_model, eq_inf=eq_weight, leq_infinity=leq_weight,
    with_void = with_void,  torch_dtype=TORCH_DTYPE, Gumbel_sinkhorn=Gumbel_sinkhorn,
    torch_device=TORCH_DEVICE) # pre-constrained Q matrix

n_atoms = len(atoms) # number of atoms
num_classes = (n_atoms + 1) if with_void else n_atoms # number of node classes
n_variables = num_positions*num_classes
print("Crystal Structure Prediction of " + file_to_parse.split('.')[0] + " with " + str(num_positions)
      + " positions" + " and " + str(atoms) + " atoms.")
print(n_variables, "Variables in the MultiClass Classification Problem")
# Problem size (e.g. graph size)
n = num_positions # number of graph nodes = number of crystal's positions

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
        'scaling': trials_history.argmin["scaling"] if temp_scaling else 1,
    }
# Default meta parameters
solver_hypers = {
    'tolerance': 1e-3,           # Loss must change by more than tolerance, or add towards patience count
    'number_epochs': int(1e5),   # Max number training steps
    'patience': 1000,             # Number early stopping triggers before breaking loop
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
num_tests = 10 # number of seeds to test
best_hard_loss = np.zeros((num_tests, 1))
gnn_time = np.zeros((num_tests, 1))
min_energy_found = 0
relative_optimality_gap = np.zeros((num_tests, 1))
for seed_value in range(10, 10 + num_tests):
    random.seed(seed_value)        # seed python RNG
    np.random.seed(seed_value)     # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG

    hypers["seed"] = seed_value
    net, embed, optimizer, scheduler = get_gnn(
        graph_dgl, nx_graph.number_of_nodes(), hypers, opt_hypers, scheduler_bool, TORCH_DEVICE, TORCH_DTYPE)

    print('Running GNN...')
    gnn_start = time()

    probs, epoch, final_bitstring, best_bitstring = run_gnn_training(filename,
        Q, bqm_model.offset, stoich_const, Gumbel_sinkhorn, graph_dgl, net, embed, optimizer, scheduler, hypers["temperature"],
        hypers["scaling"], hypers['number_epochs'], hypers['patience'], hypers['tolerance'], flag = True, seed = seed_value)

    gnn_time[seed_value - 10] = time() - gnn_start

    final_hard_loss = loss_func(Q, final_bitstring.float(), bqm_model.offset)
    best_hard_loss[seed_value - 10] = loss_func(Q, best_bitstring.float(), bqm_model.offset).item()
    relative_optimality_gap[seed_value - 10] = \
        (bqm_model.minimum_energy - best_hard_loss[seed_value - 10])/bqm_model.minimum_energy
    if np.round(best_hard_loss[seed_value - 10], 3) == bqm_model.minimum_energy:
        min_energy_found += 1
    print('Training took: ' + str(np.round(gnn_time[seed_value - 10], 2)) + 'secs')
    print('Final Hard loss: ', final_hard_loss.item())
    print('Best Hard loss: ', best_hard_loss[seed_value - 10])
    print('Final bitstring: ', final_bitstring)
    print('Best bitstring: ', best_bitstring)
print('Ground state energy = ', bqm_model.minimum_energy)
print('Average Best Hard loss = ' + str(best_hard_loss.mean()) + ' , Standard deviation = ' + str(best_hard_loss.std()))
print('Best Hard Loss = ' + str(min(best_hard_loss)))
print('Average Relative Optimality gap = ' + str(relative_optimality_gap.mean()) + ' , Standard deviation = ' + str(relative_optimality_gap.std()))
print('Best Relative Optimality gap = ' + str(min(relative_optimality_gap)))
print('Average Time = ' + str(gnn_time.mean()) + ' , Standard deviation = ' + str(gnn_time.std()))
print('Hit rate = ' + str((min_energy_found/num_tests)*100) + '%')

