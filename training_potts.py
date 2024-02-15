import dgl
import torch
import random
import numpy as np
import networkx as nx
import matplotlib
from time import time
from scipy.spatial.distance import pdist, squareform
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

from utils import generate_graph, crystal_coordinates, gaussian_kernel
from potts_utils import get_gnn, run_gnn_training, loss_func
from lp_to_bqm import BQM, qubo_to_torch

Gumbel_sinkhorn = True # if False Potts model, if True G.S. Potts
file_to_parse = 'SrTiO3G4.lp'
struct_info = file_to_parse.split('G')
struct_name = struct_info[0] #
g = int(struct_info[1][0]) # cell discretization
file_to_write = file_to_parse.split('.')[0]
filename = file_to_write + '_Test'

bqm_model = BQM(Potts = True, Gumbel_sinkhorn = Gumbel_sinkhorn)
bqm_model.parse_lp("instances/" + file_to_parse)

# Graph hypers
d = 3
p = None
graph_type = 'complete'
graph_encoder = 'GraphSAGE'
# NN learning hypers #
number_epochs = int(1e5)
PROB_THRESHOLD = 0.5
sparsity_threshold = 0.0
leq_weight = 200 # penalty weight for atom or void per position constraints for Potts model
eq_weight = 200 # penalty weight for stoichiometry constraints for Potts model
lr_scheduler_type = "No" # learning rate scheduler
cutoff_dist = 5 if graph_type == "cutoff" else 0
temp_scaling = False
scaling_str = "_dynscal" if temp_scaling else "_no_scaling"
trials_filename = 'trials_file_GS_' + str(Gumbel_sinkhorn) + '_' + file_to_write if graph_type=='complete' \
    else 'trials_file_GS_' + str(Gumbel_sinkhorn) + '_' + file_to_write + scaling_str + str(cutoff_dist)
trials_history = joblib.load(trials_filename)

# In case we want to inspect more models rather than just the best one
losses = []
for i in range(len(trials_history)):
    losses.append(trials_history.results[i]['loss'])
desc_losses_ind = np.argsort(losses)
with_void = True # include void for each position
Q, num_positions, atoms, stoich_const = qubo_to_torch(
    bqm_model, eq_inf=eq_weight, leq_infinity=leq_weight,
    with_void = with_void,  torch_dtype=TORCH_DTYPE, Gumbel_sinkhorn=Gumbel_sinkhorn,
    torch_device=TORCH_DEVICE) # pre-constrained Q matrix

pos = crystal_coordinates(struct_name, g, num_positions) # atoms' candidate 3d coordinates within the unit cell #except for SrO
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
        'model': graph_encoder,
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
    'patience': 1000 if Gumbel_sinkhorn else 5000, # Number early stopping triggers before breaking loop
    'layer_agg_type': 'mean',    # How aggregate neighbors sampled within graphSAGE
    'number_classes': num_classes
}

# Combine into a single set
hypers.update(solver_hypers)
# Constructs a random d-regular or p-probabilistic graph
nx_graph = generate_graph(n=n, d=d, p=p, pos=pos, cutoff_dist=cutoff_dist, graph_type=graph_type)

# get DGL graph from networkx graph, load onto device
graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
distvec = pdist(pos)
square_dist = squareform(distvec) # square distance matrix between positions
node_dist = list(square_dist.reshape(square_dist.shape[0]*square_dist.shape[1])) #flatten node distances
node_dist = [i for i in node_dist if i != 0] # remove 0s
if cutoff_dist>0:
    node_dist = [i for i in node_dist if i<cutoff_dist] # remove distances of nodes with distance greater than cutoff
node_dist = torch.tensor(node_dist)
gaussian_dist = gaussian_kernel(node_dist, sigma=1)
pos = (pos-np.min(pos))/(np.max(pos)-np.min(pos))
graph_dgl.ndata['pos'] = torch.tensor(pos).type(TORCH_DTYPE) # add 3D coordinates as nodes' features
graph_dgl.edata["edge_weights"] = gaussian_dist.type(TORCH_DTYPE)
if graph_encoder == "GAT":
    graph_dgl = dgl.add_self_loop(graph_dgl) # required for GAT
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
final_hard_loss = np.zeros((num_tests, 1))
gnn_time = np.zeros((num_tests, 1))
min_energy_found = 0
relative_optimality_gap = np.zeros((num_tests, 1))
best_bitstring = []
deg = nx_graph.degree
avg_degree = sum([i[1] for i in deg])/nx_graph.number_of_nodes() # average graph degree
graph_density = nx_graph.number_of_edges()/(nx_graph.number_of_nodes()*(nx_graph.number_of_nodes()-1)/2)
for seed_value in range(10, 10 + num_tests):
    random.seed(seed_value)        # seed python RNG
    np.random.seed(seed_value)     # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG

    hypers["seed"] = seed_value
    net, embed, optimizer, lr_scheduler = get_gnn(
        graph_dgl, nx_graph.number_of_nodes(), hypers, opt_hypers, lr_scheduler_type, TORCH_DEVICE, TORCH_DTYPE)

    print('Running GNN...')
    gnn_start = time()

    probs, epoch, final_bitstring, best_bitstring_seed = run_gnn_training(filename, Q, bqm_model.offset, stoich_const,
                                                                     Gumbel_sinkhorn, graph_dgl, cutoff_dist, net,
                                                                     embed, optimizer, lr_scheduler, hypers["temperature"],
                                                                    hypers["scaling"], hypers['number_epochs'],
                                                                     hypers['patience'], hypers['tolerance'],
                                                                     flag = True, seed = seed_value)

    gnn_time[seed_value - 10] = time() - gnn_start
    best_bitstring.append(best_bitstring_seed.float())
    final_hard_loss[seed_value - 10] = loss_func(Q, final_bitstring.float(), bqm_model.offset).item()
    best_hard_loss[seed_value - 10] = loss_func(Q, best_bitstring_seed.float(), bqm_model.offset).item()
    relative_optimality_gap[seed_value - 10] = \
        (bqm_model.minimum_energy - best_hard_loss[seed_value - 10])/bqm_model.minimum_energy
    if np.round(best_hard_loss[seed_value - 10], 3) == bqm_model.minimum_energy:
        min_energy_found += 1
with open('Results/' + trials_filename + '.txt', 'w') as f:
    # add model's parameters
    f.write('Ground state energy = ' + str(bqm_model.minimum_energy) + '\n')
    f.write("Number of correct assignments\n")
    for i in range(len(atoms)):
        f.write(str(int(stoich_const[i].item())) + " atom(s) of " + atoms[i] + '\n')
    f.write(str(int(stoich_const[-1].item())) + " void positions\n")
    f.write("Model's parameters\n")
    f.write("   Architecture: " + hypers["model"] + "\n")
    f.write("   dim_embedding: " + str(int(hypers["dim_embedding"])) + "\n")
    f.write("   hidden_dim: " + str(int(hypers["hidden_dim"])) + "\n")
    f.write("   dropout: " + str(hypers["dropout"]) + "\n")
    f.write("Message-passing on the complete graph\n") if graph_type=="complete" \
        else f.write("Message-passing on the cutoff graph with cutoff distance = " + str(cutoff_dist) + " Ankstroms\n")
    f.write("Average graph degree is: " + str(avg_degree))
    f.write("Graph density is: " + str(graph_density))
    for seed_value in range(10, 10+num_tests):
        f.write('Results for seed ' + str(seed_value) + ': \n')
        f.write('   Training took: ' + str(np.round(gnn_time[seed_value - 10].item(), 2)) + ' secs\n')
        f.write('   Final Hard loss: ' + str(np.round(final_hard_loss[seed_value - 10].item(), 2)) + '\n')
        f.write('   Best Hard loss: ' + str(np.round(best_hard_loss[seed_value - 10].item(), 2)) + '\n')
        for i in range(len(atoms)):
            f.write('   There are ' + str(int(best_bitstring[seed_value - 10].sum(dim=0)[i].item())) + ' atoms of type '
                    + atoms[i] + '\n')
        f.write('   There are ' + str(int(best_bitstring[seed_value - 10].sum(dim=0)[-1].item())) + ' void positions \n')
    f.write('In the best solution of all seeds:\n')
    for i in range(len(atoms)):
        f.write('   There are ' + str(int(best_bitstring[np.argmin(best_hard_loss)].sum(dim=0)[i].item())) +
                ' atoms of type ' + atoms[i] + '\n')
    f.write('   There are ' + str(int(best_bitstring[np.argmin(best_hard_loss)].sum(dim=0)[-1].item())) + ' void positions \n')
    f.write('Best Hard Loss = ' + str(min(best_hard_loss)[0]) + '\n')
    f.write('Average Best Hard loss = ' + str(best_hard_loss.mean()) +
            ' , Standard deviation = ' + str(best_hard_loss.std()) + '\n')
    f.write('Best Relative Optimality gap = ' + str(min(relative_optimality_gap)[0]) + '\n')
    f.write('Average Relative Optimality gap = ' + str(relative_optimality_gap.mean()) +
            ' , Standard deviation = ' + str(relative_optimality_gap.std()) + '\n')
    f.write('Average Time = ' + str(gnn_time.mean()) + ' , Standard deviation = ' + str(gnn_time.std()) + '\n')
    f.write('Hit rate = ' + str((min_energy_found/num_tests)*100) + '%')
print('Ground state energy = ', bqm_model.minimum_energy)
print('Average Best Hard loss = ' + str(best_hard_loss.mean()) + ' , Standard deviation = ' + str(best_hard_loss.std()))
print('Best Hard Loss = ' + str(min(best_hard_loss)))
print('Average Relative Optimality gap = ' + str(relative_optimality_gap.mean()) + ' , Standard deviation = ' + str(relative_optimality_gap.std()))
print('Best Relative Optimality gap = ' + str(min(relative_optimality_gap)))
print('Average Time = ' + str(gnn_time.mean()) + ' , Standard deviation = ' + str(gnn_time.std()))
print('Hit rate = ' + str((min_energy_found/num_tests)*100) + '%')

