import dgl
import torch
import random
import numpy as np
import functools
from hyperopt import tpe, hp, STATUS_OK, Trials
from mltb.hyperopt import fmin
from hyperopt.pyll import scope
from time import time

# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')

from utils import generate_graph
from potts_utils import get_gnn, run_gnn_training, loss_func
from lp_to_bqm import BQM, qubo_to_torch

# Graph hypers
d = 3
p = None
graph_type = 'complete'

# GNN hypers
graph_encoder = 'GraphSAGE'
number_epochs = int(1e5) # max epoch number
PROB_THRESHOLD = 0.5

# Problem parameters/hypers
sparsity_threshold = 0

leq_weight = 200 # penalty weight for atom or void per position constraints
eq_weight = 200 # penalty weight for stoichiometry constraints
scheduler_bool = False # learning rate scheduler

# Early stopping to allow NN to train to near-completion
tol = 1e-3         # loss must change by more than tol, or trigger

Gumbel_sinkhorn = False # if False Potts model, if True G.S. Potts
# Set up problem to solve
bqm_model = BQM(Potts = True, Gumbel_sinkhorn=Gumbel_sinkhorn)
file_to_parse = 'SrO.lp'
bqm_model.parse_lp("instances/" + file_to_parse)
filename = file_to_parse.split('.')[0] + '_Train_' # for writing the Sinkhorn convergence iterations and temperature
with_void = True #
# pre-constrained Q matrix
Q, num_positions, atoms, stoich_const = qubo_to_torch(
    bqm_model, eq_inf=eq_weight, leq_infinity=leq_weight, with_void = with_void, Gumbel_sinkhorn=Gumbel_sinkhorn,
    torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE
)
n_atoms = len(atoms)
num_classes = (n_atoms + 1) if with_void else n_atoms # number of node classes
n = num_positions # number of graph nodes = number of crystal's positions
print("Crystal Structure Prediction of " + file_to_parse.split('.')[0] + " with " + str(num_positions)
      + " positions" + " and " + str(atoms) + " atoms.")
# Constructs a random d-regular or p-probabilistic graph
nx_graph = generate_graph(n=n, d=d, p=p, Q = Q, sparsity_threshold=0, graph_type=graph_type)

# get DGL graph from networkx graph, load onto device
graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
graph_dgl = graph_dgl.to(TORCH_DEVICE)

def model_step(hypers, Q, bqm_model, graph_dgl, torch_device, torch_dtype):
    # Parse hyperparameters
    gnn_hypers = hypers
    opt_keys = ['lr', 'weight_decay']
    opt_params = {k: gnn_hypers.pop(k) for k in opt_keys}

    scheduler_bool = gnn_hypers.pop('scheduler')
    # graph_encoder = gnn_hypers.pop('model')

    print(f'Weight decay: {opt_params["weight_decay"]}')
    print(f'Learning rate: {opt_params["lr"]}')
    print(f'Temperature Scaling of logits: {gnn_hypers["temperature"]}')

    num_seeds = 5 # how many seeds to check for every set of hypers
    print('Running GNN...')
    total_final_loss = 0
    total_best_loss = 0
    for seed_value in range(num_seeds):
        random.seed(seed_value)        # seed python RNG
        np.random.seed(seed_value)     # seed global NumPy RNG
        torch.manual_seed(seed_value)  # seed torch RNG
        gnn_hypers["seed"] = seed_value
        net, embed, optimizer, scheduler = get_gnn(
            graph_dgl, nx_graph.number_of_nodes(), gnn_hypers, opt_params, scheduler_bool,
            torch_device, torch_dtype
        )

        # For tracking hyperparameters in results object
        gnn_hypers.update(opt_params)
        file_to_write = filename + "temp_" + str(hypers["temperature"]) + "_"
        gnn_start = time()
        probs, epoch, final_bitstring, best_bitstring = run_gnn_training(file_to_write,
            Q, bqm_model.offset, stoich_const, Gumbel_sinkhorn, graph_dgl, net, embed, optimizer, scheduler, hypers["temperature"],
            hypers["scaling"], hypers['number_epochs'], hypers['patience'], hypers['tolerance'], hypers['prob_threshold'], flag= False, seed=seed_value)

        gnn_time = time() - gnn_start
        final_hard_loss = loss_func(Q, final_bitstring.float(), bqm_model.offset)
        best_hard_loss = loss_func(Q, best_bitstring.float(), bqm_model.offset)
        print(f'Final Hard loss: {final_hard_loss.item()}')
        print(f'Best Hard loss: {best_hard_loss.item()}')
        print(f'Step took: {round(gnn_time, 2)}s')
        total_final_loss += final_hard_loss.item()
        total_best_loss += best_hard_loss.item()
    avg_final_hard_loss = total_final_loss/num_seeds
    avg_best_hard_loss = total_best_loss/num_seeds
    print(f'Average Final Hard loss: {avg_final_hard_loss}')
    print(f'Average Best Hard loss: {avg_best_hard_loss}')
    return {'loss': avg_best_hard_loss, 'final_bitstring': final_bitstring, 'best_bitstring': best_bitstring,
            'status': STATUS_OK}


patience = 1000 if Gumbel_sinkhorn else 10000
agg_type = 'mean'
search_space = {
    # Params to search over
    'dim_embedding': scope.int(hp.uniform('dim_embedding', 16, 64)),
    'hidden_dim': scope.int(hp.uniform('hidden_dim', 16, 64)),
    'dropout': scope.float(hp.uniform('dropout', 0.0, 0.5)),
    'weight_decay': scope.float(hp.loguniform('weight_decay', -5, -1)),
    'lr': scope.float(hp.loguniform('lr', -5, -2)),
    'temperature': scope.float(hp.uniform('temperature', 0.01, 1)) if Gumbel_sinkhorn else scope.float(hp.uniform('temperature', 1.5, 3)),
    'scaling': scope.float(hp.uniform('scaling', 1, 2)) if Gumbel_sinkhorn else scope.float(hp.uniform('scaling', 0, 1)),
    #possibly experiment with layer_agg_type
    # Fixed params - GNN
    'number_classes': num_classes,
    'prob_threshold': PROB_THRESHOLD,
    'number_epochs': number_epochs,
    'tolerance': tol,
    'patience': patience,  # turn off patience (temporarily)
    'model' : graph_encoder,
    'layer_agg_type': agg_type,
    # Fixed params - problem
    'scheduler': scheduler_bool,
    'sparsity threshold': sparsity_threshold,
    'leq_weight': leq_weight,
    'eq_weight': eq_weight
}


obj_func = functools.partial(
    model_step, Q = Q, bqm_model=bqm_model, graph_dgl=graph_dgl,
    torch_device=TORCH_DEVICE, torch_dtype=TORCH_DTYPE
)
# # do hpo
file_to_write = file_to_parse.split('.')[0]
num_searches = 10 # number of hyperparameters searches
scaling_str = "_no_scaling" if search_space["scaling"] == 1 else "_dynscal"
print("Hyperparameter Search over " + str(num_searches) + " sets of hypers")
best, trials = fmin(
    obj_func,
    space=search_space,
    algo=tpe.suggest,
    max_evals = num_searches,
    filename = 'trials_file_GS_' + str(Gumbel_sinkhorn) + '_' + file_to_write + scaling_str
)

print('best:', best)
print('number of trials:', len(trials.trials))

