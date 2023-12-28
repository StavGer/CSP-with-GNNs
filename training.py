import dgl
import torch
import random
import pickle
import numpy as np
import networkx as nx
import wandb
from time import time


# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TORCH_DEVICE = 'cpu'
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')

from utils import generate_graph, get_gnn, run_gnn_training, qubo_dict_to_torch, loss_func
from lp_to_bqm import BQM, qubo_to_torch


# Graph hypers
d = 3
p = None
graph_type = 'sparse'
graph_encoder = 'GraphSAGE'
# NN learning hypers #
number_epochs = 5*int(1e3)
learning_rate = 0.008109344
PROB_THRESHOLD = 0.5
sparsity_threshold = 0
leq_weight = 200
eq_weight = 200
scheduler_bool = True

# Early stopping to allow NN to train to near-completion
tol = 1e-4         # loss must change by more than tol, or trigger
patience = 10    # number early stopping triggers before breaking loop

bqm_model = BQM(Potts = False)
bqm_model.parse_lp("instances/SrO.lp")

# print(bqm_model.constraints_dict)
# print(bqm_model.variables)
# print(bqm_model.quadratic)
with open('best_hypers/best_hypers_False2023-10-24 14-47-46.pkl', 'rb') as fp:
    best_hypers = pickle.load(fp)

with_void = False
Q, Qunc, C1, C2, variables, n_atoms = qubo_to_torch(bqm_model, eq_inf=eq_weight, leq_infinity=leq_weight,
                                                    with_void = with_void,
                                                   torch_dtype=TORCH_DTYPE,
                                                   torch_device=TORCH_DEVICE) # pre-constrained Q matrix

num_positions = int(len(variables)/n_atoms)
num_variables = (n_atoms + 1)*num_positions if with_void else n_atoms*num_positions
print(num_variables, "Variables in the binary program")
# b = torch.tensor([1, 0, 0, 0, 1, 0, 0, 1, 0,  1, 0, 0, 0, 1, 0,  1, 0, 0, 1, 0, 0,  0, 1, 0], dtype= torch.float32, requires_grad=True, device=TORCH_DEVICE)
# b = torch.tensor([1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,  0, 1, 0, 1, 0 , 0], dtype= torch.float32, requires_grad=True, device=TORCH_DEVICE)

# l = (b.T @ Q @ b).squeeze() + bqm_model.offset


# Problem size (e.g. graph size)
n = num_variables # number of nodes


# Constructs a random d-regular or p-probabilistic graph
nx_graph = generate_graph(n=n, d=d, p=p, Q=Q, sparsity_threshold=0, graph_type=graph_type)


graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
graph_dgl = graph_dgl.to(TORCH_DEVICE)

# weight_decay = 1e-1

# Establish pytorch GNN + optimizer
opt_params = {'lr': best_hypers["lr"], 'weight_decay': best_hypers["weight_decay"]}

gnn_hypers = {
    'dim_embedding': int(best_hypers["dim_embedding"]),
    'hidden_dim': int(best_hypers["hidden_dim"]),
    'dropout': best_hypers["dropout"],
    'number_classes': 1,
    'prob_threshold': PROB_THRESHOLD,
    'number_epochs': number_epochs,
    'tolerance': best_hypers["tolerance"],
    'patience': int(best_hypers["patience"]),
    'temperature': best_hypers["temperature"]
}


for seed_value in range(10):
    random.seed(seed_value)        # seed python RNG
    np.random.seed(seed_value)     # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    run_hypers = {'lr': best_hypers["lr"],
                    'weight_decay': best_hypers["weight_decay"],
                    'scheduler': scheduler_bool,
                    'seed': seed_value,
                    'sparsity threshold': sparsity_threshold,
                    'graph encoder': graph_encoder,
                    'leq_weight': leq_weight,
                    'eq_weight': eq_weight}
    run_hypers.update(gnn_hypers)

    # wandb.init(
    #         # set the wandb project where this run will be logged
    #         project="CSP with GNNs",
    #
    #         # track hyperparameters and run metadata
    #         config=run_hypers
    # )
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


    final_soft_loss, _, _, _ = loss_func(final_bitstring.float(), Qunc, C1, C2, bqm_model.offset)
    best_soft_loss, _ , _, _ = loss_func(best_bitstring.float(), Qunc, C1, C2, bqm_model.offset)
    print('Final Soft loss: ', final_soft_loss.item())
    print('Best Soft loss: ', best_soft_loss.item())
    print('Final bitstring: ', final_bitstring)
    print('Best bitstring: ', best_bitstring)
    # wandb.run.summary["final_bitstring"] = final_bitstring
    # wandb.run.summary["best_bitstring"] = best_bitstring
    # wandb.finish(exit_code = None, quiet = None)
