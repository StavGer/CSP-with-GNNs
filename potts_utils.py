import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from models.models_pyg import SAGE_with_EdgeConv
from models.models_dgl import GIN, GATnet, GNNConv, GNNSage
from itertools import chain
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from scipy.stats import entropy

torch.autograd.set_detect_anomaly(True)

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

# GNN construction
def get_gnn(g, n_nodes, gnn_hypers, opt_params, lr_scheduler_type, torch_device, torch_dtype):
    """
    Helper function to load in GNN object, optimizer, and initial embedding layer.

    :param n_nodes: Number of nodes in graph
    :type n_nodes: int
    :param gnn_hypers: Hyperparameters to provide to GNN constructor
    :type gnn_hypers: dict
    :param opt_params: Hyperparameters to provide to optimizer constructor
    :type opt_params: dict
    :param torch_device: Compute device to map computations onto (CPU vs GPU)
    :type torch_dtype: str
    :param torch_dtype: Specification of pytorch datatype to use for matrix
    :type torch_dtype: str
    :return: Initialized GNN instance, embedding layer, initialized optimizer instance
    :rtype: GNN_Conv or GNN_SAGE, torch.nn.Embedding, torch.optim.AdamW
    """

    try:
        print(f'Function get_gnn(): Setting seed to {gnn_hypers["seed"]}')
        set_seed(gnn_hypers['seed'])
    except KeyError:
        print('!! Function get_gnn(): Seed not specified in gnn_hypers object. Defaulting to 0 !!')
        set_seed(0)

    model = gnn_hypers['model']
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']
    agg_type = gnn_hypers['layer_agg_type'] or 'mean'

    # instantiate the GNN
    print(f'Building {model} model...')
    if model == "GraphConv":
        net = GNNConv(g, dim_embedding, hidden_dim, number_classes, dropout)
    elif model == "GraphSAGE":
        net = GNNSage(g, dim_embedding, hidden_dim, number_classes, dropout, agg_type)
    elif model == "GAT":
        net = GATnet(g, g.ndata['pos'].shape[1], hidden_dim, 4, number_classes, dropout)
    elif model == "GIN":
        net = GIN(g, dim_embedding, hidden_dim, number_classes, dropout)
    elif model == "GraphSAGE-Edge":
        net = SAGE_with_EdgeConv(dim_embedding, g.edge_attr.shape[1], hidden_dim, number_classes, dropout)
    else:
        raise ValueError("Invalid model type input! Model type has to be in one of these two options: ['GraphConv', 'GraphSAGE']")

    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())
    # params = nn.ParameterList(net.parameters())
    print('Building ADAM-W optimizer...')
    optimizer = torch.optim.AdamW(params, **opt_params)
    # optimizer = torch.optim.SGD(params, **opt_params)
    if lr_scheduler_type == "Plateau" :
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100)
    elif lr_scheduler_type == "Cosine Warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=1000
        )
    elif lr_scheduler_type == "Cosine warmup with restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer = optimizer,
            num_warmup_steps=50,
            num_training_steps=1000,
            num_cycles=3
        )
    else:
        scheduler = None
    return net, embed, optimizer, scheduler

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def loss_func(Q, probs, offset):

    probs_flattened = torch.flatten(probs)
    p = torch.kron(probs_flattened, probs_flattened)
    Q_flatten = torch.flatten(Q)
    loss = torch.unsqueeze(p, 0)@torch.unsqueeze(Q_flatten, 1) + offset
    return loss

def log_sinkhorn(log_alpha, r, n_iter,  eps = 0.05):
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the successive row and column
    normalization.

    [1] Sinkhorn, Richard and Knopp, Paul.
    Concerning nonnegative matrices and doubly stochastic
    matrices. Pacific Journal of Mathematics, 1967
    Args:
      log_alpha: 2D tensor (a matrix of shape [N, N])
        or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
      n_iters: number of sinkhorn iterations (in practice, as little as 20
        iterations are needed to achieve decent convergence for N~100)
    Returns:
      A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
        converted to 3D tensors with batch_size equals to 1)
    """
    stop = 0
    for i in range(n_iter):
        prev_log_alpha = log_alpha
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = torch.log(r) + log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
        alpha = log_alpha.exp()
        violations_r = torch.where(torch.abs(alpha.sum(dim = -1) - 1) > eps) #row sum violations
        violations_c = torch.where(torch.abs(alpha.sum(dim = -2) - r) > eps) #column sum violations
        if (len(violations_r[0]) == 0 and len(violations_c[0]) == 0) or torch.max(torch.abs(log_alpha-prev_log_alpha)) < 1e-5:
            stop+=1
            if stop==100:
                break
        if i == n_iter - 1:
            print('Sinkhorn did not converge!')
    return alpha, i

def sample_gumbel(shape, device='cpu', eps=1e-20):
    """Samples arbitrary-shaped standard gumbel variables.
    Args:
      shape: list of integers
      eps: float, for numerical stability
    Returns:
      A sample of standard Gumbel random variables
    """
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)

def gumbel_sinkhorn(log_alpha, stoich_const, tau, n_iter):
    """ Sample a permutation matrix from the Gumbel-Sinkhorn distribution
    with parameters given by log_alpha and temperature tau.

    Args:
      log_alpha: Logarithm of assignment probabilities. In our case this is
        of dimensionality [num_pieces, num_pieces].
      tau: Temperature parameter, the lower the value for tau the more closely
        we follow a categorical sampling.
    """
    # Sample Gumbel noise.
    gumbel_noise = sample_gumbel(log_alpha.shape, device=log_alpha.device)
    # Apply the Sinkhorn operator!
    sampled_perm_mat = log_sinkhorn((log_alpha + gumbel_noise)/tau, stoich_const, n_iter)
    return sampled_perm_mat

def run_gnn_training(filename, Q, offset, stoich_const, Gumbel_sinkhorn, graph_dgl, cutoff_dist, net, embed, optimizer,
                     lr_scheduler, temperature, scaling, number_epochs=int(1e5), patience=100, tolerance=1e-4,
                     prob_threshold = 0.5, flag = False, seed=1):
    """
    Function to run model training for given graph, GNN, optimizer, and set of hypers.
    Includes basic early stopping criteria. Prints regular updates on progress as well as
    final decision.

    :param graph_dgl: Graph instance to solve
    :param net: GNN instance to train
    :type net: GNN_Conv or GNN_SAGE
    :param embed: Initial embedding layer
    :type embed: torch.nn.Embedding
    :param optimizer: Optimizer instance used to fit model parameters
    :type optimizer: torch.optim.AdamW
    :param number_epochs: Limit on number of training epochs to run
    :type number_epochs: int
    :param patience: Number of epochs to wait before triggering early stopping
    :type patience: int
    :param tolerance: Minimum change in cost to be considered non-converged (i.e.
        any change less than tolerance will add to early stopping count)
    :type tolerance: float

    :return: Final model probabilities, best color vector found during training, best loss found during training,
    final color vector of training, final loss of training, number of epochs used in training
    :rtype: torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, int
    """

    # Ensure RNG seeds are reset each training run
    print(f'Function run_gnn_training(): Setting seed to {seed}')
    set_seed(seed)

    inputs = embed.weight
    t = (temperature*torch.ones(1)).to(graph_dgl.device)
    best_bitstring = torch.zeros((graph_dgl.number_of_nodes(), stoich_const.shape[0])).type(Q.dtype).to(graph_dgl.device)
    bitstring = torch.zeros((graph_dgl.number_of_nodes(), stoich_const.shape[0])).type(Q.dtype).to(graph_dgl.device)
    # Early stopping to allow NN to train to near-completion
    prev_loss = 1.  # initial loss value (arbitrary)
    cnt = 0  # track number times early stopping is triggered
    best_loss = 1e8
    ANNEAL_RATE = 0.00003
    temp_min = torch.tensor(0.5)
    if flag and Gumbel_sinkhorn:
        gs_iters = []
        temp_logger = []
    # Training logic
    assgnmnts_change = []
    bitstring_o = bitstring
    probs_save = []
    bitstring_save = []
    i = 0
    assgnd_nodes = torch.tensor([], device=graph_dgl.device)
    assgnd_nodes_num = []
    pos_ind_per_atom= [torch.tensor([], device= 'cuda') for i in range(stoich_const.shape[0])]
    unique_pos_per_atom = [[] for i in range(stoich_const.shape[0])]
    for epoch in range(number_epochs):
        # get soft prob assignments
        # logits = net(graph_dgl) for GAT
        logits = net(inputs)
        if Gumbel_sinkhorn:
            if flag:
                temp_logger.append(t.item())
            # if i>=1000:
            #     t = scaling*t
            #     print("Temperature scaled by " + str(scaling) + ' ,t = ' + str(t))
            probs, i = gumbel_sinkhorn(logits, stoich_const, tau=t, n_iter=50000)
            if flag: gs_iters.append(i)
        else:
            if cnt%1000==0 and cnt>0:
                t = scaling*t
                print("Temperature scaled by " + str(scaling) + ' ,t = ' + str(t))
            logits = torch.div(logits, t)
            # apply softmax for normalization
            probs = F.softmax(logits, dim=1)
        loss = loss_func(Q, probs, offset)
        # t = torch.maximum(t * torch.exp(torch.tensor(-ANNEAL_RATE * epoch)), temp_min)
        bitstring_n = (probs.detach() >= prob_threshold) * 1
        probs_save.append(probs.detach())
        bitstring_save.append(bitstring_n.detach())
        changes = bitstring_n.type(torch.uint8) ^ bitstring_o.type(torch.uint8)
        assgnmnts_change.append(torch.tensor([torch.where(changes[:, col] == 1)[0].numel() for col in range(bitstring.shape[1])]))
        for atom_index in range(bitstring_n.shape[1]):
            z = torch.where(bitstring_n[:, atom_index]==1)[0]
            pos_ind_per_atom[atom_index] = torch.cat((pos_ind_per_atom[atom_index], z))
            unique_pos_per_atom[atom_index].append(torch.tensor(torch.unique(pos_ind_per_atom[atom_index]).shape))
        z = torch.where(bitstring_n[:, :bitstring_n.shape[1]-1]==1)[0]
        assgnd_nodes = torch.cat((assgnd_nodes, z))
        assgnd_nodes_num.append(torch.tensor(torch.unique(assgnd_nodes).shape))
        # a different projector from probs to bitstring
        # for i,l in enumerate(stoich_const):
        #     s, indices = torch.sort(probs[:, i])
        #     z = indices[int((probs.shape[0]- l).item()):]
        #     bitstring[z, i] = 1
        hard_loss = loss_func(Q, bitstring_n.float(), offset)

        # assert(bitstring.sum(dim=0).any()==stoich_const.any())
        # Early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss - prev_loss) <= tolerance) | (loss >= best_loss ):
            cnt += 1
        else:
            cnt = 0
        if loss < best_loss:
            best_loss = loss
            best_bitstring = bitstring_n
            best_loss_epoch = epoch
            best_probs = probs
        # update loss tracking

        prev_loss = loss

        if cnt >= patience:
            print(f'Stopping early on epoch {epoch}. Patience count: {cnt}')
            break
        # run optimization with backpropagation
        optimizer.zero_grad()  # clear gradient for step
        loss.backward()  # calculate gradient through compute graph
        optimizer.step()  # take step, update weights
        if lr_scheduler is not None:
            lr_scheduler.step()
        # tracking: print intermediate loss at regular interval
        print('Epoch %d | Total Soft Loss: %.5f' % (epoch, loss.item()))
        print('Epoch %d | Total Hard Loss: %.5f' % (epoch, loss_func(Q, bitstring_n.float(), offset).item()))
        bitstring_o = bitstring_n
        # if epoch % 100 == 0:
        #     print('Epoch %d | Total Soft Loss: %.5f' % (epoch, loss.item()))
            # print('Best Soft Loss so far: %.5f' % (best_loss.item()))
    if flag and Gumbel_sinkhorn:
        folder_to_write = "plots_no_scaling/" if scaling==1 else "plots_with_scaling/"
        if not os.path.exists(folder_to_write):
            os.makedirs(folder_to_write)
        if cutoff_dist != 0:
            folder_to_write = folder_to_write + "cutoff_" + str(cutoff_dist)
            if not os.path.exists(folder_to_write):
                os.makedirs(folder_to_write)
        gs_iters = np.array(gs_iters)
        epochs = np.arange(gs_iters.shape[0])
        temp_logger = np.array(temp_logger)
        GS_iters_file = "GS_iters_" + filename + 'seed_' + str(seed) + '.pdf'
        temp_file = "Temp" + filename + 'seed_' + str(seed) + '.pdf'
        plt.plot(epochs, gs_iters, color = 'blue')
        plt.savefig(os.path.join(folder_to_write, GS_iters_file))
        plt.clf()
        plt.plot(epochs, temp_logger, color = 'blue')
        plt.savefig(os.path.join(folder_to_write, temp_file))
        plt.clf()
    # Print final loss
    print('Last Epoch %d | Final Soft loss: %.5f' % (epoch, loss.item()))
    print('Best Loss Epoch %d | Best Soft loss: %.5f' % (best_loss_epoch, best_loss.item()))
    # Final coloring
    for atom_index in range(bitstring_n.shape[1]):
        unique_pos_per_atom[atom_index]= torch.stack(unique_pos_per_atom[atom_index])
    unique_pos_per_atom = torch.squeeze(torch.stack(unique_pos_per_atom))
    probs_save = torch.stack(probs_save)
    bitstring_save = torch.stack(bitstring_save)
    assgnmnts_change = torch.stack(assgnmnts_change)
    assgnd_nodes_num = torch.stack(assgnd_nodes_num)
    final_bitstring = bitstring_n
    return probs, best_loss_epoch, final_bitstring, best_bitstring, probs_save, bitstring_save, assgnmnts_change, assgnd_nodes_num, unique_pos_per_atom