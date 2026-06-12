import torch
import numpy as np
import torch.nn.functional as F
from torch import linalg as LA


def gt_energy_lookup(problem_name):

    """Ground truth energies for the compositions discussed in the paper are computed
    "with IPCSP, Gusev et al. [2023] using this repo https://github.com/lrcfmd/ipcsp """
    
    
    gt_energy = {'SrTiO3G4_1': -158.76, 'SrTiO3G8_1': -158.76, 'SrTiO3G8_2': -1268.67, 
                 'SrTiO3G8_3': -4281.76, 'Y2O3G8_1': -2191.57, 'Y2TiO7G8_1': -3093.53,
                 'LiMgAlOG8_1': -1620.89, 'CaAlSiOG8_1': -2199.99}

    if problem_name not in gt_energy.keys():
        print("Ground truth energy is unknown, ROG cannot be computed")
        return False
    else:
        print("Ground truth energy found, ROG will be computed")
        return gt_energy[problem_name]


def crystal_coordinates(problem_name, num_positions):

    cell_param = {'SrTiO3G4_1': 3.9, 'SrTiO3G8_1': 3.9, 'SrTiO3G8_1': 7.8, 
                  'SrTiO3G8_3': 11.9, 'Y2O3G8_1': 10.7, 'Y2Ti2O7G8_1': 10.2, 
                  'LiMgAlOG8_1': 8.2, 'CaAlSiOG_1': 11.9}
    
    struct_info = problem_name.split('G')
    struct_name = struct_info[0]
    g = int(struct_info[1].split('_')[0]) 

    discrete_dist = cell_param[problem_name] / g
    pos = np.zeros((num_positions, 3))
    row = 0
    for (i, j, k) in np.ndindex(g, g, g):
        pos[row,] = np.array([i * discrete_dist, j * discrete_dist, k * discrete_dist])
        row = row + 1
    return pos, cell_param[problem_name]


def interactions_to_edge_features(Q, n_atoms, normalize):
    """ Create edge features according to the entries of Q matrix"""
    edge_features = []
    self_loop_edge_features = []
    ind = []
    self_loop_ind = []
    
    Z = Q + torch.transpose(Q, 0, 1) - torch.diag(torch.diagonal(Q))
    rows, cols = torch.triu_indices(n_atoms, n_atoms)
    
    for c1, i in enumerate(range(0, Q.shape[0] - n_atoms, n_atoms + 1)):
        for c2, j in enumerate(range(0, Q.shape[1] - n_atoms, n_atoms + 1)):
            features = Z[i:i + n_atoms, j:j + n_atoms][..., rows, cols]
            if i != j:
                edge_features.append(features)
                ind.append([c1, c2])
            else:
                self_loop_edge_features.append(features)
                self_loop_ind.append([c1, c2])
    
    total_edge_features = edge_features + self_loop_edge_features
    edge_index = ind + self_loop_ind
    edge_features = torch.stack(total_edge_features, dim=0)
    
    if normalize:
        edge_features = F.normalize(edge_features, dim=0)
    
    return edge_features, edge_index





def loss_func(Q, probs, offset):

    probs_flattened = torch.flatten(probs)
    p = torch.kron(probs_flattened, probs_flattened)
    Q_flatten = torch.flatten(Q)
    loss = torch.unsqueeze(p, 0)@torch.unsqueeze(Q_flatten, 1) + offset
    return loss


def log_sinkhorn(log_alpha, r, n_iter, eps = 0.01):
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
    alpha_prev = log_alpha.exp()
    alpha_diff = []
    for i in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = torch.log(r) + log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)   
        alpha = log_alpha.exp()
        alpha_diff.append(torch.norm(alpha.detach()-alpha_prev.detach())**2)
        if LA.vector_norm((alpha.sum(dim=-1)-1), ord=float('inf')) < eps and LA.vector_norm((alpha.sum(dim=-2)-r), ord=float('inf')) < eps :
            break
        if i == n_iter - 1:
            print('Sinkhorn did not converge!')
        alpha_prev = alpha.clone()
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

