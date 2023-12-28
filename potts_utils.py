import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import os

from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GraphConv
from itertools import chain
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



# Define GNN GraphSage object
class GNNSage(nn.Module):
    """
    Basic GraphSAGE-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters - in this
    case, just dropout.
    """

    def __init__(self, g, in_feats, hidden_size, num_classes, dropout, agg_type='mean'):
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
        :param agg_type: Aggregation type for each SAGEConv layer. All layers will use the same agg_type
        :type agg_type: str
        """

        super(GNNSage, self).__init__()

        self.g = g
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_size, agg_type, activation=F.relu))
        # output layer
        self.layers.append(SAGEConv(hidden_size, num_classes, agg_type))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        """
        Define forward step of netowrk. In this example, pass inputs through convolution, apply relu
        and dropout, then pass through second convolution.

        :param features: Input node representations
        :type features: torch.tensor
        :return: Final layer representation, pre-activation (i.e. class logits)
        :rtype: torch.tensor
        """
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)

        return h


# Define GNN GraphConv object
class GNNConv(nn.Module):
    """
    Basic GraphConv-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters - in this
    case, just dropout.
    """

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
        self.layers.append(GraphConv(in_feats, hidden_size, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(hidden_size, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        """
        Define forward step of netowrk. In this example, pass inputs through convolution, apply relu
        and dropout, then pass through second convolution.

        :param features: Input node representations
        :type features: torch.tensor
        :return: Final layer representation, pre-activation (i.e. class logits)
        :rtype: torch.tensor
        """

        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


# Construct graph to learn on #
def get_gnn(g, n_nodes, gnn_hypers, opt_params, scheduler_bool, torch_device, torch_dtype):
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
    if scheduler_bool:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100)
    else:
        scheduler = None
    return net, embed, optimizer, scheduler


def loss_func(Q, probs, offset):

    probs_flattened = torch.flatten(probs)
    p = torch.kron(probs_flattened, probs_flattened)
    Q_flatten = torch.flatten(Q)
    loss = torch.unsqueeze(p, 0)@torch.unsqueeze(Q_flatten, 1) + offset
    return loss

def log_sinkhorn(log_alpha, r, n_iter,  Q, offset, eps = 0.01):
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
    # Sinkhorn_start = time()
    log_alpha[0, -1] = -100
    for i in range(n_iter):

        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = torch.log(r) + log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
        alpha = log_alpha.exp()
        violations_r = torch.where(torch.abs(alpha.sum(dim = -1) - 1) > eps)
        violations_c = torch.where(torch.abs(alpha.sum(dim = -2) - r) > eps)
        if (len(violations_r[0]) == 0 and len(violations_c[0]) == 0) :
            break
        if i == n_iter - 1:
            print('Sinkhorn did not converge!')
    if torch.abs(alpha[0,-1]) > eps:
        print("Violation")
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

def gumbel_sinkhorn(log_alpha, stoich_const, tau, n_iter, Q, offset):
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
    sampled_perm_mat = log_sinkhorn((log_alpha + gumbel_noise)/tau, stoich_const, n_iter, Q, offset)
    return sampled_perm_mat

def run_gnn_training(filename, Q, offset, stoich_const, Gumbel_sinkhorn, graph_dgl, net, embed, optimizer, scheduler, temperature,
                     scaling, number_epochs=int(1e5), patience=100, tolerance=1e-4, prob_threshold = 0.5, flag = False,  seed=1):
    """
    Function to run model training for given graph, GNN, optimizer, and set of hypers.
    Includes basic early stopping criteria. Prints regular updates on progress as well as
    final decision.

    :param nx_graph: Graph instance to solve
    :param graph_dgl: Graph instance to solve
    :param adj_mat: Adjacency matrix for provided graph
    :type adj_mat: torch.tensor
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
    # Tracking
    best_loss = torch.tensor(float('Inf'))
    best_bitstring = torch.zeros((graph_dgl.number_of_nodes(),stoich_const.shape[0])).type(Q.dtype).to(graph_dgl.device)
    bitstring = torch.zeros((graph_dgl.number_of_nodes(),stoich_const.shape[0])).type(Q.dtype).to(graph_dgl.device)
    # Early stopping to allow NN to train to near-completion
    prev_loss = 1.  # initial loss value (arbitrary)
    cnt = 0  # track number times early stopping is triggered
    best_loss = 1e8
    i = 0
    if flag and Gumbel_sinkhorn: gs_iters = []
    # Training logic
    for epoch in range(number_epochs):
        # get soft prob assignments
        logits = net(inputs)
        if Gumbel_sinkhorn:
            if i>=500:
                t = scaling*t
                print("Temperature scaled by " + str(scaling) + ' ,t = ' + str(t))
            probs, i = gumbel_sinkhorn(logits, stoich_const, tau=t, n_iter=5000, Q= Q, offset = offset)
            if flag: gs_iters.append(i)
        else:
            if cnt%1000==0 and cnt>0:
                t = scaling*t
                print("Temperature scaled by " + str(scaling) + ' ,t = ' + str(t))
            logits = torch.div(logits, t)
            # apply softmax for normalization
            probs = F.softmax(logits, dim=1)
        loss = loss_func(Q, probs, offset)
        # bitstring = torch.zeros((graph_dgl.number_of_nodes(),stoich_const.shape[0])).type(Q.dtype).to(graph_dgl.device)
        bitstring = (probs.detach() >= prob_threshold) * 1
        # a different projector from probs to bitstring
        # for i,l in enumerate(stoich_const):
        #     s, indices = torch.sort(probs[:, i])
        #     z = indices[int((probs.shape[0]- l).item()):]
        #     bitstring[z, i] = 1
        hard_loss = loss_func(Q, bitstring.float(), offset)

        # assert(bitstring.sum(dim=0).any()==stoich_const.any())
        # Early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss - prev_loss) <= tolerance) | (loss >= best_loss ):
            cnt += 1
        else:
            cnt = 0

        if loss < best_loss:
            best_loss = loss
            best_bitstring = bitstring
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
        if scheduler is not None:
            scheduler.step(loss)
        # tracking: print intermediate loss at regular interval
        print('Epoch %d | Total Soft Loss: %.5f' % (epoch, loss.item()))
        print('Epoch %d | Total Hard Loss: %.5f' % (epoch, loss_func(Q, bitstring.float(), offset).item()))
        # if epoch % 100 == 0:
        #     print('Epoch %d | Total Soft Loss: %.5f' % (epoch, loss.item()))
            # print('Best Soft Loss so far: %.5f' % (best_loss.item()))

    if flag and Gumbel_sinkhorn:
        folder_to_write = "plots_no_scaling/" if scaling==1 else "plots_with_scaling/"
        if not os.path.exists(folder_to_write):
            os.makedirs(folder_to_write)
        gs_iters = np.array(gs_iters)
        epochs = np.arange(gs_iters.shape[0])
        f_name = "GS_iters_" + filename + 'seed_' + str(seed) + '.pdf'
        plt.plot(epochs, gs_iters, color = 'blue')
        plt.savefig(os.path.join(folder_to_write, f_name))
    # Print final loss
    print('Last Epoch %d | Final Soft loss: %.5f' % (epoch, loss.item()))
    print('Best Loss Epoch %d | Best Soft loss: %.5f' % (best_loss_epoch, best_loss.item()))
    # Final coloring
    final_bitstring = bitstring
    return probs, epoch, final_bitstring, best_bitstring