import torch
import math
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from dgl.nn.pytorch import GraphConv, SAGEConv
from itertools import chain, islice
from time import time



# GNN class to be instantiated with specified param values
class GCN_dev(nn.Module):
    def __init__(self, in_feats, hidden_size, number_classes, dropout, temperature, device):
        """
        Initialize a new instance of the core GCN model of provided size.
        Dropout is added in forward step.

        Inputs:
            in_feats: Dimension of the input (embedding) layer
            hidden_size: Hidden layer size
            dropout: Fraction of dropout to add between intermediate layer. Value is cached for later use.
            device: Specifies device (CPU vs GPU) to load variables onto
        """
        super(GCN_dev, self).__init__()

        self.dropout_frac = dropout
        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = GraphConv(hidden_size, number_classes).to(device)
        self.temperature_scaling = nn.Parameter(temperature*torch.ones(1)).to(device)

    def forward(self, g, inputs):
        """
        Run forward propagation step of instantiated model.

        Input:
            self: GCN_dev instance
            g: DGL graph object, i.e. problem definition
            inputs: Input (embedding) layer weights, to be propagated through network
        Output:
            h: Output layer weights
        """

        # input step
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)

        # output step
        h = self.conv2(g, h)
        h = torch.div(h, self.temperature_scaling)
        h = torch.sigmoid(h)


        return h

# GNN class to be instantiated with specified param values
class GraphSAGE_dev(nn.Module):
    def __init__(self, in_feats, hidden_size, aggregator, number_classes, dropout, temperature, device):
        """
        Initialize a new instance of the core GCN model of provided size.
        Dropout is added in forward step.

        Inputs:
            in_feats: Dimension of the input (embedding) layer
            hidden_size: Hidden layer size
            dropout: Fraction of dropout to add between intermediate layer. Value is cached for later use.
            device: Specifies device (CPU vs GPU) to load variables onto
        """
        super(GraphSAGE_dev, self).__init__()

        self.dropout_frac = dropout
        self.conv1 = SAGEConv(in_feats, hidden_size, aggregator).to(device)
        self.conv2 = SAGEConv(hidden_size, number_classes, aggregator).to(device)
        self.temperature_scaling = nn.Parameter(temperature*torch.ones(1)).to(device)

    def forward(self, g, inputs):
        """
        Run forward propagation step of instantiated model.

        Input:
            self: GCN_dev instance
            g: DGL graph object, i.e. problem definition
            inputs: Input (embedding) layer weights, to be propagated through network
        Output:
            h: Output layer weights
        """

        # input step
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac)

        # output step
        h = self.conv2(g, h)
        h = torch.div(h, self.temperature_scaling)
        h = torch.sigmoid(h)

        return h


# Generate random graph of specified size and type,
# with specified degree (d) or edge probability (p)
def generate_graph(n, d=None, p=None, pos=None, cutoff_dist = None, graph_type='reg',
                   random_seed=0):
    """
    Helper function to generate a NetworkX random graph of specified type,
    given specified parameters (e.g. d-regular, d=3). Must provide one of
    d or p, d with graph_type='reg', and p with graph_type in ['prob', 'erdos'].

    Input:
        n: Problem size
        d: [Optional] Degree of each node in graph
        p: [Optional] Probability of edge between two nodes
        graph_type: Specifies graph type to generate
        random_seed: Seed value for random generator
    Output:
        nx_graph: NetworkX OrderedGraph of specified type and parameters
    """
    if graph_type == 'reg':
        print(f'Generating d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'prob':
        print(f'Generating p-probabilistic graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.fast_gnp_random_graph(n, p, seed=random_seed)
    elif graph_type == 'erdos':
        print(f'Generating erdos-renyi graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.erdos_renyi_graph(n, p, seed=random_seed)
    elif graph_type == 'complete':
        nx_temp = nx.complete_graph(n=n)
    elif graph_type == 'cutoff':
        nx_temp = nx.complete_graph(n=n)
        distvec = pdist(pos)
        square_dist = squareform(distvec) # square distance matrix between positions
        ind = np.where(square_dist>cutoff_dist) # find pairs of positions with distance > cutoff distance
        A = nx.adjacency_matrix(nx_temp).todense()
        A[ind] = 0
        nx_temp = nx.from_numpy_matrix(A)
    else:
        raise NotImplementedError(f'!! Graph type {graph_type} not handled !!')
    # Networkx does not enforce node order by default
    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)
    # Need to pull nx graph into OrderedGraph so training will work properly
    nx_graph = nx.OrderedGraph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)
    # if edge_attrs is not None:
    #     for i, l in enumerate(nx_graph.edges):
    #         nx.set_edge_attributes(nx_graph, {l:{'interaction':edge_attrs[i]}})
    # nx_graph = from_networkx(nx_graph) for pyg
    return nx_graph




# helper function to convert Q dictionary to torch tensor
def qubo_dict_to_torch(nx_G, Q, torch_dtype=None, torch_device=None):
    """
    Output Q matrix as torch tensor for given Q in dictionary format.

    Input:
        Q: QUBO matrix as defaultdict
        nx_G: graph as networkx object (needed for node lables can vary 0,1,... vs 1,2,... vs a,b,...)
    Output:
        Q: QUBO as torch tensor
    """

    # get number of nodes
    n_nodes = len(nx_G.nodes)

    # get QUBO Q as torch tensor
    Q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val

    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)

    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)

    return Q_mat


# Chunk long list
def gen_combinations(combs, chunk_size):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])


# helper function for custom loss according to Q matrix
def loss_func(probs, Qunc, C1, C2, offset):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """

    probs_ = torch.unsqueeze(probs, 1)

    # minimize cost = x.T * Q * x
    total_loss = (probs_.T @ Qunc @ probs_).squeeze() + (probs_.T @ C1 @ probs_).squeeze() + (probs_.T @ C2 @ probs_).squeeze() + offset
    uncon_loss = (probs_.T @ Qunc @ probs_).squeeze()
    C1_loss = (probs_.T @ C1 @ probs_).squeeze()
    C2_loss = (probs_.T @ C2 @ probs_).squeeze() + offset
    return total_loss, uncon_loss, C1_loss, C2_loss

def T_scaling(logits, temperature):
    return torch.div(logits, temperature)

# Construct graph to learn on
def get_gnn(n_nodes, graph_encoder, gnn_hypers, opt_params, scheduler_bool, torch_device, torch_dtype):
    """
    Generate GNN instance with specified structure. Creates GNN, retrieves embedding layer,
    and instantiates ADAM optimizer given those.

    Input:
        n_nodes: Problem size (number of nodes in graph)
        gnn_hypers: Hyperparameters relevant to GNN structure
        opt_params: Hyperparameters relevant to ADAM optimizer
        torch_device: Whether to load pytorch variables onto CPU or GPU
        torch_dtype: Datatype to use for pytorch variables
    Output:
        net: GNN instance
        embed: Embedding layer to use as input to GNN
        optimizer: ADAM optimizer instance
    """
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']
    weight_decay = opt_params['weight_decay']
    temperature = gnn_hypers["temperature"]
    # instantiate the GNN
    if graph_encoder == 'GCN':
        net = GCN_dev(dim_embedding, hidden_dim, number_classes, dropout, temperature, torch_device)
    elif graph_encoder == 'GraphSAGE':
        aggregator = 'mean'
        net = GraphSAGE_dev(dim_embedding, hidden_dim, aggregator, number_classes, dropout, temperature, torch_device)
    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())
    if weight_decay is not None:
        optimizer = torch.optim.AdamW(params, **opt_params)
    else:
        optimizer = torch.optim.Adam(params, **opt_params)
    if scheduler_bool:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100)
    else:
        scheduler = None
    return net, embed, optimizer, scheduler


# Parent function to run GNN training given input config
def run_gnn_training(Qunc, C1, C2, offset, dgl_graph, net, embed, optimizer, scheduler, number_epochs, tol, patience, prob_threshold):
    """
    Wrapper function to run and monitor GNN training. Includes early stopping.
    """
    # Assign variable for user reference
    inputs = embed.weight

    prev_loss = 1.  # initial loss value (arbitrary)
    count = 0       # track number times early stopping is triggered
    # initialize optimal solution
    best_bitstring = torch.zeros((dgl_graph.number_of_nodes(),)).type(Qunc.dtype).to(Qunc.device)
    best_loss, best_uncon_loss, best_C1_loss, best_C2_loss = loss_func(best_bitstring.float(), Qunc, C1, C2, offset)

    t_gnn_start = time()
    loss_ = []
    uncon_loss_ = []
    LEQ_loss_ = []
    EQ_loss_ = []
    # Training logic
    lrs = []
    print(f'Offset: {offset}')
    for epoch in range(number_epochs):

        # get logits/activations
        # if epoch > 0:
        probs = net(dgl_graph, inputs)[:, 0]  # collapse extra dimension output from model
        # build cost value with QUBO cost function
        loss, uncon_loss, LEQ_loss, EQ_loss = loss_func(probs, Qunc, C1, C2, offset)
        loss_.append(loss.detach().item())
        uncon_loss_.append(uncon_loss.detach().item())
        LEQ_loss_.append(LEQ_loss.detach().item())
        EQ_loss_.append(EQ_loss.detach().item())
        # Apply projection
        bitstring = (probs.detach() >= prob_threshold) * 1
        if loss < best_loss:
            best_loss = loss
            best_bitstring = bitstring

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Total Loss: {loss_[epoch]}, Unconstrained Loss: {uncon_loss_[epoch]}, LEQ loss: {LEQ_loss_[epoch]}, EQ loss: {EQ_loss_[epoch]}')
        # early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss_[epoch] - prev_loss) <= tol) | ((loss_[epoch] - best_loss) > 0):
            # print(loss_[epoch], prev_loss)
            count += 1
        else:
            count = 0
        # if (abs(loss_ - prev_loss) <= tol):
        #     break
        if count >= patience:
            print(f'Stopping early on epoch {epoch} (patience: {patience})')
            break

        # update loss tracking
        prev_loss = loss_[epoch]

        # run optimization with backpropagation
        optimizer.zero_grad()  # clear gradient for step
        loss.backward()        # calculate gradient through compute graph
        optimizer.step()       # take step, update weights
        lrs.append(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step(loss)

    t_gnn = time() - t_gnn_start
    print(f'GNN training (n={dgl_graph.number_of_nodes()}) took {round(t_gnn, 3)}')
    print(f'GNN final continuous loss: {loss_[epoch]}')
    print(f'GNN best continuous loss: {best_loss}')

    final_bitstring = (probs.detach() >= prob_threshold) * 1
    # wandb.run.summary["best_loss"] = best_loss
    return net, epoch, final_bitstring, best_bitstring


def crystal_coordinates(struct_name, g, num_positions):
    cell_param = {'SrTiO3': 3.9, 'Y2O3': 10.7, 'Y2Ti2O7': 10.2, 'LiMgAlPO': 8.2}
    discrete_dist = cell_param[struct_name] / g
    pos = np.zeros((num_positions, 3))
    row = 0
    for (i, j, k) in np.ndindex(g, g, g):
        pos[row,] = np.array([i * discrete_dist, j * discrete_dist, k * discrete_dist])
        row = row + 1
    return pos

def gaussian_kernel(x, sigma):
    return torch.exp(- (x**2/(2*sigma**2)))

def interactions_to_edge_features(Q, n_atoms):
    interactions = []
    edge_features = []
    Z = Q + torch.transpose(Q, 0 ,1) - torch.diag(torch.diagonal(Q))
    for i in range(0, Q.shape[0]-n_atoms, n_atoms+1):
        for j in range(0, Q.shape[1]-n_atoms, n_atoms+1):
            if i!=j: # do not consider self-loops for now
                interactions.append(torch.flatten(Z[i:i+n_atoms, j:j+n_atoms]))
                l = interactions[-1]
                edge_features.append(torch.tensor([x for i, x in enumerate(l) if x not in l[:i]]))
    edge_features = torch.stack(edge_features, dim = 0)
    return F.normalize(edge_features, dim=0)











