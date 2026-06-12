import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from pymatgen.core import Lattice, Structure
from torch_geometric.utils import add_self_loops
import yaml

from utils import interactions_to_edge_features
from expander import stacked_planes_edges, add_affine_edges_for_all_planes


def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as file:
        configs = yaml.safe_load(file)
    return configs

def generate_pyg_data(graph_type, Q, input_feats, n_nodes, n_atoms, positions, edge_attr=True, normalize=True):

    x = input_feats

    # Compute pairwise distances using node positions
    
    if graph_type == 'Margulis':
        g = int(np.round(n_nodes ** (1. / 3)))
        edge_index = stacked_planes_edges(g)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_nodes)
    elif graph_type == 'Gabber-Galil':
        g = int(np.round(n_nodes ** (1. / 3)))
        edge_index = add_affine_edges_for_all_planes(g)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_nodes)
    elif graph_type == 'Cutoff':
        config = load_config("config.yaml")["local_cutoff"]
        cell_param = config['cell_param']  
        cutoff_radius = config['cutoff_radius']
        max_local_nbr = config['max_local_nbr']
        edge_index = radius_graph(cell_param, positions, cutoff_radius, max_local_nbr)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_nodes)
    else:
        raise Exception("Graph construction not implemented")
    if edge_attr:
        all_edge_attr, all_edge_index = interactions_to_edge_features(Q, n_atoms, normalize=normalize)
        edge_index_set = set(map(tuple, edge_index.t().tolist()))  # Use a set for faster membership test

        cutoff_edge_attr = [edge_attr for ind, edge_attr in zip(all_edge_index, all_edge_attr) if tuple(ind) in edge_index_set]
        cutoff_edge_index = [ind for ind in all_edge_index if tuple(ind) in edge_index_set]

        edge_attr = torch.stack(cutoff_edge_attr, dim=0)
        edge_attr = F.normalize(edge_attr, dim=0)  # normalize after the cutoff
        edge_index = torch.tensor(cutoff_edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_attr = None
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def add_edge(node, neighbor, edge_index, neighbor_counts, neighbor_dict, total_neighbors):
    # Ensure that both nodes haven't exceeded their neighbor count
    if neighbor_counts[node] < total_neighbors and neighbor_counts[neighbor] < total_neighbors:
        # Check if the edge already exists
        if neighbor not in neighbor_dict[node]:
            # Add the undirected edge (both node -> neighbor and neighbor -> node)
            edge_index[0].append(node)
            edge_index[1].append(neighbor)
            edge_index[0].append(neighbor)
            edge_index[1].append(node)
            
            # Update neighbor dictionaries
            neighbor_dict[node].add(neighbor)
            neighbor_dict[neighbor].add(node)
            
            # Increment the neighbor count for both nodes
            neighbor_counts[node] += 1
            neighbor_counts[neighbor] += 1
            
    return edge_index, neighbor_counts, neighbor_dict


def radius_graph(cell_param, pos, radius, max_num_nbr):
    lattice = Lattice.cubic(cell_param)
    species = ['' for _ in range(pos.shape[0])]
    frac_coords = lattice.get_fractional_coords(pos)  # Converts to fractional
    crystal = Structure(lattice, species, frac_coords) # change frac_coords to pos to get the best results for now
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx = []
    for nbr in all_nbrs:
        if max_num_nbr is None:
            nbr_fea_idx.append(list(map(lambda x: x[2],
                                        nbr[:])))
        else:        
            if len(nbr) < max_num_nbr:
                print("Neighbors are less than the maximum")
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                    [0] * (max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:max_num_nbr]))) 
    N, M = len(nbr_fea_idx), len(nbr_fea_idx[0])  # N = number of atoms/sites, M = number of neighbors

    edge_index = [[], []]  # List to store the edges
    for i in range(N):
        M = len(nbr_fea_idx[i])
        for j in range(M):
            neighbor_idx = nbr_fea_idx[i][j]
            # Add edge from site i to its neighbor at neighbor_idx
            edge_index[0].append(i)              # Source node (current site)
            edge_index[1].append(neighbor_idx)   # Target node (neighbor)
    # Convert to a PyTorch tensor for use in a GNN
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

    return edge_index