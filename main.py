import torch
import random
import numpy as np
from time import time
from itertools import chain
from pathlib import Path
import torch.nn as nn
import yaml
import argparse

from utils import crystal_coordinates, loss_func, gumbel_sinkhorn, gt_energy_lookup
from lp_to_bqm import BQM, qubo_to_torch
from model import GINModel, BondEncoderLinear
from get_data import generate_pyg_data, load_config

    
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

def structure_search(args, device, dtype):

    instance = args.instance
    multiple = args.multiple
    graph_type = args.graph_type
    seed_value = args.seed

    instances_folder = Path("instances/")
    file_names = [f.name for f in instances_folder.iterdir() if f.is_file()]

    file_to_parse = f"{instance}_{multiple}.lp" # model file to parse
    if file_to_parse not in file_names:
        raise Exception("Composition .lp file not found!")
    

    structure_with_g = file_to_parse.split('.')[0]
    gt_energy = gt_energy_lookup(structure_with_g)


    bqm_model = BQM()
    start = time()
    bqm_model.parse_lp("instances/" + file_to_parse)
    end = np.round((time() - start), 1)
    print(f"Parsed the model .lp file in {end} seconds")


    start = time()
    problem_info = qubo_to_torch(
        bqm_model, with_void=True, torch_dtype=dtype, torch_device=device
    )
    end = np.round((time() - start), 1)
    print(f"Constructed Q matrix in {end} seconds")


    Q, num_positions, atoms, stoich_const, offset = problem_info
    n_atoms = len(atoms) # number of atoms
    num_classes = (n_atoms + 1)
    n_variables = num_positions*num_classes
    print("Crystal Structure Prediction of " + file_to_parse.split('.')[0] + " with " + str(num_positions)
        + " positions" + " and " + str(atoms) + " atoms.")
    print(n_variables, "Variables in the MultiClass Classification Problem")


    pos, cell_param = crystal_coordinates(structure_with_g, num_positions) # atoms' candidate 3d coordinates within the unit cell


    if graph_type == 'cutoff':
        config = load_config('config.yaml')
        config["local_cutoff"]['cell_param'] = cell_param
        with open("config.yaml", "w") as f:
            yaml.safe_dump(config, f)
        


    nn_config = load_config('config.yaml')['hypers']
    dim_embedding = nn_config['dim_embedding']
    hidden_dims = nn_config['hidden_channels']
    bond_encoder_out_dims = nn_config['bond_encoder_out']
    weight_decay = nn_config['weight_decay']
    lr = nn_config['lr']
    num_layers = nn_config['num_layers']

    sink_it = nn_config['sink_it'] # Sinkhorn iterations
    PROB_THRESHOLD = nn_config['prob_threshold']
    temperature = nn_config['temperature']
    scaling = nn_config['scaling']      

    opt_hypers = {
        'lr': lr,
        'weight_decay' : weight_decay
    }

    number_epochs = 100000

    edge_weight = None

    set_seed(seed_value)
    embed = nn.Embedding(num_positions, dim_embedding)
    embed = embed.type(dtype).to(device)
    normalize = True # for edge features
    start = time()
    train_data = generate_pyg_data(graph_type, Q, embed.weight, num_positions, n_atoms, pos, edge_attr=True, normalize=normalize)
    end = np.round((time() - start), 1)
    print(f"Constructed {graph_type} Graph in {end} seconds")
    train_data.to(device)
    best_hard_loss = np.inf

    net = GINModel(in_channels=dim_embedding,
                hidden_channels=hidden_dims,
                out_channels=num_classes,
                num_layers=num_layers, 
                device=device,
                bond_encoder= BondEncoderLinear(in_channels=train_data.edge_attr.shape[1],
                                                hidden_channels=bond_encoder_out_dims,
                                                num_layers=1),
                readout_dim=num_classes,
                use_readout=False).to(device)
    
    print(f'Function run_gnn_training(): Setting seed to {seed_value}')
    params = chain(net.parameters(), embed.parameters())
    print('Building ADAM-W optimizer...')
    optimizer = torch.optim.AdamW(params, **opt_hypers)
    print('Running GNN...')
    i = 0
    t = temperature
    best_rog = np.inf
    for epoch in range(number_epochs):
        logits = net(train_data, edge_weight)
        if i>=5000:
            t = scaling*t
            print(f"Temperature scaled by {scaling}, new temperature is {t}")
        probs, i = gumbel_sinkhorn(logits, stoich_const, tau=t, n_iter=sink_it)
        loss = loss_func(Q, probs, offset) 
        bitstring = (probs.detach() >= PROB_THRESHOLD) * 1
        hard_loss = loss_func(Q, bitstring.float(), offset)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if hard_loss.item() < best_hard_loss and torch.equal(bitstring.sum(dim=0)[:-1], stoich_const[:-1]):
            best_hard_loss = hard_loss.item()
            print(f"Found new best structure with energy: {best_hard_loss} at epoch {epoch}")
        if epoch%50==0:
            print('Epoch %d | Total Soft Loss: %.5f' % (epoch, loss.item()))
            print('Epoch %d | Total Hard Loss: %.5f' % (epoch, hard_loss.item()))
            print('Epoch %d | Predicted stoichiometry: %s' % (epoch, bitstring.sum(dim=0)))

    print(f'Ground state energy = {gt_energy}')
    print(f'Energy of best structure = {best_hard_loss}')
    if gt_energy:
        best_rog = (gt_energy - best_hard_loss)/gt_energy
        print(f'ROG of best structure = {best_rog}')


if __name__ =='__main__':
    
    parser = argparse.ArgumentParser(description='CSP using Graph Neural Transportation')
    parser.add_argument('--instance', type=str, default='SrTiO3G4', metavar='N',
                            help='Instance name to run structure search on (default: SrTiO3G4)')
    parser.add_argument('--multiple', type=int, default=1, metavar='N',
                            help='Number of unit cell repetitions (default: 1)')
    parser.add_argument('--graph-type', type=str, default='Gabber-Galil', metavar='N', choices=['Gabber-Galil', 'Margulis', 'Cutoff'],
                            help='Type of graph to construct for message-passing (default: Gabber-Galil)')
    parser.add_argument('--seed', type=int, default=10, metavar='N',
                            help='Seed value')
    args = parser.parse_args()


    # Set GPU/CPU
    TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TORCH_DTYPE = torch.float32
    print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')

    structure_search(args, TORCH_DEVICE, TORCH_DTYPE)