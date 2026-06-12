import numpy as np
import torch
import random
import pickle
import argparse

from lp_to_bqm import BQM, qubo_to_torch
from utils import loss_func, gt_energy_lookup


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


def initialize_bitstring(num_positions, n_atoms, stoich_const, seed, TORCH_DEVICE, TORCH_DTYPE):
    set_seed(seed)
    bitstring = torch.zeros((num_positions, n_atoms)).type(TORCH_DTYPE).to(TORCH_DEVICE)
    pos_atom = random.sample(range(0, num_positions-1), int(stoich_const.sum())) # randomly select some positions to allocate with atoms
    bitstring_shrunk = pos_atom
    start = 0
    for atom_type in range(n_atoms):
        bitstring[pos_atom[start:start+int(stoich_const[atom_type])], atom_type] = 1
        start += int(stoich_const[atom_type])
    void_positions = [i for i in np.arange(num_positions) if i not in pos_atom]# list of empty positions (void assignment)
    return bitstring, bitstring_shrunk, void_positions


def Greedy_Baseline(file_to_parse, num_steps, num_seeds):

    bqm_model = BQM()
    bqm_model.parse_lp("instances/" + file_to_parse)
    with_void = False  # include void for each position

    structure_with_g = file_to_parse.split('.')[0]
    gt_energy = gt_energy_lookup(structure_with_g)
    Q, num_positions, atoms, stoich_const, offset = qubo_to_torch(
        bqm_model,
        with_void = with_void,  torch_dtype=TORCH_DTYPE,
        torch_device=TORCH_DEVICE)  # pre-constrained Q matrix

    n_atoms = len(atoms)  # number of atoms
    num_classes = (n_atoms + 1) if with_void else n_atoms  # number of node classes
    n_variables = num_positions*num_classes
    print("Crystal Structure Prediction of " + file_to_parse.split('.')[0] + " with " + str(num_positions)
          + " positions" + " and " + str(atoms) + " atoms.")
    print(n_variables, "Variables in the MultiClass Classification Problem")
    best_loss = np.zeros(num_seeds)
    best_rog = np.zeros(num_seeds)
    print("---Performing Greedy---")
    for seed in range(num_seeds):
        best_loss_seed = 1e8
        set_seed(seed)
        bitstring, bitstring_shrunk, void_positions = initialize_bitstring(num_positions, 
                                                                                      n_atoms, 
                                                                                      stoich_const, 
                                                                                      seed, 
                                                                                      TORCH_DEVICE,
                                                                                      TORCH_DTYPE)
        loss = torch.squeeze(loss_func(Q, bitstring, offset)).item()                                                                                                                                        
        best_bitstring = bitstring.clone()
        best_loss[seed] = loss
        for _ in range(num_steps):
            pos_to_empty = random.sample(range(0, len(bitstring_shrunk)), 1) # pointer to position that becomes empty
            pos_to_fill = random.sample(range(0, len(void_positions)), 1) # pointer to position that becomes allocated
            pos = int(bitstring_shrunk[pos_to_empty[0]]) # pos that becomes empty
            atom_type = int(torch.where(bitstring[pos]==1)[0])
            bitstring[pos, atom_type] = 0
            bitstring[void_positions[pos_to_fill[0]], atom_type] = 1
            # accept new state if it has lower energy than the best
            if loss_func(Q, bitstring, bqm_model.offset).item() < best_loss_seed:
                bitstring_shrunk[pos_to_empty[0]] = void_positions[pos_to_fill[0]]
                best_loss_seed = loss_func(Q, bitstring, bqm_model.offset).item()
                best_bitstring = bitstring.clone()
                void_positions[pos_to_fill[0]] = pos
            else: # reverse bitstring updates to stay with the previous best loss
                bitstring[pos, atom_type] = 1
                bitstring[void_positions[pos_to_fill[0]], atom_type] = 0
        best_loss[seed] = best_loss_seed
        print(f"Best structure of seed {seed} has energy: {best_loss[seed]}")
        if gt_energy:
            best_rog[seed] = \
            (gt_energy - best_loss_seed)/gt_energy
            print(f"Best structure of seed {seed} has ROG: {best_rog[seed]}")
        for i in range(len(atoms)):
            print(f"{atoms[i]} atoms in positions {torch.where(best_bitstring[:, i]==1)[0].numpy()}")


def Simulated_Annealing(file_to_parse, num_seeds, num_steps, temp_init, sweep_factor, cooling_fac=0.99):

    bqm_model = BQM()
    bqm_model.parse_lp("instances/" + file_to_parse)
    with_void = False  # include void for each position
    structure_with_g = file_to_parse.split('.')[0]
    gt_energy = gt_energy_lookup(structure_with_g)
    Q, num_positions, atoms, stoich_const, offset = qubo_to_torch(
        bqm_model,with_void = with_void,  torch_dtype=TORCH_DTYPE,
        torch_device=TORCH_DEVICE)  # pre-constrained Q matrix
    n_atoms = len(atoms)  # number of atoms
    num_classes = (n_atoms + 1) if with_void else n_atoms  # number of node classes
    n_variables = num_positions*num_classes
    print(f"Crystal Structure Prediction of  {file_to_parse.split('.')[0]} with {num_positions} positions and {atoms} atoms.")
    print(n_variables, "Variables in the MultiClass Classification Problem")
    # current_loss = torch.zeros(num_seeds)
    best_rog = np.zeros(num_seeds)
    print("---Performing Simulated Annealing---")
    best_loss_seed = np.zeros((num_seeds, 1))
    best_bitstring = torch.zeros((num_positions, n_atoms+1)).type(TORCH_DTYPE).to(TORCH_DEVICE)
    for seed in range(num_seeds):
        temp = temp_init
        set_seed(seed)
        bitstring, bitstring_shrunk, void_positions = initialize_bitstring(num_positions, n_atoms, stoich_const, seed, TORCH_DEVICE, TORCH_DTYPE)
        loss = torch.squeeze(loss_func(Q, bitstring, offset)).item()
        best_loss = loss
        current_loss = loss
        for step in range(num_steps):
            # find new candidate state by allocating the atom of a filled position to an empty one
            # to make sure that the stoichiometry constraints are satisfied
            pos_to_empty = random.sample(range(0, len(bitstring_shrunk)), 1) # pointer to position that becomes empty
            pos_to_fill = random.sample(range(0, len(void_positions)), 1) # pointer to position that becomes allocated
            pos = int(bitstring_shrunk[pos_to_empty[0]]) # pos that becomes empty
            atom_type = int(torch.where(bitstring[pos] == 1)[0])
            bitstring[pos, atom_type] = 0
            bitstring[void_positions[pos_to_fill[0]], atom_type] = 1
            candidate_loss = torch.squeeze(loss_func(Q, bitstring, offset)).item() # loss of new candidate state
            if step % (sweep_factor*stoich_const.sum()) == 0 and step != 0:
                temp = temp*cooling_fac if temp > 0.01 else 0.01
            # accept candidate state if it provides a new best loss or metropolis criterion is satisfied
            if candidate_loss < current_loss or random.uniform(0, 1) < np.exp((current_loss-candidate_loss)/temp): # update the solution
                if candidate_loss < best_loss:
                    best_bitstring = bitstring.clone()
                    best_loss = candidate_loss
                bitstring_shrunk[pos_to_empty[0]] = void_positions[pos_to_fill[0]]
                current_loss = candidate_loss # loss of new state
                void_positions[pos_to_fill[0]] = pos
            else: # reverse candidate bitstring updates to stay with the previous state
                bitstring[pos, atom_type] = 1
                bitstring[void_positions[pos_to_fill[0]], atom_type] = 0

        best_loss_seed[seed] = best_loss
        print(f"Best structure of seed {seed} has energy: {best_loss_seed[seed]}")
        if gt_energy:
            best_rog[seed] = (gt_energy - best_loss)/gt_energy
            print(f"Best structure of seed {seed} has ROG: {best_rog[seed]}")
        for i in range(len(atoms)):
            print(f"{atoms[i]} atoms in positions {torch.where(best_bitstring[:, i]==1)[0].numpy()}")

# Set GPU/CPU
TORCH_DEVICE = 'cpu'
TORCH_DTYPE = torch.float32

if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='CSP using Classical methods')
    parser.add_argument('--instance', type=str, default='SrTiO3G4', metavar='N',
                            help='Instance name to run structure search on (default: SrTiO3G4)')
    parser.add_argument('--multiple', type=int, default=1, metavar='N',
                            help='Number of unit cell repetitions (default: 1)')
    parser.add_argument('--method', type=str, default='SA', metavar='N', choices = ['Greedy', 'SA'],
                            help='Type of method to use for structure search (default: SA)')
    parser.add_argument('--num-seeds', type=int, default=5, metavar='N',
                            help='Number of random seeds to run the method with (default: 5)')
    parser.add_argument('--num-steps', type=int, default=1000, metavar='N',
                            help='Number of steps to run the method for (default: 1000)')
    args = parser.parse_args()

    file_to_parse = f"{args.instance}_{args.multiple}.lp" # model file to parse
    if args.method == 'Greedy':
        Greedy_Baseline(file_to_parse=file_to_parse, num_steps=args.num_steps, num_seeds=args.num_seeds)
    elif args.method == 'SA':
        temp_init = {'SrTiO3G4_1': 1000, 'SrTiO3G8_1': 1000, 'SrTiO3G8_2': 10000, 'SrTiO3G8_3': 5000,'Y2O3G8_1': 3000, 'Y2Ti2O7G8_1': 1000, 'LiMgAlOG8_1': 3000, 'CaAlSiOG8_1': 10000}
        sweep_factor = 2 if args.instance == 'Y2O3G8' else 1
        Simulated_Annealing(file_to_parse=file_to_parse, num_seeds=args.num_seeds, num_steps=args.num_steps, temp_init=temp_init[file_to_parse.split('.')[0]], sweep_factor=sweep_factor)
    else:
        raise Exception("Not implemented")