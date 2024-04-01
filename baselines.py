import numpy as np
import torch
import random
from time import time
from lp_to_bqm import BQM, qubo_to_torch
from potts_utils import loss_func, set_seed

def initialize_bitstring(num_positions, n_atoms, stoich_const, seed, TORCH_DEVICE, TORCH_DTYPE):
    set_seed(seed)
    bitstring_shrunk = n_atoms*torch.ones((num_positions, 1)).type(TORCH_DTYPE).to(TORCH_DEVICE) # index corresponds to position, entry corresponds to atom_type
    bitstring = torch.zeros((num_positions, n_atoms+1)).type(TORCH_DTYPE).to(TORCH_DEVICE)
    bitstring[:, -1] = 1
    pos_atom = random.sample(range(0, num_positions-1), int(stoich_const[:-1].sum())) # randomly select some positions to allocate with atoms
    start = 0
    for atom_type in range(n_atoms):
        bitstring_shrunk[pos_atom[start:start+int(stoich_const[atom_type])]] = atom_type
        bitstring[pos_atom[start:start+int(stoich_const[atom_type])], atom_type] = 1
        bitstring[pos_atom[start:start+int(stoich_const[atom_type])], -1] = 0
        start += int(stoich_const[atom_type])
    void_positions = [i for i in np.arange(num_positions) if i not in pos_atom]# list of empty positions (void assignment)
    return bitstring, bitstring_shrunk, pos_atom, void_positions


# Set GPU/CPU
TORCH_DEVICE = 'cpu'
TORCH_DTYPE = torch.float32
Gumbel_sinkhorn = False  # False to consider both position and stoichiometry constraints when constructing Q



def Greedy_Baseline(file_to_parse, num_steps, num_seeds):
    struct_info = file_to_parse.split('G')
    struct_name = struct_info[0] #
    g = int(struct_info[1][0])  # cell discretization
    structure_with_g = file_to_parse.split('.')[0]
    bqm_model = BQM(Potts = True, Gumbel_sinkhorn = Gumbel_sinkhorn)
    bqm_model.parse_lp("instances/" + file_to_parse)
    leq_weight = 100  # penalty weight for atom or void per position constraints for Potts model
    eq_weight = 100  # penalty weight for stoichiometry constraints for Potts model
    with_void = True  # include void for each position
    Q, num_positions, atoms, stoich_const = qubo_to_torch(
        bqm_model, eq_inf=eq_weight, leq_infinity=leq_weight,
        with_void = with_void,  torch_dtype=TORCH_DTYPE, Gumbel_sinkhorn=Gumbel_sinkhorn,
        torch_device=TORCH_DEVICE)  # pre-constrained Q matrix

    n_atoms = len(atoms)  # number of atoms
    num_classes = (n_atoms + 1) if with_void else n_atoms  # number of node classes
    n_variables = num_positions*num_classes
    print("Crystal Structure Prediction of " + file_to_parse.split('.')[0] + " with " + str(num_positions)
          + " positions" + " and " + str(atoms) + " atoms.")
    print(n_variables, "Variables in the MultiClass Classification Problem")
    best_loss = torch.zeros(num_seeds)
    relative_optimality_gap = np.zeros(num_seeds)
    min_energy_found = 0
    run_time = np.zeros((num_seeds, 1))
    ev_atom = np.zeros((num_seeds, 1))
    print("---Performing Greedy Baseline---")
    for seed in range(num_seeds):
        set_seed(seed)
        time_start = time()
        bitstring, bitstring_shrunk, pos_atom, void_positions = initialize_bitstring(num_positions, n_atoms, stoich_const, seed, TORCH_DEVICE, TORCH_DTYPE)
        best_bitstring = bitstring
        loss = []
        loss.append(loss_func(Q, bitstring, bqm_model.offset))
        best_loss[seed] = loss[-1]
        for step in range(num_steps):
            pos_to_empty = random.sample(range(0, len(pos_atom)), 1) # select position to empty
            pos_to_fill = random.sample(range(0, len(void_positions)), 1) # select empty position to allocate an atom
            atom_type = int(bitstring_shrunk[pos_atom[pos_to_empty[0]]]) # atom type that changes position
            bitstring[pos_atom[pos_to_empty[0]], atom_type] = 0
            bitstring[pos_atom[pos_to_empty[0]], -1] = 1
            bitstring[void_positions[pos_to_fill[0]], atom_type] = 1
            bitstring[void_positions[pos_to_fill[0]], -1] = 0
            if loss_func(Q, bitstring, bqm_model.offset) < best_loss[seed]: # update the best solution
                bitstring_shrunk[pos_atom[pos_to_empty[0]]] = n_atoms
                bitstring_shrunk[void_positions[pos_to_fill[0]]] = atom_type
                best_loss[seed] = loss_func(Q, bitstring, bqm_model.offset)
                loss.append(best_loss[seed])
                best_bitstring = bitstring
                change = pos_atom[pos_to_empty[0]]
                pos_atom[pos_to_empty[0]] = void_positions[pos_to_fill[0]]
                void_positions[pos_to_fill[0]] = change
            else: # reverse update to stay with the previous best solution
                bitstring[pos_atom[pos_to_empty[0]], atom_type] = 1
                bitstring[pos_atom[pos_to_empty[0]], -1] = 0
                bitstring[void_positions[pos_to_fill[0]], atom_type] = 0
                bitstring[void_positions[pos_to_fill[0]], -1] = 1
                loss.append(best_loss[seed])
        run_time[seed] = time() - time_start
        relative_optimality_gap[seed] = \
            (bqm_model.minimum_energy - best_loss[seed])/bqm_model.minimum_energy
        if np.round(best_loss[seed], 3) == bqm_model.minimum_energy:
            min_energy_found += 1
        ev_atom[seed] = best_loss[seed]/stoich_const[:-1].sum()
    print('Number of different seeds is: ' + str(num_seeds))
    print('Number of steps for each seed is: ' + str(num_steps))
    print('Ground state energy = ' + str(bqm_model.minimum_energy))
    print('Ground truth eV/atom = ' + str(bqm_model.minimum_energy/stoich_const[:-1].sum().item()))
    print('Average Best loss = ' + str(torch.mean(best_loss).item()) + ' , Standard deviation = ' + str(torch.std(best_loss).item()))
    print('Best Loss = ' + str(torch.min(best_loss).item()))
    print('Average Relative Optimality gap = ' + str(relative_optimality_gap.mean()) + ' , Standard deviation = ' + str(relative_optimality_gap.std()))
    print('Best Relative Optimality gap = ' + str(relative_optimality_gap.min()))
    print('Hit rate = ' + str((min_energy_found/num_seeds)*100) + '%')
    print('Minimum eV/atom = ' + str(min(ev_atom)))
    print('Average eV/atom = ' + str(ev_atom.mean()) + ' , Standard deviation = ' + str(ev_atom.std()))
    print('Average Time = ' + str(np.mean(run_time)) + ' , Standard deviation = ' + str(np.std(run_time)))


def Simulated_Annealing(file_to_parse, temp_init, num_steps, num_seeds):
    struct_info = file_to_parse.split('G')
    struct_name = struct_info[0] #
    g = int(struct_info[1][0])  # cell discretization
    structure_with_g = file_to_parse.split('.')[0]
    bqm_model = BQM(Potts = True, Gumbel_sinkhorn = Gumbel_sinkhorn)
    bqm_model.parse_lp("instances/" + file_to_parse)
    leq_weight = 100  # penalty weight for atom or void per position constraints for Potts model
    eq_weight = 100  # penalty weight for stoichiometry constraints for Potts model
    with_void = True  # include void for each position
    Q, num_positions, atoms, stoich_const = qubo_to_torch(
        bqm_model, eq_inf=eq_weight, leq_infinity=leq_weight,
        with_void = with_void,  torch_dtype=TORCH_DTYPE, Gumbel_sinkhorn=Gumbel_sinkhorn,
        torch_device=TORCH_DEVICE)  # pre-constrained Q matrix

    n_atoms = len(atoms)  # number of atoms
    num_classes = (n_atoms + 1) if with_void else n_atoms  # number of node classes
    n_variables = num_positions*num_classes
    print("Crystal Structure Prediction of " + file_to_parse.split('.')[0] + " with " + str(num_positions)
          + " positions" + " and " + str(atoms) + " atoms.")
    print(n_variables, "Variables in the MultiClass Classification Problem")
    # current_loss = torch.zeros(num_seeds)
    relative_optimality_gap = np.zeros(num_seeds)
    min_energy_found = 0
    print("---Performing Simulated Annealing---")
    run_time = np.zeros((num_seeds, 1))
    ev_atom = np.zeros((num_seeds, 1))
    best_loss = np.zeros((num_seeds, 1))
    for seed in range(num_seeds):
        temp = temp_init
        set_seed(seed)
        time_start = time()
        bitstring, bitstring_shrunk, pos_atom, void_positions = initialize_bitstring(num_positions, n_atoms, stoich_const, seed, TORCH_DEVICE, TORCH_DTYPE)
        best_bitstring = bitstring
        loss = []
        loss.append(loss_func(Q, bitstring, bqm_model.offset))
        current_loss = loss[-1]
        for step in range(num_steps):
            pos_to_empty = random.sample(range(0, len(pos_atom)), 1) # select position to empty
            pos_to_fill = random.sample(range(0, len(void_positions)), 1) # select empty position to allocate an atom
            atom_type = int(bitstring_shrunk[pos_atom[pos_to_empty[0]]]) # atom type that changes position
            bitstring[pos_atom[pos_to_empty[0]], atom_type] = 0
            bitstring[pos_atom[pos_to_empty[0]], -1] = 1
            bitstring[void_positions[pos_to_fill[0]], atom_type] = 1
            bitstring[void_positions[pos_to_fill[0]], -1] = 0
            new_loss = loss_func(Q, bitstring, bqm_model.offset)
            temp *= 0.9999
            if torch.exp((new_loss-current_loss)/temp)<0.5:
                print(new_loss, current_loss, temp, step)
            if new_loss < current_loss or random.uniform(0, 1)<torch.exp((new_loss-current_loss)/temp): # update the solution
                if new_loss < current_loss:
                    best_bitstring = bitstring
                bitstring_shrunk[pos_atom[pos_to_empty[0]]] = n_atoms
                bitstring_shrunk[void_positions[pos_to_fill[0]]] = atom_type
                current_loss = new_loss
                loss.append(current_loss)
                change = pos_atom[pos_to_empty[0]]
                pos_atom[pos_to_empty[0]] = void_positions[pos_to_fill[0]]
                void_positions[pos_to_fill[0]] = change
            else: # reverse updates to stay with the previous(best) solution
                bitstring[pos_atom[pos_to_empty[0]], atom_type] = 1
                bitstring[pos_atom[pos_to_empty[0]], -1] = 0
                bitstring[void_positions[pos_to_fill[0]], atom_type] = 0
                bitstring[void_positions[pos_to_fill[0]], -1] = 1
                loss.append(current_loss)
        best_loss[seed] = min(loss).item()
        run_time[seed] = time() - time_start
        relative_optimality_gap[seed] = \
            (bqm_model.minimum_energy - best_loss[seed])/bqm_model.minimum_energy
        if np.round(best_loss[seed], 3) == bqm_model.minimum_energy:
            min_energy_found += 1
        ev_atom[seed] = best_loss[seed]/stoich_const[:-1].sum()
    print('Number of different seeds is: ' + str(num_seeds))
    print('Number of steps for each seed is: ' + str(num_steps))
    print('Ground state energy = ' + str(bqm_model.minimum_energy))
    print('Ground truth eV/atom = ' + str(bqm_model.minimum_energy/stoich_const[:-1].sum().item()))
    print('Average Best loss = ' + str(np.mean(best_loss)) + ' , Standard deviation = ' + str(np.std(best_loss)))
    print('Best Loss = ' + str(np.min(best_loss)))
    print('Average Relative Optimality gap = ' + str(relative_optimality_gap.mean()) + ' , Standard deviation = ' + str(relative_optimality_gap.std()))
    print('Best Relative Optimality gap = ' + str(relative_optimality_gap.min()))
    print('Hit rate = ' + str((min_energy_found/num_seeds)*100) + '%')
    print('Minimum eV/atom = ' + str(np.min(ev_atom)))
    print('Average eV/atom = ' + str(ev_atom.mean()) + ' , Standard deviation = ' + str(ev_atom.std()))
    print('Average Time = ' + str(np.mean(run_time)) + ' , Standard deviation = ' + str(np.std(run_time)))

# Greedy_Baseline(file_to_parse='SrTiO3G4.lp', num_steps=10000, num_seeds=10)
print("---- ---- ---- ----")
Simulated_Annealing(file_to_parse='SrTiO3G4.lp', temp_init=1e6, num_steps=100000, num_seeds=5)