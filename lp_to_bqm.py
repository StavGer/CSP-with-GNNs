import re
import torch

import multiprocessing
import torch
from scipy.sparse import coo_matrix

try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2  # arbitrary default
DEBUG = False


class BQM():

    p_ion = re.compile(r'(?P<specie>\S+)_(?P<pos>\d+)')
    p_rhs = re.compile(r'(=|<=)\s*(?P<int>\d+)')

    def __init__(self):
        self.linear = {}
        self.quadratic = {}
        self.offset = 0
        self.variables = []
        self.constraints_str = []

        # constraints are kept as dict with the keys:
        # 'type' is LEQ or EQ
        # 'lhs' [(coefficient, (specie, pos))]
        #  'rhs' integer
        self.constraints_dict = []

    def parse_lp(self, filename, mult=0.5):
        """
        mult deals with the /2 in the energy representation
        """

        ''' Old version that can't handle the engineering notation
        p_square = re.compile(r'(?P<coeff>[\+-]?\s*\d*\.?\d+)\s*(?P<var>\S+_\d+)\s*\^2')
        p_product = re.compile(r'(?P<coeff>[\+-]?\s*\d*\.?\d+)\s*(?P<var_1>\S+_\d+)\s*\*\s*(?P<var_2>\S+_\d+)')
        # p_ion = re.compile(r'(?P<specie>\S+)_(?P<pos>\d+)')
        '''
        p_square = re.compile(r'(?P<coeff>[-+]?\s*(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*(?P<var>\S+_\d+)\s*\^2')
        p_product = re.compile(r'(?P<coeff>[-+]?\s*(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*(?P<var_1>\S+_\d+)\s*\*\s*('
                               r'?P<var_2>\S+_\d+)')
        with open(filename) as f:

            line = ""
            while not line.startswith("\ Model Ion"):
                line = f.readline()
            line = f.readline()
            # Skipping the beginning till the object function
            while not line.startswith("Minimize"):
                line = f.readline()

            # Parse till we hit the constraints
            line = f.readline()
            while not line.startswith("Subject To"):
                # squares = p_square.findall(line)
                squares = [m.groupdict() for m in p_square.finditer(line)]
                if squares:
                    # print(squares)
                    for match in squares:
                        energy, name = match['coeff'], match['var']
                        # m = BQM.p_ion.search(name)
                        # k = (m.group('specie'), int(m.group('pos')))
                        k = BQM.specie_pos(name)

                        if k in self.linear:
                            print(f"Another quadratic term for {name} was encountered")
                        else:
                            self.linear[k] = float(energy.replace(' ', ''))
                            # print(k, float(energy.replace(' ', '')))
                            if name not in self.variables:
                                self.variables.append(name)

                # products = p_product.findall(line)
                products = [m.groupdict() for m in p_product.finditer(line)]
                if products:
                    # print(products)
                    for match in products:
                        # energy, name_1, name_2 = match
                        energy, name_1, name_2 = match['coeff'], match['var_1'], match['var_2']
                        # m1 = BQM.p_ion.search(name_1)
                        # m2 = BQM.p_ion.search(name_2)
                        #


                        t1 = BQM.specie_pos(name_1)
                        t2 = BQM.specie_pos(name_2)

                        k = BQM.order(t1, t2)


                        if k in self.quadratic:
                            print(f"Another quadratic term for {name_1} and {name_2} was encountered")
                        else:
                            self.quadratic[k] = float(energy.replace(' ', ''))
                            # print(k, float(energy.replace(' ', '')))
                            if name_1 not in self.variables:
                                self.variables.append(name_1)
                            if name_2 not in self.variables:
                                self.variables.append(name_2)

                line = f.readline()

            # Parse the constraints till we hit bounds
            # Since the constraints are can be multiline,
            # we glue the lines on the go and then process them
            # we assume that all variables are binary afterwards
            line = f.readline()  # it either has first constraint or "Bounds"
            line_prev = line.strip()
            while not line.startswith("Bounds"):

                if ':' in line:
                    self.constraints_str.append(line_prev)
                    line_prev = line.strip()
                else:
                    line_prev += line.strip()

                line = f.readline()
            if not line_prev.startswith("Bounds"):
                self.constraints_str.append(line_prev)

            if len(self.constraints_str) > 1:
                self.constraints_str = self.constraints_str[1:]


        if mult != 1:
            for k in self.quadratic:
                self.quadratic[k] = self.quadratic[k] * mult
            for k in self.linear:
                self.linear[k] = self.linear[k] * mult
    def parse_constraints(self):
        for constraint in self.constraints_str:
            dict_con = {}
            if '<=' in constraint:
                dict_con['type'] = 'LEQ'
            elif '=' in constraint:
                dict_con['type'] = 'EQ'
            else:
                print(f"Skipping constraint of unknown type {constraint}")

            m = BQM.p_rhs.search(constraint)
            dict_con['rhs'] = int(m.group('int'))

            # Getting the variables with coefficients
            dict_con['lhs'] = []
            for var in self.variables:

                p_var = re.compile(r'(?P<coeff>\d+)?\s*' + var + r'\D')
                m = p_var.search(constraint)

                if m:
                    if m.group('coeff') is not None:
                        dict_con['lhs'].append((int(m.group('coeff')), BQM.specie_pos(var)))
                    else:
                        dict_con['lhs'].append((1, BQM.specie_pos(var)))

            self.constraints_dict.append(dict_con)

    def max_bound(self, max=None):
        """
        bound the positive coefficients, that tend to
        become very large for high symmetry structures
        """

        # print(f"Before {self.linear}")
        # print(f"Before {self.quadratic}")
        if max is None:
            lowest = min([v for k, v in self.linear.items()])
            lowest = min(lowest, min([v for k, v in self.quadratic.items()]))

            # max = -3*lowest
            max = -lowest

        for k, v in self.linear.items():
            self.linear[k] = min(self.linear[k], max)

        for k, v in self.quadratic.items():
            self.quadratic[k] = min(self.quadratic[k], max)

        # print(f"After {self.linear}")
        # print(f"After {self.quadratic}")
        print(f"The maximum coefficient was set to {max}")

    def qubofy(self, eq_inf, leq_infinity):
        """
        The procedure to get rid of constraints
        Not the general approach,
        here only <= 1 and = constraints are assumed!

        Penalties for breaking constraints
        """
        if DEBUG:
            print(len(self.linear) + len(self.quadratic), " terms")
            print(len(self.constraints_dict), " constraints")
            print(self.constraints_dict)

        for dict_con in self.constraints_dict:
            if dict_con['type'] == 'EQ':
                if DEBUG:
                    print("=====================", dict_con)
                N = len(dict_con['lhs'])
                vars = dict_con['lhs']
                # Terms involving the rhs constant
                self.offset += eq_inf * dict_con['rhs'] ** 2
                if DEBUG:
                    print(f"Offset is increased by {eq_inf} * {dict_con['rhs']} ** 2")
                for i in range(N):
                    if vars[i][1] in self.linear:
                        if DEBUG:
                            print(vars[i][1],
                                  f"added - 2 * {eq_inf} * {vars[i][0]} * {dict_con['rhs']} + {eq_inf} * {vars[i][0]} ** 2", "to",
                                  self.linear[vars[i][1]])
                        self.linear[vars[i][1]] = self.linear[vars[i][1]] - 2 * eq_inf * vars[i][0] * dict_con['rhs'] \
                                                  + eq_inf * vars[i][0] ** 2
                        if self.C2.get(vars[i][1]):
                            self.C2[vars[i][1]] = self.C2[vars[i][1]] - 2 * eq_inf * vars[i][0] * dict_con['rhs'] + eq_inf * vars[i][0] ** 2
                        else:
                            self.C2[vars[i][1]] = - 2 * eq_inf * vars[i][0] * dict_con['rhs'] + eq_inf * vars[i][0] ** 2
                    else:

                        self.linear[vars[i][1]] = -2 * eq_inf * vars[i][0] * dict_con['rhs'] + eq_inf * vars[i][0] ** 2
                        self.C2[vars[i][1]] = -2 * eq_inf * vars[i][0] * dict_con['rhs'] + eq_inf * vars[i][0] ** 2
                        if DEBUG:
                            print(vars[i][1], f"added -2 * {eq_inf} * {vars[i][0]} * {dict_con['rhs']} + {eq_inf} * {vars[i][0]} ** 2")
                            print("Adding a new linear term for placement", vars[i][1])

                # Terms involving pairwise products
                for i in range(N):
                    for j in range(i + 1, N):
                        pair = BQM.order(vars[i][1], vars[j][1])

                        if pair in self.quadratic:
                            if DEBUG:
                                print(pair, f"added 2 * {eq_inf} * {vars[i][0]} * {vars[j][0]}", "to", self.quadratic[pair])
                            self.quadratic[pair] = self.quadratic[pair] + 2 * eq_inf * vars[i][0] * vars[j][0]
                            self.C2[pair] = 2 * eq_inf * vars[i][0] * vars[j][0]
                        else:
                            self.quadratic[pair] = 2 * eq_inf * vars[i][0] * vars[j][0]
                            self.C2[pair] = 2 * eq_inf * vars[i][0] * vars[j][0]
                            if DEBUG:
                                print(pair, f"added 2 * {eq_inf} * {vars[i][0]} * {vars[j][0]}")
                                print("Adding new quadratic term for placement", pair)

            else:
                print("Encountered constraint of unknown type")

    @staticmethod
    def specie_pos(name):
        m = BQM.p_ion.search(name)
        return m.group('specie'), int(m.group('pos'))

    @staticmethod
    def order(t1, t2):
        '''
        tuples are of the form ('O', 2)
        order is based on position first, where lower is better
        and then on the species

        Return t1, t2 in the correct order
        '''

        swap = False

        if t1[1] == t2[1]:
            if t1[0] > t2[0]:
                swap = True

        if t1[1] > t2[1]:
            swap = True

        if swap:
            return t2, t1
        else:
            return t1, t2


def qubo_to_torch(bqm_model, with_void, torch_dtype=None, torch_device=None):
    """
    Output Q matrix as torch tensor for given Q in dictionary format.
    """
    elements = list(bqm_model.linear.keys())
    n_variables = len(elements)  # without void

    atoms = [element[0] for element in elements if element[1] == 0]
    n_atoms = len(atoms)

    num_positions = int(n_variables / n_atoms)
    n_total_variables = int((n_atoms + 1) * num_positions)  # with void

    bqm_model.parse_constraints()

    # Create sparse matrix for Q
    row, col, data = [], [], []
    linear_map = {element: idx for idx, element in enumerate(elements)}
    for k, v in bqm_model.linear.items():
        idx = linear_map[k]
        row.append(idx)
        col.append(idx)
        data.append(v)
    for (k0, k1), v in bqm_model.quadratic.items():
        idx0 = linear_map[k0]
        idx1 = linear_map[k1]
        row.append(idx0)
        col.append(idx1)
        data.append(v)

    Q = coo_matrix((data, (row, col)), shape=(n_variables, n_variables))
    eq_const = [bqm_model.constraints_dict[i]['rhs'] for i in range(1, len(bqm_model.constraints_dict)) if bqm_model.constraints_dict[i]['type'] == 'EQ']
    
    if with_void:
        row_offsets = Q.row//n_atoms
        col_offsets = Q.col//n_atoms
        new_row = Q.row + row_offsets
        new_col = Q.col + col_offsets
        Q = coo_matrix((Q.data, (new_row, new_col)), shape=(n_total_variables, n_total_variables))
        eq_const.append(num_positions-sum(eq_const))

    Q = torch.tensor(Q.toarray(), dtype=torch_dtype, device=torch_device)
    stoich_const = torch.tensor(eq_const, dtype=torch_dtype, device=torch_device)
    return Q, num_positions, atoms, stoich_const, bqm_model.offset


