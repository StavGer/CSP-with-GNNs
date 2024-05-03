"""
A routine to convert quadratic binary programs with linear constraints arising
during CSP to to Quadratic Unconstraint Binary Optimisation problems
suitable for Quantum Annealers

IP Model is represented in lp format
and implementation is CSP specific
"""

import re
import torch, numpy
import itertools

DEBUG = False


class BQM():

    p_ion = re.compile(r'(?P<specie>\S+)_(?P<pos>\d+)')
    p_rhs = re.compile(r'(=|<=)\s*(?P<int>\d+)')

    def __init__(self, Potts : bool, Gumbel_sinkhorn: bool):
        self.linear = {}
        self.quadratic = {}
        self.offset = 0
        self.minimum_energy = 0.0
        self.variables = []
        self.constraints_str = []
        self.C1 = {}
        self.C2 = {}
        self.Potts = Potts
        self.Gumber_sinkhorn = Gumbel_sinkhorn

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
        p_product = re.compile(r'(?P<coeff>[-+]?\s*(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*(?P<var_1>\S+_\d+)\s*\*\s*(?P<var_2>\S+_\d+)')
        with open(filename) as f:
            line = ""
            while not line.startswith("\ Model Ion"):
                line = f.readline()
            line = f.readline()
            self.minimum_energy = float(line.split()[-1])
            # Skipping the beginning till the object function
            while not line.startswith("Minimize"):
                line = f.readline()

            # Parse till we hit the constraints
            line = f.readline()
            while not line.startswith("Subject To"):
                # squares = p_square.findall(line)
                squares = [m.groupdict() for m in p_square.finditer(line)]
                if squares:
                    for match in squares:
                        energy, name = match['coeff'], match['var']
                        k = BQM.specie_pos(name)

                        if k in self.linear:
                            print(f"Another quadratic term for {name} was encountered")
                        else:
                            self.linear[k] = float(energy.replace(' ', ''))
                            # print(k, float(energy.replace(' ', '')))
                            if name not in self.variables:
                                self.variables.append(name)

                products = [m.groupdict() for m in p_product.finditer(line)]
                if products:
                    for match in products:
                        energy, name_1, name_2 = match['coeff'], match['var_1'], match['var_2']

                        t1 = BQM.specie_pos(name_1)
                        t2 = BQM.specie_pos(name_2)

                        k = BQM.order(t1, t2)

                        if k in self.quadratic:
                            print(f"Another quadratic term for {name_1} and {name_2} was encountered")
                        else:
                            self.quadratic[k] = float(energy.replace(' ', ''))
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

        for k in self.quadratic:
            self.quadratic[k] *= mult
        for k in self.linear:
            self.linear[k] *= mult

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

    def qubofy(self, eq_inf, leq_inf):
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

        # TODO - leaving subselection option in, in case we want to revert
        for dict_con in self.constraints_dict[:]:
            # Potts model will place exactly one element per position, so these LEQ
            # constraints, which limit the placement at each position to at most 1, are
            # not useful for the Potts model.
            if dict_con['type'] == 'LEQ':
                if not self.Potts:
                    if DEBUG:
                        print("=====================", dict_con)

                    # I know, you'd forget
                    if dict_con['rhs'] != 1:
                        print("Encountered unsupported constraint! Treating RHS as 1")

                    N = len(dict_con['lhs'])
                    for i in range(N):
                        for j in range(i + 1, N):
                            pair = BQM.order(dict_con['lhs'][i][1], dict_con['lhs'][j][1])
                            if pair in self.quadratic:
                                if DEBUG:
                                    print(pair, "added", leq_inf, "to", self.quadratic[pair])
                                self.quadratic[pair] += leq_inf
                                self.C1[pair] = leq_inf
                            else:
                                self.quadratic[pair] = leq_inf
                                self.C1[pair] = leq_inf
                                if DEBUG:
                                    print("Adding new quadratic term for orbits", pair)
                                    print(pair, leq_inf)

            elif dict_con['type'] == 'EQ':
                """
                Equality constraints take the form sum(x^t_i) = k
                where x^t_i represents an atom of species t at position i, and k is
                some fixed integer, i.e. k=3 for Oxygen
                These constraints are added into the problem Hamiltonian as positive
                energy terms of the form [sum(x^t_i) - k]^2 .
                In order to map these to contributions in the Q matrix, we need to
                expand this equation and identify which Q entries relate to which
                x^t_i value. For instance, in the above, we would expand to
                sum(x^t_i)^2 - 2*k*sum(x^t_i) + k^2
                we need to expand the sums: sum(x^t_i) = (x^t_0 + x^t_1 + ... )
                -> (x^t_0)^2 - 2*x^t_0*k + 2*x^t_0*x^t_1 + (x^t_1)^2 - 2*x^t_1*k + ... + k^2
                We can group like terms, and assume aribtrary prefactors on x of form
                a_i, and we see
                
                (x^t_0)^2*(a_0^2 - 2*a_0*k) + ... + (x^t_i)^2*(a_i^2 - 2*a_i*k) + 2*a_0*a_1*x^t_0*x^t_1 + ... + 2*a_i*a_j*x^t_i*x^t_j + k^2
                
                we can group then add these contributions to the appropriate linear terms
                (x^t_i)^2*(a_i^2 - 2*a_i*k)
                quadratic terms
                2*a_i*a_j*x^t_i*x^t_j
                and offset k^2
                
                And we can scale them with a penalty term `eq_inf`.
                """
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
                            print(vars[i][1], f" added {eq_inf} * ({vars[i][0]}**2 - 2*{vars[i][0]}*{dict_con['rhs']}) ", "to ", self.linear[vars[i][1]])
                        # Add linear terms of form (x^t_i)^2*(a_i^2 - 2k)
                        self.linear[vars[i][1]] += eq_inf * (vars[i][0]**2 - 2*vars[i][0]*dict_con['rhs'])
                        if self.C2.get(vars[i][1]):
                            self.C2[vars[i][1]] += eq_inf * (vars[i][0]**2 - 2*vars[i][0]*dict_con['rhs'])
                        else:
                            self.C2[vars[i][1]] = eq_inf * (vars[i][0]**2 - 2*vars[i][0]*dict_con['rhs'])
                    else:
                        self.linear[vars[i][1]] = eq_inf * (vars[i][0]**2 - 2*vars[i][0]*dict_con['rhs'])
                        self.C2[vars[i][1]] = eq_inf * (vars[i][0]**2 + 2*vars[i][0]*dict_con['rhs'])
                        if DEBUG:
                            print(vars[i][1], f" added {eq_inf} * ({vars[i][0]}**2 - 2*{vars[i][0]}*{dict_con['rhs']})")
                            print("Adding a new linear term for placement", vars[i][1])

                # Terms involving pairwise products
                for i in range(N):
                    for j in range(i + 1, N):
                        pair = BQM.order(vars[i][1], vars[j][1])

                        if pair in self.quadratic:
                            if DEBUG:
                                print(pair, f" added {eq_inf} * 2*{vars[i][0]}*{vars[j][0]} ", "to ", self.quadratic[pair])
                            # Adding quadratic terms of form 2*a_i*a_j*x^t_i*x^t_j
                            self.quadratic[pair] += eq_inf * 2*vars[i][0]*vars[j][0]
                            self.C2[pair] = eq_inf * 2*vars[i][0]*vars[j][0]
                        else:
                            self.quadratic[pair] = eq_inf * 2*vars[i][0]*vars[j][0]
                            self.C2[pair] = eq_inf * 2*vars[i][0]*vars[j][0]
                            if DEBUG:
                                print(pair, f" added {eq_inf} * 2*{vars[i][0]}*{vars[j][0]}")
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


def qubo_to_torch(bqm_model, eq_inf, leq_inf, with_void=False, Gumbel_sinkhorn=False, torch_dtype=None, torch_device=None):
    """
    Output Q matrix as torch tensor for given Q in dictionary format.
    """

    # update quadratic dict with linear dict entries
    # bqm_model.quadratic.update(bqm_model.linear)

    elements = list(bqm_model.linear.keys())
    n_variables = len(elements)  # without void

    atoms = [element[0] for element in elements if element[1]==0]
    n_atoms = len(atoms)

    num_positions = int(n_variables / n_atoms)
    n_total_variables = int((n_atoms + 1) * num_positions)  # with void

    bqm_model.parse_constraints()
    if not Gumbel_sinkhorn:
        bqm_model.qubofy(eq_inf, leq_inf)
        # bqm_model.quadratic.update(bqm_model.linear)

    # get number of nodes
    elements = list(bqm_model.linear.keys())
    n_variables = len(elements)
    Q = torch.zeros((n_variables, n_variables))

    for k in elements:
        # Retrieve list index of value k, i.e. k = (0, 1)
        idx = elements.index(k)
        Q[idx, idx] = bqm_model.linear[k]

    for k in bqm_model.quadratic.keys():
        # Retrieve list index of each value k[0], k[1] in pair k,
        # i.e. k = ((0, 1), (Sr, 2))
        idx0 = elements.index(k[0])
        idx1 = elements.index(k[1])
        Q[idx0, idx1] = bqm_model.quadratic[k]

    eq_const = []  # values for stoichiometry constraints
    for i in range(1, len(bqm_model.constraints_dict)):
        if bqm_model.constraints_dict[i]['type'] == 'EQ':
            eq_const.append(bqm_model.constraints_dict[i]['rhs'])
    stoich_const = torch.tensor(eq_const, dtype=Q.dtype, device=torch_device)

    if with_void:
        Q_with_void = [[0]*n_total_variables]*n_total_variables

        # TODO - better way to do this?
        void_idx = range(n_atoms, n_total_variables, n_atoms+1)
        c = 0
        for i in range(n_total_variables):
            if i in void_idx:
                # row corresponding to void
                c += 1
            else:
                # Insert row from Q into Q_with_void, adjusting for void rows every
                # `n_atoms` row
                Q_with_void[i] = [Q[i-c, j] for j in range(n_variables)]
                # Expand row i by inserting 0 every j elements (0 for void)
                [Q_with_void[i].insert(j, 0) for j in void_idx] # add 0s to columns for void

        # number of positions assigned to void
        n_void = num_positions - sum(eq_const)
        eq_const.append(n_void) if num_positions - sum(eq_const) else eq_const.append(1e-10)
        stoich_const = torch.tensor(eq_const, dtype=Q.dtype, device=torch_device)

        Q_with_void = torch.tensor(Q_with_void)

        Q_ret = Q_with_void
    else:
        Q_ret = Q

    if torch_dtype is not None:
        Q_ret = Q_ret.type(torch_dtype)
    if torch_device is not None:
        Q_ret = Q_ret.to(torch_device)

    return Q_ret, num_positions, atoms, stoich_const
