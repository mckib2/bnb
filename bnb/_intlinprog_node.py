'''Search Node class for intlinprog branch and bound solver.'''

from copy import deepcopy

import numpy as np
from scipy.optimize import linprog

class _Node:
    '''Encapsulate an LP in the search.'''
    def __init__(self, ilp, parent_cost=-1*np.inf, depth=None, parent_id=None):
        self.id = self.take_num()
        self.depth = depth # tree depth
        self.parent_id = parent_id
        self.ilp = deepcopy(ilp)
        self.parent_cost = parent_cost # used for best-first search

        # Populated by self.solve():
        self.feasible = None
        self.z = None
        self.x = None
        self.lp_res = None # reference to the LP solution

    def solve(self):
        '''Solve the LP relaxation.'''
        try:
            # method = 'interior-point'
            method = 'revised simplex'
            lp_res = linprog(
                -1*self.ilp.c,
                self.ilp.A_ub, self.ilp.b_ub,
                self.ilp.A_eq, self.ilp.b_eq,
                bounds=self.ilp.bounds, method=method,
                options=self.lp_solver_options)
        except ValueError as _e:
            # This is an infeasibility exception -- assume infeasible
            # print(e)
            lp_res = {}
            lp_res['success'] = False

        self.lp_res = lp_res
        if not lp_res['success']:
            self.feasible = False
        else:
            self.feasible = True
            self.z = -1*lp_res['fun']
            self.x = lp_res['x']

    def change_upper_bound(self, idx, upper):
        '''Constrain a single variable to be below a value.'''
        prev = self.ilp.bounds[idx]

        # If we are setting bounds to be equal, then we actually
        # need a constraint
        if prev[0] == upper:
            self.add_eq_constraint(idx, upper)
            self.ilp.bounds[idx] = (0, None)
        else:
            self.ilp.bounds[idx] = (prev[0], upper)

    def change_lower_bound(self, idx, lower):
        '''Constrain a single variable to be above a value.'''
        prev = self.ilp.bounds[idx]

        # If we are setting bounds to be equal, then we actually
        # need a constraint
        if prev[0] == lower:
            self.add_eq_constraint(idx, lower)
            self.ilp.bounds[idx] = (0, None)
        else:
            self.ilp.bounds[idx] = (lower, prev[1])

    def add_eq_constraint(self, idx, val):
        '''Add a row to A_eq and b_eq.'''

        # Hacky solution relying on provate method of namedtuple:
        if self.ilp.A_eq is None:
            self.ilp = self.ilp._replace(
                A_eq=np.zeros((0, self.ilp.c.size)),
                b_eq=np.zeros(0))
        A_eq_new = np.zeros((1, self.ilp.A_eq.shape[1]))
        A_eq_new[:, idx] = 1
        self.ilp = self.ilp._replace(
            A_eq=np.concatenate((self.ilp.A_eq, A_eq_new)),
            b_eq=np.concatenate((self.ilp.b_eq, [val])))

    def __lt__(self, other):
        '''Node comparison for best-first search strategy.'''
        return self.parent_cost < other.parent_cost

    def __repr__(self):
        return(
            'node with id: {id}\n'
            '\tfeasible: {feasible}\n'
            '\t  bounds: {bounds}\n'
            '\t       z: {z}\n'
            '\t       x: {x}').format(
                id=self.id, feasible=self.feasible,
                bounds=self.ilp.bounds, z=self.z, x=str(self.x))

    # Class attributes and methods
    lp_solver_options = {}
    @classmethod
    def global_set_lp_solver_options(cls, lp_solver_options):
        '''Solver options will never change for associated LP.'''
        cls.lp_solver_options = lp_solver_options
    _node_ctr = 0
    @classmethod
    def take_num(cls):
        '''Return a number unique to the node instance.  This is used
        for keeping track of the cost of leaf nodes when updating
        zbar (global upper bound).'''
        cls._node_ctr += 1
        return cls._node_ctr
