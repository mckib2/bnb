'''Interface for mixed integer linear programs.'''

import logging
from queue import LifoQueue, Queue, PriorityQueue
from time import time
from collections import namedtuple
from copy import deepcopy

import numpy as np
from scipy.optimize import linprog

_ILPProblem = namedtuple(
    '_ILPProblem', 'c A_ub b_ub A_eq b_eq binary real_valued bounds')

class _Node:
    '''Encapsulate an LP in the search.'''
    def __init__(self, ilp, parent_cost=-1*np.inf):
        self.id = self.take_num()
        self.ilp = deepcopy(ilp)
        self.parent_cost = parent_cost # used for best-first search

        # Populated by self.solve():
        self.feasible = None
        self.z = None
        self.x = None
        self.lp_res = None # reference to the LP solution

    def solve(self):
        '''Solve the LP relaxation.'''
        lp_res = linprog(
            -1*self.ilp.c,
            self.ilp.A_ub, self.ilp.b_ub,
            self.ilp.A_eq, self.ilp.b_eq,
            bounds=self.ilp.bounds, method='revised simplex')

        if not lp_res['success']:
            self.feasible = False
        else:
            self.feasible = True
            self.z = -1*lp_res['fun']
            self.x = lp_res['x']
            self.lp_res = lp_res

    def change_upper_bound(self, idx, upper):
        '''Constrain a single variable to be below a value.'''
        prev = self.ilp.bounds[idx]
        self.ilp.bounds[idx] = (prev[0], upper)

    def change_lower_bound(self, idx, lower):
        '''Constrain a single variable to be above a value.'''
        prev = self.ilp.bounds[idx]
        self.ilp.bounds[idx] = (lower, prev[1])

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

    # Class attributes
    _node_ctr = 0
    @classmethod
    def take_num(cls):
        '''Return a number that should be unique to the node instance.
        '''
        cls._node_ctr += 1
        return cls._node_ctr

def  _process_intlinprog_args(
        c, A_ub, b_ub, A_eq, b_eq, binary, real_valued, bounds):
    '''Sanitize input to intlinprog.'''

    # Deal with coefficients:
    c = np.array(c).flatten()

    # Deal with constraints:
    if A_ub is not None:
        A_ub = np.array(A_ub)
        assert A_ub.ndim == 2, 'Inequality constraint matrix must be 2D!'
        assert A_ub.shape[1] == c.size, ( # pylint: disable=E1136
            'Inequality constraint matrix must match size of c!')
    if b_ub is not None:
        b_ub = np.array(b_ub).flatten()
        assert b_ub.size == A_ub.shape[0], (
            'Inequality constraint vector must have match size of matrix!')
    if A_eq is not None:
        A_eq = np.array(A_eq)
        assert A_eq.ndim == 2, 'Inequality constraint matrix must be 2D!'
        assert A_eq.shape[1] == c.size, ( # pylint: disable=E1136
            'Inequality constraint matrix must match size of c!')
    if b_eq is not None:
        b_eq = np.array(b_eq).flatten()
        assert b_eq.size == A_eq.shape[0], (
            'Inequality constraint vector must have match size of matrix!')

    # Deal with binary condition:
    assert isinstance(binary, bool), 'binary must be a boolean!'
    if binary:
        if bounds is not None:
            logging.warning('Ignoring supplied bounds, using binary.')
        bounds = [(0, 1)]*c.size
    elif bounds is None:
        bounds = [(0, None)]*c.size
    else: # not binary and bounds is not None
        assert len(bounds) == c.size, (
            'bounds must have c.size number of elements!')
        assert all([len(b) == 2 and (
            (b[0] is None or b[1] is None) or (b[0] < b[1])) for b in bounds]), (
                'bounds are ill-formed!')

    # mask of real valued variables
    if real_valued is not None:
        real_valued = np.array(real_valued, dtype=bool).flatten()
        if real_valued.size != c.size:
            raise ValueError(
                'Expected real_valued mask of size %d but got %d' % (
                    c.size, real_valued.size))
        if np.sum(real_valued) == c.size:
            logging.warning('All variables are real-valued, this is a linear program!')
    else:
        # default: all variables are integer valued
        real_valued = np.zeros(c.size, dtype=bool)

    # Return sanitized values as an _ILPProblem:
    return _ILPProblem(c, A_ub, b_ub, A_eq, b_eq, binary, real_valued, bounds)

def intlinprog(
        c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, binary=False,
        real_valued=None, bounds=None, search_strategy='depth-first'):
    '''Use branch and bound to solve mixed linear programs.'''

    ilp = _process_intlinprog_args(
        c, A_ub, b_ub, A_eq, b_eq, binary, real_valued, bounds)

    # Choose queue type based on search strategy, see
    # https://en.wikipedia.org/wiki/Branch_and_bound#Generic_version
    search_strat = search_strategy.lower()
    try:
        Q = {
            'depth-first': LifoQueue,
            'breadth-first': Queue,
            'best-first': PriorityQueue,
        }[search_strat]()
    except KeyError:
        raise ValueError('"%s" is not a valid search strategy!' % search_strategy)

    # Get integer valued variables mask
    integer_valued = ~ilp.real_valued

    # We are looking for the best node
    best_node = _Node(ilp)

    # Solve the associated LP
    cur_node = _Node(ilp)
    cur_node.solve()

    # Let zbar be its optimal value
    zbar = cur_node.z

    # Let uz = value of best known feasible solution (-inf if none known)
    uz = -1*np.inf

    logging.info('Initialized beginning node:')
    logging.info(str(cur_node))

    nit = 0
    start_time = time()

    # tag: terminate
    def _terminate():
        '''Termination: return the best node as the solution.'''
        # z = uz is optimal => node corresponding to uz is solution node
        if best_node.x is None:
            logging.warning('No solution found, returning empty node.')

        # Use LP OptimizationResult as a starter
        res = best_node.lp_res
        res['execution_time'] = time() - start_time
        res['fun'] = best_node.z
        res['nit'] = nit
        res['x'] = best_node.x
        return res

    # Run the thing
    while True:

        # tag: LP_infeasibility_test
        # Linear program infeasible over subdivision?
        # YES
        if not cur_node.feasible:
            logging.info('Node is infeasible!')
            # GOTO: exhausted_test
        else: # NO

            # Let z be its optimal value; is z <= uz?
            # YES
            if cur_node.z <= uz:
                # Fathomed by bound
                logging.info('Node fathomed by bound!')
                # GOTO exhausted_test
            else: # NO

                # tag: integer_sol_test
                # Solution to LP has all variables integer?
                # YES
                if np.allclose(
                        cur_node.x[integer_valued],
                        np.round(cur_node.x[integer_valued])):

                    logging.info('Found integer solution!')
                    # Change uz to z
                    uz = cur_node.z
                    logging.info('Updated bounds: %g <= z* <= %g', uz, zbar)

                    # If we did enough to change the bound, then we're the
                    # best so far
                    best_node = cur_node

                    # YES
                    if np.allclose(uz, zbar):
                        logging.info('Solution is optimal! Terminating.')
                        return _terminate()
                    # NO -- fathomed by integrality
                    logging.info('Fathomed by integrality!')
                    # GOTO: exhausted_test
                else: # NO

                    # Generate new subdivisions from a fractional variable in the
                    # LP solution
                    n1 = _Node(cur_node.ilp, cur_node.z)
                    n2 = _Node(cur_node.ilp, cur_node.z)

                    # Rule for branching: choose variable with largest residual
                    # as in [2]_.
                    x0 = cur_node.x.copy()
                    x0[ilp.real_valued] = np.nan # Only choose from integer-valued variables
                    idx = np.nanargmax(x0 - np.floor(x0))
                    n1.change_upper_bound(idx, np.floor(cur_node.x[idx]))
                    n2.change_lower_bound(idx, np.ceil(cur_node.x[idx]))
                    logging.info('Generating new node with bounds: %s', str(n1.ilp.bounds))
                    logging.info('Generating new node with bounds: %s', str(n2.ilp.bounds))

                    # Put new division on the Queue
                    logging.info('New nodes stashed in the Queue.')
                    Q.put(n1)
                    Q.put(n2)
                    # GOTO: exhausted_test

        # tag: exhausted_test
        # Every subdivision analyzed completely?
        # YES
        if Q.empty():
            # Termination
            logging.info('No nodes on the Queue, terminating!')
            return _terminate()
        # NO
        # Select subdivision not yet analyzed completely
        cur_node = Q.get()

        # Solve linear program over subdivision
        cur_node.solve()
        logging.info('Solved node:')
        logging.info(str(cur_node))
        # GOTO: LP_infeasibility_test

        nit += 1

    # We should never reach here
    raise ValueError('Something has gone terribly wrong...')

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    c = [100, 150]
    A = [
        [8000, 4000],
        [15, 30],
    ]
    b = [40000, 200]
    res = intlinprog(c, A, b, search_strategy='depth-first')
    print(res)
