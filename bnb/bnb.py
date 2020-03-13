'''Branch and bound integer programming solver.

References
----------
.. [1] Bradley, Stephen P., Arnoldo C. Hax, and Thomas L. Magnanti.
       "Applied mathematical programming." (1977).
.. [2] Taylor, Bernard W., et al. Introduction to management science.
       Prentice Hall, 2002.
'''

import logging
from queue import LifoQueue
from time import time

import numpy as np
from scipy.optimize import linprog

class Node:
    '''Encapsulate an LP in the search.'''
    def __init__(self, c, A_ub, b_ub, A_eq, b_eq, bounds):
        self.id = self.take_num()
        self.c = c.copy()
        self.A_ub = A_ub.copy()
        self.b_ub = b_ub.copy()

        if A_eq is not None:
            self.A_eq = A_eq.copy()
            self.b_eq = b_eq.copy()
        else:
            self.A_eq = None
            self.b_eq = None

        self.bounds = bounds.copy()

        # Populated by self.solve():
        self.feasible = None
        self.z = None
        self.x = None

    def solve(self):
        '''Solve the LP relaxation.'''
        res = linprog(
            -1*self.c, self.A_ub, self.b_ub, self.A_eq, self.b_eq,
            bounds=self.bounds, method='revised simplex')

        if not res['success']:
            self.feasible = False
        else:
            self.feasible = True
            self.z = -1*res['fun']
            self.x = res['x']

    def change_upper_bound(self, idx, upper):
        '''Constrain a single variable to be below a value.'''
        prev = self.bounds[idx]
        self.bounds[idx] = (prev[0], upper)

    def change_lower_bound(self, idx, lower):
        '''Constrain a single variable to be above a value.'''
        prev = self.bounds[idx]
        self.bounds[idx] = (lower, prev[1])

    def __repr__(self):
        return(
            'node with id: {id}\n'
            '\tfeasible: {feasible}\n'
            '\t  bounds: {bounds}\n'
            '\t       z: {z}\n'
            '\t       x: {x}').format(
                id=self.id, feasible=self.feasible,
                bounds=self.bounds, z=self.z, x=str(self.x))

    # Class attributes
    _node_ctr = 0
    @classmethod
    def take_num(cls):
        '''Return a number that should be unique to the node instance.
        '''
        cls._node_ctr += 1
        return cls._node_ctr

class BNB:
    '''Relaxed LP branch and bound method for solving general ILPs.

    Notes
    -----
    Implements the algorithm described in [1]_ Figure 9.17.
    '''

    def __init__(self, c, A_ub, b_ub, A_eq, b_eq, bounds, real_valued):

        # Which variables are real valued (mixed integer programming)
        self.real_valued = real_valued
        self.integer_valued = ~real_valued

        # Initialization
        # Solve the associated LP
        self.cur_node = Node(c, A_ub, b_ub, A_eq, b_eq, bounds)
        self.cur_node.solve()

        # Let zbar be its optimal value
        self.zbar = self.cur_node.z

        # Let uz = value of best known feasible solution (-inf if none known)
        self.uz = -1*np.inf

        # Grab a Queue to stash leafs on
        self.Q = LifoQueue()

        # Keep track of the best incumbent node
        self.best_node = Node(c, A_ub, b_ub, A_eq, b_eq, bounds)

        # track how long it takes to run
        self.start_time = None

        logging.info('Initialized beginning node:')
        logging.info(str(self.cur_node))

    def run(self):
        '''Start the control flow.'''
        self.start_time = time()
        return self.LP_infeasibility_test()

    def LP_infeasibility_test(self):
        '''Linear program infeasible over subdivision?'''

        # YES
        if not self.cur_node.feasible:
            logging.info('Node is infeasible!')
            return self.exhausted_test()
        # NO

        # Let z be its optimal value; is z <= uz?
        # YES
        if self.cur_node.z <= self.uz:
            # Fathomed by bound
            logging.info('Node fathomed by bound!')
            return self.exhausted_test()
        # NO
        return self.integer_sol_test()

    def exhausted_test(self):
        '''Every subdivision analyzed completely?'''

        # YES
        if self.Q.empty():
            # Termination
            logging.info('No nodes on the Queue, terminating!')
            return self.terminate()
        # NO
        # Select subdivision not yet analyzed completely
        self.cur_node = self.Q.get()

        # Solve linear program over subdivision
        self.cur_node.solve()
        logging.info('Solved node:')
        logging.info(str(self.cur_node))
        return self.LP_infeasibility_test()

    def terminate(self):
        '''Termination'''
        # z = uz is optimal => node corresponding to uz is solution node
        if self.best_node.x is None:
            logging.warning('No solution found, returning empty node.')
        return {
            'x': self.best_node.x,
            'fun': self.best_node.z,
            'execution_time': time() - self.start_time,
        }

    def integer_sol_test(self):
        '''Solution to LP has all variables integer?'''

        # YES
        if np.allclose(
                self.cur_node.x[self.integer_valued],
                np.round(self.cur_node.x[self.integer_valued])):
            logging.info('Found integer solution!')
            # Change uz to z
            self.uz = self.cur_node.z
            logging.info('Updated bounds: %g <= z* <= %g', self.uz, self.zbar)

            # If we did enough to change the bound, then we're the
            # best so far
            self.best_node = self.cur_node

            # YES
            if np.allclose(self.uz, self.zbar):
            # if True:
                logging.info('Solution is optimal! Terminating.')
                return self.terminate()
            # NO -- fathomed by integrality
            logging.info('Fathomed by integrality!')
            return self.exhausted_test()
        # NO


        # Generate new subdivisions from a fractional variable in the
        # LP solution
        n1 = Node(
            self.cur_node.c,
            self.cur_node.A_ub, self.cur_node.b_ub,
            self.cur_node.A_eq, self.cur_node.b_eq,
            self.cur_node.bounds)
        n2 = Node(
            self.cur_node.c,
            self.cur_node.A_ub, self.cur_node.b_ub,
            self.cur_node.A_eq, self.cur_node.b_eq,
            self.cur_node.bounds)

        # Rule for branching: choose variable with largest residual
        # as in [2]_.
        x0 = self.cur_node.x.copy()
        x0[self.real_valued] = np.nan # Only choose from integer-valued variables
        idx = np.nanargmax(x0 - np.floor(x0))
        n1.change_upper_bound(idx, np.floor(self.cur_node.x[idx]))
        n2.change_lower_bound(idx, np.ceil(self.cur_node.x[idx]))
        logging.info('Generating new node with bounds: %s', str(n1.bounds))
        logging.info('Generating new node with bounds: %s', str(n2.bounds))

        # Put new division on the Queue
        logging.info('New nodes stashed in the Queue.')
        self.Q.put(n1)
        self.Q.put(n2)

        return self.exhausted_test()


def intprog(c, A_ub, b_ub, A_eq=None, b_eq=None, bounds=None, binary=False, real_valued=None, method='bnb'):
    '''Integer program solver.

    Parameters
    ----------
    c :

    Notes
    -----
    For some binary variables and others just integer, explcitly
    supply bounds, e.g., for 2 binary and 2 integer variables:

    .. code-block:: python
        bounds = [(0, 1)]*2 + [(0, None)]*2
    '''

    # Input matrices are assumed from here on out to be numpy arrays
    c = np.array(c)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    if A_eq is not None and b_eq is not None:
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

    if c.ndim > 1:
        logging.warning('Flattening coefficient vector!')
        c = np.flatten(c)
    if A_ub.ndim != 2:
        raise ValueError('Inequality constraint matrix should be 2D array!')
    if b_ub.ndim > 1:
        logging.warning('Flattening inequality constraint vector!')
        b = np.flatten(b)

    # Add bounds for binary
    if binary:
        if bounds is not None:
            logging.warning('Ignoring supplied bounds, using binary.')
        bounds = [(0, 1)]*c.size
    elif bounds is None:
        bounds = [(0, None)]*c.size

    # mask of real valued variables
    if real_valued is not None:
        real_valued = np.array(real_valued, dtype=bool).flatten()
        if real_valued.size != c.size:
            raise ValueError(
                'Expected real_valued mask of size %d but got %d' % (c.size, real_valued.size))
        if np.sum(real_valued) == c.size:
            logging.warning('All variables are real-valued, this is a linear program!')
    else:
        # default: all variables are integer valued
        real_valued = np.zeros(c.size, dtype=bool)

    # Call the appropriate method
    if method == 'bnb':
        bnb = BNB(c, A_ub, b_ub, A_eq, b_eq, bounds, real_valued)
        return bnb.run()
    else:
        raise ValueError('"%s" not a valid method!' % method)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    c = [5, 8]
    A = [
        [1, 1],
        [5, 9],
    ]
    b = [6, 45]
    res = intprog(c, A, b)
    print(res)
