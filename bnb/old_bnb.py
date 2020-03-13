'''Use branch and bound to solve integer program.'''

import logging
from queue import LifoQueue

import numpy as np
import networkx as nx
from scipy.optimize import linprog

class Node:
    _id_counter = 0
    @classmethod
    def get_id(cls):
        ret = cls._id_counter
        cls._id_counter += 1
        return ret

    def __init__(self, c, A_ub=None, b_ub=None, bounds=None, integer_lbnd=None):
        self.c = np.array(c)*-1 # turn max problem into min for scipy.optimize.linprog
        self.A_ub = A_ub
        self.b_ub = b_ub

        # Make bounds -- default x_i > 0
        self.bnds = bounds
        if bounds is None:
            self.bnds = [(0, None)]*self.c.size

        # The maximum integer solution lower bound and holder for cost
        self.integer_lbnd = integer_lbnd
        self.cost = None

        # Create A_ub if I don't have it
        if self.A_ub is None:
            self.A_ub = np.zeros((0, self.c.size))
            self.b_ub = np.zeros(0)

        # Make sure we got arrays
        self.A_ub = np.array(self.A_ub)
        self.b_ub = np.array(self.b_ub)

        # Get a unique id
        self.id = self.get_id()

    def label(self):
        '''Return label for plotting.'''
        if self.cost is None:
            return str(self.id) + '\n' + str(self.cost)
        return str(self.id) + '\n' + '%g' % self.cost

    def add_A_ub(self, idx, val):
        '''Add an upper bound constraint.'''

        # Add on a new row
        self.A_ub = np.concatenate((self.A_ub, np.zeros((1, self.A_ub.shape[1]))), axis=0)
        self.A_ub[-1, idx] = 1
        self.b_ub = np.concatenate((self.b_ub, [val]))

    def change_bound(self, idx, lower, upper):
        '''Constrain a single variable to be between lower and upper.'''
        self.bnds[idx] = (lower, upper)

    def change_upper_bound(self, idx, upper):
        '''Constrain a single variable to be below a value.'''
        prev = self.bnds[idx]
        self.bnds[idx] = (prev[0], upper)

    def change_lower_bound(self, idx, lower):
        '''Constrain a single variable to be above a value.'''
        prev = self.bnds[idx]
        self.bnds[idx] = (lower, prev[1])

    def solve(self):
        '''Solve the LP relaxation of this node's problem instance.'''

        # Use revised simplex, more accurate than interior-point
        # Won't work currently with interior point
        res = linprog(self.c, A_ub=self.A_ub, b_ub=self.b_ub, bounds=self.bnds, method='revised simplex')

        # if we can't solve it, quit
        if not res['success']:
            return(None, (self.integer_lbnd, -np.inf))

        # Get the upper and lower bound
        ubnd = res['fun']*-1
        lbnd = -1*self.c @ np.floor(res['x'])

        # Save known integer lower bound if needed
        if self.integer_lbnd is None:
            self.integer_lbnd = lbnd

        # Save the cost for the label
        self.cost = ubnd

        # Return solution consisting of coefficients and bounds
        return(res['x'], (self.integer_lbnd, ubnd))

    def spawn(self):
        '''Return a copy of myself.'''
        return Node(-1*self.c.copy(), self.A_ub.copy(), self.b_ub.copy(), self.bnds.copy(), self.integer_lbnd)

def bnb(c, A_ub, b_ub, bounds=None, ret_tree=False):
    '''BNB using LP relaxations to solve integer programs.

    Parameters
    ----------
    c : 1D array
        Objective coefficients (for max problem).
    A_ub : 2D array or None, optional
        Upper bound constraints (<=).
    b_ub : 1D array or None, optional
        Upper bound inequality constraint vector.
    bounds : list of tuple or None, optional
        Bounds for each variable.
    ret_tree : bool, optional
        Return the networkx.DiGraph object.

    Returns
    -------
    x : 1D array or None
        Optimal integer solution or None if the problem is infeasible.
    G : networkx.DiGraph
        Tree of solution nodes.

    Notes
    -----
    Follows the algorithm described in [1].  Assumes that objective should be maximized.

    References
    ----------
    .. [1] http://web.tecnico.ulisboa.pt/mcasquilho/compute/_linpro/TaylorB_module_c.pdf
    '''

    # Initial node
    n0 = Node(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    x0, (lbnd0, ubnd0) = n0.solve()

    # Check exit conditions
    if x0 is not None and np.all(np.isclose(x0, x0.round())):
        logging.info('Parent has integer solution!')
        return {
            'x': x0,
            'fun': ubnd0,
        }
    elif x0 is None:
        warning.info('Problem is not feasible!')
        return(None, G)

    # Keep track of backup nodes in case all children are duds
    Q = LifoQueue()
    Q.put(n0)

    # Result object
    res = dict()

    if ret_tree:
        # Make a tree and include in return object
        G = nx.DiGraph()
        G.add_node(n0)
        res['tree'] = G

    # Keep track of the greatest upper bound of any ending node
    GUB = dict()

    while True:
        # Solve parent node
        logging.info('Parent has bounds: (%g, %g)', lbnd0, ubnd0)

        # branch on non-integer variable with largest fractional part
        # there are better ways of doing this, but do this for simplicity for now
        idx = np.argmax(x0 - np.floor(x0)).squeeze()
        n1 = n0.spawn()
        n2 = n0.spawn()

        # Add new bounds
        n1.change_upper_bound(idx, np.floor(x0[idx]))
        n2.change_lower_bound(idx, np.ceil(x0[idx]))

        # Solve children
        x1, (lbnd1, ubnd1) = n1.solve()
        x2, (lbnd2, ubnd2) = n2.solve()
        logging.info('First node has upper bound: %g', ubnd1)
        logging.info('Second node has upper bound: %g', ubnd2)

        # If n0 has children, then it is no longer an ending node
        GUB.pop(n0, None)

        # Add to tree
        if ret_tree:
            G.add_edge(n0, n1)
            G.add_edge(n0, n2)

        # Are any of these integer?
        cond1 = x1 is not None and np.all(np.isclose(x1, np.round(x1)))
        cond2 = x2 is not None and np.all(np.isclose(x2, np.round(x2)))
        if cond1:
            logging.info('Found integer solution x1! Need to see if optimal...')

            if all([ubnd1 >= ub for ub in GUB.values()]) and ubnd1 >= ubnd2:
                logging.info('x1 is the optimal integer solution!')
                res['x'] = x1
                res['fun'] = ubnd1
                return res
            else:
                logging.info('x1 is not optimal, but is now in GUB')
                GUB[n1] = ubnd1

        if cond2:
            logging.info('Found integer solution x2! Need to see if optimal...')

            if all([ubnd2 >= ub for ub in GUB.values()]) and ubnd2 >= ubnd1:
                logging.info('x2 is the optimal integer solution!')
                res['x'] = x2
                res['fun'] = ubnd2
                return res
            else:
                logging.info('x2 is not optimal, but is now in GUB')
                GUB[n2] = ubnd2

        # Find suitable parent if both children are infeasible
        if (x1 is None and x2 is None) or (cond1 and x2 is None) or (cond2 and x1 is None):
            logging.info('Both children are duds! Pulling off stack!')
            n0 = Q.get()
            x0, (lbnd0, ubnd0) = n0.solve()
            GUB.pop(n0, None)
            continue

        # If not, select the node with the highest max upper bound and set as new parent
        del n0, x0, lbnd0, ubnd0
        if ubnd1 > ubnd2:
            logging.info('Choosing first node as new parent.')
            n0, x0, lbnd0, ubnd0 = n1, x1, lbnd1, ubnd1

            # Save n2 for later if it's feasible
            if x2 is not None and ubnd2 > lbnd2:
                logging.info('Stashing n2 in queue')
                GUB[n2] = ubnd2
                Q.put(n2)
        else:
            logging.info('Choosing second node as new parent.')
            n0, x0, lbnd0, ubnd0 = n2, x2, lbnd2, ubnd2

            # Save n1 for later if it's feasible
            if x1 is not None and ubnd1 > lbnd1:
                logging.info('Stashing n1 in queue')
                GUB[n1] = ubnd1
                Q.put(n1)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    c = [8, 11, 6, 4]
    A = [
        [5, 7, 4, 3],
    ]
    b = [14]
    bnds = [(0, 1)]*4
    res = bnb(c, A_ub=A, b_ub=b, bounds=bnds)
    print(res)
