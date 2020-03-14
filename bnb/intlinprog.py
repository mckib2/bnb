'''Interface for mixed integer linear programs.'''

# import logging # use for debugging
from queue import LifoQueue, Queue, PriorityQueue
from time import time
from collections import namedtuple
from copy import deepcopy
from warnings import warn
from functools import partial

import numpy as np
from scipy.optimize import linprog, OptimizeWarning, OptimizeResult

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
            bounds=self.ilp.bounds, method='revised simplex',
            options=self.lp_solver_options)

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

    # Class attributes and methods
    lp_solver_options = {}
    @classmethod
    def global_set_lp_solver_options(cls, lp_solver_options):
        '''Solver options will never change for associated LP.'''
        cls.lp_solver_options = lp_solver_options
    _node_ctr = 0
    @classmethod
    def take_num(cls):
        '''Return a number unique to the node instance.

        This is intended for use when plotting search trees.
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
        assert A_ub.ndim == 2, (
            'Inequality constraint matrix must be 2D!')
        assert A_ub.shape[1] == c.size, ( # pylint: disable=E1136
            'Inequality constraint matrix must match size of c!')
    if b_ub is not None:
        b_ub = np.array(b_ub).flatten()
        assert b_ub.size == A_ub.shape[0], (
            'Inequality constraint vector must have match size of '
            'matrix!')
    if A_eq is not None:
        A_eq = np.array(A_eq)
        assert A_eq.ndim == 2, (
            'Inequality constraint matrix must be 2D!')
        assert A_eq.shape[1] == c.size, ( # pylint: disable=E1136
            'Inequality constraint matrix must match size of c!')
    if b_eq is not None:
        b_eq = np.array(b_eq).flatten()
        assert b_eq.size == A_eq.shape[0], (
            'Inequality constraint vector must have match size of '
            'matrix!')

    # Deal with binary condition:
    assert isinstance(binary, bool), 'binary must be a boolean!'
    if binary:
        if bounds is not None:
            msg = 'Ignoring supplied bounds, using binary constraint.'
            warn(msg, OptimizeWarning)
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
            raise ValueError('Expected array of size %d but got %d'
                             '' % (c.size, real_valued.size))
        if np.sum(real_valued) == c.size:
            msg = ('All variables are real-valued, this is a LP! '
                   'Would be better to use dedicated LP solver.')
            warn(msg, OptimizeWarning)
    else:
        # default: all variables are integer valued
        real_valued = np.zeros(c.size, dtype=bool)

    # Return sanitized values as an _ILPProblem:
    return _ILPProblem(
        c, A_ub, b_ub, A_eq, b_eq, binary, real_valued, bounds)

def _add_info_from_lp_result(res, lp_res):
    '''Scrape information about associated linear program solution.'''

    # shift LP statuses past 1 up one to convert to ILP status; we
    # inserted an additional status at position 1 for maxiter timeout
    # with/without feasible solution
    if lp_res['status'] > 1:
        lp_res['status'] += 1

    for k in ['con', 'slack', 'success', 'message', 'status']:
        # Only update if the key wasn't already set by ILP solver
        if k not in res:
            res[k] = lp_res[k]

    return res

def _terminate(best_node, res, start_time, nit, integer_valued):
    '''Termination: return the best node as the solution.'''
    # z = uz is optimal => node corresponding to uz is
    # the solution node
    if best_node.x is None:
        msg = 'No solution found, returning empty node.'
        warn(msg, OptimizeWarning)

        # Variables we don't have because associated LP failed
        res['con'] = None
        res['slack'] = None
    else:
        # Grab info about solution node from LP OptimizationResult
        res = _add_info_from_lp_result(res, best_node.lp_res)

    res['execution_time'] = time() - start_time
    res['fun'] = best_node.z
    res['nit'] = nit
    res['x'] = best_node.x

    # Make integer values true integers
    if best_node.x is not None:
        res['x'][integer_valued] = np.round(res['x'][integer_valued])

    return res

def intlinprog(
        c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, binary=False,
        real_valued=None, bounds=None, search_strategy='depth-first',
        options=None, lp_options=None):
    '''Use branch and bound to solve mixed linear programs.

    Parameters
    ----------
    c : 1-D array
        The coefficients of the linear objective function to be
        minimized.
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub``
        specifies the coefficients of a linear inequality constraint
        on ``x``.
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    A_eq : 2-D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies
        the coefficients of a linear equality constraint on ``x``.
    b_eq : 1-D array, optional
        The equality constraint vector. Each element of ``A_eq @ x``
        must equal the corresponding element of ``b_eq``.
    binary : bool, optional
        A convienience flag that indicates all solution variables are
        binary, i.e., integer-valued with ``bnds = [(0, 1)]*c.size``.
        Integer valued solution (``False``) is assumed by default.
    real_valued : 1-D array, optional
        Vector of real-valued constraints, specified as a boolean
        mask. The ```True`` entries in ``real_valued`` indicate the
        components of the solution ``x`` that are real-valued.  The
        ``False`` entries indicate that the variable is
        integer-valued.  By default, the mask is set to all ``False``
        indicating all solution variables should be integral.
    bounds : sequence, optional
        A sequence of ``(min, max)`` pairs for each element in ``x``,
        defining the minimum and maximum values of that decision
        variable. Use ``None`` to indicate that there is no bound. By
        default, bounds are ``(0, None)`` (all decision variables are
        non-negative). If a single tuple ``(min, max)`` is provided,
        then ``min`` and ``max`` will serve as bounds for all decision
        variables.
    search_strategy : {'depth-first', 'breadth-first', 'best-first'}, optional
        The strategy for branch and bound to choose the next node in
        the search tree.  By default `'depth-first'`is chosen.  "The
        `'depth-first'` variant is recommended ... because it quickly
        produces full solutions, and therefore upper bounds" [3]_.
    options : dict, optional
        A dictionary of integer linear programming solver options. The
        following fields are accepted options:

            maxiter : int
                Maximum number of nodes to evaluate.
                Default: ``inf`` (keep going until provably optimal
                solution is found).
            disp : bool
                Set to ``True`` to print convergence messages.
                Default: ``False``.

    lp_options: dict, optional
        A dictionary of linear program solver options to pass to
        :ref:`'revised simplex' <optimize.linprog-revised_simplex>`.
        For a list of valid options, see the corresponding
        :ref:`linprog <optimize.linprog>` documentation.

    Returns
    -------
    res : OptimizationResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the
        fields:

            x : 1-D array
                The values of the integral and real-valued decision
                variables from an associated linear program that
                minimizes the objective function while satisfying the
                constraints.
            fun : float
                The optimal value of the objective function ``c @ x``.
            slack : 1-D array
                The (nominally positive) values of the slack
                variables of the associated linear program which
                satisfies all integral constraints;
                ``b_ub - A_ub @ x``
            con : 1-D array
                The (nominally zero) residuals of the equality
                constraints, ``b_eq - A_eq @ x``.
            success : bool
                ``True`` when the algorithm succeeds in finding an
                optimal solution.
            status : int
                An integer representing the exit status of the
                algorithm.

                ``0`` : Optimization terminated successfully.

                ``1`` : Iteration limit reached with feasible
                        solution.

                ``2`` : Iteration limit reached without feasible
                        solution.

                ``3`` : Problem appears to be infeasible.

                ``4`` : Problem appears to be unbounded.

                ``5`` : Numerical difficulties encountered.

            nit : int
                The total number of iterations performed in all
                phases, i.e. the number of nodes in the search tree.
            execution_time : float
                The number of seconds taken to find the optimal
                solution with integral constraints.
            message : str
                A string descriptor of the exit status of the
                algorithm.

    Notes
    -----

    References
    ----------
    .. [1] Bradley, Stephen P., Arnoldo C. Hax, and Thomas L.
           Magnanti. "Applied mathematical programming." (1977).
    .. [2] Taylor, Bernard W., et al. Introduction to management
           science. Prentice Hall, 2002.
    .. [3] https://en.wikipedia.org/wiki/Branch_and_bound
    '''

    try:
        ilp = _process_intlinprog_args(
            c, A_ub, b_ub, A_eq, b_eq, binary, real_valued, bounds)
    except AssertionError as e:
        # _process_intlinprog_args uses assertions as a clean way to
        # describe requirements on variables, but the correct
        # Exception is probably ValueError:
        raise ValueError(str(e))

    # Get options for both ILP and LP and let the _Node class know
    # what the LP options are
    if options is None:
        options = {}
    solver_options = {k: v for k, v in options.items()} # pylint: disable=R1721
    if lp_options is None:
        lp_options = {}
    lp_solver_options = {k: v for k, v in lp_options.items()} # pylint: disable=R1721
    _Node.global_set_lp_solver_options(lp_solver_options)

    # Choose queue type based on search strategy, see [3]_.
    search_strat = search_strategy.lower()
    try:
        Q = {
            'depth-first': LifoQueue,
            'breadth-first': Queue,
            'best-first': PriorityQueue,
        }[search_strat]()
    except KeyError:
        raise ValueError('Unknown strategy %s' % search_strategy)

    # Get integer valued variables mask
    integer_valued = ~ilp.real_valued

    # We are looking for the best node
    best_node = _Node(ilp)

    # Solve the associated LP
    cur_node = _Node(ilp)
    cur_node.solve()

    # Let zbar be its optimal value
    zbar = cur_node.z

    # Let uz = value of best known feasible solution
    # (-inf if none known)
    uz = -1*np.inf

    # logging.info('Initialized beginning node:')
    # logging.info(str(cur_node))

    nit = 0
    maxit = solver_options.get('maxiter', np.inf)
    res = OptimizeResult()
    messages = {
        1: 'Iteration limit reached with feasible solution.',
        2: 'Iteration limit reached without feasible solution.',
    }
    start_time = time()

    # tag: terminate
    terminate = partial(
        _terminate, start_time=start_time,
        integer_valued=integer_valued)

    # Start branchin' and boundin'
    while True:

        # tag: LP_infeasibility_test
        # Linear program infeasible over subdivision?
        # YES
        # if not cur_node.feasible:
        #     logging.info('Node is infeasible!')
        #     # GOTO: exhausted_test
        # NO
        if cur_node.feasible:

            # Let z be its optimal value; is z <= uz?
            # YES
            # if cur_node.z <= uz:
            #     # Fathomed by bound
            #     logging.info('Node fathomed by bound!')
            #     # GOTO exhausted_test
            # NO
            if cur_node.z > uz:

                # tag: integer_sol_test
                # Solution to LP has all variables integer?
                # YES
                if np.allclose(
                        cur_node.x[integer_valued],
                        np.round(cur_node.x[integer_valued])):

                    # logging.info('Found integer solution!')
                    # Change uz to z
                    uz = cur_node.z
                    # logging.info(
                    #     'Updated bounds: %g <= z* <= %g', uz, zbar)

                    # If we did enough to change the bound, then we're
                    # the best so far
                    best_node = cur_node

                    # YES
                    if np.allclose(uz, zbar):
                        # logging.info(
                        #     'Solution is optimal! Terminating.')
                        return terminate(
                            res=res, best_node=best_node, nit=nit)
                    # NO -- fathomed by integrality
                    # logging.info('Fathomed by integrality!')
                    # GOTO: exhausted_test
                # NO
                else:

                    # Generate new subdivisions from a fractional
                    # variable in the LP solution
                    n1 = _Node(cur_node.ilp, cur_node.z)
                    n2 = _Node(cur_node.ilp, cur_node.z)

                    # Rule for branching: choose variable with largest
                    # residual as in [2]_.
                    x0 = cur_node.x.copy()
                    # Only choose from integer-valued variables:
                    x0[ilp.real_valued] = np.nan
                    idx = np.nanargmax(x0 - np.floor(x0))
                    n1.change_upper_bound(
                        idx, np.floor(cur_node.x[idx]))
                    n2.change_lower_bound(
                        idx, np.ceil(cur_node.x[idx]))
                    # logging.info(
                    #     'Generating new node with bounds: %s',
                    #     str(n1.ilp.bounds))
                    # logging.info(
                    #     'Generating new node with bounds: %s',
                    #     str(n2.ilp.bounds))

                    # Put new division on the Queue
                    # logging.info('New nodes stashed in the Queue.')
                    Q.put(n1)
                    Q.put(n2)
                    # GOTO: exhausted_test

        # tag: exhausted_test
        # Every subdivision analyzed completely?
        # YES
        if Q.empty():
            # Termination
            # logging.info('No nodes on the Queue, terminating!')
            return terminate(res=res, best_node=best_node, nit=nit)
        # NO
        # Select subdivision not yet analyzed completely
        cur_node = Q.get()

        # Call it an iteration before we solve the next LP
        nit += 1
        if nit >= maxit:
            if best_node.x is not None:
                res['status'] = 1 # we have a feasible node
            else:
                res['status'] = 2 # no feasible node found
            res['message'] = messages[res['status']]

            # Declare success if we found a feasible solution, might
            # not be optimal though:
            res['success'] = best_node.x is not None
            return terminate(res=res, best_node=best_node, nit=nit)

        # Solve linear program over subdivision
        cur_node.solve()
        # logging.info('Solved node:')
        # logging.info(str(cur_node))
        # GOTO: LP_infeasibility_test

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
    res = intlinprog(c, A, b, search_strategy='depth-first', options={'maxiter': 6})
    print(res)
