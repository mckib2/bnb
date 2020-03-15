'''Interface for mixed integer linear programs.'''

# import logging # use for debugging
from queue import LifoQueue, Queue, PriorityQueue
from time import time
from collections import namedtuple
from copy import deepcopy
from warnings import warn

import numpy as np
from scipy.optimize import linprog, OptimizeWarning, OptimizeResult

_ILPProblem = namedtuple(
    '_ILPProblem', 'c A_ub b_ub A_eq b_eq binary real_valued bounds')

class _Node:
    '''Encapsulate an LP in the search.'''
    def __init__(self, ilp, parent_cost=-1*np.inf, depth=None):
        self.id = self.take_num()
        self.depth = depth # tree depth
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
        c, A_ub, b_ub, A_eq, b_eq, binary, real_valued, bounds, x0):
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

    # Initial values are solution variables or None
    if x0 is not None:
        try:
            x0 = np.array(x0).flatten()
            assert x0.size == c.size, 'x0 must have same size as c!'

            # Make sure it satisfies constraints
            msg = 'x0 is not a feasible solution! Ignoring.'
            if A_ub is not None:
                assert np.all(A_ub @ x0 <= b_ub), msg
            if A_eq is not None:
                assert np.allclose(A_eq @ x0, b_eq), msg

            # Make sure integral entries in x0 are approx. integral
            assert np.allclose(
                x0[~real_valued], np.round(x0[~real_valued])), msg

            # Mae sure entries in x0 are within bounds
            for x00, b in zip(x0, bounds):
                if b[0] is not None:
                    assert x00 >= b[0], msg
                if b[1] is not None:
                    assert x00 <= b[1], msg
        except AssertionError as e:
            warn(str(e), OptimizeWarning)

    # Return sanitized values as an _ILPProblem:
    # We only use x0 for the initial solution, so return it separately
    return (_ILPProblem(
        c, A_ub, b_ub, A_eq, b_eq, binary, real_valued, bounds), x0)

def _make_result(node, nit, maxiter, start_time, is_callback=False):
    '''Make an OptimizeResult object from a search tree _Node.'''

    res = OptimizeResult()

    # messages unique to ILP OptimizeResult: status -> message
    messages = {
        0 : [
            'Optimization terminated successfully (with optimal solution).',
            'Optimization proceeding nominally.',
        ],
        1 : [
            'Iteration limit reached with feasible solution (maybe suboptimal).',
            'Iteration limit reached without feasible solution.',
        ],
    }
    # Copy messages from LP OptimizeResult
    if node.lp_res is not None:
        for ii in range(2, 5):
            messages[ii] = node.lp_res['message']

    # Did we exceed allowed number of iterations?
    if nit >= maxiter:
        res['status'] = 1
        # collapse message list into a single item:
        if node.x is None:
            messages[1] = messages[1][1]
        else:
            messages[1] = messages[1][0]

    # If we found a feasible node, then we won (even if it's not
    # provably optimal)
    res['con'] = np.zeros(0)
    res['slack'] = np.zeros(0)
    if node.x is None:
        # We only fail if we don't find a feasible solution, the
        # associated LP result will know more about why we failed
        res['success'] = False
        res['status'] = node.lp_res['status']
    else:
        # Info about associated LP solution
        if node.ilp.A_eq is not None:
            res['con'] = node.ilp.b_eq - node.ilp.A_eq @ node.x
        if node.ilp.A_ub is not None:
            res['slack'] = node.ilp.b_ub - node.ilp.A_ub @ node.x

        # If we have a feasible solution then we declare success
        res['success'] = True
        if 'status' not in res:
            # Resolve status 0 message into a single message based
            # on whether we are in a callback or not
            messages[0] = messages[0][is_callback]
            res['status'] = 0

    res['execution_time'] = time() - start_time
    res['fun'] = node.z
    res['nit'] = nit
    res['x'] = node.x
    res['depth'] = node.depth

    # Make integer values true integers
    if node.x is not None:
        res['x'][~node.ilp.real_valued] = np.round(
            res['x'][~node.ilp.real_valued])

    # Look up message and return result
    res['message'] = messages[res['status']]
    return res

def _terminate(best_node, start_time, nit, maxiter):
    '''Termination: return the best node as the solution.'''
    # z = uz is optimal => node corresponding to uz is
    # the solution node
    if best_node.x is None:
        msg = 'No solution found, returning empty node.'
        warn(msg, OptimizeWarning)

    return _make_result(best_node, nit, maxiter, start_time)

def intlinprog(
        c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, binary=False,
        real_valued=None, bounds=None, search_strategy='depth-first',
        callback=None, options=None, lp_options=None, x0=None):
    '''Use branch and bound to solve mixed integer linear programs.

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
        the search tree.  By default `'depth-first'` is chosen.  "The
        `'depth-first'` variant is recommended ... because it quickly
        produces full solutions, and therefore upper bounds" [3]_.
    callback : callable, optional
        If a callback function is provided, it will be called at least
        once per iteration of the algorithm. The callback function
        must accept a single `scipy.optimize.OptimizeResult`
        consisting of the following fields:

            x : 1-D array
                The current solution vector of the associated linear
                program (may or may not be integral feasible).
            fun : float
                The current value of the objective function ``c @ x``.
            success : bool
                ``True`` when the algorithm has found a feasible
                solution.
            slack : 1-D array
                The (nominally positive) values of the slack of the
                associated linear program, ``b_ub - A_ub @ x``.
            con : 1-D array
                The (nominally zero) residuals of the equality
                constraints of the associated linear program,
                ``b_eq - A_eq @ x``.
            status : int
                An integer representing the status of the algorithm.

                ``0`` : Optimization proceeding nominally.

                ``1`` : Iteration limit reached.

                ``2`` : Problem appears to be infeasible.

                ``3`` : Problem appears to be unbounded.

                ``4`` : Numerical difficulties encountered.

            nit : int
                The current iteration number (number of nodes
                evaluated in the search tree).
            message : str
                A string descriptor of the algorithm status.

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
    x0 : 1-D array, optional
        Guess values of the decision variables, which will be used
        for additional pruning of the search tree by the optimization
        algorithm. This argument is only used if ``x0`` represents a
        feasible solution.

    Returns
    -------
    res : OptimizationResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the
        fields:

            x : 1-D array
                The values of the integral and real-valued decision
                variables that minimize the objective function while
                satisfying the constraints.
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
                ``True`` when the algorithm succeeds in finding a
                feasible solution (not necessarily optimal: see
                ``status``).
            status : int
                An integer representing the exit status of the
                algorithm.

                ``0`` : Optimization terminated successfully (with
                        optimal solution).

                ``1`` : Iteration limit reached.

                ``2`` : Problem appears to be infeasible.

                ``3`` : Problem appears to be unbounded.

                ``4`` : Numerical difficulties encountered.

            nit : int
                The total number of iterations performed in all
                phases, i.e. the number of nodes evaluated in the
                search tree.
            execution_time : float
                The number of seconds taken to find the optimal
                solution with integral constraints.
            message : str
                A string descriptor of the exit status of the
                algorithm.

    Notes
    -----
    When ``x0`` is provided and is feasible, it is set as the best
    solution currently known.  This will assist in pruning branches
    that will never beat ``x0``, but the search will still start
    from the initial associated linear program as the state of the
    search tree is not captured by ``x0`` alone.  This means that
    stopping ``intlinprog`` and resuming from the last known best
    solution will still require a search starting from the first node,
    but which might proceed more quickly due to the possibility of
    more aggressive pruning.

    References
    ----------
    .. [1] Bradley, Stephen P., Arnoldo C. Hax, and Thomas L.
           Magnanti. "Applied mathematical programming." (1977).
    .. [2] Taylor, Bernard W., et al. Introduction to management
           science. Prentice Hall, 2002.
    .. [3] https://en.wikipedia.org/wiki/Branch_and_bound
    '''

    try:
        ilp, x0 = _process_intlinprog_args(
            c, A_ub, b_ub, A_eq, b_eq,
            binary, real_valued, bounds, x0)
    except AssertionError as e:
        # _process_intlinprog_args uses assertions as a clean way to
        # describe requirements on variables, but the correct
        # Exception is probably ValueError:
        raise ValueError(str(e))

    # The callback with always be called, so make sure it's actually
    # callable
    if callback is None:
        do_callback = lambda *args: None
    elif not callable(callback):
        warn('callback is not callable! Ignoring.', OptimizeWarning)
        do_callback = lambda *args: None
    else:
        do_callback = lambda *args: callback(_make_result(
            cur_node, nit, maxit, start_time, is_callback=True))

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

    # Let uz = value of best known feasible solution (-inf if none
    # known).  We are looking for the best node -- initialize as the
    # initial feasible node if given
    best_node = _Node(ilp)
    if x0 is not None:
        # Populate fields in best_node and OptimizeResult based on
        # known initial solution x0
        uz = c @ x0
        best_node.feasible = True
        best_node.z = uz
        best_node.x = x0
    else:
        uz = -1*np.inf

    # Solve the associated LP
    cur_node = _Node(ilp, depth=0)
    cur_node.solve()

    # Let zbar be its optimal value
    zbar = cur_node.z

    # Run the solver until the optimal solution is found or maxiter
    # is hit
    maxit = solver_options.get('maxiter', np.inf)
    nit = 0
    start_time = time()

    # tag: terminate
    terminate = lambda *args: _terminate(
        best_node, start_time, nit, maxit)

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
                        return terminate()
                    # NO -- fathomed by integrality
                    # logging.info('Fathomed by integrality!')
                    # GOTO: exhausted_test
                # NO
                else:

                    # Generate new subdivisions from a fractional
                    # variable in the LP solution
                    n1 = _Node(
                        cur_node.ilp, cur_node.z,
                        depth=cur_node.depth+1)
                    n2 = _Node(
                        cur_node.ilp, cur_node.z,
                        depth=cur_node.depth+1)

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
            return terminate()
        # NO

        # Call callback before we grab the next node from the Queue;
        # uses cur_node to generate OptimizeResult
        do_callback()

        # Select subdivision not yet analyzed completely
        cur_node = Q.get()

        # Call it an iteration before we solve the next LP
        nit += 1
        if nit >= maxit:
            # Quit if we timeout on number of iterations
            return terminate()

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
    x0 = [0, 6]
    res = intlinprog(
        c, A, b,
        search_strategy='depth-first',
        options={'maxiter': np.inf},
        x0=x0)
    print(res)
