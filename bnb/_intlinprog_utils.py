'''Utility functions for intlinprog.'''

from warnings import warn
from time import time
from collections import namedtuple
from queue import LifoQueue, Queue, PriorityQueue

import numpy as np
from scipy.optimize import OptimizeWarning, OptimizeResult

_ILPProblem = namedtuple(
    '_ILPProblem', 'c A_ub b_ub A_eq b_eq binary real_valued bounds')

def _get_Queue(search_strategy):
    '''Return the correct Queue type based on search_strategy.'''
    search_strat = search_strategy.lower()
    try:
        return {
            'depth-first': LifoQueue,
            'breadth-first': Queue,
            'best-first': PriorityQueue,
        }[search_strat]()
    except KeyError:
        raise ValueError('Unknown strategy %s' % search_strategy)

def _get_branch_rule_function(branch_rule):
    '''Return function that gives branching rule.'''
    if isinstance(branch_rule, str):
        branch_rule = branch_rule.lower()

    if callable(branch_rule):
        return branch_rule
    if branch_rule == 'max fraction':
        return _branch_on_max_fraction
    if branch_rule == 'most infeasible':
        return _branch_on_most_infeasible
    if branch_rule == 'max fun':
        return _branch_on_max_fun

    raise ValueError('Unknown branch rule %s' % branch_rule)

def _process_intlinprog_args(
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

def _make_result(
        node, nit, num_integral_sol, maxiter, start_time,
        is_callback=False):
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
    res['nsol'] = num_integral_sol

    # Make integer values true integers
    if node.x is not None:
        res['x'][~node.ilp.real_valued] = np.round(
            res['x'][~node.ilp.real_valued])

    # Look up message and return result
    res['message'] = messages[res['status']]
    return res

def _print_disp_hdr(fmt=None):
    '''Print the header of the disp output.'''
    cols = [
        'num int\nsolution',
        'nodes\nexplored',
        'total\ntime (s)',
        'integer\nfval',
        'relative\ngap (%)',
    ]

    # Split out into different header rows
    num_hdr_rows = max([len(col.split('\n')) for col in cols])
    hdr_rows = []
    for ii in range(num_hdr_rows):
        hdr_rows.append([])
        for col in cols:
            split_col = col.split('\n')
            hdr_rows[ii].append(split_col[ii])

    if fmt is None:
        fmt = []
        for ii, col in enumerate(cols):
            fmt.append('{%d:%d}' % (ii, max(len(c) for c in col.split('\n'))))
        fmt = '  '.join(fmt)

    print('\nBranch and bound:\n')
    for ii, hdr_row in enumerate(hdr_rows):
        print(fmt.format(*hdr_row))
    return fmt

def _print_iter_info(fmt, nsol, nit, start_time, fval, uz, zbar):
    '''Print a single line of the disp output.'''
    rgap = 100*(1 - uz/zbar)
    ttol = time() - start_time
    cols = [
        str(nsol),
        str(nit),
        '%.3g' % ttol,
        '%g' % fval,
        '%g' % rgap,
    ]
    print(fmt.format(*cols))

def _branch_on_most_infeasible(node, _rtol, _atol):
    '''Branch on non-integral variable with fraction closest to 1/2.
    '''
    return np.argmin(np.abs(node.x - np.floor(node.x) - 0.5))

def _branch_on_max_fun(node, rtol, atol):
    '''Branch on non-integral variable with max corresponding
    component in the absolute value of the objective function.'''

    # Filter out basically integral variables
    x0 = node.x.copy()
    x0[np.abs(x0 - np.round(x0)) < (atol + rtol*np.abs(x0))] = np.nan

    # Only choose from integer-valued variables:
    x0[node.ilp.real_valued] = np.nan
    return np.nanargmax(np.abs(node.ilp.c*x0))

def _branch_on_max_fraction(node, rtol, atol):
    '''Branch on non-integral variable with highest fractional part.
    '''

    # Filter out basically integral variables
    x0 = node.x.copy()
    x0[np.abs(x0 - np.round(x0)) < (atol + rtol*np.abs(x0))] = np.nan

    # Only choose from integer-valued variables:
    x0[node.ilp.real_valued] = np.nan

    return np.nanargmax(x0 - np.floor(x0))
