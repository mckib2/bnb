'''Python branch and bound.'''

import logging

import numpy as np

def real_valued_bnb(lbound, ubound, branch, Qinit, eps=np.finfo(float).eps):
    '''Python implementation of real-valued branch and bound.

    Parameters
    ----------
    lbound : callable
        Bounding function that computes lower bound of objective
        function over rectangular region, Q.
    ubound : callable
        Bounding function that computes upper bound of objective
        function over rectangular region, Q.
    branch : callable
        Problem specific branch rule. Returns list of Q.
    Qinit : tuple of tuples of float
        Description of an initial m-dimensional rectangle. Has the
        form ((d1_low, d1_hi), (d2_low, d2_hi), ..., (dM_low, dM_hi)).
    eps : float, optional
        Small positive number; tolerance.

    Returns
    -------
    res : dict
        Dictionary containing:

        - x : Estimate of the minimum of the objective function.
        - Q : Final m-dimensional rectange in which x exists.
        - niter : Number of iterations performed.

    Notes
    -----
    Implements generic algorithm for real-valued objective functions
    as described in [1]_.  This branch and bound function relies on
    lower and upper bounding functions over a rectanglular region
    that become tight as as the rectangles shrink.

    The branching function should return disjoint sets to avoid
    infinite looping.

    References
    ----------
    .. [1] https://web.stanford.edu/class/ee364b/lectures/bb_slides.pdf
    '''

    # Make sure our m-dimensional rectangle makes sense
    for b in Qinit:
        assert b[0] < b[1], (
            '%g >= %g! Bounds of Qinit must have b[0] < b[1]!' % b)

    # Intialize
    Q = Qinit
    L1 = lbound(Q)
    U1 = ubound(Q)

    # See if we can leave early
    if U1 - L1 <= eps:
        return Q

    # Loop until we hit tolerance
    U2, L2 = np.finfo(float).max/10, -1*np.finfo(float).max/10
    niter = 0
    cond = U2 - L2
    last_cond = cond*2
    while cond > eps and cond != last_cond:

        # Make sure we don't get stuck in infinite loop
        last_cond = cond
        
        # Apply branching rule
        Qs = branch(Q)
        
        # Compute and update lower and upper bounds
        nQs = len(Qs)
        lbnds, ubnds = np.empty(nQs), np.empty(nQs)
        for ii, Q0 in enumerate(Qs):
            lbnds[ii] = lbound(Q0)
            ubnds[ii] = ubound(Q0)
        idx = np.argmin(lbnds)
        L2, U2 = lbnds[idx], np.min(ubnds)

        # Find the new region (take the one with lowest lower bnd)
        Q = Qs[idx]
        niter += 1

        # Make sure we break when we hit tolerance
        cond = U2 - L2

    # If we don't meet condition, then something bad happened
    if cond >= eps:
        logging.warning(
            'Breaking out of loop prematurely! '
            'Beware of infinite loop!')

    # Call the solution the point in the middle of the resulting
    # m-dimensional rectange
    x = [(d[0] + d[1])/2 for d in Q]
        
    return {
        'x': x,
        'Q': Q,
        'niter': niter,
    }

if __name__ == '__main__':
    pass
