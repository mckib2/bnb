'''Python implementation of simplex method.'''

import numpy as np

def simplex(c, A, b):
    '''Danztig's simplex method.

    Notes
    -----
    Implements simplex algorithm as described in [1]_.

    References
    ----------
    .. [1] https://personal.utdallas.edu/~scniu/OPRE-6201/documents/LP06-Simplex-Tableau.pdf
    '''

    # Sanity checks
    m, n = A.shape[:]
    assert b.size == m, 'b should be same size as A.shape[0]!'
    assert c.size == n, 'c should be same size as A.shape[1]!'

    # Make everything be sizes we expect
    b = np.reshape(b, (m, 1))
    c = np.reshape(c, (n, 1))
    
    # Construct the initial simplex tableau
    top = np.concatenate((np.ones((1, 1)), -1*c.T, np.zeros((1, m))), axis=1)
    bot = np.concatenate((np.zeros((m, 1)), A, np.eye(m)), axis=1)
    right = np.concatenate((np.zeros((1, 1)), b), axis=0)
    tableau = np.concatenate((top, bot), axis=0)
    tableau = np.concatenate((tableau, right), axis=1)

    # Column labels (1-indexed)
    col_labels = ['z'] + ['x%d' % (ii+1) for ii in range(n)] + ['s%d' % (ii+1) for ii in range(m)]
    
    # Basic variables are initially the slack variables
    basic_vars = col_labels[1+n:]

    # Nonbasic variables are initially xs
    nonbasic_vars = col_labels[1:n+1]

    niter = 0
    while True:
        # The most negative non-basic variable is the pivot column
        nonbasic_idx = [ii for ii, val in enumerate(col_labels) if val in nonbasic_vars]
        nonbasic = np.concatenate((tableau[0, nonbasic_idx][:, None], np.array(nonbasic_idx)[:, None]), axis=1)
        if all(nonbasic[:, 0] >= 0): # optimum when no more negative coeff of nonbasic variables
            break
        idx = np.argmin(nonbasic[:, 0])
        pivot_col = int(nonbasic[idx, 1])
        
        # Do ratio test
        RHS = tableau[1:, -1]
        den = tableau[1:, pivot_col]
        ratio = np.zeros(den.size)*np.nan
        mask = den > 0 # ignore nonpositive pivot col coeff
        ratio[mask] = RHS[mask]/den[mask]
        
        # Pivot row has minimum ratio
        pivot_row = np.nanargmin(ratio) + 1
        
        # Entry in R4 is leaving var, pivot col is entering var
        basic_vars[pivot_row-1] = col_labels[pivot_col]
        
        # Row operations to make pivot 1
        tableau[pivot_row, :] *= 1/tableau[pivot_row, pivot_col]
        
        # Row operations to zero out rest of pivot column
        for ii in range(tableau.shape[0]):
            if ii == pivot_row:
                continue
            tableau[ii, 1:] -= tableau[ii, pivot_col]*tableau[pivot_row, 1:]

        # Keep track of how many iterations we do
        niter += 1

    # Pull out the solution from the tableau
    sol = np.zeros(n+m)
    for ii, val in enumerate(col_labels[1:]):
        if val in basic_vars:
            val_idx = basic_vars.index(val)+1
            sol[ii] = tableau[val_idx, -1]
    fopt = tableau[0, -1]
            
    return {
        'x': sol[:n],
        'slack': sol[n:],
        'niter': niter,
        'fopt': fopt,
    }

if __name__ == '__main__':

    c = np.array([4, 3])
    A = np.array([
        [2, 3],
        [-3, 2],
        [0, 2],
        [2, 1],
    ])
    b = np.array([6, 3, 5, 4])

    res = simplex(c, A, b)
    print('    x: %s' % str(res['x']))
    print('slack: %s' % str(res['slack']))
    print(' fopt: %g' % res['fopt'])
    print('niter: %d' % res['niter'])
    
    '''
    # Second test case
    # https://sites.math.washington.edu/~burke/crs/407/notes/section2.pdf
    c = np.array([3, 2, -4])
    A = np.array([
        [1, 4, 0],
        [2, 4, -2],
        [1, 1, -2],
    ])
    b = np.array([5, 6, 2])

    res = simplex(c, A, b)
    print('    x: %s' % str(res['x']))
    print('slack: %s' % str(res['slack']))
    print(' fopt: %g' % res['fopt'])
    print('niter: %d' % res['niter'])
    '''
