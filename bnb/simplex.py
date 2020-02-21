'''Python implementation of simplex method.'''

import numpy as np
from tabulate import tabulate

class Tableau:
    '''Keep track of current problem tableau.'''

    def __init__(self, c, A, b):

        m, n = A.shape[:]
        self.m, self.n = m, n
        
        # Construct the initial simplex tableau
        top = np.concatenate((np.ones((1, 1)), -1*c.T, np.zeros((1, m))), axis=1)
        bot = np.concatenate((np.zeros((m, 1)), A, np.eye(m)), axis=1)
        right = np.concatenate((np.zeros((1, 1)), b), axis=0)
        self.tableau = np.concatenate((top, bot), axis=0)
        self.tableau = np.concatenate((self.tableau, right), axis=1)
    
        # Column labels (1-indexed)
        self.col_labels = ['z'] + ['x%d' % (ii+1) for ii in range(n)] + ['s%d' % (ii+1) for ii in range(m)]

        # Basic variables are initially the slack variables
        self.basic_vars = self.col_labels[1+n:]

        # Nonbasic variables are initially xs
        self.nonbasic_vars = self.col_labels[1:n+1]

        # Keep track of how many pivots
        self.niter = 0

    def pivot(self):
        '''Perform a single pivot operation.'''

        # The most negative non-basic variable is the pivot column
        nonbasic_idx = [ii for ii, val in enumerate(self.col_labels) if val in self.nonbasic_vars]
        nonbasic = np.concatenate((self.tableau[0, nonbasic_idx][:, None], np.array(nonbasic_idx)[:, None]), axis=1)
        if all(nonbasic[:, 0] >= 0): # optimum when no more negative coeff of nonbasic variables
            return False
        idx = np.argmin(nonbasic[:, 0])
        pivot_col = int(nonbasic[idx, 1])
        
        # Do ratio test
        RHS = self.tableau[1:, -1]
        den = self.tableau[1:, pivot_col]
        ratio = np.zeros(den.size)*np.nan
        mask = den > 0 # ignore nonpositive pivot col coeff
        ratio[mask] = RHS[mask]/den[mask]
        
        # Pivot row has minimum ratio
        pivot_row = np.nanargmin(ratio) + 1
        
        # Entry in at pivot_row is leaving var, pivot col is entering var
        #self.basic_vars[pivot_row-1] = self.col_labels[pivot_col]
        # nonbasic_vars was never changing, so change to this:
        (self.basic_vars[pivot_row-1],
         self.nonbasic_vars[pivot_col-1]) = (
             self.col_labels[pivot_col],
             self.basic_vars[pivot_row-1])
        
        # Row operations to make pivot 1
        self.tableau[pivot_row, :] *= 1/self.tableau[pivot_row, pivot_col]
        
        # Row operations to zero out rest of pivot column
        for ii in range(self.tableau.shape[0]):
            if ii == pivot_row:
                continue
            self.tableau[ii, 1:] -= self.tableau[ii, pivot_col]*self.tableau[pivot_row, 1:]

        self.niter += 1
        return True

    def solution(self):
        # Pull out the solution from the tableau
    
        sol = np.zeros(self.n + self.m)
        for ii, val in enumerate(self.col_labels[1:]):
            if val in self.basic_vars:
                val_idx = self.basic_vars.index(val)+1
                sol[ii] = self.tableau[val_idx, -1]
                fopt = self.tableau[0, -1]
        return {
            'x': sol[:self.n],
            'slack': sol[self.n:],
            'niter': self.niter,
            'fopt': fopt,
        }

    def __repr__(self):
        return tabulate(
            [[val] + self.tableau[ii, :].tolist() for ii, val in enumerate(self.basic_vars)],
            headers=[''] + self.col_labels + [''],
            tablefmt='orgtbl',
            floatfmt=".1f")
            
def simplex(c, A, b):
    '''Danztig's simplex method.

    Notes
    -----
    Implements simplex algorithm as described in [1]_.

    References
    ----------
    .. [1] https://personal.utdallas.edu/~scniu/OPRE-6201/documents/LP06-Simplex-Tableau.pdf
    '''

    # Make sure we are working with numpy stuffs
    A = np.array(A)
    assert A.ndim == 2, 'A must 2 dimensional!'
    b = np.array(b)
    assert b.squeeze().ndim == 1, 'b must be column vector!'
    c = np.array(c)
    assert c.squeeze().ndim == 1, 'c must be column vector!'
    
    # Sanity checks
    m, n = A.shape[:]
    assert b.size == m, 'b should be same size as A.shape[0]!'
    assert c.size == n, 'c should be same size as A.shape[1]!'

    # Make everything be sizes we expect
    b = np.reshape(b, (m, 1))
    c = np.reshape(c, (n, 1))

    # Construct the initial simplex tableau
    tableau = Tableau(c, A, b)
    
    # Pivot till we just can't pivot any more
    res = True
    while res:
        res = tableau.pivot()

    # return solution after all pivoting has done
    return tableau.solution()
    
if __name__ == '__main__':
    pass
