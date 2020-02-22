'''Try simplex again.'''

from tabulate import tabulate

class Tableau:
    '''Tableau class.'''

    def __init__(self, c, A_ub, b_ub, A_eq, b_eq, A_lb, b_lb):

        self.c = c
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_lb = A_lb
        self.b_lb = b_lb
        self.A_eq = A_eq
        self.b_eq = b_eq

        # Do big M method to deal with equality/geq constraints
        self.bigM = 1e6
        
        # Allocate the table
        self.num_vars = len(c)
        self.num_slack = len(b_ub)
        self.num_surplus = len(b_lb)
        self.num_artificial = len(b_lb) + len(b_eq)
        self.nbasic_var_labels = 1
        self.nrhs_col = 1
        self.nobjective_row = 1
        self.nhdr_row = 1
        self.nA_ub = len(b_ub)
        self.nA_lb = len(b_lb)
        self.nA_eq = len(b_eq)
        
        self.num_cols = (
            self.num_vars +
            self.num_slack +
            self.num_surplus +
            self.num_artificial +
            self.nbasic_var_labels +
            self.nrhs_col
        )
        self.num_rows = (
            self.nhdr_row +
            self.nobjective_row +
            self.nA_ub +
            self.nA_lb +
            self.nA_eq
        )
        self.tableau = []
        for ii in range(self.num_rows):
            self.tableau.append([0]*self.num_cols)

        # A_ub gets a slack variable for each row
        # A_eq gets an artificial variable for each row
        # A_lb gets a surplus variable (negative slack variable) and artificial variable for each row

        # Fill in headers of the tableau
        self.init_hdr()

        # Fill in values
        self.init_tableau_values()

        # Start pivoting
        done = False
        while not done:
            self.pivot()

    def pivot(self):
        '''Perform a pivot.'''
        raise NotImplementedError()
        
    def init_tableau_values(self):
        '''Fill in initial values for constraints.'''

        # header and objective rows come first along rows
        # basic variable headers come first along columns
        row_offset = self.nhdr_row + self.nobjective_row
        col_offset = self.nbasic_var_labels
        
        slack_idx = self.num_vars + col_offset
        surplus_idx = self.num_vars + self.num_slack + col_offset
        artificial_idx = self.num_vars + self.num_slack + self.num_surplus + col_offset

        # Do objective row
        for ii, val in enumerate(self.c):
            self.tableau[self.nhdr_row][ii+col_offset] = -1*val
        
        # leq constraints
        for ii, row in enumerate(self.A_ub):
            # Include slack variables to be appended on this row
            for jj, val in enumerate(row):
                self.tableau[ii+row_offset][jj+col_offset] = val
            self.tableau[ii+row_offset][slack_idx] = 1
            slack_idx += 1

            # Also get RHS values
            for jj, val in enumerate(self.b_ub):
                self.tableau[ii+row_offset][-1] = val

        # eq constraints
        for ii, row in enumerate(self.A_eq):
            # Include artificial variable on this row
            for jj, val in enumerate(row):
                self.tableau[self.nA_ub+ii+row_offset][jj+col_offset] = val
            self.tableau[self.nA_ub+ii+row_offset][artificial_idx] = 1
            self.tableau[self.nhdr_row][artificial_idx] = -1*self.bigM
            artificial_idx += 1

            # Also get RHS values
            for jj, val in enumerate(self.b_eq):
                self.tableau[self.nA_ub+ii+row_offset][-1] = val

        # geq constraints -- big M method
        for ii, row in enumerate(self.A_lb):
            # Include suprplus and artificial variables on this row
            for jj, val in enumerate(row):
                self.tableau[self.nA_ub+self.nA_eq+ii+row_offset][jj+col_offset] = val
            self.tableau[self.nA_ub+self.nA_eq+ii+row_offset][artificial_idx] = 1
            self.tableau[self.nhdr_row][artificial_idx] = -1*self.bigM
            self.tableau[self.nA_ub+self.nA_eq+ii+row_offset][surplus_idx] = -1
            artificial_idx += 1
            surplus_idx += 1

            # Also get RHS values
            for jj, val in enumerate(self.b_lb):
                self.tableau[self.nA_ub+self.nA_eq+ii+row_offset][-1] = val
                
    def init_hdr(self):
        '''Fill in initial values for tableau header.'''
        idx = 0

        # Column tracking basic variables is first
        self.tableau[0][idx] = 'basic'
        idx += 1
        
        # Start with variables that appear in objective function,
        # then do slack, surplus, and artificial
        for ii in range(self.num_vars):
            self.tableau[0][idx] = 'x' + str(ii)
            idx += 1
        for ii in range(self.num_slack):
            self.tableau[0][idx] = 'sl' + str(ii)
            idx += 1
        for ii in range(self.num_surplus):
            self.tableau[0][idx] = 'su' + str(ii)
            idx += 1
        for ii in range(self.num_artificial):
            self.tableau[0][idx] = 'a' + str(ii)
            idx += 1
        # End with RHS
        self.tableau[0][idx] = 'RHS'
        idx += 1

        # Did we get all of them?
        assert idx == self.num_cols

        # Now do row headers
        self.tableau[1][0] = '' # objective row, no basic var here
        
        # Get a feasible solution by setting slack to RHS for A_ub
        col_offset = self.nbasic_var_labels
        row_offset = self.nhdr_row + self.nobjective_row
        for jj in range(self.nA_ub):
            self.tableau[row_offset + jj][0] = self.tableau[0][self.num_vars + col_offset + jj]

        # Set artificial to RHS for A_eq (because surplus is negative)
        offset = self.num_vars + self.num_slack + self.num_slack + col_offset
        for jj in range(self.nA_eq):
            self.tableau[row_offset + self.nA_ub + jj][0] = self.tableau[0][offset + jj]

        # Set artificial to RHS for A_lb
        offset += self.nA_eq
        for jj in range(self.nA_lb):
            self.tableau[row_offset + self.nA_ub + self.nA_eq + jj][0] = self.tableau[0][offset + jj]
            
    def __repr__(self):
        return tabulate(
            self.tableau,
            headers="firstrow",
            tablefmt='orgtbl',
            floatfmt=".1f")

        
if __name__ == '__main__':
    c = [2, -1, 3]
    A_ub = [
        [0, 2, 1],
    ]
    b_ub = [2]
    A_lb = [
        [1, -2, 1],
    ]
    b_lb = [3]
    A_eq = [
        [1, 1, 1],
    ]
    b_eq = [4]

    t = Tableau(c, A_ub, b_ub, A_eq, b_eq, A_lb, b_lb)
    print(t)
