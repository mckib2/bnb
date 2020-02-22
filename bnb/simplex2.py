'''Try simplex again.'''

from tabulate import tabulate

class Tableau:
    '''Tableau class.'''

    def __init__(self, c, A_ub=[], b_ub=[], A_eq=[], b_eq=[], A_lb=[], b_lb=[]):

        self.c = c
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_lb = A_lb
        self.b_lb = b_lb
        self.A_eq = A_eq
        self.b_eq = b_eq

        # Find any negative entries in b and make positive
        remove_idx = set()
        for ii, val in enumerate(self.b_ub):
            if val < 0:
                # Multiply the leq inequality by -1 turning it into geq
                self.A_lb.append([-1*v0 for v0 in self.A_ub[ii]])
                self.b_lb.append(-1*val)
                remove_idx.add(ii)
        self.A_ub = [row for ii, row in enumerate(self.A_ub) if ii not in remove_idx]
        self.b_ub = [val for ii, val in enumerate(self.b_ub) if ii not in remove_idx]

        remove_idx = set()
        for ii,  val in enumerate(self.b_lb):
            if val < 0:
                # turns into leq inequality
                self.A_ub.append([-1*v0 for v0 in self.A_lb[ii]])
                self.b_ub.append([-1*val])
                remove_idx.add(ii)
        self.A_lb = [row for ii, row in enumerate(self.A_lb) if ii not in remove_idx]
        self.b_lb = [val for ii, val in enumerate(self.b_lb) if ii not in remove_idx]

        for ii, val in enumerate(self.b_eq):
            if val < 0:
                # Remains an equality
                self.A_eq[ii] = [-1*v0 for v0 in self.A_eq[ii]]
                self.b_eq[ii] *= -1
        
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

        # Determine if we need one or two phase solution
        self.two_phase = self.num_artificial > 0
        
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

        # Solve phase 1 if we need to
        if self.two_phase:

            # Change the problem to minimize sum of artificial variables
            phase1_c = [''] + [0]*(self.num_vars+self.num_slack+self.num_surplus) + [1]*(self.num_artificial)

            # Add the new objective row to the top -- we now have two objective functions we're tracking!
            self.tableau.insert(self.nhdr_row, phase1_c + [0])
            self.nobjective_row += 1
            self.num_rows += 1

            # Since we have no negatives in the objective row, use artificial variable rows to
            # eliminate entries in the objective column
            offset = self.nbasic_var_labels + self.num_vars + self.num_slack + self.num_surplus
            for ii in range(self.num_artificial):

                # Find current artificial variable row
                for jj in range(self.nhdr_row+self.nobjective_row, self.num_rows):
                    if self.tableau[jj][offset+ii] == 1: # should be 1
                        row_idx = jj
                        break

                # Update basic variable labels
                self.tableau[row_idx][0] = self.tableau[0][offset+ii]

                # Remove entry from objective
                row_vals = self.tableau[row_idx]
                for jj in range(self.nbasic_var_labels, self.num_cols):
                    self.tableau[self.nhdr_row][jj] -= row_vals[jj]

            # Solve the phase 1 problem
            self.solve()

            # If we get an objective value other than 0, problem is infeasible
            if self.get_obj() != 0:
                raise ValueError('There is no solution!')

            # Now solve phase 2 by removing artificial variable columns and phase 1 objective row
            del self.tableau[self.nhdr_row]
            self.nobjective_row -= 1
            self.num_rows -= 1

            offset = self.nbasic_var_labels+self.num_vars+self.num_slack+self.num_surplus
            for ii in range(self.num_rows):
                del self.tableau[ii][offset:offset+self.num_artificial]
            self.num_cols -= self.num_artificial
            self.num_artificial = 0

            # Can solve the phase 2 problem as normal

        self.solve()
        
    def solve(self):
        '''Pivot till you can't pivot no more.'''
        # Start pivoting
        done = False
        while not done:
            done = self.pivot()

    def get_obj(self):
        '''Find the current objective function value.'''
        return self.tableau[self.nhdr_row][-1]
            
    def pivot(self):
        '''Perform a pivot.'''

        # Check out starting state
        print(self)
        
        # Check exit conditions
        obj_row = self.tableau[self.nhdr_row][self.nbasic_var_labels:-self.nrhs_col]
        is_neg = [val < 0 for val in obj_row]
        if not any(is_neg):
            return True # we're done!

        # Choose incoming variable from strictly negative objective cols
        min_index, _min_value = min(enumerate(zip(obj_row, is_neg)), key=lambda x0: x0[1][0]*x0[1][1])
        pivot_col = min_index + self.nbasic_var_labels

        # Choose the outgoing variable using ratio test
        ratios = [
            (self.tableau[ii][-1]/self.tableau[ii][pivot_col], ii) for ii in range(
                self.nhdr_row+self.nobjective_row, self.num_rows) if self.tableau[ii][pivot_col] > 0]
        if len(ratios) == 0:
            raise ValueError('Unbounded solution!')
        _min_ratio, pivot_row = min(ratios, key=lambda x0: x0[0])

        # Update the basic variable labels with the incoming variable
        self.tableau[pivot_row][0] = self.tableau[0][pivot_col]
        
        # Pivot! Pivot row divided by pivot value
        pivot_val = self.tableau[pivot_row][pivot_col]
        for jj in range(self.nbasic_var_labels, self.num_cols):
            self.tableau[pivot_row][jj] /= pivot_val
        pivot_row_vals = self.tableau[pivot_row]

        # Now use pivot row to remove remaining entries in the pivot col
        pivot_col_vals = [(ii, self.tableau[ii][pivot_col]) for ii in range(self.nhdr_row, self.num_rows) if ii != pivot_row and abs(self.tableau[ii][pivot_col]) > 0]
        for ii, val in pivot_col_vals:
            for jj in range(self.nbasic_var_labels, len(self.tableau[ii])):
                self.tableau[ii][jj] -= pivot_row_vals[jj]*val

        # Tell caller we didn't detect exiting condition and pivoted
        return False
    
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
            self.tableau[ii+row_offset][-1] = self.b_ub[ii]

        # eq constraints
        for ii, row in enumerate(self.A_eq):
            # Include artificial variable on this row
            for jj, val in enumerate(row):
                self.tableau[self.nA_ub+ii+row_offset][jj+col_offset] = val
            self.tableau[self.nA_ub+ii+row_offset][artificial_idx] = 1
            #self.tableau[self.nhdr_row][artificial_idx] = -1*self.bigM # if bigM
            artificial_idx += 1

            # Also get RHS values
            self.tableau[self.nA_ub+ii+row_offset][-1] = self.b_eq[ii]

        # geq constraints
        for ii, row in enumerate(self.A_lb):
            # Include suprplus and artificial variables on this row
            for jj, val in enumerate(row):
                self.tableau[self.nA_ub+self.nA_eq+ii+row_offset][jj+col_offset] = val
            self.tableau[self.nA_ub+self.nA_eq+ii+row_offset][artificial_idx] = 1
            #self.tableau[self.nhdr_row][artificial_idx] = -1*self.bigM # if bigM
            self.tableau[self.nA_ub+self.nA_eq+ii+row_offset][surplus_idx] = -1
            artificial_idx += 1
            surplus_idx += 1

            # Also get RHS values
            self.tableau[self.nA_ub+self.nA_eq+ii+row_offset][-1] = self.b_lb[ii]
                
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

    def solution(self):
        '''Return all useful info about solution.'''

        fopt = self.get_obj()

        # Extract basic variables from tableau
        basic_vars = [r[0] for r in self.tableau[self.nhdr_row+self.nobjective_row:]]
        x = [0]*self.num_vars
        s = [0]*(self.num_slack+self.num_surplus) # consider surplus variables as slack
        for ii, v in enumerate(basic_vars):
            if 'x' in v:
                idx = int(v[1:])
                x[idx] = self.tableau[self.nhdr_row+self.nobjective_row+ii][-1]
            elif 'sl' in v:
                idx = int(v[2:])
                s[idx] = self.tableau[self.nhdr_row+self.nobjective_row+ii][-1]
            elif 'su' in v:
                idx = int(v[2:])
                s[idx+self.num_slack] = self.tableau[self.nhdr_row+self.nobjective_row+ii][-1]
            else:
                raise ValueError('Got unexpected basic variable label: %s' % v)

        return {
            'fopt': fopt,
            'x': x,
            'slack': s,
        }
            
    def __repr__(self):
        return '\n' + tabulate(
            self.tableau,
            headers="firstrow",
            tablefmt='orgtbl',
            floatfmt=".1f") + '\n'

def simplex(*args, **kwargs):
    '''Simplex pass to Tableau class.'''

    t = Tableau(*args, **kwargs)
    return t.solution()
        
        
if __name__ == '__main__':

    import json
    
    # https://faculty.math.illinois.edu/~mlavrov/docs/482-fall-2019/lecture6.pdf
    c = [-1, -1, 1, 1]
    A_eq = [
        [1, 2, 1, 1],
        [2, -1, -1, -3],
    ]
    b_eq = [7, -1]
    res = simplex(c, A_eq=A_eq, b_eq=b_eq)
    print(json.dumps(res, indent=4))

    # http://web.mit.edu/15.053/www/AMP-Chapter-02.pdf
    c = [6, 14, 13]
    A_ub = [
        [1/2, 2, 1],
        [1, 2, 4],
    ]
    b_ub = [24, 60]
    res = simplex(c, A_ub=A_ub, b_ub=b_ub)
    print(json.dumps(res, indent=4))

    # https://nptel.ac.in/content/storage2/courses/105108127/pdf/Module_3/M3L4_LN.pdf
    c = [3, 5]
    A_ub = [
        [0, 1],
    ]
    b_ub = [6]
    A_eq = [
        [3, 2],
    ]
    b_eq = [18]
    A_lb = [
        [3, 5],
    ]
    b_lb = [2]

    res = simplex(
        c=c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        A_lb=A_lb, b_lb=b_lb)
    print(json.dumps(res, indent=4))

    # https://nptel.ac.in/content/storage2/courses/105108127/pdf/Module_3/M3L4_LN.pdf
    c = [3, 2]
    A_ub = [
        [0, 1],
    ]
    b_ub = [6]
    A_eq = [
        [3, 2],
    ]
    b_eq = [18]
    A_lb = [
        [1, 1],
    ]
    b_lb = [2]
    res = simplex(
        c=c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        A_lb=A_lb, b_lb=b_lb)
    print(json.dumps(res, indent=4))
