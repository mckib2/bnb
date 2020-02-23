'''Make sure simplex algorithm is working as expected.

References
----------
.. [1] https://sites.math.washington.edu/~burke/crs/407/notes/section2.pdf
.. [2] https://personal.utdallas.edu/~scniu/OPRE-6201/documents/LP06-Simplex-Tableau.pdf
.. [3] http://people.brunel.ac.uk/~mastjjb/jeb/or/morelp.html
.. [4] https://en.wikipedia.org/wiki/Simplex_algorithm
.. [5] http://www.math.wsu.edu/faculty/dzhang/201/Guideline%20to%20Simplex%20Method.pdf
.. [6] https://faculty.math.illinois.edu/~mlavrov/docs/482-fall-2019/lecture6.pdf
.. [7] http://web.mit.edu/15.053/www/AMP-Chapter-02.pdf
.. [8] https://nptel.ac.in/content/storage2/courses/105108127/pdf/Module_3/M3L4_LN.pdf
.. [9] https://faculty.math.illinois.edu/~mlavrov/docs/482-fall-2019/lecture13.pdf
.. [10] https://ocw.ehu.eus/pluginfile.php/8161/mod_resource/content/1/3_Duality.pdf
.. [11] https://web.stanford.edu/~ashishg/msande111/notes/chapter4.pdf
'''

import unittest
import numpy as np
from bnb import simplex

class TestSimplex(unittest.TestCase):
    '''Run simplex algorithm through the ringer.'''

    def test_prob1(self):
        '''Example prob (2.1) from [1]_.'''
        c = [5, 4, 3]
        A_ub = [
            [2, 3, 1],
            [4, 1, 2],
            [3, 4, 2],
        ]
        b_ub = [5, 11, 8]
        res = simplex(c, A_ub=A_ub, b_ub=b_ub)
        
        x = [2, 0, 1]
        s = [0, 1, 0]
        z = 13
        self.assertEqual(x, res['x'])
        self.assertEqual(s, res['slack'])
        self.assertEqual(z, res['fopt'])

    def test_prob2(self):
        '''Example prob from bottom of pg 25 in [1]_.'''
        c = [3, 2, -4]
        A = [
            [1, 4, 0],
            [2, 4, -2],
            [1, 1, -2],
        ]
        b = [5, 6, 2]
        res = simplex(c, A, b)

        x = [4, 0, 1]
        fopt = 8
        self.assertEqual(x, res['x'])
        self.assertEqual(fopt, res['fopt'])

    def test_prob3(self):
        '''Example prob from [2]_.'''
        c = [4, 3]
        A = [
            [2, 3],
            [-3, 2],
            [0, 2],
            [2, 1],
        ]
        b = [6, 3, 5, 4]
        res = simplex(c, A, b)

        x = [3/2, 1]
        s = [0, 11/2, 3, 0]
        fopt = 9
        self.assertEqual(x, res['x'])
        self.assertEqual(s, res['slack'])
        self.assertEqual(fopt, res['fopt'])

    def test_1997_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [1, 1]
        A_ub = [
            [50, 24],
            [30, 33],
        ]
        b_ub = [40*60, 35*60]
        A_lb = [
            [1, 0],
            [0, 1],
        ]
        b_lb = [45, 5]
        
        res = simplex(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_lb=A_lb, b_lb=b_lb,
        )

        x = [45, 6.25]
        fopt = 1.25
        self.assertEqual(x, res['x'])
        self.assertEqual(fopt, res['fopt'] - 50)

    def test_1995_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [13, 5]
        A_ub = [
            [15, 7],
            [25, 45],
            [1, 0],
            [0, 1],
        ]
        b_ub = [20*60, 15*60, 37, 14]
        res = simplex(c, A_ub=A_ub, b_ub=b_ub)

        x = [36, 0]
        fopt = 343
        self.assertEqual(x, res['x'])
        self.assertEqual(fopt, res['fopt'] - 125)

    def test_1994_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [20 - 10*13/60 - 2*20/60, 30 - 10*19/60 - 2*29/60]
        A_ub = [
            [13, 19],
            [20, 29],
        ]
        b_ub = [40*60, 35*60]
        A_lb = [
            [1, 0],
        ]
        b_lb = [10]
        res = simplex(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_lb=A_lb, b_lb=b_lb,
        )

        # Need to get rounded to match given exam solution
        xres = [round(x0, 2) for x0 in res['x']]
        fres = round(sum([c0*x0 for c0, x0 in zip(c, xres)]), 1)

        x = [10, 65.52]
        fopt = 1866.5
        self.assertEqual(x, xres)
        self.assertEqual(fopt, fres)

    def test_1992_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [3, 5]
        A_ub = [
            [12, 25],
            [2/5, -1],
        ]
        b_ub = [30*60, 0]
        res = simplex(c, A_ub=A_ub, b_ub=b_ub)

        # Need to get rounded to match given exam solutions
        xres = [round(x0, 1) for x0 in res['x']]
        fres = round(sum([c0*x0 for c0, x0 in zip(c, xres)]), 1)

        x = [81.8, 32.7]
        fopt = 408.9
        self.assertEqual(x, xres)
        self.assertEqual(fopt, fres)

    def test_1988_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [-4, -5, -6]
        A_ub = [
            [1, -1, 0],
        ]
        b_ub = [5]
        A_eq = [
            [-1, -1, 1],
        ]
        b_eq = [0]
        A_lb = [
            [1, 1, 0],
            [7, 12, 0],
        ]
        b_lb = [11, 35]
        res = simplex(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            A_lb=A_lb, b_lb=b_lb
        )
        xres = [round(x0, 6) for x0 in res['x']]

        x = [8, 3, 11]
        fopt = 113
        self.assertEqual(x, xres)
        self.assertEqual(fopt, -1*res['fopt'])

    def test_1987_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [5, 6]
        A_ub = [
            [1, 1],
            [5, 4],
        ]
        b_ub = [10, 35]
        A_lb = [
            [1, -1],
        ]
        b_lb = [3]
        res = simplex(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_lb=A_lb, b_lb=b_lb,
        )

        x = [47/9, 20/9]
        fopt = 355/9
        self.assertEqual(x, res['x'])
        self.assertEqual(fopt, res['fopt'])

    def test_1986_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [30, 10]
        A = [
            [6, 3],
            [3, -1],
            [1, 1/4],
        ]
        b = [40, 0, 4]
        res = simplex(c, A, b)
        dig = 6
        xres = [round(x0, dig) for x0 in res['x']]
        fres = round(res['fopt'], dig)

        x = [round(4/3, dig), round(64/6, dig)]
        fopt = round(146.666666666666666666666666, dig)
        self.assertEqual(x, xres)
        self.assertEqual(fopt, fres)

    def test_wikipedia_example(self):
        '''Example from [4]_.'''
        c = [2, 3, 4]
        A = [
            [3, 2, 1],
            [2, 5, 3],
        ]
        b = [10, 15]
        res = simplex(c, A, b)

        x = [0, 0, 5]
        s = [5, 0]
        fopt = -20
        self.assertEqual(x, res['x'])
        self.assertEqual(s, res['slack'])
        self.assertEqual(fopt, -1*res['fopt'])

    def test_wsu_example(self):
        '''Example from WSU guide [5]_.'''
        c = [3, 1]
        A = [
            [2, 1],
            [2, 3],
        ]
        b = [8, 12]
        res = simplex(c, A, b)

        x = [4, 0]
        s = [0, 4]
        fopt = 12
        self.assertEqual(x, res['x'])
        self.assertEqual(s, res['slack'])
        self.assertEqual(fopt, res['fopt'])
    
    def test_two_phase_example(self):
        '''Two phase simplex example from [6]_.'''
        c = [-1, -1, 1, 1]
        A_eq = [
            [1, 2, 1, 1],
            [2, -1, -1, -3],
        ]
        b_eq = [7, -1]
        res = simplex(c, A_eq=A_eq, b_eq=b_eq)

        x = [2, 0, 5, 0]
        fopt = -3
        self.assertEqual(x, res['x'])
        self.assertEqual(fopt, -1*res['fopt'])

    def test_simple_example(self):
        '''Simple example on pg 50 from [7]_.'''
        c = [6, 14, 13]
        A_ub = [
            [1/2, 2, 1],
            [1, 2, 4],
        ]
        b_ub = [24, 60]
        res = simplex(c, A_ub=A_ub, b_ub=b_ub)

        # Round to match solution
        xres = [round(x0, 6) for x0 in res['x']]
        
        x = [36, 0, 6]
        fopt = 294
        self.assertEqual(x, xres)
        self.assertEqual(fopt, res['fopt'])

    def test_compare_two_phase_to_bigm(self):
        '''Greater than and equality constraints from pg 1 of [8]_.'''
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

        # Round to match given solution
        xres = [round(x0, 6) for x0 in res['x']]
        
        x = [2, 6]
        fopt = 36
        self.assertEqual(x, xres)
        self.assertEqual(fopt, res['fopt'])        

    def test_test_compare_two_phase_to_bigm2(self):
        '''Greater than and equality constraints from pg 5 of [8]_.'''
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

        x = [6, 0]
        fopt = 18
        self.assertEqual(x, res['x'])
        self.assertEqual(fopt, res['fopt'])

    def test_symmetric_dual(self):
        '''Extract dual solution from tableau when prob has A_ub only.
        
        Uses example prob from pg 1 of [9]_.
        '''
        c = [2, 3]
        A_ub = [
            [-1, 1],
            [1, -2],
            [3, 4],
        ]
        b_ub = [3, 2, 26]
        res = simplex(c=c, A_ub=A_ub, b_ub=b_ub)

        # Make sure primal solution is correct
        x = [2, 5]
        s = [0, 10, 0]
        fopt = 19
        self.assertEqual(x, res['x'])
        self.assertEqual(s, res['slack'])
        self.assertEqual(fopt, res['fopt'])
        
        # Round to resolve small differences
        dig = 8
        ures = [round(u0, dig) for u0 in res['dual']]
        u = [round(1/7, dig), 0, round(5/7, dig)]
        self.assertEqual(u, ures)

    def test_symmetric_dual2(self):
        '''Try another dual from pg 104 of [10]_.'''
        c = [3, 1, -2]
        A_ub = [
            [1, 2, 1],
            [2, -1, 3],
        ]
        b_ub = [5, 4]
        res = simplex(c, A_ub=A_ub, b_ub=b_ub)

        # Make sure primal is correct
        x = [13/5, 6/5, 0]
        s = [0, 0]
        fopt = 9
        self.assertEqual(x, res['x'])
        self.assertEqual(s, res['slack'])
        self.assertEqual(fopt, res['fopt'])

        # Now make sure dual is correct
        u = [1, 1]
        self.assertEqual(u, res['dual'])

    def test_dual_with_A_ub_and_A_lb(self):
        '''Try a dual with multiple kinds of constraints from [10]_.'''
        c = [-1, -2]
        A_ub = [
            [4, 3],
        ]
        b_ub = [12]
        A_lb = [
            [1, 3],
            [2, 1],
        ]
        b_lb = [6, 4]
        res = simplex(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_lb=A_lb, b_lb=b_lb,
        )

        # Make sure primal checks out
        x = [6/5, 8/5]
        s = [12/5, 0, 0]
        fopt = 22/5
        self.assertEqual(x, res['x'])
        self.assertEqual(s, [round(s0, 8) for s0 in res['slack']])
        self.assertEqual(fopt, -1*res['fopt'])

        # Check dual
        u = [0, 3/5, 1/5]
        self.assertEqual(u, res['dual'])

    def test_primal_dual_equality(self):
        '''Example from stanford [11]_.'''
        c = [3, 2.5]
        A_ub = [
            [4.44, 0],
            [0, 6.67],
            [4, 2.86],
            [3, 6],
        ]
        b_ub = [100, 100, 100, 100]
        res = simplex(c=c, A_ub=A_ub, b_ub=b_ub)
        print(res)

        # Round do precision given in ref
        xres = [round(x0, 1) for x0 in res['x']]
        fres = round(res['fopt'], 1)

        # Make sure primal is good
        x = [20.4, 6.5]
        fopt = 77.3
        self.assertEqual(x, xres)
        self.assertEqual(fopt, fres)

        # Check dual while we're at it
        ures = [round(u0, 3) for u0 in res['dual']]
        u = [0, 0, .681, .092]
        self.assertEqual(u, ures)

        # Solve the dual directly
        dres = simplex(
            c=[-1*b0 for b0 in b_ub],
            A_lb=[list(i) for i in zip(*A_ub)],
            b_lb=c,
        )

        # Make sure both solutions agree
        self.assertEqual(
            [round(u0, 9) for u0 in res['dual']],
            [round(x0, 9) for x0 in dres['x']]
        )
        self.assertEqual(
            [round(x0, 9) for x0 in res['x']],
            [round(u0, 9) for u0 in dres['dual']]
        )

if __name__ == '__main__':
    unittest.main()
