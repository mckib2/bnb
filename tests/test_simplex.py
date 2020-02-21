'''Make sure simplex algorithm is working as expected.

References
----------
.. [1] https://sites.math.washington.edu/~burke/crs/407/notes/section2.pdf
.. [2] https://personal.utdallas.edu/~scniu/OPRE-6201/documents/LP06-Simplex-Tableau.pdf
.. [3] http://people.brunel.ac.uk/~mastjjb/jeb/or/morelp.html
'''

import unittest
import numpy as np
from bnb import simplex

class TestSimplex(unittest.TestCase):
    '''Run simplex algorithm through the ringer.'''

    def test_prob1(self):
        '''Example prob (2.1) from [1]_.'''
        c = [5, 4, 3]
        A = [
            [2, 3, 1],
            [4, 1, 2],
            [3, 4, 2],
        ]
        b = [5, 11, 8]
        res = simplex(c, A, b)

        x = np.array([2, 0, 1])
        s = np.array([0, 1, 0])
        z = 13
        self.assertTrue(np.allclose(x, res['x']))
        self.assertTrue(np.allclose(s, res['slack']))
        self.assertTrue(np.allclose(z, res['fopt']))

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

        x = np.array([4, 0, 1])
        fopt = 8
        self.assertTrue(np.allclose(x, res['x']))
        self.assertTrue(np.allclose(fopt, res['fopt']))

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

        x = np.array([3/2, 1])
        s = np.array([0, 11/2, 3, 0])
        fopt = 9
        self.assertTrue(np.allclose(x, res['x']))
        self.assertTrue(np.allclose(s, res['slack']))
        self.assertTrue(np.allclose(fopt, res['fopt']))

    def test_1997_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [1, 1]
        A= [
            [50, 24],
            [30, 33],
            [-1, 0],
            [0, -1],
        ]
        b = [40*60, 35*60, -45, -5]
        res = simplex(c, A, b)

        x = np.array([45, 6.25])
        fopt = 1.25
        self.assertTrue(np.allclose(x, res['x']))
        self.assertTrue(np.allclose(fopt, res['fopt'] - 50))

    def test_1995_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [13, 5]
        A = [
            [15, 7],
            [25, 45],
            [1, 0],
            [0, 1],
        ]
        b = [20*60, 15*60, 37, 14]
        res = simplex(c, A, b)

        x = np.array([36, 0])
        fopt = 343
        self.assertTrue(np.allclose(x, res['x']))
        self.assertTrue(np.allclose(fopt, res['fopt'] - 125))

    def test_1994_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [20 - 10*13/60 - 2*20/60, 30 - 10*19/60 - 2*29/60]
        A = [
            [13, 19],
            [20, 29],
            [-1, 0]
        ]
        b = [40*60, 35*60, -10]
        res = simplex(c, A, b)

        x = np.array([10, 65.52])
        fopt = 1866.5
        self.assertTrue(np.allclose(x, res['x']))
        self.assertTrue(np.allclose(fopt, res['fopt']))

    def test_1992_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [3, 5]
        A = [
            [12, 25],
            [.4, -1],
        ]
        b = [30*60, 0]
        res = simplex(c, A, b)

        x = np.array([81.8, 32.7])
        fopt = 408.9
        xres = np.around(res['x'], decimals=1)
        self.assertTrue(np.allclose(x, xres))
        self.assertTrue(np.allclose(fopt, np.array(c).T @ xres))

    def test_1988_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [-10, -11]
        A = [
            [-1, -1],
            [1, -1],
            [-7, -12],
        ]
        b = [-11, 5, -35]
        res = simplex(c, A, b)

        x = np.array([8, 3])
        fopt = 113
        self.assertTrue(np.allclose(x, res['x']))
        self.assertTrue(np.allclose(fopt, -1*res['fopt']))

    def test_1987_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [5, 6]
        A = [
            [1, 1],
            [-1, 1],
            [5, 4],
        ]
        b = [10, -3, 35]
        res = simplex(c, A, b)

        x = np.array([47/9, 20/9])
        fopt = 355/9
        self.assertTrue(np.allclose(x, res['x']))
        self.assertTrue(np.allclose(fopt, res['fopt']))

    def test_1986_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [30, 10]
        A = [
            [6, 3],
            [3, -1], # Works without this constraint
            [1, 1/4],
        ]
        b = [40, 0, 4]
        res = simplex(c, A, b)
        print(res)
        
if __name__ == '__main__':
    unittest.main()
