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
        
if __name__ == '__main__':
    unittest.main()
