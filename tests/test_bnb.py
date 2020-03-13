'''Test problems.

References
----------
.. [1] http://web.tecnico.ulisboa.pt/mcasquilho/compute/_linpro/TaylorB_module_c.pdf
.. [2] https://en.wikipedia.org/wiki/Integer_programming#Example
.. [3] http://web.mit.edu/15.053/www/AMP-Chapter-09.pdf
.. [4] http://people.brunel.ac.uk/~mastjjb/jeb/or/ip.html#capbud
.. [5] http://www.math.clemson.edu/~mjs/courses/mthsc.440/integer.pdf
.. [6] http://web.pdx.edu/~stipakb/download/PA557/ReadingsPA557sec6.pdf
.. [7] https://www.isical.ac.in/~arijit/courses/autumn2016/ILP-Lecture-1.pdf
.. [8] https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Mixed_Integer_Programming.pdf
.. [9] https://www.cs.upc.edu/~erodri/webpage/cps/theory/lp/milp/slides.pdf
'''

import unittest

import numpy as np
from bnb import intprog

class TestProblems(unittest.TestCase):
    '''Validation integer programming problems.'''

    def test_prob1(self):
        '''First problem from [1]_.'''
        c = [100, 150]
        A = [
            [8000, 4000],
            [15, 30],
        ]
        b = [40000, 200]
        res = intprog(c, A, b)
        self.assertEqual(res['x'].tolist(), [1, 6])
        self.assertEqual(res['fun'], 1000)

    def test_prob2(self):
        '''Second problem from [1]_.'''
        c = [300, 90, 400, 150]
        A = [
            [35000, 10000, 25000, 90000],
            [4, 2, 7, 3],
            [1, 1, 0, 0],
        ]
        b = [120000, 12, 1]
        res = intprog(c, A, b)
        self.assertEqual(res['x'].tolist(), [1, 0, 1, 0])
        self.assertEqual(res['fun'], 700)

    def test_prob3(self):
        '''Example problem from Wikipedia.'''
        c = [0, 1]
        A = [
            [-1, 1],
            [3, 2],
            [2, 3],
        ]
        b = [1, 12, 12]
        res = intprog(c, A, b)
        self.assertTrue(res['x'].tolist() == [1, 2] or res['x'].tolist() == [2, 2])
        self.assertEqual(res['fun'], 2)

    def test_prob4(self):
        '''Problem starting on page 287 of [3]_.'''
        c = [5, 8]
        A = [
            [1, 1],
            [5, 9],
        ]
        b = [6, 45]
        res = intprog(c, A, b)
        self.assertEqual(res['x'].tolist(), [0, 5])
        self.assertEqual(res['fun'], 40)

    def test_prob5(self):
        '''Capital budgeting extension from [4]_.'''
        c = [.2, .3, .5, .1]
        A = [
            [.5, 1, 1.5, .1],
            [.3, .8, 1.5, .4],
            [.2, .2, .3, .1],
        ]
        b = [3.1, 2.5, .4]
        res = intprog(c, A, b, binary=True)
        self.assertEqual(res['x'].tolist(), [0, 0, 1, 1])
        self.assertEqual(res['fun'], .6)

    def test_prob6(self):
        '''Example problem from page 4 of [5]_.'''
        c = [8, 11, 6, 4]
        A = [
            [5, 7, 4, 3],
        ]
        b = [14]
        res = intprog(c, A, b, binary=True)
        self.assertEqual(res['x'].tolist(), [0, 1, 1, 1])
        self.assertEqual(res['fun'], 21)

    def test_prob7(self):
        '''Example problem from page 1 of [6]_.'''
        c = [12, 40, 15, 20, 10]
        A = [
            [.2, .5, .2, .2, .3],
        ]
        b = [.8]
        res = intprog(c, A, b, binary=True)
        self.assertEqual(res['x'].tolist(), [0, 1, 0, 1, 0])
        self.assertEqual(res['fun'], 60)

    def test_prob8(self):
        '''Example problem from page 9 of [7]_.'''
        c = [8, 11, 6, 4]
        A = [
            [5, 7, 4, 3],
            [7, 0, 2, 1],
        ]
        b = [14, 10]
        res = intprog(c, A, b, binary=True)
        self.assertEqual(res['x'].tolist(), [0, 1, 1, 1])
        self.assertEqual(res['fun'], 21)

    def test_prob10(self):
        '''Example problem from page 2 of [8]_'''
        # The problem is stated incorrectly for the solution they
        # give in the text, but if you look at the Dataset table on
        # page 2 they type it in different than stated above
        c = [1, 1, 2, -2]
        A_ub = [
            [1, 0, 2, 0],
            [0, 2, 0, -8],
            [0, -1, 2, -1],
        ]
        b_ub = [700, 0, -1]
        A_eq = [
            [1, 1, 1, 1],
        ]
        b_eq = [10]
        bnds = [(0, 10)]*4
        res = intprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bnds)
        self.assertTrue(np.allclose(res['x'], [3, 4, 2, 1]))
        self.assertEqual(res['fun'], 9)

    def test_prob11(self):
        '''Example problem from pg 5 of [9]_.'''
        c = [1, 1]
        A = [
            [2, -2],
            [-8, 10],
        ]
        b = [-1, 13]
        res = intprog(c, A, b)
        self.assertEqual(res['x'].tolist(), [1, 2])
        self.assertEqual(res['fun'], 3)

if __name__ == '__main__':
    unittest.main()
