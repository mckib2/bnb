'''Test problems.

References
----------
.. [1] http://web.tecnico.ulisboa.pt/mcasquilho/compute/_linpro/TaylorB_module_c.pdf
.. [2] https://en.wikipedia.org/wiki/Integer_programming#Example
.. [3] http://web.mit.edu/15.053/www/AMP-Chapter-09.pdf
.. [4] http://people.brunel.ac.uk/~mastjjb/jeb/or/ip.html#capbud
.. [5] http://www.math.clemson.edu/~mjs/courses/mthsc.440/integer.pdf
'''

import unittest

from bnb import intprog

class TestProblems(unittest.TestCase):

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
        bnds = [(0, 1)]*4
        res = intprog(c, A, b, bounds=bnds)
        self.assertEqual(res['x'].tolist(), [0, 0, 1, 1])
        self.assertEqual(res['fun'], .6)

    def test_prob6(self):
        '''Example problem from page 4 of [5]_.'''
        c = [8, 11, 6, 4]
        A = [
            [5, 7, 4, 3],
        ]
        b = [14]
        bnds = [(0, 1)]*4
        res = intprog(c, A, b, bounds=bnds)
        self.assertEqual(res['x'].tolist(), [0, 1, 1, 1])
        self.assertEqual(res['fun'], 21)

if __name__ == '__main__':
    unittest.main()
