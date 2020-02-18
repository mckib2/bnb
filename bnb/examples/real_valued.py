'''Show how to use BNB to solve a simple real-valued problem.'''

import json
import numpy as np
from bnb import real_valued_bnb

if __name__ == '__main__':

    # Make an m-dimensional rectangle to search in
    Qinit = ((-10, 10), (-10, 10))

    # Define a function we'd like minimize
    f = lambda x: x[0]**2 + x[1]**2

    # Define an upper and lower bound
    lbound = lambda Q: .5*(
        np.min([q**2 for q in Q[0]]) +
        np.min([q**2 for q in Q[1]]))
    ubound = lambda Q: 2*(
        np.max([q**2 for q in Q[0]]) +
        np.max([q**2 for q in Q[1]]))
    
    # Define a branching rule
    def branch(Q):
        # Split along longest edge
        idx = np.argmax([d[1] - d[0] for d in Q])
        midpoint = (Q[idx][0] + Q[idx][1])/2

        Q1 = [d for d in Q]
        Q1[idx] = (Q1[idx][0], midpoint)

        # Add eps to make disjoint -- might get booted out of
        # loop if not!
        Q2 = [d for d in Q]
        Q2[idx] = (midpoint+np.finfo(float).eps, Q2[idx][1])

        return(Q1, Q2)
    
    res = real_valued_bnb(lbound, ubound, branch, Qinit)
    print(json.dumps(res, indent=4))
