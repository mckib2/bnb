# distutils: language=c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "simplex.hpp" namespace "simplex" nogil:
    cdef cppclass Simplex[T]:
        Simplex()
        Simplex(const vector[T] c_,
		const vector[vector[T]] A_ub_, const vector[T] b_ub_,
		const vector[vector[T]] A_eq_, const vector[T] b_eq_,
		const vector[vector[T]] A_lb_, const vector[T] b_lb_,
		const bool solve_dual_)
        void solve()
        void show()

#cdef void csimplex[T](const vector[T] c_,
#	              const vector[vector[T]] A_ub_, const vector[T] b_ub_,
#	              const vector[vector[T]] A_eq_, const vector[T] b_eq_,
#	              const vector[vector[T]] A_lb_, const vector[T] b_lb_,
#	              const bool solve_dual_):
#    cdef Simplex[T] * S = new Simplex[T](
#        c_, A_ub_, b_ub_, A_eq_, b_eq_, A_lb_, b_lb_, solve_dual_)
#    S.solve()
#    S.show()
#
#    del S

def simplex(c, A_ub=[], b_ub=[], A_eq=[], b_eq=[], A_lb=[], b_lb=[], const bool solve_dual=False):

    # Just try double for now
    cdef vector[double] c_ = c
    cdef vector[vector[double]] A_ub_ = A_ub
    cdef vector[vector[double]] A_eq_ = A_eq
    cdef vector[vector[double]] A_lb_ = A_lb
    cdef vector[double] b_ub_ = b_ub
    cdef vector[double] b_eq_ = b_eq
    cdef vector[double] b_lb_ = b_lb
    
    #csimplex[double](c_, A_ub_, b_ub_, A_eq_, b_eq_, A_lb_, b_lb_, solve_dual)
    cdef Simplex[double] * S = new Simplex[double](
        c_, A_ub_, b_ub_, A_eq_, b_eq_, A_lb_, b_lb_, solve_dual)
    S.solve()
    S.show()
    
    del S
