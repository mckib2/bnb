#include <vector>
#include <stack>
#include <algorithm>
#include <functional>
#include <assert.h>
#include "simplex.hpp"

namespace simplex {

  template<class T>
  Simplex<T>::Simplex(const std::vector<T> c,
		      const std::vector<std::vector<T> > A_ub, const std::vector<T> b_ub,
		      const std::vector<std::vector<T> > A_eq, const std::vector<T> b_eq,
		      const std::vector<std::vector<T> > A_lb, const std::vector<T> b_lb,
		      const bool solve_dual)
    : solve_dual(false)
  {
    
    // Store copy of coefficients and constraints
    this->c = std::vector<T>(c);
    this->A_ub = std::vector<std::vector<T> >(A_ub);
    this->A_eq = std::vector<std::vector<T> >(A_eq);
    this->A_lb = std::vector<std::vector<T> >(A_lb);
    this->b_ub = std::vector<T>(b_ub);
    this->b_eq = std::vector<T>(b_eq);
    this->b_lb = std::vector<T>(b_lb);

    // Go ahead and make RHS all non-negative. No longer trust function
    // arguments to have correct constraints after this! Use Simplex::A_ub, etc.
    this->make_rhs_nonnegative();
    
    this->num_vars = this->c.size();
    this->num_slack = this->b_ub.size();
    this->num_surplus = this->b_lb.size();
    this->num_artificial = this->num_surplus + this->b_eq.size();

    // Table specific stuff
    this->num_obj_rows = 1;
    
    // See if we need to run two phases
    this->two_phase = this->num_artificial > 0;
    
    // Initialize the tableau to all 0s
    this->allocate_tableau();

    // Make headers for the tableau
    this->make_hdrs();

    // Maybe fill in tableau before FBS?
    
    // Get an intitial feasible solution
    this->initial_fbs();
  }

  template<class T>
  void Simplex<T>::initial_fbs() {
    
  }
  
  template<class T>
  void Simplex<T>::make_hdrs() {

    // Column tracking basic variables is first
    this->hdrs.push_back("basic");
        
    // Start with variables that appear in objective function,
    // then do slack, surplus, and artificial
    for (auto ii = 0; ii < this->num_vars; ++ii) {
      this->hdrs.push_back("x" + std::string(ii));
    }
    for (auto ii = 0; ii < this->num_slack; ++ii) {
      this->hdrs.push_back("sl" + std::string(ii));
    }
    for (auto ii = 0; ii < this->num_surplus; ++ii) {
      this->hdrs.push_back("su" + std::string(ii));
    }
    for (auto ii = 0; ii < this->num_artificial; ++ii) {
      this->hdrs.push_back("a" + std::string(ii));
    }
    // End with RHS
    this->hdrs.push_back("RHS");

    // Did we get all of them?
    assert(this->hdrs.size() == this->num_cols());
  }
  
  template<class T>
  void Simplex<T>::make_rhs_nonnegative() {
    // Find any negative entries in b and make positive
    auto remove_idx = std::stack<std::size_t>();
    std::size_t ii = 0;
    for (auto val : this->b_ub) {
      if (val < 0) {
	// Multiply the leq inequality by -1 turning it into geq
	auto r = std::vector<T>(this->A_ub[ii]);
	std::transform(r.cbegin(),
		       r.cend(),
		       r.begin(),
		       std::negate<T>());
	this->A_lb.push_back(r);
	this->b_lb.push_back(-1*val);
	remove_idx.push(ii);
      }
      ii += 1;
    }
    // Remove entries marked for deletion;
    // use stack so indices aren't messed up!
    while (!remove_idx.empty()) {
      auto row_idx = remove_idx.top();
      this->A_ub[row_idx] = this->A_ub.back();
      this->A_ub.pop_back();
      this->b_ub[row_idx] = this->b_ub.back();
      this->b_ub.pop_back();
      remove_idx.pop();
    }

    // Reset index counter and do the same for geq constraints
    ii = 0;
    for (auto val : this->b_lb) {
      if (val < 0) {
	// turns into leq inequality
	auto r = std::vector<T>(this->A_lb[ii]);
	std::transform(r.cbegin(),
		       r.cend(),
		       r.begin(),
		       std::negate<T>());
	this->A_ub.push_back(r);
	this->b_ub.push_back(-1*val);
	remove_idx.push(ii);
      }
    }
    while (!remove_idx.empty()) {
      auto row_idx = remove_idx.top();
      this->A_lb[row_idx] = this->A_lb.back();
      this->A_lb.pop_back();
      this->b_lb[row_idx] = this->b_lb.back();
      this->b_lb.pop_back();
      remove_idx.pop();
    }

    // Equality constraints just need to be negated
    ii = 0;
    for (auto val : this->b_eq) {
      if (val < 0) {
	// Remains an equality
	std::transform(this->A_eq[ii].cbegin(),
		       this->A_eq[ii].cend(),
		       this->A_eq[ii].begin(),
		       std::negate<T>());
	this->b_eq[ii] *= -1;
      }
    }
  }
  
  template<class T>
  void Simplex<T>::allocate_tableau() {
    this->tableau(this->num_rows(), std::vector<T>(this->num_cols()));
  }

  template<class T>
  std::size_t Simplex<T>::num_rows() {
    return this->num_obj_rows +
      this->num_slack +
      this->num_artificial;
  }

  template<class T>
  std::size_t Simplex<T>::obj_row_start_idx() {
    return this->num_obj_rows - 1;
  }

  template<class T>
  std::size_t Simplex<T>::phase1_obj_row_idx() {
    return this->obj_row_start_idx() - 1;
  }

  template<class T>
  std::size_t Simplex<T>::rhs_col_idx() {
    return this->num_cols() - 1;
  }

  template<class T>
  T Simplex<T>::get_obj_val() {
    return this->tableau[this->obj_row_start_idx()][this->rhs_col_idx()];
  }

  template<class T>
  T Simplex<T>::get_phase1_obj_val() {
    return this->tableau[this->phase1_obj_row_idx()][this->rhs_col_idx()];
  }
  
}

int main() {
  return 0;
}
