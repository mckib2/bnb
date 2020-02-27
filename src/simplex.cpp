#include <vector>
#include <stack>
#include <string>
#include <algorithm>
#include <functional>
#include <iterator>
#include <assert.h> // assert
#include <iostream>

#include "simplex.hpp"

namespace simplex {

  template<class T>
  Simplex<T>::Simplex(const std::vector<T> c_,
		      const std::vector<std::vector<T> > A_ub_, const std::vector<T> b_ub_,
		      const std::vector<std::vector<T> > A_eq_, const std::vector<T> b_eq_,
		      const std::vector<std::vector<T> > A_lb_, const std::vector<T> b_lb_,
		      const bool solve_dual_)
  {

    // May be computationally advantageous to solve dual instead of primal
    this->solve_dual = solve_dual_;
    assert(!this->solve_dual); // we don't do this yet
    
    // Store copy of coefficients and constraints
    this->c = std::vector<T>(c_);
    this->A_ub = std::vector<std::vector<T> >(A_ub_);
    this->A_eq = std::vector<std::vector<T> >(A_eq_);
    this->A_lb = std::vector<std::vector<T> >(A_lb_);
    this->b_ub = std::vector<T>(b_ub_);
    this->b_eq = std::vector<T>(b_eq_);
    this->b_lb = std::vector<T>(b_lb_);

    // Go ahead and make RHS all non-negative. No longer trust function
    // arguments to have correct constraints after this! Use Simplex::A_ub, etc.
    this->make_rhs_nonnegative();
    
    this->num_vars = this->c.size();
    this->num_slack = this->b_ub.size();
    this->num_surplus = this->b_lb.size();
    this->num_artificial = this->num_surplus + this->b_eq.size();

    // We have a two phase problem if we got artificial variables
    this->two_phase = this->num_artificial > 0;
    
    // Table specific stuff
    this->num_obj_rows = 1;
    
    // See if we need to run two phases
    this->two_phase = this->num_artificial > 0;
    
    // Initialize the tableau to all 0s
    this->allocate_tableau();

    // Make headers for the tableau
    this->make_hdrs();

    // Maybe fill in tableau before FBS?
    this->fill_initial_tableau();
    
    // Get an intitial feasible solution
    this->label_initial_fbs();
  }

  template<class T>
  void Simplex<T>::solve() {

    // Solve the phase 1 problem if we need to
    if (this->two_phase) {
      
      // Change the problem to minimize sum of artificial variables
      auto phase1_c = std::vector<T>(this->num_cols());
      std::fill_n(std::next(phase1_c.begin(), this->artificial_col_start_idx()), this->num_artificial, 1);

      // Add the new objective row to the top -- we now have two objective functions we're tracking!
      this->tableau.insert(this->tableau.cbegin(), phase1_c);
      this->num_obj_rows++;
      this->basis.insert(this->basis.cbegin(), "phase1");

      // Since we have no negatives in the objective row, use artificial variable rows to
      // eliminate entries in the objective column
      auto pivot_row = this->num_rows() - this->num_artificial;
      for (auto ii = this->artificial_col_start_idx(); ii < this->rhs_col_idx(); ++ii) {
	this->pivot(pivot_row, ii);
	pivot_row++;
      }

      // Solve the phase 1 problem
      this->two_phase = false;
      this->solve();

      // If we get an objective value other than 0, problem is infeasible
      if (this->get_obj_val() != 0) {
	std::cout << "There is no solution!" << std::endl;
	return;
      }

      // Now solve phase 2 by removing artificial variable columns and phase 1 objective row
      this->tableau.erase(this->tableau.cbegin());
      this->basis.erase(this->basis.cbegin());
      this->num_obj_rows--;
      for (auto & row : this->tableau) {
	row.erase(std::next(row.cbegin(), this->rhs_col_idx() - this->num_artificial),
		  std::prev(row.cend(), 1));
      }
      this->hdrs.erase(std::next(this->hdrs.cbegin(), this->rhs_col_idx() - this->num_artificial + this->hdr_var_start_idx()),
		       std::prev(this->hdrs.cend(), 1));
      this->num_artificial = 0;

      // Can solve the phase 2 problem as normal
    }
    
    // Keep pivoting till we can't pivot no more!
    // break to get out of loop
    while (true) {

      // Take a look
      this->show();

      // Check exit conditions
      auto obj_row_idx = this->obj_row_start_idx();
      auto & tol = this->tolerance;
      if (std::all_of(this->tableau[obj_row_idx].cbegin(),
		      std::prev(this->tableau[obj_row_idx].cend(), 1), // don't include RHS
		      [tol](const auto & el) {
			return el >= 0 || -1*el < tol;
		      })) {
	// All objective row elements are positive, we're done!
	break;
      }
		      
      // Choose incoming variable from strictly negative objective cols
      auto min_it = std::min_element(this->tableau[obj_row_idx].cbegin(),
				     std::prev(this->tableau[obj_row_idx].cend(), 1)); // don't include RHS
      assert(*min_it < 0);
      auto pivot_col = std::distance(this->tableau[obj_row_idx].cbegin(), min_it);
      
      // Choose the outgoing variable using ratio test.
      // If we store references to column major tableau we could get these easier
      auto ratios = std::vector<std::pair<std::size_t, T> >();
      auto rhs_col_idx = this->rhs_col_idx();
      std::size_t idx = 0;
      for (const auto & row : this->tableau) {
	if (row[pivot_col] > 0) {
	  ratios.push_back(std::make_pair(idx, row[rhs_col_idx]/row[pivot_col]));
	}
	idx++;
      }
      if (ratios.size() == 0) {
	std::cout << "Unbounded solution!" << std::endl;
	break;
      }
      auto ratio_min_it = std::min_element(ratios.cbegin(),
					   ratios.cend(),
					   [](const auto & a, const auto & b) -> bool {
					     return std::get<1>(a) < std::get<1>(b);
					   });
      auto pivot_row = std::get<0>(*ratio_min_it);
      
      // Do the pivot
      this->pivot(pivot_row, pivot_col);
    }
  }
  
  template<class T>
  void Simplex<T>::pivot(const std::size_t pivot_row, const std::size_t pivot_col) {

    // Update the basic variable labels with the incoming variable
    this->basis[pivot_row] = this->hdrs[this->hdr_var_start_idx() + pivot_col];
   
    // Pivot! Pivot row divided by pivot value
    auto pivot_val = this->tableau[pivot_row][pivot_col];
    std::transform(this->tableau[pivot_row].cbegin(),
		   this->tableau[pivot_row].cend(),
		   this->tableau[pivot_row].begin(),
		   [pivot_val](const auto & el) {
		     return el/pivot_val;
		   });

    // Now use pivot row to remove remaining entries in the pivot col
    for (std::size_t ii = 0; ii < this->num_rows(); ++ii) {
      if (ii == pivot_row) {
	continue;
      }
      auto val = this->tableau[ii][pivot_col];
      zip2(this->tableau[pivot_row].cbegin(),
	   this->tableau[pivot_row].cend(),
	   this->tableau[ii].begin(),
	   [val](const auto & el1, auto & el2) {
	     el2 -= val*el1;
	   });
    }    
  }

  template<class T>
  inline std::size_t Simplex<T>::hdr_var_start_idx() {
    return 1; // 'basic' is first entry
  }

  template<class T>
  inline std::size_t Simplex<T>::hdr_slack_start_idx() {
    return this->hdr_var_start_idx() + this->num_vars;
  }

  template<class T>
  inline std::size_t Simplex<T>::hdr_surplus_start_idx() {
    return this->hdr_slack_start_idx() + this->num_slack;
  }
  
  template<class T>
  inline std::size_t Simplex<T>::hdr_artificial_start_idx() {
    return this->hdr_surplus_start_idx() + this->num_surplus;
  }
  
  template<class T>
  void Simplex<T>::label_initial_fbs() {

    // First entry of the basis labels is reserved for objective row
    this->basis.push_back("obj");
    
    // Get a feasible solution by setting slack to RHS for A_ub
    std::copy_n(std::next(this->hdrs.cbegin(), this->hdr_slack_start_idx()),
		this->num_slack,
		std::back_inserter(this->basis));

    // Set artificial to RHS for A_eq (because surplus is negative)
    std::copy_n(std::next(this->hdrs.cbegin(), this->hdr_artificial_start_idx()),
		this->num_surplus,
		std::back_inserter(this->basis));

    // Set artificial to RHS for A_lb
    std::copy_n(std::next(this->hdrs.cbegin(), this->hdr_artificial_start_idx() + this->num_surplus),
		this->num_artificial - this->num_surplus,
		std::back_inserter(this->basis));

  }
  
  template<class T>
  void Simplex<T>::fill_initial_tableau() {

    // Do objective row
    auto obj_row_idx = this->obj_row_start_idx();
    std::copy(this->c.cbegin(), this->c.cend(), this->tableau[obj_row_idx].begin());
    transform_n(this->tableau[obj_row_idx].cbegin(),
		this->num_vars,
		this->tableau[obj_row_idx].begin(),
		std::negate<T>());

    // leq constraints
    auto slack_col = this->slack_col_start_idx();
    auto rhs_col = this->rhs_col_idx();
    zip3(this->A_ub.cbegin(),
	 this->A_ub.cend(),
	 this->b_ub.cbegin(),
	 std::next(this->tableau.begin(), this->A_ub_row_start_idx()),
	 [&slack_col, rhs_col](const auto & Ael, const auto & bel, auto & tableau_row) {

	   // Copy coefficients from A_ub to tableau
	   std::copy(Ael.cbegin(), Ael.cend(), tableau_row.begin());

	   // Include slack variable for each constraint
	   tableau_row[slack_col] = 1;

	   // Copy RHS value for this constraint
	   tableau_row[rhs_col] = bel;

	   // Move to next slack col
	   slack_col++;
	 });

    // eq constraints
    auto artificial_col = this->artificial_col_start_idx();
    zip3(this->A_eq.cbegin(),
	 this->A_eq.cend(),
	 this->b_eq.cbegin(),
	 this->tableau.begin() + this->A_eq_row_start_idx(),
	 [&artificial_col, rhs_col](const auto & Ael, const auto & bel, auto & tableau_row) {

	   // Copy coefficients from A_eq to tableau
	   std::copy(Ael.cbegin(), Ael.cend(), tableau_row.begin());

	   // Include artificial variable for each constraint
	   tableau_row[artificial_col] = 1;

	   // Copy RHS value for this constraint
	   tableau_row[rhs_col] = bel;

	   // Increment artificial col
	   artificial_col++;
	 });

    // geq constraints
    auto surplus_col = this->surplus_col_start_idx();
    zip3(this->A_lb.cbegin(),
	 this->A_lb.cend(),
	 this->b_lb.cbegin(),
	 this->tableau.begin() + this->A_lb_row_start_idx(),
	 [&surplus_col, &artificial_col, rhs_col](const auto & Ael, const auto & bel, auto & tableau_row) {

	   // Copy coefficients from A_eq to tableau
	   std::copy(Ael.cbegin(), Ael.cend(), tableau_row.begin());

	   // Include surplus and artificial variable
	   tableau_row[surplus_col] = -1;
	   tableau_row[artificial_col] = 1;

	   // Copy RHS value for this constraint
	   tableau_row[rhs_col] = bel;

	   // Increment cols
	   surplus_col++;
	   artificial_col++;
	 });

  }

  template<class T>
  void Simplex<T>::make_hdrs() {

    // Column tracking basic variables is first
    this->hdrs.push_back("basic");
        
    // Start with variables that appear in objective function,
    // then do slack, surplus, and artificial
    for (std::size_t ii = 0; ii < this->num_vars; ++ii) {
      this->hdrs.push_back("x" + std::to_string(ii));
    }
    for (std::size_t ii = 0; ii < this->num_slack; ++ii) {
      this->hdrs.push_back("sl" + std::to_string(ii));
    }
    for (std::size_t ii = 0; ii < this->num_surplus; ++ii) {
      this->hdrs.push_back("su" + std::to_string(ii));
    }
    for (std::size_t ii = 0; ii < this->num_artificial; ++ii) {
      this->hdrs.push_back("a" + std::to_string(ii));
    }
    // End with RHS
    this->hdrs.push_back("RHS");

    // Did we get all of them?
    assert(this->hdrs.size() - this->num_basic_label_cols == this->num_cols());
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
    this->tableau.resize(this->num_rows(), std::vector<T>(this->num_cols()));
  }

  template<class T>
  inline std::size_t Simplex<T>::num_rows() {
    return this->num_obj_rows +
      this->b_ub.size() +
      this->b_eq.size() +
      this->b_lb.size();
  }

  template<class T>
  inline std::size_t Simplex<T>::num_cols() {
    return this->num_vars +
      this->num_slack +
      this->num_surplus +
      this->num_artificial +
      this->num_rhs_cols;
  }

  template<class T>
  inline std::size_t Simplex<T>::obj_row_start_idx() {
    return 0; // Always start at beginning
  }

  template<class T>
  inline std::size_t Simplex<T>::phase1_obj_row_idx() {
    return this->obj_row_start_idx() + 1; // always the second obj
  }

  template<class T>
  inline std::size_t Simplex<T>::rhs_col_idx() {
    return this->num_cols() - 1;
  }

  template<class T>
  inline T Simplex<T>::get_obj_val() {
    return this->tableau[this->obj_row_start_idx()][this->rhs_col_idx()];
  }

  template<class T>
  inline T Simplex<T>::get_phase1_obj_val() {
    return this->tableau[this->phase1_obj_row_idx()][this->rhs_col_idx()];
  }
  
  template<class T>
  inline std::size_t Simplex<T>::A_ub_row_start_idx() {
    return this->obj_row_start_idx() + this->num_obj_rows;
  }

  template<class T>
  inline std::size_t Simplex<T>::A_eq_row_start_idx() {
    return this->A_ub_row_start_idx() + this->b_ub.size();
  }

  template<class T>
  inline std::size_t Simplex<T>::A_lb_row_start_idx() {
    return this->A_eq_row_start_idx() + this->b_eq.size();
  }

  template<class T>
  inline std::size_t Simplex<T>::slack_col_start_idx() {
    return this->num_vars;
  }

  template<class T>
  inline std::size_t Simplex<T>::surplus_col_start_idx() {
    return this->slack_col_start_idx() + this->num_slack;
  }
  
  template<class T>
  inline std::size_t Simplex<T>::artificial_col_start_idx() {
    return this->surplus_col_start_idx() + this->num_surplus;
  }

  template<class T>
  inline std::size_t Simplex<T>::slack_col_by_A_ub_row(std::size_t row_idx) {
    return this->slack_col_start_idx() + row_idx;
  }

  template<class T>
  void Simplex<T>::show() {

    // Spit out header
    for (const auto & el : this->hdrs) {
      std::cout << el << ' ';
    }
    std::cout << '\n';

    // Spit out table underneath
    zip2(this->tableau.cbegin(),
	 this->tableau.cend(),
	 this->basis.cbegin(),
	 [](const auto & row, const auto & row_hdr) {
	   std::cout << row_hdr << ' ';
	   for (const auto & col : row) {
	     std::cout << col << ' ';
	   }
	   std::cout << '\n';
	 });
    std::cout << std::endl; // flush
  }

  // Explicit template instantiations to make sure we only work with numeric types
  template class Simplex<float>;
  template class Simplex<double>;
  template class Simplex<long double>;

}

/*
int main() {

  auto c = std::vector<double>({3, 5});

  auto A_ub = std::vector<std::vector<double> >();
  A_ub.push_back(std::vector<double>({0, 1}));
  auto b_ub = std::vector<double>({6});

  auto A_eq = std::vector<std::vector<double> >();
  A_eq.push_back(std::vector<double>({3, 2}));
  auto b_eq = std::vector<double>({18});

  auto A_lb = std::vector<std::vector<double> >();
  A_lb.push_back(std::vector<double>({3, 5}));
  auto b_lb = std::vector<double>({2});

  auto s = simplex::Simplex<double>(c,
				    A_ub, b_ub,
				    A_eq, b_eq,
				    A_lb, b_lb,
				    false);

  s.show();

  s.solve();
  
  return 0;
}
*/
