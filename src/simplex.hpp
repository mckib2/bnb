
#include <memory>
#include <vector>

namespace simplex {

  template<class T>
  class Solution {
  public:
    std::unique_ptr<T> x;
    std::unique_ptr<T> slack;
    std::unique_ptr<T> dual;
    T fobj;
  };
  
  template<class T>
  class Simplex {
  public:
    explicit Simplex(const std::vector<T> c,
		     const std::vector<std::vector<T> > A_ub, const std::vector<T> b_ub,
		     const std::vector<std::vector<T> > A_eq, const std::vector<T> b_eq,
		     const std::vector<std::vector<T> > A_lb, const std::vector<T> b_lb,
		     const bool solve_dual);
    ~Simplex() { }

    std::vector<T> c;
    std::vector<std::vector<T> > A_ub;
    std::vector<std::vector<T> > A_eq;
    std::vector<std::vector<T> > A_lb;
    std::vector<T> b_ub;
    std::vector<T> b_eq;
    std::vector<T> b_lb;
    
    bool two_phase;
    bool solve_dual;
    T tolerance = 1e-9;
    std::size_t num_vars;
    std::size_t num_slack;
    std::size_t num_surplus;
    std::size_t num_artificial;

    std::size_t num_obj_rows;

    // The actual tableau
    std::vector<std::string> hdrs;
    std::vector<std::string> basis;
    std::vector<std::vector<T> > tableau;
    
    // Functions that do work:
    void make_rhs_nonnegative(void);
    void allocate_tableau(void);
    void make_hdrs(void);
    void initial_fbs(void);
    bool pivot(const std::size_t row_idx, const std::size_t col_idx);
    void add_phase1_obj(void);
    void initial_fbs_before_phase1(void);
    void remove_artificial_vars(void);
    void remove_phase1_obj(void);
    void initial_fbs_after_phase1(void);
    
    // Functions to help out ineteraction with the table
    std::size_t num_rows(void);
    std::size_t num_cols(void);
    std::size_t obj_row_start_idx(void);
    std::size_t phase1_obj_row_idx(void);
    std::size_t rhs_col_idx(void);

    T get_obj_val(void);
    T get_phase1_obj_val(void);
    std::unique_ptr<std::string> get_basis(void);
    std::unique_ptr<T> get_vars(void);
    std::unique_ptr<T> get_slack_vars(void);
    std::unique_ptr<T> get_surplus_vars(void);
    std::unique_ptr<T> get_artificial_vars(void);
    std::unique_ptr<T> get_row(const std::size_t row_idx);
    std::unique_ptr<T> get_col(const std::size_t col_idx);
    std::size_t get_idx_of_most_neg_in_col(const std::size_t col_idx);
    std::size_t get_idx_of_most_neg_in_row(const std::size_t row_idx);
    std::size_t get_idx_of_most_neg_ratio(const std::size_t row_idx, const std::size_t col_idx);
    void normalize_row(const std::size_t row_idx);
    void add_rows(const std::size_t row1_idx, const std::size_t row2_idx, const T row1_mult, const T row2_mult);
    std::size_t get_num_neg_in_row(const std::size_t row_idx, const bool use_tol);
    std::string diplay(void);
    
  };

} // namespace simplex

