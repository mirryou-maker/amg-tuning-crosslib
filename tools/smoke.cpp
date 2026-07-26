// Build smoke test: does AMGCL + Boost property_tree runtime parameter
// selection compile and run? Solves a 2D Poisson system on a small grid.
//
// If this builds, the Phase 1 sweep harness is viable as written.

#include <iostream>
#include <vector>
#include <boost/property_tree/ptree.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

// 5-point Laplacian on an n x n grid, CSR.
static int poisson(int n, std::vector<int> &ptr, std::vector<int> &col,
                   std::vector<double> &val)
{
    int n2 = n * n;
    ptr.clear(); col.clear(); val.clear();
    ptr.reserve(n2 + 1); ptr.push_back(0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int k = i + j * n;
            if (j > 0) { col.push_back(k - n); val.push_back(-1.0); }
            if (i > 0) { col.push_back(k - 1); val.push_back(-1.0); }
            col.push_back(k); val.push_back(4.0);
            if (i + 1 < n) { col.push_back(k + 1); val.push_back(-1.0); }
            if (j + 1 < n) { col.push_back(k + n); val.push_back(-1.0); }
            ptr.push_back(static_cast<int>(col.size()));
        }
    }
    return n2;
}

int main(int argc, char *argv[]) {
    // Parameters chosen at RUNTIME by string -- the capability the sweep needs.
    std::string coarsening = (argc > 1) ? argv[1] : "smoothed_aggregation";
    std::string relaxation = (argc > 2) ? argv[2] : "spai0";
    std::string solver     = (argc > 3) ? argv[3] : "cg";

    std::vector<int> ptr, col;
    std::vector<double> val;
    int n = poisson(64, ptr, col, val);

    boost::property_tree::ptree prm;
    prm.put("precond.coarsening.type", coarsening);
    prm.put("precond.relax.type", relaxation);
    prm.put("solver.type", solver);
    prm.put("solver.tol", 1e-8);
    prm.put("solver.maxiter", 1000);

    typedef amgcl::backend::builtin<double> Backend;
    typedef amgcl::make_solver<
        amgcl::amg<Backend,
            amgcl::runtime::coarsening::wrapper,
            amgcl::runtime::relaxation::wrapper>,
        amgcl::runtime::solver::wrapper<Backend>
        > Solver;

    Solver solve(std::tie(n, ptr, col, val), prm);

    std::vector<double> b(n, 1.0), x(n, 0.0);
    int iters; double error;
    std::tie(iters, error) = solve(b, x);

    std::cout << "coarsening=" << coarsening
              << " relax=" << relaxation
              << " solver=" << solver
              << " n=" << n
              << " iters=" << iters
              << " resid=" << error << "\n";
    return 0;
}
