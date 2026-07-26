// Phase 1 sweep runner: solve one Matrix Market system with one AMGCL
// parameter combination, emit a single JSON line.
//
// Design notes that matter for the validity of the pilot:
//   * setup and solve are timed separately; the sweep compares setup+solve,
//     but a config that is cheap to build and slow to run must stay visible.
//   * Matrix Market 'symmetric' files store only the lower triangle. They are
//     expanded to full CSR here -- feeding the half-stored matrix to AMGCL
//     would silently solve a different problem.
//   * every failure mode gets its own status string. A run that diverges, one
//     that hits maxiter, and one that throws are NOT the same event, and
//     collapsing them would destroy the failure-prediction analysis.
//
// Usage:
//   runner <matrix.mtx> <coarsening> <relaxation> <solver> [tol] [maxiter]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/property_tree/ptree.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

using clk = std::chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

struct COO {
    int n = 0;
    std::vector<int> i, j;
    std::vector<double> v;
};

// Read a coordinate-format Matrix Market file, expanding symmetric /
// skew-symmetric / hermitian storage into explicit full entries.
static COO read_mm(const std::string &path, std::string &symmetry) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);

    std::string line;
    if (!std::getline(f, line)) throw std::runtime_error("empty file");
    {
        std::string lower = line;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.rfind("%%matrixmarket", 0) != 0)
            throw std::runtime_error("not a Matrix Market file");
        std::istringstream hs(lower);
        std::vector<std::string> tok;
        for (std::string t; hs >> t;) tok.push_back(t);
        if (tok.size() < 5) throw std::runtime_error("short MM banner");
        if (tok[2] != "coordinate")
            throw std::runtime_error("unsupported MM format: " + tok[2]);
        if (tok[3] == "complex")
            throw std::runtime_error("complex matrices not supported");
        symmetry = tok[4];
    }

    while (std::getline(f, line)) {
        if (!line.empty() && line[0] != '%') break;
    }

    int nrows, ncols; long long nnz;
    {
        std::istringstream ds(line);
        if (!(ds >> nrows >> ncols >> nnz))
            throw std::runtime_error("bad MM size line");
    }
    if (nrows != ncols) throw std::runtime_error("matrix is not square");

    const bool pattern_only = false;  // filtered upstream; values expected
    const bool expand = (symmetry == "symmetric" ||
                         symmetry == "skew-symmetric" ||
                         symmetry == "hermitian");
    const double mirror_sign = (symmetry == "skew-symmetric") ? -1.0 : 1.0;

    COO A;
    A.n = nrows;
    A.i.reserve(expand ? nnz * 2 : nnz);
    A.j.reserve(expand ? nnz * 2 : nnz);
    A.v.reserve(expand ? nnz * 2 : nnz);

    for (long long k = 0; k < nnz; ++k) {
        int r, c; double val = 1.0;
        if (!(f >> r >> c)) throw std::runtime_error("truncated MM data");
        if (!pattern_only && !(f >> val))
            throw std::runtime_error("truncated MM values");
        --r; --c;                       // Matrix Market is 1-indexed
        A.i.push_back(r); A.j.push_back(c); A.v.push_back(val);
        if (expand && r != c) {
            A.i.push_back(c); A.j.push_back(r);
            A.v.push_back(mirror_sign * val);
        }
    }
    return A;
}

// COO -> CSR, summing duplicate entries.
static void to_csr(const COO &A, std::vector<int> &ptr,
                   std::vector<int> &col, std::vector<double> &val) {
    const int n = A.n;
    const size_t nz = A.v.size();
    ptr.assign(n + 1, 0);
    for (size_t k = 0; k < nz; ++k) ++ptr[A.i[k] + 1];
    for (int r = 0; r < n; ++r) ptr[r + 1] += ptr[r];

    col.assign(nz, 0);
    val.assign(nz, 0.0);
    std::vector<int> next(ptr.begin(), ptr.end() - 1);
    for (size_t k = 0; k < nz; ++k) {
        int p = next[A.i[k]]++;
        col[p] = A.j[k];
        val[p] = A.v[k];
    }
}

static std::string esc(const std::string &s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr <<
            "usage: runner <matrix.mtx> [key=value ...]\n"
            "\n"
            "Any AMGCL parameter path may be given as key=value. Parameter\n"
            "paths differ between families, so they cannot be positional:\n"
            "  precond.coarsening.type=ruge_stuben\n"
            "  precond.coarsening.eps_strong=0.25        (ruge_stuben)\n"
            "  precond.coarsening.aggr.eps_strong=0.08   (aggregation family)\n"
            "  precond.relax.type=iluk\n"
            "  precond.relax.k=2                         (iluk fill level)\n"
            "  solver.type=cg  solver.tol=1e-8  solver.maxiter=1000\n"
            "\n"
            "The pseudo-key 'repeat=N' is consumed by the runner, not AMGCL.\n";
        return 2;
    }
    const std::string path = argv[1];

    boost::property_tree::ptree prm;
    prm.put("precond.coarsening.type", "smoothed_aggregation");
    prm.put("precond.relax.type", "spai0");
    prm.put("solver.type", "cg");
    prm.put("solver.tol", 1e-8);
    prm.put("solver.maxiter", 1000);

    // Setup+solve is often tens of milliseconds -- the same order as process
    // startup and scheduler noise. Repeating in-process and reporting the
    // MINIMUM amortises that: noise can only ever add time, so the minimum is
    // the closest sample to an undisturbed run.
    int repeat = 1;
    bool scale = false;
    std::string config;
    for (int a = 2; a < argc; ++a) {
        std::string kv = argv[a];
        size_t eq = kv.find('=');
        if (eq == std::string::npos) {
            std::cerr << "bad argument (expected key=value): " << kv << "\n";
            return 2;
        }
        std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        if (k == "repeat") { repeat = std::max(1, std::atoi(v.c_str())); continue; }
        if (k == "scale") {                 // runner-level, not an AMGCL param
            scale = (v == "1" || v == "true");
            if (!config.empty()) config += " ";
            config += kv;
            continue;
        }
        prm.put(k, v);
        if (!config.empty()) config += " ";
        config += kv;
    }

    const std::string coarsening = prm.get<std::string>("precond.coarsening.type");
    const std::string relaxation = prm.get<std::string>("precond.relax.type");
    const std::string solver = prm.get<std::string>("solver.type");
    const double tol = prm.get<double>("solver.tol");
    const int maxiter = prm.get<int>("solver.maxiter");

    std::string status = "ok", message, symmetry = "?";
    double t_read = 0, t_setup = 0, t_solve = 0, resid = 0;
    double t_total_med = 0, t_total_cv = 0;
    int iters = 0, n = 0;
    long long nnz = 0;

    try {
        auto t0 = clk::now();
        std::string sym;
        COO coo = read_mm(path, sym);
        symmetry = sym;
        std::vector<int> ptr, col;
        std::vector<double> val;
        to_csr(coo, ptr, col, val);
        n = coo.n;
        nnz = static_cast<long long>(val.size());

        // Optional symmetric diagonal (Jacobi) equilibration:
        //   A' = D^{-1/2} A D^{-1/2},  D_ii = max(|a_ii|, eps).
        // A rescue attempt for matrices where every unscaled config diverges
        // or errors -- badly scaled diagonals break both strength-of-
        // connection heuristics and ILU pivots. Applied BEFORE b is formed,
        // so b' = D^{-1/2} A D^{-1/2} x_exact and the reported residual is
        // for the scaled system (comparable across configs, which is all the
        // sweep needs). Scaling time is charged to t_read, not t_setup:
        // it is a fixed preprocessing cost shared by every config.
        if (scale) {
            std::vector<double> dinv(n, 1.0);
            for (int r = 0; r < n; ++r)
                for (int p = ptr[r]; p < ptr[r + 1]; ++p)
                    if (col[p] == r && std::fabs(val[p]) > 1e-300)
                        dinv[r] = 1.0 / std::sqrt(std::fabs(val[p]));
            for (int r = 0; r < n; ++r)
                for (int p = ptr[r]; p < ptr[r + 1]; ++p)
                    val[p] *= dinv[r] * dinv[col[p]];
        }

        auto t1 = clk::now();
        t_read = secs(t0, t1);

        typedef amgcl::backend::builtin<double> Backend;
        typedef amgcl::make_solver<
            amgcl::amg<Backend,
                amgcl::runtime::coarsening::wrapper,
                amgcl::runtime::relaxation::wrapper>,
            amgcl::runtime::solver::wrapper<Backend>
            > Solver;

        // Reproducible right-hand side: b = A * x_exact with a linear-ramp
        // x_exact. The all-ones vector used previously is DEGENERATE for any
        // matrix with zero row sums (Laplacians, pure-Neumann FEM): b becomes
        // exactly 0, x = 0 is the solution, and every solver "converges" in 0
        // iterations measuring nothing but setup (observed on Andrews and
        // denormal in the Phase 1 sweep). The ramp has no such null-space
        // alignment, stays deterministic, and needs no RNG.
        std::vector<double> x_exact(n), b(n, 0.0);
        for (int r = 0; r < n; ++r)
            x_exact[r] = 1.0 + static_cast<double>(r) / n;
        for (int r = 0; r < n; ++r)
            for (int p = ptr[r]; p < ptr[r + 1]; ++p)
                b[r] += val[p] * x_exact[col[p]];

        std::vector<double> setups, solves, totals;
        for (int rep = 0; rep < repeat; ++rep) {
            auto t2 = clk::now();
            Solver S(std::tie(n, ptr, col, val), prm);
            auto t3 = clk::now();

            std::vector<double> x(n, 0.0);
            auto t4 = clk::now();
            std::tie(iters, resid) = S(b, x);
            auto t5 = clk::now();

            setups.push_back(secs(t2, t3));
            solves.push_back(secs(t4, t5));
            totals.push_back(secs(t2, t3) + secs(t4, t5));
        }

        t_setup = *std::min_element(setups.begin(), setups.end());
        t_solve = *std::min_element(solves.begin(), solves.end());

        std::vector<double> sorted = totals;
        std::sort(sorted.begin(), sorted.end());
        t_total_med = sorted[sorted.size() / 2];

        double mean = 0;
        for (double t : totals) mean += t;
        mean /= totals.size();
        double var = 0;
        for (double t : totals) var += (t - mean) * (t - mean);
        var /= totals.size();
        t_total_cv = (mean > 0) ? std::sqrt(var) / mean : 0.0;

        if (!std::isfinite(resid))       status = "diverged";
        else if (resid > tol && iters >= maxiter) status = "maxiter";
        else if (resid > tol)            status = "stalled";
    } catch (const std::exception &e) {
        status = "error";
        message = e.what();
    } catch (...) {
        status = "error";
        message = "unknown exception";
    }

    // C prints non-finite doubles as "nan"/"inf"; JSON parsers accept neither
    // (Python's json requires the capitalised "NaN"). Emitting null keeps the
    // record machine-readable -- silently unparseable lines were previously
    // being misfiled as harness crashes.
    char resid_str[32];
    if (std::isfinite(resid)) std::snprintf(resid_str, sizeof resid_str, "%g", resid);
    else std::snprintf(resid_str, sizeof resid_str, "null");

    std::printf(
        "{\"matrix\":\"%s\",\"coarsening\":\"%s\",\"relaxation\":\"%s\","
        "\"solver\":\"%s\",\"config\":\"%s\",\"symmetry\":\"%s\",\"n\":%d,\"nnz\":%lld,"
        "\"tol\":%g,\"maxiter\":%d,\"repeat\":%d,\"status\":\"%s\",\"iters\":%d,"
        "\"resid\":%s,\"t_read\":%.6f,\"t_setup\":%.6f,\"t_solve\":%.6f,"
        "\"t_total\":%.6f,\"t_total_med\":%.6f,\"t_total_cv\":%.4f,"
        "\"message\":\"%s\"}\n",
        esc(path).c_str(), coarsening.c_str(), relaxation.c_str(),
        solver.c_str(), esc(config).c_str(), symmetry.c_str(), n, nnz,
        tol, maxiter, repeat,
        status.c_str(), iters, resid_str, t_read, t_setup, t_solve,
        t_setup + t_solve, t_total_med, t_total_cv, esc(message).c_str());

    return status == "ok" ? 0 : 1;
}
