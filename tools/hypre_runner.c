/* hypre BoomerAMG sweep runner -- the hypre counterpart of runner.cpp.
 *
 * Solves one Matrix Market system with BoomerAMG-preconditioned PCG (SPD) or
 * GMRES (nonsymmetric), with coarsening / relaxation / strong-threshold chosen
 * at RUNTIME by key=value arguments, and emits one JSON line. Kept
 * deliberately parallel to runner.cpp so the cross-library comparison controls
 * everything except the library:
 *   - same ramp RHS  b = A * x_exact,  x_exact_i = 1 + i/n
 *   - same reporting: setup/solve split, status codes, in-process repeat min
 *   - same relative-residual tolerance and iteration cap
 *
 * Built against a sequential hypre (HYPRE_ENABLE_MPI=OFF): our sweep runs one
 * single-threaded process per matrix, so no MPI is needed and the biggest
 * build risk disappears.
 *
 * Parameters (key=value), mapped to hypre enums:
 *   coarsen=6|8|10        Falgout | PMIS | HMIS   (HYPRE_BoomerAMGSetCoarsenType)
 *   relax=3|6|8|18        hybrid-GS | sym-GS | L1-sym-GS | L1-Jacobi
 *   strong=0.25|0.5|0.7   strength threshold      (SetStrongThreshold)
 *   interp=6|0            ext+i | classical        (SetInterpType)
 *   solver=cg|gmres
 *   tol=1e-8  maxiter=1000  repeat=1
 *
 * Usage: hypre_runner <matrix.mtx> [key=value ...]
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Minimal Matrix Market coordinate reader with symmetric expansion.
 * Returns 0 on success. Fills n, and malloc'd rows/cols/vals of length *nnz. */
static int read_mm(const char *path, int *n_out, long *nnz_out,
                   int **rp, int **cp, double **vp, char *sym_out) {
    FILE *f = fopen(path, "r");
    if (!f) return 1;
    char line[4096];
    if (!fgets(line, sizeof line, f)) { fclose(f); return 2; }
    /* banner: %%MatrixMarket matrix coordinate real <sym> */
    char w0[64], w1[64], w2[64], w3[64], w4[64];
    for (char *p = line; *p; ++p) *p = (char)tolower((unsigned char)*p);
    if (sscanf(line, "%63s %63s %63s %63s %63s", w0, w1, w2, w3, w4) < 5) {
        fclose(f); return 3;
    }
    if (strcmp(w2, "coordinate") != 0 || strcmp(w3, "complex") == 0) {
        fclose(f); return 4;
    }
    strncpy(sym_out, w4, 31);
    int expand = (strcmp(w4, "symmetric") == 0 ||
                  strcmp(w4, "skew-symmetric") == 0 ||
                  strcmp(w4, "hermitian") == 0);
    double sign = (strcmp(w4, "skew-symmetric") == 0) ? -1.0 : 1.0;

    /* skip comments */
    do {
        if (!fgets(line, sizeof line, f)) { fclose(f); return 5; }
    } while (line[0] == '%');

    int nrow, ncol; long nnz;
    if (sscanf(line, "%d %d %ld", &nrow, &ncol, &nnz) != 3) { fclose(f); return 6; }
    if (nrow != ncol) { fclose(f); return 7; }

    long cap = expand ? nnz * 2 : nnz;
    int *R = malloc(cap * sizeof(int)), *C = malloc(cap * sizeof(int));
    double *V = malloc(cap * sizeof(double));
    long k = 0;
    for (long e = 0; e < nnz; ++e) {
        int r, c; double v = 1.0;
        if (fscanf(f, "%d %d %lf", &r, &c, &v) < 2) { fclose(f); return 8; }
        R[k] = r - 1; C[k] = c - 1; V[k] = v; ++k;
        if (expand && r != c) { R[k] = c - 1; C[k] = r - 1; V[k] = sign * v; ++k; }
    }
    fclose(f);
    *n_out = nrow; *nnz_out = k; *rp = R; *cp = C; *vp = V;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: hypre_runner <matrix.mtx> [key=value ...]\n");
        return 2;
    }
    const char *path = argv[1];
    int coarsen = 6, relax = 8, interp = 6, maxiter = 1000, repeat = 1;
    double strong = 0.25, tol = 1e-8;
    char solver[16] = "cg";
    char config[512] = "";

    for (int a = 2; a < argc; ++a) {
        char *eq = strchr(argv[a], '=');
        if (!eq) continue;
        *eq = 0; char *k = argv[a], *v = eq + 1;
        if      (!strcmp(k, "coarsen")) coarsen = atoi(v);
        else if (!strcmp(k, "relax"))   relax = atoi(v);
        else if (!strcmp(k, "interp"))  interp = atoi(v);
        else if (!strcmp(k, "strong"))  strong = atof(v);
        else if (!strcmp(k, "solver"))  strncpy(solver, v, 15);
        else if (!strcmp(k, "tol"))     tol = atof(v);
        else if (!strcmp(k, "maxiter")) maxiter = atoi(v);
        else if (!strcmp(k, "repeat"))  repeat = atoi(v) > 0 ? atoi(v) : 1;
        *eq = '=';
        if (strcmp(k, "repeat")) {
            if (config[0]) strncat(config, " ", sizeof config - strlen(config) - 1);
            strncat(config, argv[a], sizeof config - strlen(config) - 1);
        }
    }

    HYPRE_Initialize();

    const char *status = "ok", *message = "";
    int n = 0, iters = 0; long nnz = 0;
    double t_read = 0, t_setup = 0, t_solve = 0, resid = 0;
    char sym[32] = "?";
    int *R = NULL, *C = NULL; double *V = NULL;

    double t0 = now_sec();
    int rc = read_mm(path, &n, &nnz, &R, &C, &V, sym);
    if (rc) {
        printf("{\"matrix\":\"%s\",\"library\":\"hypre\",\"status\":\"error\","
               "\"message\":\"read_mm rc=%d\"}\n", path, rc);
        return 1;
    }
    t_read = now_sec() - t0;

    /* ramp exact solution and its RHS b = A x_exact, assembled per row.
     * Build once; the matrix is rebuilt each repeat only for setup timing. */
    double *x_exact = malloc(n * sizeof(double));
    double *b = calloc(n, sizeof(double));
    for (int i = 0; i < n; ++i) x_exact[i] = 1.0 + (double)i / n;
    for (long e = 0; e < nnz; ++e) b[R[e]] += V[e] * x_exact[C[e]];

    /* count entries per row for IJ SetValues */
    int *rowcnt = calloc(n, sizeof(int));
    for (long e = 0; e < nnz; ++e) rowcnt[R[e]]++;

    double best_total = 1e300;
    for (int rep = 0; rep < repeat; ++rep) {
        HYPRE_IJMatrix Aij; HYPRE_ParCSRMatrix Apar;
        HYPRE_IJMatrixCreate(0, 0, n - 1, 0, n - 1, &Aij);
        HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(Aij);
        /* insert entries one triple at a time (simple, robust) */
        for (long e = 0; e < nnz; ++e) {
            int nrows = 1, ncols = 1, row = R[e], col = C[e];
            double val = V[e];
            HYPRE_IJMatrixAddToValues(Aij, nrows, &ncols, &row, &col, &val);
        }
        HYPRE_IJMatrixAssemble(Aij);
        HYPRE_IJMatrixGetObject(Aij, (void **)&Apar);

        HYPRE_IJVector bij, xij; HYPRE_ParVector bpar, xpar;
        HYPRE_IJVectorCreate(0, 0, n - 1, &bij);
        HYPRE_IJVectorSetObjectType(bij, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(bij);
        HYPRE_IJVectorCreate(0, 0, n - 1, &xij);
        HYPRE_IJVectorSetObjectType(xij, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(xij);
        for (int i = 0; i < n; ++i) {
            double z = 0.0; HYPRE_IJVectorSetValues(bij, 1, &i, &b[i]);
            HYPRE_IJVectorSetValues(xij, 1, &i, &z);
        }
        HYPRE_IJVectorAssemble(bij); HYPRE_IJVectorGetObject(bij, (void **)&bpar);
        HYPRE_IJVectorAssemble(xij); HYPRE_IJVectorGetObject(xij, (void **)&xpar);

        /* BoomerAMG preconditioner */
        HYPRE_Solver amg;
        HYPRE_BoomerAMGCreate(&amg);
        HYPRE_BoomerAMGSetCoarsenType(amg, coarsen);
        HYPRE_BoomerAMGSetRelaxType(amg, relax);
        HYPRE_BoomerAMGSetInterpType(amg, interp);
        HYPRE_BoomerAMGSetStrongThreshold(amg, strong);
        HYPRE_BoomerAMGSetMaxIter(amg, 1);       /* one V-cycle as preconditioner */
        HYPRE_BoomerAMGSetTol(amg, 0.0);
        HYPRE_BoomerAMGSetPrintLevel(amg, 0);

        double ts = now_sec();
        HYPRE_Solver krylov;
        int is_cg = (strcmp(solver, "cg") == 0);
        if (is_cg) {
            HYPRE_ParCSRPCGCreate(0, &krylov);
            HYPRE_PCGSetMaxIter(krylov, maxiter);
            HYPRE_PCGSetTol(krylov, tol);
            HYPRE_PCGSetPrintLevel(krylov, 0);
            HYPRE_PCGSetPrecond(krylov,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, amg);
            HYPRE_ParCSRPCGSetup(krylov, Apar, bpar, xpar);
            double tm = now_sec(); t_setup = tm - ts;
            HYPRE_ParCSRPCGSolve(krylov, Apar, bpar, xpar);
            t_solve = now_sec() - tm;
            HYPRE_PCGGetNumIterations(krylov, &iters);
            HYPRE_PCGGetFinalRelativeResidualNorm(krylov, &resid);
            HYPRE_ParCSRPCGDestroy(krylov);
        } else {
            HYPRE_ParCSRGMRESCreate(0, &krylov);
            HYPRE_GMRESSetKDim(krylov, 50);
            HYPRE_GMRESSetMaxIter(krylov, maxiter);
            HYPRE_GMRESSetTol(krylov, tol);
            HYPRE_GMRESSetPrintLevel(krylov, 0);
            HYPRE_GMRESSetPrecond(krylov,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, amg);
            HYPRE_ParCSRGMRESSetup(krylov, Apar, bpar, xpar);
            double tm = now_sec(); t_setup = tm - ts;
            HYPRE_ParCSRGMRESSolve(krylov, Apar, bpar, xpar);
            t_solve = now_sec() - tm;
            HYPRE_GMRESGetNumIterations(krylov, &iters);
            HYPRE_GMRESGetFinalRelativeResidualNorm(krylov, &resid);
            HYPRE_ParCSRGMRESDestroy(krylov);
        }
        HYPRE_BoomerAMGDestroy(amg);
        HYPRE_IJMatrixDestroy(Aij);
        HYPRE_IJVectorDestroy(bij);
        HYPRE_IJVectorDestroy(xij);

        double total = t_setup + t_solve;
        if (total < best_total) best_total = total;
    }

    if (!isfinite(resid))                 status = "diverged";
    else if (resid > tol && iters >= maxiter) status = "maxiter";
    else if (resid > tol)                 status = "stalled";

    char resid_str[32];
    if (isfinite(resid)) snprintf(resid_str, sizeof resid_str, "%g", resid);
    else                 snprintf(resid_str, sizeof resid_str, "null");

    printf("{\"matrix\":\"%s\",\"library\":\"hypre\",\"config\":\"%s\","
           "\"coarsen\":%d,\"relax\":%d,\"interp\":%d,\"strong\":%g,"
           "\"solver\":\"%s\",\"symmetry\":\"%s\",\"n\":%d,\"nnz\":%ld,"
           "\"tol\":%g,\"maxiter\":%d,\"repeat\":%d,\"status\":\"%s\","
           "\"iters\":%d,\"resid\":%s,\"t_read\":%.6f,\"t_setup\":%.6f,"
           "\"t_solve\":%.6f,\"t_total\":%.6f,\"message\":\"%s\"}\n",
           path, config, coarsen, relax, interp, strong, solver, sym, n, nnz,
           tol, maxiter, repeat, status, iters, resid_str, t_read, t_setup,
           t_solve, best_total, message);

    free(R); free(C); free(V); free(x_exact); free(b); free(rowcnt);
    HYPRE_Finalize();
    return strcmp(status, "ok") == 0 ? 0 : 1;
}
