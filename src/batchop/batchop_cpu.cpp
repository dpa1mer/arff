#include "tbb/parallel_for.h"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/partitioner.h"
#include <vector>
#include "mex.h"
#include "matrix.h"
#include "blas.h"
#include "lapacke.h"

static char const * const errInputId = "batchop_cpu:InvalidInput";

static
void doMult(mxArray * plhs[],
            const mxArray * A, const size_t * dimA, const std::string& transpA,
            const mxArray * B, const size_t * dimB, const std::string& transpB,
            size_t nPages) {

    lapack_int dimA0 = dimA[0], dimA1 = dimA[1], dimB0 = dimB[0];

    lapack_int m = dimA[0], r1 = dimA[1], r2 = dimB[0], n = dimB[1];
    if (transpA == "T") {
        m = dimA[1];
        r1 = dimA[0];
    }
    if (transpB == "T") {
        r2 = dimB[1];
        n = dimB[0];
    }

    if (r1 != r2) {
        mexErrMsgIdAndTxt(errInputId, "Inner dimensions do not match.");
    }

    double * A_ptr = mxGetDoubles(A);
    double * B_ptr = mxGetDoubles(B);

    size_t dimC[] = {static_cast<size_t>(m), static_cast<size_t>(n), nPages};
    mxArray * C = mxCreateUninitNumericArray(3, dimC, mxDOUBLE_CLASS, mxREAL);
    double * C_ptr = mxGetDoubles(C);
    
    tbb::affinity_partitioner ap;
    tbb::parallel_for(0, static_cast<int>(nPages), 1, [&](int k) {
    // #pragma omp parallel for
    // for (int k = 0; k < N; ++k) {
        const double one = 1, zero = 0;
        const lapack_int inc = 1;

        const double * A_k = &A_ptr[dimA[0] * dimA[1] * k];
        const double * B_k = &B_ptr[dimB[0] * dimB[1] * k];
        double * C_k = &C_ptr[m * n * k];
        if (n == 1) { // Vector rhs
            // cblas_dgemv(CblasColMajor, CblasNoTrans, m, r, 1, A_k, m, B_k, 1, 0, C_k, 1);
            dgemv_(transpA.c_str(), &dimA0, &dimA1, &one, A_k, &dimA0, B_k, &inc, &zero, C_k, &inc);
        } else { // Matrix rhs
            // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, r, 1, A_k, m, B_k, r, 0, C_k, m);
            dgemm_(transpA.c_str(), transpB.c_str(), &m, &n, &r1, &one, A_k, &dimA0, B_k, &dimB0, &zero, C_k, &m);
        }
    }, ap);

    plhs[0] = C;
}

static
void doTrisolve(mxArray * plhs[],
                const mxArray * A, const size_t * dimA,
                const mxArray * B, const size_t * dimB,
                const std::string& uplo,
                size_t nPages) {
    lapack_int m = dimA[0];
    if (dimA[1] != m || dimB[0] != m) {
        mexErrMsgIdAndTxt(errInputId, "Dimensions do not match.");
    }
    lapack_int n = dimB[1];

    const double * A_ptr = mxGetDoubles(A);
    const double * B_ptr = mxGetDoubles(B);

    size_t dimX[] = {static_cast<size_t>(m), static_cast<size_t>(n), nPages};
    mxArray * X = mxCreateUninitNumericArray(3, dimX, mxDOUBLE_CLASS, mxREAL);
    double * X_ptr = mxGetDoubles(X);

    double one = 1, zero = 0;
    lapack_int inc = 1;
    
    tbb::parallel_for(0, static_cast<int>(nPages), 1, [&](int k){
        const double * A_k = &A_ptr[m * m * k];
        const double * B_k = &B_ptr[m * n * k];
        double * X_k = &X_ptr[m * n * k];

        // Copy B_k to X_k. This will be overwritten during computation.
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 0, m, n, B_k, m, X_k, m);
        if (n == 1) { // Vector rhs
            dtrsv_(uplo.c_str(), "N", "N", &m, A_k, &m, X_k, &inc);
        } else { // Matrix rhs
            dtrsm_("L", uplo.c_str(), "N", "N", &m, &n, &one, A_k, &m, X_k, &m);
        }
    });

    plhs[0] = X;
}

static
void doCholesky(mxArray * plhs[], const mxArray * A, const size_t * dimA, size_t nPages) {
    size_t m = dimA[0];
    if (dimA[1] != m) {
        mexErrMsgIdAndTxt(errInputId, "Matrices are not square.");
    }

    size_t dimL[] = {m, m, nPages};
    mxArray * L = mxCreateNumericArray(3, dimL, mxDOUBLE_CLASS, mxREAL);
    mxArray * info = mxCreateUninitNumericArray(1, &nPages, mxINT64_CLASS, mxREAL);

    const double * A_ptr = mxGetDoubles(A);
    double * L_ptr = mxGetDoubles(L);
    lapack_int * info_ptr = reinterpret_cast<lapack_int *>(mxGetInt64s(info));

    tbb::parallel_for(0, static_cast<int>(nPages), 1, [&](int k){
        const double * A_k = &A_ptr[m * m * k];
        double * L_k = &L_ptr[m * m * k];

        // Copy lower triangle of A_k to L_k. This will be overwritten with the factorization.
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'L', m, m, A_k, m, L_k, m);
        info_ptr[k] = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', m, L_k, m);
    });

    plhs[0] = L;
    plhs[1] = info;
}

static
void doCholsolve(mxArray * plhs[],
                 const mxArray * L, const size_t * dimL,
                 const mxArray * B, const size_t * dimB,
                 size_t nPages) {
    size_t m = dimL[0], n = dimB[1];
    if (dimL[1] != m || dimB[0] != m) {
        mexErrMsgIdAndTxt(errInputId, "Dimensions do not match.");
    }

    size_t dimX[] = {m, n, nPages};
    mxArray * X = mxCreateUninitNumericArray(3, dimX, mxDOUBLE_CLASS, mxREAL);

    const double * L_ptr = mxGetDoubles(L);
    const double * B_ptr = mxGetDoubles(B);
    double * X_ptr = mxGetDoubles(X);

    tbb::parallel_for(0, static_cast<int>(nPages), 1, [&](int k){
        const double * L_k = &L_ptr[m * m * k];
        const double * B_k = &B_ptr[m * n * k];
        double * X_k = &X_ptr[m * n * k];

        // Copy B_k to X_k. This will be overwritten with the solution.
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 0, m, n, B_k, m, X_k, m);
        LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', m, n, L_k, m, X_k, m);
    });

    plhs[0] = X;
}

static
void doCholCong(mxArray * plhs[],
                 const mxArray * L, const size_t * dimL,
                 const mxArray * B, const size_t * dimB,
                 size_t nPages) {
    lapack_int m = dimL[0];
    if (dimL[1] != m || dimB[0] != m || dimB[1] != m) {
        mexErrMsgIdAndTxt(errInputId, "Dimensions do not match.");
    }

    size_t dimX[] = {static_cast<size_t>(m), static_cast<size_t>(m), nPages};
    mxArray * X = mxCreateUninitNumericArray(3, dimX, mxDOUBLE_CLASS, mxREAL);

    const double * L_ptr = mxGetDoubles(L);
    const double * B_ptr = mxGetDoubles(B);
    double * X_ptr = mxGetDoubles(X);

    double one = 1, zero = 0;
    
    tbb::parallel_for(0, static_cast<int>(nPages), 1, [&](int k){
        const double * L_k = &L_ptr[m * m * k];
        const double * B_k = &B_ptr[m * m * k];
        double * X_k = &X_ptr[m * m * k];

        // Copy B_k to X_k. This will be overwritten during computation.
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 0, m, m, B_k, m, X_k, m);
        dtrsm_("L", "L", "N", "N", &m, &m, &one, L_k, &m, X_k, &m);
        dtrsm_("R", "L", "T", "N", &m, &m, &one, L_k, &m, X_k, &m);
    });

    plhs[0] = X;
}

static
void doQR(mxArray * plhs[], const mxArray * A, const size_t * dimA, size_t nPages) {
    size_t m = dimA[0], n = dimA[1];
    size_t r = std::min(m, n);

    const double * A_ptr = mxGetDoubles(A);

    size_t dimQ[] = {m, r, nPages};
    mxArray * Q = mxCreateUninitNumericArray(3, dimQ, mxDOUBLE_CLASS, mxREAL);
    double * Q_ptr = mxGetDoubles(Q);

    size_t dimR[] = {m, n, nPages};
    mxArray * R = mxCreateNumericArray(3, dimR, mxDOUBLE_CLASS, mxREAL);
    double * R_ptr = mxGetDoubles(R);

    size_t dimP[] = {n, nPages};
    mxArray * P = mxCreateNumericArray(2, dimP, mxINT64_CLASS, mxREAL);
    lapack_int * P_ptr = reinterpret_cast<lapack_int *>(mxGetInt64s(P));
    
    // Create a temporary A for each thread
    tbb::enumerable_thread_specific< std::vector<double, tbb::cache_aligned_allocator<double>> > thread_A(m * n);

    tbb::enumerable_thread_specific< std::vector<double, tbb::cache_aligned_allocator<double>> > thread_tau(r);
    
    // Figure out how much workspace is required for each routine
    double query;
    LAPACKE_dgeqp3_work(LAPACK_COL_MAJOR, m, n, nullptr, m, nullptr, nullptr, &query, -1);
    size_t work_size = static_cast<size_t>(query);
    LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, m, r, r, nullptr, m, nullptr, &query, -1);
    work_size = std::max(work_size, static_cast<size_t>(query));
    tbb::enumerable_thread_specific< std::vector<double, tbb::cache_aligned_allocator<double> > > thread_work(work_size);
    
    tbb::parallel_for(0, static_cast<int>(nPages), 1, [&](int k){
        const double * A_k = &A_ptr[m * n * k];
        double * A_local = thread_A.local().data();
        double * work_local = thread_work.local().data();
        double * tau_local = thread_tau.local().data();
        double * Q_k     = &Q_ptr[m * r * k];
        double * R_k     = &R_ptr[m * n * k];
        lapack_int * P_k  = &P_ptr[n * k];
        
        // Make a local copy of A_k. This will be overwritten during computation.
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 0, m, n, A_k, m, A_local, m);
        // Compute R
        LAPACKE_dgeqp3_work(LAPACK_COL_MAJOR, m, n, A_local, m, P_k, tau_local, work_local, work_size);
        // Copy R from upper triangle of A
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', m, n, A_local, m, R_k, m);
        // left columns of A get replaced with Q
        LAPACKE_dorgqr_work(LAPACK_COL_MAJOR, m, r, r, A_local, m, tau_local, work_local, work_size);
        // Copy Q into packed array
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 0, m, r, A_local, m, Q_k, m);
    });
    
    plhs[0] = Q;
    plhs[1] = R;
    plhs[2] = P;
}

static
void doLS(mxArray * plhs[],
          const mxArray * A, const size_t * dimA,
          const mxArray * B, const size_t * dimB,
          size_t nPages) {
    lapack_int m = dimA[0], r = dimA[1], n = dimB[1];
    if (dimB[0] != m) {
        mexErrMsgIdAndTxt(errInputId, "Dimensions do not match.");
    }

    const double * A_ptr = mxGetDoubles(A);
    const double * B_ptr = mxGetDoubles(B);

    mxArray * info = mxCreateUninitNumericArray(1, &nPages, mxINT64_CLASS, mxREAL);
    lapack_int * info_ptr = reinterpret_cast<lapack_int *>(mxGetInt64s(info));

    size_t dimX[] = {static_cast<size_t>(r), static_cast<size_t>(n), static_cast<size_t>(nPages)};
    mxArray * X = mxCreateUninitNumericArray(3, dimX, mxDOUBLE_CLASS, mxREAL);
    double * X_ptr = mxGetDoubles(X);

    // Create a temporary A and B for each thread
    tbb::enumerable_thread_specific< std::vector<double, tbb::cache_aligned_allocator<double>> > thread_A(m * r);
    tbb::enumerable_thread_specific< std::vector<double, tbb::cache_aligned_allocator<double>> > thread_B(m * n);

    // Figure out how much workspace is required
    double query;
    LAPACKE_dgels_work(LAPACK_COL_MAJOR, 'N', m, r, n, nullptr, m, nullptr, m, &query, -1);
    size_t work_size = static_cast<size_t>(query);
    tbb::enumerable_thread_specific< std::vector<double, tbb::cache_aligned_allocator<double> > > thread_work(work_size);

    tbb::parallel_for(0, static_cast<int>(nPages), 1, [&](int k){
        // Make a local copy of A_k. This will be overwritten during computation.
        const double * A_k = &A_ptr[m * r * k];
        double * A_local = thread_A.local().data();
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 0, m, r, A_k, m, A_local, m);

        // Make a local copy of B_k. This will be overwritten during computation.
        const double * B_k = &B_ptr[m * n * k];
        double * B_local = thread_B.local().data();
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 0, m, n, B_k, m, B_local, m);
        
        // Solve the least-squares problem
        double * work_local = thread_work.local().data();
        info_ptr[k] = LAPACKE_dgels_work(LAPACK_COL_MAJOR, 'N', m, r, n, A_local, m, B_local, m, work_local, work_size);

        // Copy result to X
        double * X_k = &X_ptr[r * n * k];
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 0, r, n, B_local, m, X_k, r);
    });

    plhs[0] = X;
    plhs[1] = info;
}

static
void doSVD(mxArray * plhs[], const mxArray * A, const size_t * dimA, size_t rank, size_t nPages) {
    size_t m = dimA[0], n = dimA[1];
    size_t r = std::min(m, n);
    rank = std::min(rank, r);

    const double * A_ptr = mxGetDoubles(A);

    size_t dimU[] = {m, rank, nPages};
    mxArray * U = mxCreateUninitNumericArray(3, dimU, mxDOUBLE_CLASS, mxREAL);
    double * U_ptr = mxGetDoubles(U);

    size_t dimS[] = {rank, nPages};
    mxArray * S = mxCreateUninitNumericArray(2, dimS, mxDOUBLE_CLASS, mxREAL);
    double * S_ptr = mxGetDoubles(S);

    size_t dimVt[] = {rank, n, nPages};
    mxArray * Vt = mxCreateUninitNumericArray(3, dimVt, mxDOUBLE_CLASS, mxREAL);
    double * Vt_ptr = mxGetDoubles(Vt);

    mxArray  * info = mxCreateUninitNumericArray(1, &nPages, mxINT64_CLASS, mxREAL);
    lapack_int * info_ptr = reinterpret_cast<lapack_int *>(mxGetInt64s(info));

    tbb::enumerable_thread_specific< std::vector<double, tbb::cache_aligned_allocator<double> > > thread_A(m * n);
    tbb::enumerable_thread_specific< std::vector<double, tbb::cache_aligned_allocator<double> > > thread_U(m * r);
    tbb::enumerable_thread_specific< std::vector<double, tbb::cache_aligned_allocator<double> > > thread_S(r);
    tbb::enumerable_thread_specific< std::vector<double, tbb::cache_aligned_allocator<double> > > thread_Vt(r * n);

    // Figure out how much workspace is required for each routine
    double query;
    LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, 'S', 'S', m, n, nullptr, m, nullptr, nullptr, m, nullptr, r, &query, -1);
    size_t work_size = static_cast<size_t>(query);
    tbb::enumerable_thread_specific< std::vector<double, tbb::cache_aligned_allocator<double> > > thread_work(work_size);

    tbb::parallel_for(0, static_cast<int>(nPages), 1, [&](int k){
        // Make a local copy of A_k. This will be overwritten during computation.
        const double * A_k = &A_ptr[m * n * k];
        double * A_local = thread_A.local().data();
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 0, m, n, A_k, m, A_local, m);
        
        double * U_local = thread_U.local().data();
        double * S_local = thread_S.local().data();
        double * Vt_local = thread_Vt.local().data();
        double * work_local = thread_work.local().data();
        info_ptr[k] = LAPACKE_dgesvd_work(
            LAPACK_COL_MAJOR, 'S', 'S', m, n, A_local, m,
            S_local, U_local, m, Vt_local, r,
            work_local, work_size);

        // Copy only the first rank singular values and vectors.
        double * U_k     = &U_ptr[m * rank * k];
        double * S_k     = &S_ptr[rank * k];
        double * Vt_k  = &Vt_ptr[rank * n * k];
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 0, m, rank, U_local, m, U_k, m);
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 0, rank, 1, S_local, r, S_k, rank);
        LAPACKE_dlacpy(LAPACK_COL_MAJOR, 0, rank, n, Vt_local, r, Vt_k, rank);
    });
    
    plhs[0] = U;
    plhs[1] = S;
    plhs[2] = Vt;
    plhs[3] = info;
}

void mexFunction(int nlhs, mxArray * plhs[],
                 int nrhs, mxArray const * prhs[])
{
    static_assert(sizeof(ptrdiff_t) == sizeof(int64_t), "int64_t and ptrdiff_t are inequivalent");

    if (nrhs < 2 || !mxIsChar(prhs[0])) mexErrMsgIdAndTxt(errInputId, "Must provide an operation and input matrix.");

    // Get string indicating which algorithm to perform.
    char * algStr_ptr = mxArrayToUTF8String(prhs[0]);
    const std::string algStr = algStr_ptr;
    mxFree(algStr_ptr);

    const mxArray * A = prhs[1];
    if (!mxIsDouble(A)) {
        mexErrMsgIdAndTxt(errInputId, "Inputs must be of type double.");
    }
    size_t nDimA = mxGetNumberOfDimensions(A);
    if (nDimA < 2 || nDimA > 3) {
        mexErrMsgIdAndTxt(errInputId, "batchop operates on 2D or 3D arrays only.");
    }
    const size_t * dimA = mxGetDimensions(A);
    const lapack_int nPages = (nDimA == 2) ? 1 : dimA[2];

    /******** Unary Operations ********/
    if (algStr == "chol") {
        doCholesky(plhs, A, dimA, nPages);
        return;
    }

    if (algStr == "qr") {
        doQR(plhs, A, dimA, nPages);
        return;
    }
    
    if (algStr == "svd") {
        if (nrhs != 3 || !mxIsScalar(prhs[2])) {
            mexErrMsgIdAndTxt(errInputId, "Must provide desired rank.");
        }
        doSVD(plhs, A, dimA, static_cast<lapack_int>(mxGetScalar(prhs[2])), nPages);
        return;
    } 
    
    /******** Binary Operations ********/
    if (nrhs < 3) {
        mexErrMsgIdAndTxt(errInputId, "Binary operation requires two inputs.");
    }

    const mxArray * B = prhs[2];
    size_t nDimB = mxGetNumberOfDimensions(B);
    const size_t * dimB = mxGetDimensions(B);
    if (!mxIsDouble(B)) {
        mexErrMsgIdAndTxt(errInputId, "Inputs must be of type double.");
    } else if (nDimB != nDimA || (nPages > 1 && dimB[2] != nPages)) {
        mexErrMsgIdAndTxt(errInputId, "Dimensions do not match.");
    }

    if (algStr == "mult") {
        std::string transpA = "N";
        if (nrhs >= 4) {
            if (!mxIsChar(prhs[3])) {
                mexErrMsgIdAndTxt(errInputId, "Usage: batchop('mult', A, B, ['N' or 'T', 'N' or 'T'])");
            }
            char * transpA_ptr = mxArrayToUTF8String(prhs[3]);
            transpA = transpA_ptr;
            mxFree(transpA_ptr);
            if (transpA != "N" && transpA != "T") {
                mexErrMsgIdAndTxt(errInputId, "Usage: batchop('mult', A, B, ['N' or 'T', 'N' or 'T'])");
            }
        }

        std::string transpB = "N";
        if (nrhs >= 5) {
            if (!mxIsChar(prhs[4])) {
                mexErrMsgIdAndTxt(errInputId, "Usage: batchop('mult', A, B, ['N' or 'T', 'N' or 'T'])");
            }
            char * transpB_ptr = mxArrayToUTF8String(prhs[4]);
            transpB = transpB_ptr;
            mxFree(transpB_ptr);
            if (transpB != "N" && transpB != "T") {
                mexErrMsgIdAndTxt(errInputId, "Usage: batchop('mult', A, B, ['N' or 'T', 'N' or 'T'])");
            }
        }

        doMult(plhs, A, dimA, transpA,
                        B, dimB, transpB,
                        nPages);
        return;
    }
    
    if (algStr == "trisolve") {
        if (nrhs != 4 || !mxIsChar(prhs[3])) {
            mexErrMsgIdAndTxt(errInputId, "Usage: batchop('trisolve', T, B, 'U' or 'L')");
        }

        char * uplo_ptr = mxArrayToUTF8String(prhs[3]);
        const std::string uplo = uplo_ptr;
        mxFree(uplo_ptr);
        if (uplo != "U" && uplo != "L") {
            mexErrMsgIdAndTxt(errInputId, "Usage: batchop('trisolve', T, B, 'U' or 'L')");
        }

        doTrisolve(plhs, A, dimA, B, dimB, uplo, nPages);
        return;
    }

    if (algStr == "cholsolve") {
        doCholsolve(plhs, A, dimA, B, dimB, nPages);
        return;
    }

    if (algStr == "cholcong") {
        doCholCong(plhs, A, dimA, B, dimB, nPages);
        return;
    }

    if (algStr == "leastsq") {
        doLS(plhs, A, dimA, B, dimB, nPages);
        return;
    }

    mexErrMsgIdAndTxt(errInputId, "Unknown operation.");
}

