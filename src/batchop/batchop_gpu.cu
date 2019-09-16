
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <cusolverDn.h>

static char const * const errInputId = "batchop_gpu:InvalidInput";
static char const * const errCudaId = "batchop_gpu:CudaError";
static char const * const errCudaMsg = "batchop_gpu encountered a cuda error.";

static bool solverInitialized = false;
static cublasHandle_t cublasHandle = NULL;
static cusolverDnHandle_t cusolverHandle = NULL;
static syevjInfo_t syevj_params = NULL;
static gesvdjInfo_t gesvdj_params = NULL;

static mxArray * gpuCanary;

static void uninit();

static bool init() {
    if (!solverInitialized || !mxGPUIsValidGPUData(gpuCanary)) {
        // Initialize the MATLAB GPU API if not already initialized.
        if (mxInitGPU() != MX_GPU_SUCCESS) {
            return false;
        }

        const size_t one = 1;
        mxGPUArray * canary = mxGPUCreateGPUArray(1, &one, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        gpuCanary = mxGPUCreateMxArrayOnGPU(canary);
        mxGPUDestroyGPUArray(canary);
        mexMakeArrayPersistent(gpuCanary);

        cublasStatus_t blastat = cublasCreate(&cublasHandle);
        if (blastat != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        
        cusolverStatus_t status = cusolverDnCreate(&cusolverHandle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            cublasDestroy(cublasHandle);
            return false;
        }

        status = cusolverDnCreateSyevjInfo(&syevj_params);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            cusolverDnDestroy(cusolverHandle);
            cublasDestroy(cublasHandle);
            return false;
        }

        status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            cusolverDnDestroySyevjInfo(syevj_params);
            cusolverDnDestroy(cusolverHandle);
            cublasDestroy(cublasHandle);
            return false;
        }
        
        solverInitialized = true;
        mexAtExit(uninit);
    }
    return true;
}

static void uninit() {
    if (solverInitialized) {
        if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);
        if (syevj_params) cusolverDnDestroySyevjInfo(syevj_params);
        if (cusolverHandle) cusolverDnDestroy(cusolverHandle);
        if (cublasHandle) cublasDestroy(cublasHandle);
        mxDestroyArray(gpuCanary);
    }
}

static
mxGPUArray *
doMult(const mxGPUArray *A, const size_t * dimA, const std::string& transpA,
       const mxGPUArray *B, const size_t * dimB,  const std::string& transpB,
       size_t nPages) {
    size_t dimA0 = dimA[0], dimB0 = dimB[0];
    size_t m = dimA[0], r1 = dimA[1], r2 = dimB[0], n = dimB[1];
    cublasOperation_t opA = CUBLAS_OP_N, opB = CUBLAS_OP_N;
    if (transpA == "T") {
        m = dimA[1];
        r1 = dimA[0];
        opA = CUBLAS_OP_T;
    }
    if (transpB == "T") {
        r2 = dimB[1];
        n = dimB[0];
        opB = CUBLAS_OP_T;
    }

    if (r1 != r2) {
        mexErrMsgIdAndTxt(errInputId, "Inner dimensions do not match.");
    }

    size_t dimC[] = {m, n, nPages};
    mxGPUArray *C = mxGPUCreateGPUArray(3, dimC, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Scalar required by cublas.
    const double one = 1.0, zero = 0.0;

    // Compute A * B
    cublasStatus_t status = cublasDgemmStridedBatched(
        cublasHandle, opA, opB, m, n, r1,
        &one, static_cast<const double *>(mxGPUGetDataReadOnly(A)), dimA0, m * r1,
        static_cast<const double *>(mxGPUGetDataReadOnly(B)), dimB0, r1 * n,
        &zero, static_cast<double *>(mxGPUGetData(C)), m, m * n,
        nPages);
    if (status != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }

    return C;
}

static
thrust::tuple<mxGPUArray *, mxGPUArray *>
doCholesky(const mxGPUArray *A, const size_t * dimA, size_t nPages) {
    size_t m = dimA[0];
    if (dimA[1] != m) {
        mexErrMsgIdAndTxt(errInputId, "Matrices are not square.");
    }

    // Copy A into L. This will be overwritten with the Cholesky factor in place.
    mxGPUArray *L = mxGPUCopyGPUArray(A);

    // Create array to store factorization status.
    mxGPUArray *info = mxGPUCreateGPUArray(1, &nPages, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Create array of pointers into pages of L.
    thrust::device_vector<size_t> ptrL(nPages, m * m * sizeof(double));
    thrust::exclusive_scan(ptrL.begin(), ptrL.end(), ptrL.begin(),
                           reinterpret_cast<size_t>(mxGPUGetData(L)));

    // Factor L = A in place.
    cusolverStatus_t status = cusolverDnDpotrfBatched(
        cusolverHandle, CUBLAS_FILL_MODE_LOWER, m,
        reinterpret_cast<double **>(ptrL.data().get()), m,
        static_cast<int *>(mxGPUGetData(info)),
        nPages);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }

    return thrust::make_tuple(L, info);
}

static
mxGPUArray *
doCholsolve(const mxGPUArray * L, const size_t * dimL,
            const mxGPUArray * B, const size_t * dimB,
            size_t nPages) {
    size_t m = dimL[0], n = dimB[1];
    if (dimL[0] != m || dimB[0] != m) {
        mexErrMsgIdAndTxt(errInputId, "Dimensions do not match.");
    }
    
    // Copy B into X. This will be overwritten with the solution.
    mxGPUArray *X = mxGPUCopyGPUArray(B);

    // Create array of pointers into pages of L.
    thrust::device_vector<size_t> ptrL(nPages, m * m * sizeof(double));
    thrust::exclusive_scan(ptrL.begin(), ptrL.end(), ptrL.begin(),
                           reinterpret_cast<size_t>(mxGPUGetDataReadOnly(L)));


    // Create array of pointers into pages of X.
    thrust::device_vector<size_t> ptrX(nPages, m * n * sizeof(double));
    thrust::exclusive_scan(ptrX.begin(), ptrX.end(), ptrX.begin(),
                           reinterpret_cast<size_t>(mxGPUGetData(X)));


    if (n == 1) {
        // Dummy array to store solve status.
        size_t one = 1;
        mxGPUArray *info = mxGPUCreateGPUArray(1, &one, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

        cusolverStatus_t status = cusolverDnDpotrsBatched(
            cusolverHandle, CUBLAS_FILL_MODE_LOWER, m, n,
            reinterpret_cast<double **>(ptrL.data().get()), m,
            reinterpret_cast<double **>(ptrX.data().get()), m,
            static_cast<int *>(mxGPUGetData(info)),
            nPages);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
        }

        mxGPUDestroyGPUArray(info);
    } else {
        double one = 1.0;
        
        // Compute X = L \ X
        cublasStatus_t status = cublasDtrsmBatched(
            cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            m, n,
            &one, reinterpret_cast<double **>(ptrL.data().get()), m,
            reinterpret_cast<double **>(ptrX.data().get()), m,
            nPages);
        if (status != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
        }

        // Compute X = L' \ X = L' \ (L \ B)
        status = cublasDtrsmBatched(
            cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            m, n,
            &one, reinterpret_cast<double **>(ptrL.data().get()), m,
            reinterpret_cast<double **>(ptrX.data().get()), m,
            nPages);
        if (status != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
        }
    }

    return X;
}

static
mxGPUArray *
doCholCong(const mxGPUArray * L, const size_t * dimL,
           const mxGPUArray * B, const size_t * dimB,
           size_t nPages) {
    size_t m = dimL[0];
    if (dimL[1] != m || dimB[0] != m || dimB[1] != m) {
        mexErrMsgIdAndTxt(errInputId, "Dimensions do not match.");
    }

    // Copy B into X. This will be overwritten with the solution.
    mxGPUArray *X = mxGPUCopyGPUArray(B);

    // Create array of pointers into pages of L.
    thrust::device_vector<size_t> ptrL(nPages, m * m * sizeof(double));
    thrust::exclusive_scan(ptrL.begin(), ptrL.end(), ptrL.begin(),
                           reinterpret_cast<size_t>(mxGPUGetDataReadOnly(L)));


    // Create array of pointers into pages of X.
    thrust::device_vector<size_t> ptrX(nPages, m * m * sizeof(double));
    thrust::exclusive_scan(ptrX.begin(), ptrX.end(), ptrX.begin(),
                           reinterpret_cast<size_t>(mxGPUGetData(X)));

    // Scalar required by cublas.
    double one = 1.0;

    // Compute X = L \ X
    cublasStatus_t status = cublasDtrsmBatched(
        cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        m, m,
        &one, reinterpret_cast<double **>(ptrL.data().get()), m,
        reinterpret_cast<double **>(ptrX.data().get()), m,
        nPages);
    if (status != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }

    // Compute X = X / L' = L \ B / L'
    status = cublasDtrsmBatched(
        cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
        m, m,
        &one, reinterpret_cast<double **>(ptrL.data().get()), m,
        reinterpret_cast<double **>(ptrX.data().get()), m,
        nPages);
    if (status != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }

    return X;
}

static
thrust::tuple<mxGPUArray *, mxGPUArray *>
doLS(const mxGPUArray * A, const size_t * dimA,
     const mxGPUArray * B, const size_t * dimB,
     size_t nPages) {
    size_t m = dimA[0], r = dimA[1], n = dimB[1];
    if (dimB[0] != m) {
        mexErrMsgIdAndTxt(errInputId, "Dimensions do not match.");
    }

    size_t dimX[] = {r, n, nPages};
    mxGPUArray * X = mxGPUCreateGPUArray(3, dimX, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Copy A and B. Both will be overwritten
    mxGPUArray * Ac = mxGPUCopyGPUArray(A);
    mxGPUArray * Bc = mxGPUCopyGPUArray(B);

    // Create array of pointers into pages of Ac and Bc.
    thrust::device_vector<size_t> ptrAc(nPages, m * r * sizeof(double));
    thrust::exclusive_scan(ptrAc.begin(), ptrAc.end(), ptrAc.begin(),
                           reinterpret_cast<size_t>(mxGPUGetData(Ac)));
    thrust::device_vector<size_t> ptrBc(nPages, m * n * sizeof(double));
    thrust::exclusive_scan(ptrBc.begin(), ptrBc.end(), ptrBc.begin(),
                           reinterpret_cast<size_t>(mxGPUGetData(Bc)));

    mxGPUArray * info = mxGPUCreateGPUArray(1, &nPages, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    int valid;
    cublasStatus_t status = cublasDgelsBatched(
        cublasHandle, CUBLAS_OP_N, m, r, n,
        reinterpret_cast<double **>(ptrAc.data().get()), m,
        reinterpret_cast<double **>(ptrBc.data().get()), m,
        &valid, static_cast<int *>(mxGPUGetData(info)), nPages
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }

    // Copy solution to X
    const double zero = 0.0, one = 1.0;
    status = cublasDgeam(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, r, n * nPages,
        &one, static_cast<const double *>(mxGPUGetDataReadOnly(Bc)), m,
        &zero, static_cast<const double *>(mxGPUGetDataReadOnly(X)), r,
        static_cast<double *>(mxGPUGetData(X)), r
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }

    mxGPUDestroyGPUArray(Ac);
    mxGPUDestroyGPUArray(Bc);
    return thrust::make_tuple(X, info);
}

static
thrust::tuple<mxGPUArray *, mxGPUArray *, mxGPUArray *>
doEig(const mxGPUArray * A, const size_t * dimA,
      cusolverEigMode_t eigmode, size_t nPages) {
    size_t m = dimA[0];
    if (dimA[1] != m) {
        mexErrMsgIdAndTxt(errInputId, "Matrices are not square.");
    }

    // Copy A into Q. This will be overwritten with the eigenvectors.
    mxGPUArray *Q = mxGPUCopyGPUArray(A);

    // Create array to store eigenvalues.
    size_t dimE[] = {m, nPages};
    mxGPUArray *E = mxGPUCreateGPUArray(2, dimE, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Create workspace
    int workSize;
    cusolverStatus_t status = cusolverDnDsyevjBatched_bufferSize(
        cusolverHandle, eigmode, CUBLAS_FILL_MODE_LOWER, m,                
        static_cast<const double *>(mxGPUGetData(Q)), m,              
        static_cast<const double *>(mxGPUGetData(E)),
        &workSize, syevj_params, nPages);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }
    size_t workSize2 = workSize;
    mxGPUArray *work = mxGPUCreateGPUArray(1, &workSize2, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Create array to store eigendecomposition status.
    mxGPUArray *info = mxGPUCreateGPUArray(1, &nPages, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Do factorization
    status = cusolverDnDsyevjBatched(
        cusolverHandle, eigmode, CUBLAS_FILL_MODE_LOWER, m,                
        static_cast<double *>(mxGPUGetData(Q)), m,              
        static_cast<double *>(mxGPUGetData(E)),
        static_cast<double *>(mxGPUGetData(work)), workSize,
        static_cast<int *>(mxGPUGetData(info)),
        syevj_params, nPages);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }

    mxGPUDestroyGPUArray(work);

    return thrust::make_tuple(E, Q, info);
}

static
thrust::tuple<mxGPUArray *, mxGPUArray *, mxGPUArray *, mxGPUArray *>
doSVD(const mxGPUArray *A, const size_t * dimA, size_t rank, size_t nPages) {
    size_t m = dimA[0], n = dimA[1];

    // Copy A. This will be overwritten during the computation.
    mxGPUArray *Ac = mxGPUCopyGPUArray(A);

    // Create arrays to store singular values and vectors
    size_t dimS[] = {std::min(m, n), nPages};
    size_t dimU[] = {m, m, nPages};
    size_t dimV[] = {n, n, nPages};
    mxGPUArray *S = mxGPUCreateGPUArray(2, dimS, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *U = mxGPUCreateGPUArray(3, dimU, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *V = mxGPUCreateGPUArray(3, dimV, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Create workspace
    int workSize;
    cusolverStatus_t status = cusolverDnDgesvdjBatched_bufferSize(
        cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, m, n,                
        static_cast<const double *>(mxGPUGetData(Ac)), m,              
        static_cast<const double *>(mxGPUGetData(S)), 
        static_cast<const double *>(mxGPUGetData(U)), m,              
        static_cast<const double *>(mxGPUGetData(V)), n,              
        &workSize,
        gesvdj_params, nPages);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }
    size_t workSize2 = workSize;
    mxGPUArray *work = mxGPUCreateGPUArray(1, &workSize2, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Create array to store SVD status.
    mxGPUArray *info = mxGPUCreateGPUArray(1, &nPages, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Do factorization
    status = cusolverDnDgesvdjBatched(
        cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, m, n,                
        static_cast<double *>(mxGPUGetData(Ac)), m,              
        static_cast<double *>(mxGPUGetData(S)),
        static_cast<double *>(mxGPUGetData(U)), m,              
        static_cast<double *>(mxGPUGetData(V)), n,
        static_cast<double *>(mxGPUGetData(work)), workSize,
        static_cast<int *>(mxGPUGetData(info)),
        gesvdj_params, nPages);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }

    mxGPUDestroyGPUArray(work);
    mxGPUDestroyGPUArray(Ac);

    if (rank < m && rank < n) {
        // Copy partial SVD
        size_t dimSr[] = {rank, nPages};
        size_t dimUr[] = {m, rank, nPages};
        size_t dimVr[] = {n, rank, nPages};
        mxGPUArray *Sr = mxGPUCreateGPUArray(2, dimSr, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        mxGPUArray *Ur = mxGPUCreateGPUArray(3, dimUr, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        mxGPUArray *Vr = mxGPUCreateGPUArray(3, dimVr, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

        cudaError_t custat = cudaMemcpy2D(
            mxGPUGetData(Sr), rank * sizeof(double),
            mxGPUGetData(S), std::min(m, n) * sizeof(double),
            rank * sizeof(double), nPages, cudaMemcpyDeviceToDevice);
        if (custat != cudaSuccess) {
            mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
        }

        custat = cudaMemcpy2D(
            mxGPUGetData(Ur), m * rank * sizeof(double),
            mxGPUGetData(U), m * m * sizeof(double),
            m * rank * sizeof(double), nPages, cudaMemcpyDeviceToDevice);
        if (custat != cudaSuccess) {
            mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
        }

        custat = cudaMemcpy2D(
            mxGPUGetData(Vr), n * rank * sizeof(double),
            mxGPUGetData(V), n * n * sizeof(double),
            n * rank * sizeof(double), nPages, cudaMemcpyDeviceToDevice);
        if (custat != cudaSuccess) {
            mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
        }

        mxGPUDestroyGPUArray(U);
        mxGPUDestroyGPUArray(S);
        mxGPUDestroyGPUArray(V);
        return thrust::make_tuple(Ur, Sr, Vr, info);
    }

    return thrust::make_tuple(U, S, V, info);
}

#if false
static
thrust::tuple<mxGPUArray *, mxGPUArray *, mxGPUArray *, mxGPUArray *>
doSVDApprox(const mxGPUArray *A, int rank) {
    const mwSize nDimA = mxGPUGetNumberOfDimensions(A);
    const mwSize *dimA = mxGPUGetDimensions(A);
    if (nDimA != 3) {
        mexErrMsgIdAndTxt(errInputId, errInputMsg);
    }
    size_t k = dimA[0];
    size_t l = dimA[1];
    size_t n = dimA[2];
    mxFree((void *)dimA);

    // Create arrays to store subset of singular values and vectors
    size_t dimS[] = {rank, n};
    size_t dimU[] = {k, rank, n};
    size_t dimV[] = {l, rank, n};
    mxGPUArray *S = mxGPUCreateGPUArray(2, dimS, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *U = mxGPUCreateGPUArray(3, dimU, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *V = mxGPUCreateGPUArray(3, dimV, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Create workspace
    int workSize;
    cusolverStatus_t status = cusolverDnDgesvdaStridedBatched_bufferSize(
        cusolverHandle,
        CUSOLVER_EIG_MODE_VECTOR,
        rank,
        k,                
        l,                
        static_cast<const double *>(mxGPUGetDataReadOnly(A)),
        k,
        k * l,
        static_cast<const double *>(mxGPUGetData(S)),
        rank,
        static_cast<const double *>(mxGPUGetData(U)),
        k,
        k * rank,
        static_cast<const double *>(mxGPUGetData(V)),
        l,
        l * rank,
        &workSize,
        n);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }
    size_t workSize2 = workSize;
    mxGPUArray *work = mxGPUCreateGPUArray(1, &workSize2, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Create array to store SVD status.
    mxGPUArray *info = mxGPUCreateGPUArray(1, &n, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Do factorization
    status = cusolverDnDgesvdaStridedBatched(
        cusolverHandle,
        CUSOLVER_EIG_MODE_VECTOR,
        rank,
        k,                
        l,                
        static_cast<const double *>(mxGPUGetDataReadOnly(A)),      
        k,
        k * l,
        static_cast<double *>(mxGPUGetData(S)),
        rank,
        static_cast<double *>(mxGPUGetData(U)),
        k,
        k * rank,
        static_cast<double *>(mxGPUGetData(V)),
        l,
        l * rank,
        static_cast<double *>(mxGPUGetData(work)),
        workSize,
        static_cast<int *>(mxGPUGetData(info)),
        NULL,
        n);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }

    // Clean up
    mxGPUDestroyGPUArray(work);

    return thrust::make_tuple(U, S, V, info);
}
#endif

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    // Initialize solver
    if (!init()) {
        mexErrMsgIdAndTxt(errCudaId, errCudaMsg);
    }

    if (nrhs < 2) {
        mexErrMsgIdAndTxt(errInputId, "Must provide an operation and input matrix.");
    }

    // Get string indicating which algorithm to perform.
    char * algStr_ptr = mxArrayToUTF8String(prhs[0]);
    const std::string algStr = algStr_ptr;
    mxFree(algStr_ptr);

	// Throw an error if the input is not a GPU array.
    if (!mxIsGPUArray(prhs[1]) || !mxGPUIsValidGPUData(prhs[1])) {
        mexErrMsgIdAndTxt(errInputId, "Inputs to batchop_gpu must be of type double gpuArray.");
    }

    // Unwrap input to an mxGPUArray (must be real double).
    const mxGPUArray *A = mxGPUCreateFromMxArray(prhs[1]);
    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS || mxGPUGetComplexity(A) != mxREAL) {
        mexErrMsgIdAndTxt(errInputId, "Inputs to batchop_gpu must be of type double gpuArray.");
    }

    size_t nDimA = mxGPUGetNumberOfDimensions(A);
    if (nDimA < 2 || nDimA > 3) {
        mexErrMsgIdAndTxt(errInputId, "batchop operates on 2D or 3D arrays only.");
    }
    const size_t * dimA = mxGPUGetDimensions(A);
    size_t nPages = (nDimA == 2) ? 1 : dimA[2];

    if (algStr == "chol") {
        mxGPUArray *L, *info;
        thrust::tie(L, info) = doCholesky(A, dimA, nPages);
        plhs[0] = mxGPUCreateMxArrayOnGPU(L);
        plhs[1] = mxGPUCreateMxArrayOnGPU(info);
        mxGPUDestroyGPUArray(L);
        mxGPUDestroyGPUArray(info);
    } else if (algStr == "eigval") {
        mxGPUArray *E, *Q, *info;
        thrust::tie(E, Q, info) = doEig(A, dimA, CUSOLVER_EIG_MODE_NOVECTOR, nPages);
        plhs[0] = mxGPUCreateMxArrayOnGPU(E);
        plhs[1] = mxGPUCreateMxArrayOnGPU(info);
        mxGPUDestroyGPUArray(E);
        mxGPUDestroyGPUArray(Q);
        mxGPUDestroyGPUArray(info);
    } else if (algStr == "eig") {
        mxGPUArray *E, *Q, *info;
        thrust::tie(E, Q, info) = doEig(A, dimA, CUSOLVER_EIG_MODE_VECTOR, nPages);
        plhs[0] = mxGPUCreateMxArrayOnGPU(E);
        plhs[1] = mxGPUCreateMxArrayOnGPU(Q);
        plhs[2] = mxGPUCreateMxArrayOnGPU(info);
        mxGPUDestroyGPUArray(E);
        mxGPUDestroyGPUArray(Q);
        mxGPUDestroyGPUArray(info);
    } else if (algStr == "svd") {
        if (nrhs != 3 || !mxIsScalar(prhs[2])) {
            mexErrMsgIdAndTxt(errInputId, "Must provide desired rank.");
        }

        mxGPUArray *U, *S, *V, *info;
        thrust::tie(U, S, V, info) = doSVD(A, dimA, static_cast<size_t>(mxGetScalar(prhs[2])), nPages);
        plhs[0] = mxGPUCreateMxArrayOnGPU(U);
        plhs[1] = mxGPUCreateMxArrayOnGPU(S);
        plhs[2] = mxGPUCreateMxArrayOnGPU(V);
        plhs[3] = mxGPUCreateMxArrayOnGPU(info);
        mxGPUDestroyGPUArray(U);
        mxGPUDestroyGPUArray(S);
        mxGPUDestroyGPUArray(V);
        mxGPUDestroyGPUArray(info);
    } else { // Binary operations
        if (nrhs < 3) {
            mexErrMsgIdAndTxt(errInputId, "Binary operation requires two inputs.");
        } else if (!mxIsGPUArray(prhs[2]) || !mxGPUIsValidGPUData(prhs[2])) {
            mexErrMsgIdAndTxt(errInputId, "Inputs to batchop_gpu must be of type double gpuArray.");
        }
        const mxGPUArray *B = mxGPUCreateFromMxArray(prhs[2]);
        if (mxGPUGetClassID(B) != mxDOUBLE_CLASS || mxGPUGetComplexity(B) != mxREAL) {
            mexErrMsgIdAndTxt(errInputId, "Inputs to batchop_gpu must be of type double gpuArray.");
        }

        size_t nDimB = mxGPUGetNumberOfDimensions(B);
        const size_t * dimB = mxGPUGetDimensions(B);
        if (nDimB != nDimA || (nPages > 1 && dimB[2] != nPages)) {
            mexErrMsgIdAndTxt(errInputId, "Dimensions do not match.");
        }

        if (algStr == "cholcong") {
            // A actually stores Cholesky factor L. Compute L \ B / L'
            mxGPUArray *X = doCholCong(A, dimA, B, dimB, nPages);
            plhs[0] = mxGPUCreateMxArrayOnGPU(X);
            mxGPUDestroyGPUArray(X);
        } else if (algStr == "cholsolve") {
            // A actually stores Cholesky factor L. Compute L' \ (L \ B)
            mxGPUArray *X = doCholsolve(A, dimA, B, dimB, nPages);
            plhs[0] = mxGPUCreateMxArrayOnGPU(X);
            mxGPUDestroyGPUArray(X);
        } else if (algStr == "leastsq") {
            mxGPUArray *X, *info;
            thrust::tie(X, info) = doLS(A, dimA, B, dimB, nPages);
            plhs[0] = mxGPUCreateMxArrayOnGPU(X);
            plhs[1] = mxGPUCreateMxArrayOnGPU(info);
            mxGPUDestroyGPUArray(X);
            mxGPUDestroyGPUArray(info);
        } else if (algStr == "mult") {
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
            mxGPUArray *C = doMult(A, dimA, transpA, B, dimB, transpB, nPages);
            plhs[0] = mxGPUCreateMxArrayOnGPU(C);
            mxGPUDestroyGPUArray(C);
        } else {
            mexErrMsgIdAndTxt(errInputId, "Unknown operation.");
        }

        mxFree((void *)dimB);
        mxGPUDestroyGPUArray(B);
    }

    mxFree((void *)dimA);
    mxGPUDestroyGPUArray(A);
}
