/*
 * \file cuda_context.h
 */

#include "cuda_helper.h"

// container for cuda and cudnn resources
class CudaContext {
   public:
    CudaContext() {
        cublasCreate(&_cublas_handle);
        checkCudaErrors(cudaGetLastError());
        checkCudnnErrors(cudnnCreate(&_cudnn_handle));
    }
    ~CudaContext() {
        cublasDestroy(_cublas_handle);
        checkCudnnErrors(cudnnDestroy(_cudnn_handle));
    }

    cublasHandle_t cublas() { return _cublas_handle; };

    cudnnHandle_t cudnn() { return _cudnn_handle; };

    flt_type const one = 1.f;
    flt_type const zero = 0.f;
    flt_type const minus_one = -1.f;

   private:
    cublasHandle_t _cublas_handle;
    cudnnHandle_t _cudnn_handle;
};
