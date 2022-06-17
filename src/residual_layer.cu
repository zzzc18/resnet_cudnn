/*
 * \file convolutional_layer.cu
 */

#include <iostream>

#include "residual_layer.h"

void Residual::InitFeatureShape() { out_shape_ = in_shape_; }

void Residual::InitWeightsShape(std::vector<std::array<int, 4>> &w_l,
                                std::vector<std::array<int, 4>> &b_l) {
    // nothing actually
    w_l.emplace_back(std::array<int, 4>{1, 1, 1, 1});
    b_l.emplace_back(std::array<int, 4>{1, 1, 1, 1});
    return;
}

void Residual::DescriptorsAndWorkSpace() {
    output_desc_ = output_.tensor();
    input_desc_ = input_.tensor();
    return;
}

void Residual::Forward() {
    InitiateZeros<<<(output_.LengthNchw() + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D,
                    BLOCK_DIM_1D>>>(output_.CudaPtr(), output_.LengthNchw());
    // checkCudaErrors(
    //     cudaMemset(output_.CudaPtr(), 0, sizeof(float) *
    //     output_.LengthNchw()));
    cudnnAddTensor(
        cuda_->cudnn(),  // cudnnHandle_t                     handle,
        &cuda_->one,     // const void                       *alpha,
        inputLayer1_->GetOutputDesc(),  // const cudnnTensorDescriptor_t aDesc,
        inputLayer1_->GetOutput().CudaPtr(),  // const void *A,
        &cuda_->one,       // const void                       *beta,
        output_desc_,      // const cudnnTensorDescriptor_t     cDesc,
        output_.CudaPtr()  // void                             *C
    );
    cudnnAddTensor(
        cuda_->cudnn(),  // cudnnHandle_t                     handle,
        &cuda_->one,     // const void                       *alpha,
        inputLayer2_->GetOutputDesc(),  // const cudnnTensorDescriptor_t aDesc,
        inputLayer2_->GetOutput().CudaPtr(),  // const void *A,
        &cuda_->one,       // const void                       *beta,
        output_desc_,      // const cudnnTensorDescriptor_t     cDesc,
        output_.CudaPtr()  // void                             *C
    );
    return;
}

void Residual::Backward(BlobPointer<float> const &labels) {
    checkCudaErrors(cudaMemcpy(
        inputLayer1_->GetGradOutput().CudaPtr(), grad_output_.CudaPtr(),
        sizeof(float) * output_.LengthNchw(), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(
        inputLayer2_->GetGradOutput().CudaPtr(), grad_output_.CudaPtr(),
        sizeof(float) * output_.LengthNchw(), cudaMemcpyDeviceToDevice));
    return;
}
