/*
 * \file convolutional_layer.cu
 */

#include <iostream>

#include "residual_layer.h"

void Residual::InitFeatureShape() { out_shape_ = in_shape_; }

void Residual::InitWeightsShape(std::vector<std::array<int, 4>> &w_l,
                                std::vector<std::array<int, 4>> &b_l) {
    w_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});
    b_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});
    return;
}

void Residual::DescriptorsAndWorkSpace() {
    output_desc_ = output_.tensor();
    input_desc_ = input_.tensor();
    return;
}

void Residual::Forward() {
    checkCudaErrors(cudaMemset(output_.CudaPtr(), 0, output_.buf_size()));
    checkCublasErrors(cublasSaxpy(
        cuda_->cublas(), output_.LengthNchw(), &cuda_->one,
        inputLayer1_->GetOutput().CudaPtr(), 1, output_.CudaPtr(), 1));
    checkCublasErrors(cublasSaxpy(
        cuda_->cublas(), output_.LengthNchw(), &cuda_->one,
        inputLayer2_->GetOutput().CudaPtr(), 1, output_.CudaPtr(), 1));
    return;
}

void Residual::Backward(BlobPointer<float> const &labels) {
    // Special case for BackwardCopy
    if (inputLayer1_->GetLayerType() == LayerType::Split) {
        checkCublasErrors(cublasSaxpy(cuda_->cublas(),
                                      inputLayer1_->GetOutput().LengthNchw(),
                                      &cuda_->one, output_.CudaPtr(), 1,
                                      inputLayer1_->GetOutput().CudaPtr(), 1));
    } else {
        checkCudaErrors(cudaMemcpy(inputLayer1_->GetOutput().CudaPtr(),
                                   output_.CudaPtr(), output_.buf_size(),
                                   cudaMemcpyDeviceToDevice));
    }

    if (inputLayer2_->GetLayerType() == LayerType::Split) {
        checkCublasErrors(cublasSaxpy(cuda_->cublas(),
                                      inputLayer2_->GetOutput().LengthNchw(),
                                      &cuda_->one, output_.CudaPtr(), 1,
                                      inputLayer2_->GetOutput().CudaPtr(), 1));
    } else {
        checkCudaErrors(cudaMemcpy(inputLayer2_->GetOutput().CudaPtr(),
                                   output_.CudaPtr(), output_.buf_size(),
                                   cudaMemcpyDeviceToDevice));
    }
    return;
}

////////////

void Split::InitFeatureShape() { out_shape_ = in_shape_; }

void Split::InitWeightsShape(std::vector<std::array<int, 4>> &w_l,
                             std::vector<std::array<int, 4>> &b_l) {
    w_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});
    b_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});
    return;
}

void Split::DescriptorsAndWorkSpace() {
    output_desc_ = output_.tensor();
    input_desc_ = input_.tensor();
    return;
}

void Split::Forward() {
    checkCudaErrors(cudaMemcpy(output_.CudaPtr(), input_.CudaPtr(),
                               input_.buf_size(), cudaMemcpyDeviceToDevice));
    // need to clear output_ before backward
    return;
}

void Split::Backward(BlobPointer<float> const &labels) {
    if (afterSplitLayer_) {
        checkCublasErrors(cublasSaxpy(cuda_->cublas(), input_.LengthNchw(),
                                      &cuda_->one, output_.CudaPtr(), 1,
                                      input_.CudaPtr(), 1));
    } else {
        checkCudaErrors(cudaMemcpy(input_.CudaPtr(), output_.CudaPtr(),
                                   output_.buf_size(),
                                   cudaMemcpyDeviceToDevice));
    }
    return;
}