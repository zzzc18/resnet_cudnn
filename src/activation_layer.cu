/*
 * \file activation_layer.cu
 */

#include <cmath>

#include "activation_layer.h"

void Activation::InitFeatureShape() { out_shape_ = in_shape_; }

void Activation::InitWeightsShape(std::vector<std::array<int, 4>> &w_l,
                                  std::vector<std::array<int, 4>> &b_l) {
    w_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});
    b_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});

    return;
}

void Activation::Forward() {
    input_desc_ = input_.tensor();
    output_desc_ = output_.tensor();

    checkCudnnErrors(cudnnActivationForward(
        cuda_->cudnn(), act_desc_, &cuda_->one, input_desc_, input_.CudaPtr(),
        &cuda_->zero, output_desc_, output_.CudaPtr()));

    return;
}

void Activation::Backward(BlobPointer<float> const &labels) {
    checkCudnnErrors(cudnnActivationBackward(
        cuda_->cudnn(), act_desc_, &cuda_->one, output_desc_, output_.CudaPtr(),
        output_desc_, output_.CudaPtr(), input_desc_, input_.CudaPtr(),
        &cuda_->zero, input_desc_, d_temp_grad_features_));
    this->BackwardCopy();
    return;
}
