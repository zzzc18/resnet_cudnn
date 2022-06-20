/*
 * \file activation_layer.cu
 */

#include <cassert>

#include "inplace_relu.h"

void InplaceReLU::InitFeatureShape() { out_shape_ = in_shape_; }

void InplaceReLU::InitWeightsShape(std::vector<std::array<int, 4>> &w_l,
                                   std::vector<std::array<int, 4>> &b_l) {
    w_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});
    b_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});

    return;
}

void InplaceReLU::Forward() {
    assert(false);
    // input_desc_ = input_.tensor();
    // output_desc_ = output_.tensor();
    // assert(input_.CudaPtr() == output_.CudaPtr());
    // checkCudnnErrors(cudnnActivationForward(
    //     cuda_->cudnn(), act_desc_, &cuda_->one, input_desc_,
    //     input_.CudaPtr(), &cuda_->zero, output_desc_, output_.CudaPtr()));
    return;
}

void InplaceReLU::Backward(BlobPointer<float> const &labels) {
    assert(false);
    // assert(output_.CudaPtr() == input_.CudaPtr());
    // checkCudnnErrors(cudnnActivationBackward(
    //     cuda_->cudnn(), act_desc_, &cuda_->one, output_desc_,
    //     output_.CudaPtr(), output_desc_, grad_output_.CudaPtr(), input_desc_,
    //     input_.CudaPtr(), &cuda_->one, input_desc_, grad_input_.CudaPtr()));
    return;
}
