/*
 * \file activation_layer.cu
 */

#include "activation_layer.h"

std::array<int, 4> Activation::InitFeatureShape(
    std::array<int, 4> const &in_shape) {
    in_shape_ = in_shape;
    out_shape_ = in_shape;
    return out_shape_;
}

void Activation::InitWeightsShape(std::vector<std::array<int, 4>> &w_l,
                                  std::vector<std::array<int, 4>> &b_l) {
    w_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});
    b_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});

    return;
}

void Activation::Forward() {
    input_desc_ = input_.tensor();
    output_desc_ = output_.tensor();
#if (DEBUG_ACTIVATION & 0x01)
    std::cout << name_ << "[FORWARD]\n";
    input_.print(name_ + "::input", true);
#endif
    checkCudnnErrors(cudnnActivationForward(
        cuda_->cudnn(), act_desc_, &cuda_->one, input_desc_, input_.CudaPtr(),
        &cuda_->zero, output_desc_, output_.CudaPtr()));

#if (DEBUG_ACTIVATION & 0x01)
    output_.print(name_ + "::output", true);
#endif

    return;
}

void Activation::Backward(BlobPointer<float> const &labels) {
    checkCudnnErrors(cudnnActivationBackward(
        cuda_->cudnn(), act_desc_, &cuda_->one, output_desc_, output_.CudaPtr(),
        output_desc_, grad_output_.CudaPtr(), input_desc_, input_.CudaPtr(),
        &cuda_->zero, input_desc_, grad_input_.CudaPtr()));
#if (DEBUG_ACTIVATION & 0x02)
    std::cout << name_ << "[BACKWARD]\n";
    input_.print(name_ + "::input", true);
    output_.print(name_ + "::output", true);
    grad_input_.print(name_ + "::dx", true);
    grad_output_.print(name_ + "::dy", true);
#endif

    return;
}
