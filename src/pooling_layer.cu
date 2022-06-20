/*
 * \file pooling_layer.cu
 */

#include "pooling_layer.h"

void Pooling::InitFeatureShape() {
    out_shape_[0] = in_shape_[0];
    out_shape_[1] = in_shape_[1];
    out_shape_[2] = (in_shape_[2] + 2 * padding_ - kernel_size_) / stride_ + 1;
    out_shape_[3] = (in_shape_[3] + 2 * padding_ - kernel_size_) / stride_ + 1;
}

void Pooling::InitWeightsShape(std::vector<std::array<int, 4>> &w_l,
                               std::vector<std::array<int, 4>> &b_l) {
    w_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});
    b_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});
    return;
}

void Pooling::Forward() {
    input_desc_ = input_.tensor();
    output_desc_ = output_.tensor();
    cudnnPoolingForward(cuda_->cudnn(), pool_desc_, &cuda_->one, input_desc_,
                        input_.CudaPtr(), &cuda_->zero, output_desc_,
                        d_retain_output_);
    checkCudaErrors(cudaMemcpy(output_.CudaPtr(), d_retain_output_,
                               output_.buf_size(), cudaMemcpyDeviceToDevice));
#if (DEBUG_POOLING & 0x01)
    input_.print(name_ + "::input", true, input_.GetWidth());
    output_.print(name_ + "::output", true, output_.GetWidth());
#endif

    return;
}

void Pooling::Backward(BlobPointer<float> const &labels) {
    checkCudnnErrors(cudnnPoolingBackward(
        cuda_->cudnn(), pool_desc_, &cuda_->one, output_desc_, d_retain_output_,
        output_desc_, output_.CudaPtr(), input_desc_, input_.CudaPtr(),
        &cuda_->zero, input_desc_, d_temp_grad_features_));
    this->BackwardCopy();

#if (DEBUG_POOLING & 0x02)
    std::cout << name_ << "[BACKWARD]" << std::endl;
    input_.print(name_ + "::input", true, input_.GetWidth());
    output_.print(name_ + "::predict", true, output_.GetWidth());
    grad_output_.print(name_ + "::dy", true, grad_output_.GetWidth());
    grad_input_.print(name_ + "::dx", true, grad_input_.GetWidth());
#endif
    return;
}
