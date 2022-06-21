/*
 * \file fully_connected_layer.cu
 */

#include "fully_connected_layer.h"

void Fully_connected::InitFeatureShape() {
    // Reshape
    std::array<int, 4> old_in_shape(in_shape_);
    in_shape_[0] = old_in_shape[0];
    in_shape_[1] = old_in_shape[1] * old_in_shape[2] * old_in_shape[3];
    in_shape_[2] = 1;
    in_shape_[3] = 1;

    out_shape_[0] = in_shape_[0];
    out_shape_[1] = output_shape_;
    out_shape_[2] = 1;
    out_shape_[3] = 1;
}

void Fully_connected::InitWeightsShape(std::vector<std::array<int, 4>> &w_l,
                                       std::vector<std::array<int, 4>> &b_l) {
    // initialize weight, bias, and output
    input_shape_ = in_shape_[1];
    std::array<int, 4> w{1, 1, input_shape_, output_shape_},
        b{1, 1, output_shape_, 1};
    w_l.emplace_back(w);
    b_l.emplace_back(b);

    return;
}

void Fully_connected::DescriptorsAndWorkSpace() {
    if (d_one_vec != nullptr) {
        checkCudaErrors(cudaFree(d_one_vec));
    }

    int batch_size = input_.get_n();
#ifdef ZDEBUG
    std::cout << this->GetName()
              << "::d_one_vec=" << sizeof(float) * batch_size / 1024 / 1024
              << "MB\n";
#endif
    checkCudaErrors(
        cudaMalloc((void **)&d_one_vec, sizeof(float) * batch_size));
    InitiateVecOnes<<<(batch_size + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D,
                      BLOCK_DIM_1D>>>(d_one_vec, batch_size);
}

void Fully_connected::Forward() {
    int batch_size = input_.get_n();
    // output = weightsT * input (without biases)
    checkCublasErrors(cublasSgemm(cuda_->cublas(), CUBLAS_OP_T, CUBLAS_OP_N,
                                  output_shape_, batch_size, input_shape_,
                                  &cuda_->one, weights_.CudaPtr(), input_shape_,
                                  input_.CudaPtr(), input_shape_, &cuda_->zero,
                                  output_.CudaPtr(), output_shape_));

    // output += biases * d-one-vecT
    if (useBias_)
        checkCublasErrors(cublasSgemm(
            cuda_->cublas(), CUBLAS_OP_N, CUBLAS_OP_N, output_shape_,
            batch_size, 1, &cuda_->one, biases_.CudaPtr(), output_shape_,
            d_one_vec, 1, &cuda_->one, output_.CudaPtr(), output_shape_));

#if (DEBUG_DENSE & 0x01)
    input_.print(name_ + "::input", true);
    weights_.print(name_ + "::weight", true);
    biases_.print(name_ + "::bias", true);
    output_.print(name_ + "::output", true);
#endif
    return;
}

void Fully_connected::Backward(BlobPointer<float> const &labels) {
    int batch_size = input_.get_n();
    if (useBias_)
        cublasSgemv(cuda_->cublas(), CUBLAS_OP_N, output_shape_, batch_size,
                    &cuda_->one, output_.CudaPtr(), output_shape_, d_one_vec, 1,
                    &cuda_->zero, grad_biases_.CudaPtr(), 1);

    // dw = x * (dy)T
    cublasSgemm(cuda_->cublas(), CUBLAS_OP_N, CUBLAS_OP_T, input_shape_,
                output_shape_, batch_size, &cuda_->one, input_.CudaPtr(),
                input_shape_, output_.CudaPtr(), output_shape_, &cuda_->zero,
                grad_weights_.CudaPtr(), input_shape_);

    // dx = W * dy
    if (!gradient_stop_) {
        cublasSgemm(cuda_->cublas(), CUBLAS_OP_N, CUBLAS_OP_N, input_shape_,
                    batch_size, output_shape_, &cuda_->one, weights_.CudaPtr(),
                    input_shape_, output_.CudaPtr(), output_shape_,
                    &cuda_->zero, d_temp_grad_features_, input_shape_);
        this->BackwardCopy();
    }
    return;
}
