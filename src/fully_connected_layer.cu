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

    checkCudaErrors(
        cudaMalloc((void **)&d_one_vec, sizeof(float) * batch_size));
    InitiateVecOnes<<<(batch_size + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D,
                      BLOCK_DIM_1D>>>(d_one_vec, batch_size);
}

void Fully_connected::Forward() {
    int batch_size = input_.get_n();
    // output = weightsT * input (without biases)
    checkCublasErrors(cublasSgemm(
        cuda_->cublas(), CUBLAS_OP_N, CUBLAS_OP_N,  // OP(A),OP(B)
        output_shape_, batch_size, input_shape_,    // m,n,k
        &cuda_->one,                                // alpha
        weights_.CudaPtr(), output_shape_,          // A, lda
        input_.CudaPtr(), input_shape_,             // B, ldb
        &cuda_->zero,                               // beta
        output_.CudaPtr(), output_shape_            // C, ldc
        ));

    if (!useBias_) return;
    // output += biases * d-one-vecT
    checkCublasErrors(cublasSgemm(cuda_->cublas(),               // handle
                                  CUBLAS_OP_N, CUBLAS_OP_N,      // OP(A),OP(B)
                                  output_shape_, batch_size, 1,  // m,n,k
                                  &cuda_->one,                   // alpha
                                  biases_.CudaPtr(), output_shape_,  // A, lda
                                  d_one_vec, 1,                      // B, ldb
                                  &cuda_->one,                       // beta
                                  output_.CudaPtr(), output_shape_   // C, ldc
                                  ));
}

void Fully_connected::Backward(BlobPointer<float> const &labels) {
    float *xPtr = input_.CudaPtr();
    if (previousSplitLayer_ != nullptr) {
        xPtr = previousSplitLayer_->GetInput().CudaPtr();
    };

    int batch_size = input_.get_n();
    if (useBias_) {
        checkCublasErrors(
            cublasSgemv(cuda_->cublas(), CUBLAS_OP_N, output_shape_, batch_size,
                        &cuda_->one, output_.CudaPtr(), output_shape_,
                        d_one_vec, 1, &cuda_->zero, grad_biases_.CudaPtr(), 1));
    }

    // dw = xT * (dy)
    checkCublasErrors(cublasSgemm(
        cuda_->cublas(), CUBLAS_OP_N, CUBLAS_OP_T, output_shape_, input_shape_,
        batch_size, &cuda_->one, output_.CudaPtr(), output_shape_, xPtr,
        input_shape_, &cuda_->zero, grad_weights_.CudaPtr(), output_shape_));

    // dx = W * dyT
    if (!gradient_stop_) {
        checkCublasErrors(cublasSgemm(
            cuda_->cublas(), CUBLAS_OP_T, CUBLAS_OP_N,  // OP(A),OP(B)
            input_shape_, batch_size, output_shape_,    // m,n,k
            &cuda_->one,                                // alpha
            weights_.CudaPtr(), output_shape_,          // A, lda
            output_.CudaPtr(), output_shape_,           // B, ldb
            &cuda_->zero,                               // beta
            d_temp_grad_features_, input_shape_         // C, ldc
            ));
        this->BackwardCopy();
    }
    return;
}
