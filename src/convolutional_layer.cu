/*
 * \file convolutional_layer.cu
 */

#include "convolutional_layer.h"

void Conv2D::AllocateDnnWorkspace() {
    size_t temp_size = 0;

    // forward
    const int request_count = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int return_count = -1;
    cudnnConvolutionFwdAlgoPerf_t
        conv_forward_[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    cudnnConvolutionBwdDataAlgoPerf_t
        conv_backward_data_[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    cudnnConvolutionBwdFilterAlgoPerf_t
        conv_backward_filter_[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    checkCudnnErrors(cudnnFindConvolutionForwardAlgorithm(
        cuda_->cudnn(), input_desc_, filter_desc_, conv_desc_, output_desc_,
        request_count, &return_count, conv_forward_));
    conv_forward_algo_ = conv_forward_[0].algo;
    // conv_forward_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

    checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(
        cuda_->cudnn(), input_desc_, filter_desc_, conv_desc_, output_desc_,
        conv_forward_algo_, &temp_size));
    workspace_shape_ = std::max(workspace_shape_, temp_size);

    // backward - filter
    checkCudnnErrors(cudnnFindConvolutionBackwardFilterAlgorithm(
        cuda_->cudnn(), input_desc_, output_desc_, conv_desc_, filter_desc_,
        request_count, &return_count, conv_backward_filter_));
    conv_backward_filter_algo_ = conv_backward_filter_[0].algo;
    // conv_backward_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

    checkCudnnErrors(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cuda_->cudnn(), input_desc_, output_desc_, conv_desc_, filter_desc_,
        conv_backward_filter_algo_, &temp_size));
    workspace_shape_ = std::max(workspace_shape_, temp_size);

    // backward - data
    checkCudnnErrors(cudnnFindConvolutionBackwardDataAlgorithm(
        cuda_->cudnn(), filter_desc_, output_desc_, conv_desc_, input_desc_,
        request_count, &return_count, conv_backward_data_));
    conv_backward_data_algo_ = conv_backward_data_[0].algo;
    // conv_backward_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

    checkCudnnErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cuda_->cudnn(), filter_desc_, output_desc_, conv_desc_, input_desc_,
        conv_backward_data_algo_, &temp_size));
    workspace_shape_ = std::max(workspace_shape_, temp_size);

    if (workspace_shape_ > 0) {
        if (d_workspace_ != nullptr) checkCudaErrors(cudaFree(d_workspace_));
        checkCudaErrors(cudaMalloc((void **)&d_workspace_, workspace_shape_));
    }
}

std::array<int, 4> Conv2D::InitFeatureShape(
    std::array<int, 4> const &in_shape) {
    in_shape_ = in_shape;

    out_shape_[0] = in_shape[0];
    out_shape_[1] = out_channels_;
    out_shape_[2] = (in_shape_[2] + 2 * padding_ - kernel_size_) / stride_ + 1;
    out_shape_[3] = (in_shape_[3] + 2 * padding_ - kernel_size_) / stride_ + 1;

    return out_shape_;
}

void Conv2D::InitWeightsShape(std::vector<std::array<int, 4>> &w_l,
                              std::vector<std::array<int, 4>> &b_l) {
    w_l.emplace_back(std::array<int, 4>{out_channels_, in_shape_[1],
                                        kernel_size_, kernel_size_});
    b_l.emplace_back(std::array<int, 4>{1, out_channels_, 1, 1});
    return;
}

void Conv2D::DescriptorsAndWorkSpace() {
    checkCudnnErrors(cudnnSetFilter4dDescriptor(
        filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels_,
        input_.GetChannels(), kernel_size_, kernel_size_));

    biases_desc_ = biases_.tensor();
    output_desc_ = output_.tensor();
    input_desc_ = input_.tensor();

    AllocateDnnWorkspace();
    return;
}

void Conv2D::Forward() {
    checkCudnnErrors(cudnnConvolutionForward(
        cuda_->cudnn(), &cuda_->one, input_desc_, input_.CudaPtr(),
        filter_desc_, weights_.CudaPtr(), conv_desc_, conv_forward_algo_,
        d_workspace_, workspace_shape_, &cuda_->zero, output_desc_,
        output_.CudaPtr()));

    checkCudnnErrors(cudnnAddTensor(cuda_->cudnn(), &cuda_->one, biases_desc_,
                                    biases_.CudaPtr(), &cuda_->one,
                                    output_desc_, output_.CudaPtr()));

#if (DEBUG_CONV & 0x01)
    input_.print(name_ + "::input", true, input_.GetWidth());
    weights_.print(name_ + "::weight", true, weights_.GetWidth());
    biases_.print(name_ + "::bias", true, biases_.GetWidth());
    output_.print(name_ + "::output", true, output_.GetWidth());
#endif

    return;
}

void Conv2D::Backward(BlobPointer<flt_type> const &labels) {
    // gradients of biases
    checkCudnnErrors(cudnnConvolutionBackwardBias(
        cuda_->cudnn(), &cuda_->one, output_desc_, grad_output_.CudaPtr(),
        &cuda_->zero, biases_desc_, grad_biases_.CudaPtr()));

    // gradients of weights
    checkCudnnErrors(cudnnConvolutionBackwardFilter(
        cuda_->cudnn(), &cuda_->one, input_desc_, input_.CudaPtr(),
        output_desc_, grad_output_.CudaPtr(), conv_desc_,
        conv_backward_filter_algo_, d_workspace_, workspace_shape_,
        &cuda_->zero, filter_desc_, grad_weights_.CudaPtr()));

    // gradients of input data
    if (!gradient_stop_)
        checkCudnnErrors(cudnnConvolutionBackwardData(
            cuda_->cudnn(), &cuda_->one, filter_desc_, weights_.CudaPtr(),
            output_desc_, grad_output_.CudaPtr(), conv_desc_,
            conv_backward_data_algo_, d_workspace_, workspace_shape_,
            &cuda_->zero, input_desc_, grad_input_.CudaPtr()));

#if (DEBUG_CONV & 0x02)
    std::cout << name_ << "[BACKWARD]\n";
    grad_input_.print(name_ + "gdata", true);
    grad_output_.print(name_ + "::gradients", true, grad_output_.GetWidth());
    grad_biases_.print(name_ + "gbias", true);
    grad_weights_.print(name_ + "gfilter", true);
#endif
    return;
}
