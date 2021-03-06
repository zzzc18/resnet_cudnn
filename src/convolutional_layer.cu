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
#ifdef ZDEBUG
        std::cout << this->GetName()
                  << "::d_workspace_=" << workspace_shape_ / 1024 / 1024
                  << "MB\n";
#endif
        checkCudaErrors(cudaMalloc((void **)&d_workspace_, workspace_shape_));
    }
}

void Conv2D::InitFeatureShape() {
    out_shape_[0] = in_shape_[0];
    out_shape_[1] = out_channels_;
    out_shape_[2] = (in_shape_[2] + 2 * padding_ - kernel_size_) / stride_ + 1;
    out_shape_[3] = (in_shape_[3] + 2 * padding_ - kernel_size_) / stride_ + 1;
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
        cuda_->cudnn(),      // cudnnHandle_t                       handle,
        &cuda_->one,         // const void                         *alpha,
        input_desc_,         // const cudnnTensorDescriptor_t       xDesc,
        input_.CudaPtr(),    // const void                         *x,
        filter_desc_,        // const cudnnFilterDescriptor_t       wDesc,
        weights_.CudaPtr(),  // const void                         *w,
        conv_desc_,          // const cudnnConvolutionDescriptor_t  convDesc,
        conv_forward_algo_,  // cudnnConvolutionFwdAlgo_t           algo,
        d_workspace_,        // void                               *workSpace,
        workspace_shape_,    // size_t workSpaceSizeInBytes,
        &cuda_->zero,        // const void                         *beta,
        output_desc_,        // const cudnnTensorDescriptor_t       yDesc,
        output_.CudaPtr()    // void                               *y
        ));

    if (useBias_) {
        checkCudnnErrors(cudnnAddTensor(
            cuda_->cudnn(), &cuda_->one, biases_desc_, biases_.CudaPtr(),
            &cuda_->one, output_desc_, output_.CudaPtr()));
    }
    return;
}

void Conv2D::Backward(BlobPointer<float> const &labels) {
    float *xPtr = input_.CudaPtr();
    if (previousSplitLayer_ != nullptr) {
        xPtr = previousSplitLayer_->GetInput().CudaPtr();
    };

    // gradients of biases
    if (useBias_) {
        checkCudnnErrors(cudnnConvolutionBackwardBias(
            cuda_->cudnn(), &cuda_->one, output_desc_, output_.CudaPtr(),
            &cuda_->zero, biases_desc_, grad_biases_.CudaPtr()));
    }

    // gradients of weights
    checkCudnnErrors(cudnnConvolutionBackwardFilter(
        cuda_->cudnn(), &cuda_->one, input_desc_, xPtr, output_desc_,
        output_.CudaPtr(), conv_desc_, conv_backward_filter_algo_, d_workspace_,
        workspace_shape_, &cuda_->zero, filter_desc_, grad_weights_.CudaPtr()));

    // gradients of input data
    if (!gradient_stop_) {
        // for multiple nodes have one
        checkCudnnErrors(cudnnConvolutionBackwardData(
            cuda_->cudnn(), &cuda_->one, filter_desc_, weights_.CudaPtr(),
            output_desc_, output_.CudaPtr(), conv_desc_,
            conv_backward_data_algo_, d_workspace_, workspace_shape_,
            &cuda_->zero, input_desc_, d_temp_grad_features_));
        this->BackwardCopy();
    }
    return;
}
