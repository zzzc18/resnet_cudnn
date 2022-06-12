/*
 * \file convolutional_layer.h
 * \brief class for convolutional layer
 */

#pragma once

#include "layer.h"

class Conv2D : public Layer {
   public:
    Conv2D(std::string const &name, int const out_channels,
           int const kernel_size, int const stride = 1, int const padding = 0,
           int const dilation = 1)
        : out_channels_(out_channels),
          kernel_size_(kernel_size),
          stride_(stride),
          padding_(padding),
          dilation_(dilation) {
        SetName(name);
        checkCudnnErrors(cudnnCreateFilterDescriptor(&filter_desc_));

        checkCudnnErrors(cudnnCreateConvolutionDescriptor(&conv_desc_));

        checkCudnnErrors(cudnnSetConvolution2dDescriptor(
            conv_desc_, padding_, padding_, stride_, stride_, dilation_,
            dilation_, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    }

    ~Conv2D() {
        checkCudnnErrors(cudnnDestroyFilterDescriptor(filter_desc_));
        checkCudnnErrors(cudnnDestroyConvolutionDescriptor(conv_desc_));

        if (d_workspace_ != nullptr) checkCudaErrors(cudaFree(d_workspace_));
        biases_desc_ = nullptr;
    }

    virtual std::array<int, 4> InitFeatureShape(
        std::array<int, 4> const &in_shape) override;
    virtual void InitWeightsShape(
        std::vector<std::array<int, 4>> &w_p,
        std::vector<std::array<int, 4>> &b_p) override;
    virtual void Forward() override;
    virtual void Backward(BlobPointer<flt_type> const &labels) override;

    virtual void DescriptorsAndWorkSpace() override;

   private:
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int dilation_;

    std::array<int, 4> output_shape_;
    cudnnConvolutionDescriptor_t conv_desc_;

    cudnnConvolutionFwdAlgo_t conv_forward_algo_;
    cudnnConvolutionBwdDataAlgo_t conv_backward_data_algo_;
    cudnnConvolutionBwdFilterAlgo_t conv_backward_filter_algo_;

    size_t workspace_shape_{};
    void *d_workspace_{nullptr};
    cudnnFilterDescriptor_t filter_desc_{nullptr};
    cudnnTensorDescriptor_t biases_desc_{nullptr};
    void AllocateDnnWorkspace();
};
