/*
 * \file pooling_layer.h
 * \brief class for pooling layer
 */

#pragma once

#include "layer.h"

class Pooling : public Layer
{
public:
    Pooling(std::string const &name, int const kernel_size, int const padding,
            int const stride, cudnnPoolingMode_t const &mode) : kernel_size_(kernel_size), padding_(padding), stride_(stride), mode_(mode)
    {
        SetName(name);
        cudnnCreatePoolingDescriptor(&pool_desc_);
        cudnnSetPooling2dDescriptor(pool_desc_, mode_, CUDNN_PROPAGATE_NAN,
                                    kernel_size_, kernel_size_, padding_, padding_, stride_, stride_);
    }
    ~Pooling()
    {
        cudnnDestroyPoolingDescriptor(pool_desc_);
    }

    virtual std::array<int, 4> InitFeatureShape(std::array<int, 4> const &input_size) override;
    virtual void InitWeightsShape(std::vector<std::array<int, 4>> &w_p, std::vector<std::array<int, 4>> &b_p) override;
    virtual void Forward() override;
    virtual void Backward(BlobPointer<flt_type> const &labels) override;

    virtual void DescriptorsAndWorkSpace() override
    {
        Layer::DescriptorsAndWorkSpace();
    }

private:
    int kernel_size_;
    int padding_;
    int stride_;

    cudnnPoolingMode_t mode_;

    std::array<int, 4> output_shape_;
    cudnnPoolingDescriptor_t pool_desc_;
};
