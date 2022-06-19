/*
 * \file activation_layer.h
 * \brief class for activation layer
 */

#pragma once

#include "layer.h"

class InplaceReLU : public Layer {
   public:
    InplaceReLU(std::string const &name,
                cudnnActivationMode_t const &mode = CUDNN_ACTIVATION_RELU,
                float const coef = 5.f) {
        SetName(name);
        layerType_ = LayerType::InplaceReLU;
        mode_ = mode;

        cudnnCreateActivationDescriptor(&act_desc_);
        cudnnSetActivationDescriptor(act_desc_, mode, CUDNN_PROPAGATE_NAN,
                                     coef);
    }

    ~InplaceReLU() { cudnnDestroyActivationDescriptor(act_desc_); }

    virtual void InitFeatureShape() override;
    virtual void InitWeightsShape(
        std::vector<std::array<int, 4>> &w_p,
        std::vector<std::array<int, 4>> &b_p) override;
    virtual void Forward() override;
    virtual void Backward(BlobPointer<float> const &labels) override;

    virtual void DescriptorsAndWorkSpace() override {
        Layer::DescriptorsAndWorkSpace();
    }

   private:
    cudnnActivationDescriptor_t act_desc_;
    cudnnActivationMode_t mode_;
};
