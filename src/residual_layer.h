/*
 * \file convolutional_layer.h
 * \brief class for convolutional layer
 */

#pragma once

#include "layer.h"

class Residual : public Layer {
   public:
    Residual(std::string const &name, Layer *inputLayer1, Layer *inputLayer2)
        : inputLayer1_(inputLayer1), inputLayer2_(inputLayer2) {
        SetName(name);
    }

    ~Residual() {}

    virtual void InitFeatureShape() override;
    virtual void InitWeightsShape(
        std::vector<std::array<int, 4>> &w_p,
        std::vector<std::array<int, 4>> &b_p) override;
    virtual void Forward() override;
    virtual void Backward(BlobPointer<float> const &labels) override;

    virtual void DescriptorsAndWorkSpace() override;

   private:
    Layer *inputLayer1_{nullptr};
    Layer *inputLayer2_{nullptr};

    std::array<int, 4> output_shape_;
};
