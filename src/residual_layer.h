/*
 * \file convolutional_layer.h
 * \brief class for convolutional layer
 */

#pragma once

#include "layer.h"

class Residual : public Layer {
   public:
    Residual(std::string const &name, Layer *inputLayer1 = nullptr,
             Layer *inputLayer2 = nullptr)
        : inputLayer1_(inputLayer1), inputLayer2_(inputLayer2) {
        SetName(name);
        layerType_ = LayerType::Residual;
    }

    ~Residual() {}

    virtual void InitFeatureShape() override;
    virtual void InitWeightsShape(
        std::vector<std::array<int, 4>> &w_p,
        std::vector<std::array<int, 4>> &b_p) override;
    virtual void Forward() override;
    virtual void Backward(BlobPointer<float> const &labels) override;

    virtual void DescriptorsAndWorkSpace() override;

    Layer *inputLayer1_{nullptr};
    Layer *inputLayer2_{nullptr};

   private:
    std::array<int, 4> output_shape_;
};

class Split : public Layer {
   public:
    Split(std::string const &name, Layer *outputLayer1, Layer *outputLayer2)
        : outputLayer1_(outputLayer1), outputLayer2_(outputLayer2) {
        SetName(name);
        layerType_ = LayerType::Split;
    }

    ~Split() {}

    virtual void InitFeatureShape() override;
    virtual void InitWeightsShape(
        std::vector<std::array<int, 4>> &w_p,
        std::vector<std::array<int, 4>> &b_p) override;
    virtual void Forward() override;
    virtual void Backward(BlobPointer<float> const &labels) override;

    virtual void DescriptorsAndWorkSpace() override;

   private:
    Layer *outputLayer1_{nullptr};
    Layer *outputLayer2_{nullptr};

    std::array<int, 4> output_shape_;
};
