/*
 * \file softmax_layer.h
 * \brief class for softmax layer
 */

#pragma once

#include "layer.h"

class Softmax : public Layer {
   public:
    Softmax(std::string name) {
        SetName(name);
        layerType_ = LayerType::Softmax;
    }
    ~Softmax() {}

    virtual void InitFeatureShape() override;
    virtual void InitWeightsShape(
        std::vector<std::array<int, 4>> &w_p,
        std::vector<std::array<int, 4>> &b_p) override;
    virtual void Forward() override;
    virtual void Backward(BlobPointer<float> const &labels) override;

    int ObtainPredictionAccuracy(std::vector<label_t> const &labels,
                                 std::vector<int> &confusion_matrix);

    virtual void DescriptorsAndWorkSpace() override {
        Layer::DescriptorsAndWorkSpace();
    }
};
