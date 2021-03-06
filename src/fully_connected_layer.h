/*
 * \file fully_connected_layer.h
 * \brief class for fully connected layer
 */

#pragma once

#include "layer.h"

class Fully_connected : public Layer {
   public:
    Fully_connected(std::string const &name, int const output_size,
                    bool useBias = true)
        : useBias_(useBias) {
        SetName(name);
        layerType_ = LayerType::Fully_connected;
        output_shape_ = output_size;
    }

    ~Fully_connected() {
        if (d_one_vec != nullptr) cudaFree(d_one_vec);
    }

    virtual void InitFeatureShape() override;
    virtual void InitWeightsShape(
        std::vector<std::array<int, 4>> &w_p,
        std::vector<std::array<int, 4>> &b_p) override;
    virtual void Forward() override;
    virtual void Backward(BlobPointer<float> const &labels) override;

    virtual void DescriptorsAndWorkSpace() override;

   private:
    int input_shape_ = 0;
    int output_shape_ = 0;
    bool useBias_;

    float *d_one_vec{nullptr};
};
