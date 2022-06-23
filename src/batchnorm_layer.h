/*
 * \file convolutional_layer.h
 * \brief class for convolutional layer
 */

#pragma once

#include "layer.h"

class Batchnorm2D : public Layer {
   public:
    Batchnorm2D(std::string const &name, bool zeroInitWeight = false,
                const double exponentialAverageFactor = 0.1,
                const double epsilon = 1E-5)
        : zeroInitWeight_(zeroInitWeight),
          exponentialAverageFactor_(exponentialAverageFactor),
          epsilon_(epsilon) {
        SetName(name);
        layerType_ = LayerType::Batchnorm2D;

        checkCudnnErrors(cudnnCreateTensorDescriptor(&bnDesc_));
    }

    ~Batchnorm2D() {
        checkCudnnErrors(cudnnDestroyTensorDescriptor(bnDesc_));
        if (resultRunningMean_ != nullptr)
            checkCudaErrors(cudaFree(resultRunningMean_));
        if (resultRunningVariance_ != nullptr)
            checkCudaErrors(cudaFree(resultRunningVariance_));
        if (resultSaveMean_ != nullptr)
            checkCudaErrors(cudaFree(resultSaveMean_));
        if (resultSaveInvVariance_ != nullptr)
            checkCudaErrors(cudaFree(resultSaveInvVariance_));
        bnDesc_ = nullptr;
    }

    virtual void InitFeatureShape() override;
    virtual void InitWeightsShape(
        std::vector<std::array<int, 4>> &w_p,
        std::vector<std::array<int, 4>> &b_p) override;
    virtual void Forward() override;
    virtual void Backward(BlobPointer<float> const &labels) override;

    virtual void DescriptorsAndWorkSpace() override;
    virtual void InitiateWeightsAndBiases() override;

    void *resultRunningMean_{nullptr};
    void *resultRunningVariance_{nullptr};

   private:
    bool zeroInitWeight_;

    double exponentialAverageFactor_;
    double epsilon_;

    std::array<int, 4> output_shape_;

    void *resultSaveMean_{nullptr};
    void *resultSaveInvVariance_{nullptr};

    cudnnTensorDescriptor_t bnDesc_{nullptr};

    void AllocateBatchnorm2D();
};
