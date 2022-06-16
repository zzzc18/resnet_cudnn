/*
 * \file convolutional_layer.cu
 */

#include <iostream>

#include "batchnorm_layer.h"

void Batchnorm2D::InitiateWeightsAndBiases() {
    if (weights_.CudaPtr() == nullptr || biases_.CudaPtr() == nullptr) return;

    std::vector<float> weights(weights_.LengthNchw(), 1.0);
    weights_.ToDevice(weights.data(), weights.size());

    std::vector<float> biases(biases_.LengthNchw(), 0.0);
    for (size_t i = 0; i < biases.size(); i++) biases[i] = 0.f;
    biases_.ToDevice(biases.data(), biases.size());
}

void Batchnorm2D::AllocateBatchnorm2D() {
    if (resultRunningMean_ == nullptr) {
        checkCudaErrors(cudaMalloc((void **)&resultRunningMean_,
                                   sizeof(float) * input_.GetChannels()));
        checkCudaErrors(cudaMemset(resultRunningMean_, 0,
                                   sizeof(float) * input_.GetChannels()));
    }

    if (resultRunningVariance_ == nullptr) {
        checkCudaErrors(cudaMalloc((void **)&resultRunningVariance_,
                                   sizeof(float) * input_.GetChannels()));
        checkCudaErrors(cudaMemset(resultRunningVariance_, 0,
                                   sizeof(float) * input_.GetChannels()));
    }

    if (resultSaveMean_ == nullptr) {
        checkCudaErrors(cudaMalloc((void **)&resultSaveMean_,
                                   sizeof(float) * input_.GetChannels()));
    }

    if (resultSaveInvVariance_ == nullptr) {
        checkCudaErrors(cudaMalloc((void **)&resultSaveInvVariance_,
                                   sizeof(float) * input_.GetChannels()));
    }
}

std::array<int, 4> Batchnorm2D::InitFeatureShape(
    std::array<int, 4> const &in_shape) {
    out_shape_ = in_shape_ = in_shape;
    return out_shape_;
}

void Batchnorm2D::InitWeightsShape(std::vector<std::array<int, 4>> &w_l,
                                   std::vector<std::array<int, 4>> &b_l) {
    // 1xCx1x1
    w_l.emplace_back(std::array<int, 4>{1, in_shape_[1], 1, 1});
    b_l.emplace_back(std::array<int, 4>{1, in_shape_[1], 1, 1});
    return;
}

void Batchnorm2D::DescriptorsAndWorkSpace() {
    output_desc_ = output_.tensor();
    input_desc_ = input_.tensor();

    cudnnDeriveBNTensorDescriptor(bnDesc_, input_desc_,
                                  CUDNN_BATCHNORM_SPATIAL);
    AllocateBatchnorm2D();
    return;
}

void Batchnorm2D::Forward() {
    if (phase_ == WorkloadType::training) {
        checkCudnnErrors(cudnnBatchNormalizationForwardTraining(
            cuda_->cudnn(),  // cudnnHandle_t                    handle,
            CUDNN_BATCHNORM_SPATIAL,  // cudnnBatchNormMode_t             mode,
            &cuda_->one,              // const void                      *alpha,
            &cuda_->zero,             // const void                      *beta,
            input_desc_,              // const cudnnTensorDescriptor_t    xDesc,
            input_.CudaPtr(),         // const void                      *x,
            output_desc_,             // const cudnnTensorDescriptor_t    yDesc,
            output_.CudaPtr(),        // void                            *y,
            bnDesc_,                  // const cudnnTensorDescriptor_t
                                      // bnScaleBiasMeanVarDesc,
            weights_.CudaPtr(),  // const void                      *bnScale,
            biases_.CudaPtr(),   // const void                      *bnBias,
            exponentialAverageFactor_,  // double exponentialAverageFactor,
            resultRunningMean_,         // void *resultRunningMean,
            resultRunningVariance_,     // void *resultRunningVariance,
            epsilon_,               // double                           epsilon,
            resultSaveMean_,        // void                    *resultSaveMean,
            resultSaveInvVariance_  // void *resultSaveInvVariance
            ));
    } else {
        checkCudnnErrors(cudnnBatchNormalizationForwardInference(
            cuda_->cudnn(),  // cudnnHandle_t                    handle,
            CUDNN_BATCHNORM_SPATIAL,  // cudnnBatchNormMode_t             mode,
            &cuda_->one,              // const void                      *alpha,
            &cuda_->zero,             // const void                      *beta,
            input_desc_,              // const cudnnTensorDescriptor_t    xDesc,
            input_.CudaPtr(),         // const void                      *x,
            output_desc_,             // const cudnnTensorDescriptor_t    yDesc,
            output_.CudaPtr(),        // void                            *y,
            bnDesc_,                  // const cudnnTensorDescriptor_t
                                      // bnScaleBiasMeanVarDesc,
            weights_.CudaPtr(),  // const void                      *bnScale,
            biases_.CudaPtr(),   // const void                      *bnBias,
            resultRunningMean_,  // const void                *estimatedMean,
            resultRunningVariance_,  // const void *estimatedVariance,
            epsilon_                 // double                           epsilon
            ));
    }

    return;
}

void Batchnorm2D::Backward(BlobPointer<float> const &labels) {
    checkCudnnErrors(cudnnBatchNormalizationBackward(
        cuda_->cudnn(),           // cudnnHandle_t                    handle,
        CUDNN_BATCHNORM_SPATIAL,  // cudnnBatchNormMode_t             mode,
        &cuda_->one,              // const void                 *alphaDataDiff,
        &cuda_->zero,             // const void                 *betaDataDiff,
        &cuda_->one,              // const void                 *alphaParamDiff,
        &cuda_->zero,             // const void                 *betaParamDiff,
        input_desc_,              // const cudnnTensorDescriptor_t    xDesc,
        input_.CudaPtr(),         // const void                      *x,
        output_desc_,             // const cudnnTensorDescriptor_t    dyDesc,
        grad_output_.CudaPtr(),   // const void                      *dy,
        input_desc_,              // const cudnnTensorDescriptor_t    dxDesc,
        grad_input_.CudaPtr(),    // void                            *dx,
        bnDesc_,                  // const cudnnTensorDescriptor_t
                                  // bnScaleBiasDiffDesc,
        weights_.CudaPtr(),       // const void                      *bnScale,
        grad_weights_.CudaPtr(),  // void *resultBnScaleDiff,
        grad_biases_.CudaPtr(),   // void *resultBnBiasDiff,
        epsilon_,                 // double                           epsilon,
        resultSaveMean_,          // const void                      *savedMean,
        resultSaveInvVariance_    // const void *savedInvVariance
        ));
    return;
}
