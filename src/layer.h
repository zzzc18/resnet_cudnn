/*
 * \file layer.h
 */
#pragma once

#include "blob.h"
#include "cuda_context.h"
#include "cuda_helper.h"
#include "utilities_sc.h"

enum class WorkloadType { training, inference };

__global__ void InitiateZeros(float *d_one_vec, size_t length);

class Layer {
   public:
    Layer(){};
    virtual ~Layer() {
        input_desc_ = nullptr;
        output_desc_ = nullptr;
        cuda_ = nullptr;
    };

    std::string GetName() const { return name_; }
    void SetName(std::string const &name_in) { name_ = name_in; }

    void SetGradientStop() { gradient_stop_ = true; }

    cudnnTensorDescriptor_t &GetOutputDesc() { return output_desc_; }
    BlobPointer<float> &GetOutput() { return output_; }
    BlobPointer<float> &GetGradOutput() { return grad_output_; }

   protected:
    virtual void Forward() = 0;
    virtual void Backward(BlobPointer<float> const &labels) = 0;

    virtual void InitWeightsShape(std::vector<std::array<int, 4>> &w_p,
                                  std::vector<std::array<int, 4>> &b_p) = 0;
    // initialize weights along with the input size
    virtual void InitiateWeightsAndBiases();

    virtual void InitFeatureShape() = 0;
    virtual int ObtainPredictionAccuracy(std::vector<label_t> const &labels,
                                         std::vector<int> &confusion_matrix);
    virtual void DescriptorsAndWorkSpace() = 0;

    void SetCudaContext(CudaContext *context) { cuda_ = context; }
    void SetWorkloadType(WorkloadType const &in) { phase_ = in; }

    // memory pointers
    BlobPointer<float> input_;        // x
    BlobPointer<float> output_;       // y
    BlobPointer<float> grad_input_;   // dx
    BlobPointer<float> grad_output_;  // dy

    BlobPointer<float> grad_weights_;  // dw
    BlobPointer<float> grad_biases_;   // db

    BlobPointer<float> weights_;  // w
    BlobPointer<float> biases_;   // b

    // cuda and cudnn enviroments
    CudaContext *cuda_{nullptr};
    cudnnTensorDescriptor_t input_desc_{nullptr};
    cudnnTensorDescriptor_t output_desc_{nullptr};

    // stop calculating gradients or not
    bool gradient_stop_{false};
    // shapes of input and output features
    std::array<int, 4> in_shape_, out_shape_;

    // Permit Network to access the protected members of Layer
    friend class Network;

   private:
    std::string name_;

   protected:
    WorkloadType phase_{WorkloadType::inference};
};
