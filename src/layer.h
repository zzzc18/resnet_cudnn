/*
 * \file layer.h
 */
#pragma once

#include "blob.h"
#include "cuda_context.h"
#include "cuda_helper.h"

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

   protected:
    virtual void Forward() = 0;
    virtual void Backward(BlobPointer<flt_type> const &labels) = 0;

    virtual void InitWeightsShape(std::vector<std::array<int, 4>> &w_p,
                                  std::vector<std::array<int, 4>> &b_p) = 0;
    // initialize weights along with the input size
    void InitiateWeightsAndBiases();

    virtual std::array<int, 4> InitFeatureShape(
        std::array<int, 4> const &input_shape) = 0;
    virtual int ObtainPredictionAccuracy(std::vector<label_t> const &labels,
                                         std::vector<int> &confusion_matrix);
    virtual void DescriptorsAndWorkSpace() = 0;

    void SetCudaContext(CudaContext *context) { cuda_ = context; }

    // memory pointers
    BlobPointer<flt_type> input_;        // x
    BlobPointer<flt_type> output_;       // y
    BlobPointer<flt_type> grad_input_;   // dx
    BlobPointer<flt_type> grad_output_;  // dy

    BlobPointer<flt_type> grad_weights_;  // dw
    BlobPointer<flt_type> grad_biases_;   // db

    BlobPointer<flt_type> weights_;  // w
    BlobPointer<flt_type> biases_;   // b

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
    void SetGradientStop() { gradient_stop_ = true; }
    std::string name_;
};
