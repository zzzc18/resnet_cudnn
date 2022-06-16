/*
 * \file network.h
 */
#pragma once

#include "Dataset/Dataset.h"
#include "ImageNetParser/ImageNetParser.h"
#include "activation_layer.h"
#include "batchnorm_layer.h"
#include "convolutional_layer.h"
#include "fully_connected_layer.h"
#include "pooling_layer.h"
#include "residual_layer.h"
#include "softmax_layer.h"

class Network {
   public:
    Network();
    ~Network();

    int GetBatchSize() const { return batch_size_; }
    void SetBatchSize(int const in) { batch_size_ = in; }

    void AddLayers();
    void SetWorkloadType(WorkloadType const &in);
    void Train(const Dataset<dataType> *datasetPtr, float const learning_rate);

    void Predict(const Dataset<dataType> *datasetPtr);

    void AllocateMemoryForFeatures();
    void InitWeights();
    void DescriptorsAndWorkspace();

   private:
    void SetCudaContext();
    void Forward();
    void Backward(BlobPointer<float> const &labels);

    void Update(float const learning_rate = 0.02f);

    int ObtainPredictionAccuracy(std::vector<label_t> const &labels,
                                 std::vector<int> &confusion_matrix);

    // the temporal working space for training and predicting
    float *d_features_{nullptr};
    float *d_grad_features_{nullptr};

    float *d_grad_weights_{nullptr};
    float *d_grad_biases_{nullptr};

    // the memory for weights and biases
    float *d_weights_{nullptr};
    float *d_biases_{nullptr};

    bool is_memory_for_weights_allocated_{false};

    // layers in the network
    std::vector<Layer *> layers_;
    int batch_size_;

    CudaContext cuda_;

    WorkloadType phase_{WorkloadType::inference};
    size_t length_weights_{}, length_biases_{};
};
