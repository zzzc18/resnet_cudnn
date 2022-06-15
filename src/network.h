/*
 * \file network.h
 */
#pragma once

#include "Dataset/Dataset.h"
#include "ImageNetParser/ImageNetParser.h"
#include "activation_layer.h"
#include "convolutional_layer.h"
#include "fully_connected_layer.h"
#include "pooling_layer.h"
#include "softmax_layer.h"

using namespace LeNet_5;

enum class WorkloadType { training, inference };

class Network {
   public:
    Network();
    ~Network();

    int GetBatchSize() const { return batch_size_; }
    void SetBatchSize(int const in) { batch_size_ = in; }

    void AddLayers();
    void SetWorkloadType(WorkloadType const &in) { phase_ = in; }

    // void Train(std::vector<one_image> const &train_sample,
    //            std::vector<label_t> const &train_label,
    //            flt_type const learning_rate);
    void Train(const Dataset<dataType> *datasetPtr,
               flt_type const learning_rate);

    void Predict(const Dataset<dataType> *datasetPtr);
    // void Predict(std::vector<one_image> const &test_sample,
    //              std::vector<label_t> const &test_label);

    void AllocateMemoryForFeatures();
    void InitWeights();
    void DescriptorsAndWorkspace();

    //    private:
   public:
    void SetCudaContext();
    void Forward();
    void Backward(BlobPointer<flt_type> const &labels);

    void Update(flt_type const learning_rate = 0.02f);

    int ObtainPredictionAccuracy(std::vector<label_t> const &labels,
                                 std::vector<int> &confusion_matrix);

    // the temporal working space for training and predicting
    flt_type *d_features_{nullptr};
    flt_type *d_grad_features_{nullptr};

    flt_type *d_grad_weights_{nullptr};
    flt_type *d_grad_biases_{nullptr};

    // the memory for weights and biases
    flt_type *d_weights_{nullptr};
    flt_type *d_biases_{nullptr};

    bool is_memory_for_weights_allocated_{false};

    // layers in the network
    std::vector<Layer *> layers_;
    int batch_size_;

    CudaContext cuda_;

    WorkloadType phase_{WorkloadType::inference};
    size_t length_weights_{}, length_biases_{};
};
