/*
 * \file network.h
 */
#pragma once

#include <map>
#include <queue>
#include <set>
#include <vector>

#include "Dataset/Dataset.h"
#include "ImageNetParser/ImageNetParser.h"
#include "activation_layer.h"
#include "batchnorm_layer.h"
#include "convolutional_layer.h"
#include "fully_connected_layer.h"
#include "pooling_layer.h"
#include "residual_layer.h"
#include "softmax_layer.h"

class LayerGraph {
   public:
    LayerGraph() {}
    void AddEdge(Layer *from, Layer *to);
    void TopoSort();

    std::vector<Layer *> layers_;  // In topological
    std::set<Layer *> layerCollection_;
    std::map<Layer *, std::vector<Layer *>> edgeGraph_;
};

class Network {
   public:
    Network();
    ~Network();

    int GetBatchSize() const { return batch_size_; }
    void SetBatchSize(int const in) { batch_size_ = in; }

    virtual void AddLayers();
    void SetWorkloadType(WorkloadType const &in);
    void Train(const Dataset<dataType> *datasetPtr, float const learning_rate,
               float const momentum, float const weightDecay);

    void Predict(const Dataset<dataType> *datasetPtr);

    void AllocateMemoryForFeatures();
    void InitWeights();
    void DescriptorsAndWorkspace();

   protected:
    void SetCudaContext();
    void Forward();
    void Backward(BlobPointer<float> const &labels);

    void Update(float const learning_rate, float const momentum,
                float const weightDecay);

    int ObtainPredictionAccuracy(std::vector<label_t> const &labels,
                                 std::vector<int> &confusion_matrix);

    // the temporal working space for training and predicting
    float *d_features_{nullptr};
    float *d_grad_features_{nullptr};

    float *d_grad_weights_{nullptr};
    float *d_grad_biases_{nullptr};

    // for momentum-SGD
    float *d_grad_weights_history_{nullptr};
    float *d_grad_biases_history_{nullptr};

    // the memory for weights and biases
    float *d_weights_{nullptr};
    float *d_biases_{nullptr};

    bool is_memory_for_weights_allocated_{false};

    // layers in the network
    LayerGraph layerGraph_;
    int batch_size_;

    CudaContext cuda_;

    WorkloadType phase_{WorkloadType::inference};
    size_t length_weights_{}, length_biases_{}, length_features_{};
};