/*
 * \file layer.cc
 */
#include <algorithm>
#include <cassert>
#include <random>

#include "layer.h"

__global__ void InitiateZeros(float *d_one_vec, size_t length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    d_one_vec[i] = 0.f;
}

__global__ void InitiateVecOnes(float *d_one_vec, size_t length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    d_one_vec[i] = 1.f;
}

void LayerGraph::AddEdge(Layer *from, Layer *to) {
    layerCollection_.insert(from);
    layerCollection_.insert(to);
    edgeGraph_[from].emplace_back(to);
    invEdgeGraph_[to].emplace_back(from);
}

void LayerGraph::TopoSort() {
    assert(invEdgeGraph_.size() > 0);
    std::queue<Layer *> que;
    std::map<Layer *, int> indegrees;
    std::vector<Layer *>().swap(layers_);  // clear layers_

    for (auto node : layerCollection_) {
        indegrees[node] = 0;
    }

    for (auto edgeSet : invEdgeGraph_) {
        for (auto node : edgeSet.second) {
            indegrees[node]++;
        }
    }

    for (auto node : layerCollection_) {
        if (indegrees[node] == 0) {
            que.push(node);
        }
    }

    while (!que.empty()) {
        Layer *srcNode = que.front();
        layers_.emplace_back(srcNode);
        que.pop();
        for (auto dstNode : invEdgeGraph_[srcNode]) {
            int ind = --indegrees[dstNode];
            if (ind == 0) {
                que.push(dstNode);
            }
        }
    }

    std::reverse(layers_.begin(), layers_.end());
    // for (auto layer : layers_) {
    //     std::cout << layer->GetName() << "\n";
    // }
}

void Layer::InitiateWeightsAndBiases() {
    if (weights_.CudaPtr() == nullptr || biases_.CudaPtr() == nullptr) return;

    // Create random network
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(1118);

    // Xaiver like init
    float range;
    range = sqrt(6.f / input_.LengthChw());
    std::uniform_real_distribution<> uniform(-range, range);

    // He kaiming init for Conv2D
    // actually [outdim*kernel_size*kernel_size]
    std::normal_distribution<> dis(
        0, sqrt(2.0 / (weights_.GetHeight() * weights_.GetWidth() *
                       weights_.get_n())));

    // He kaiming init for Linear
    // Height is actually input dims for fc
    std::uniform_real_distribution<> disFC(-sqrt(1.0 / weights_.GetHeight()),
                                           sqrt(1.0 / weights_.GetHeight()));

    std::vector<float> weights(weights_.LengthNchw(), 0.0);
    if (this->GetLayerType() == LayerType::Conv2D) {
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] = static_cast<float>(dis(gen));
        }
    } else if (this->GetLayerType() == LayerType::Fully_connected) {
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] = static_cast<float>(disFC(gen));
        }
    } else {
        assert(false);
    }
    weights_.ToDevice(weights.data(), weights.size());

    std::vector<float> biases(biases_.LengthNchw(), 0.0);
    if (this->GetLayerType() == LayerType::Conv2D) {
        // conv2d has no bias in resnet
        for (size_t i = 0; i < biases.size(); i++) {
            biases[i] = 0.f;
        }
    } else if (this->GetLayerType() == LayerType::Fully_connected) {
        for (size_t i = 0; i < biases.size(); i++) {
            biases[i] = static_cast<float>(disFC(gen));
        }
    } else {
        assert(false);
    }

    biases_.ToDevice(biases.data(), biases.size());
}

int Layer::ObtainPredictionAccuracy(std::vector<label_t> const &labels,
                                    std::vector<int> &confusion_matrix) {
    assert("No Loss layer cannot estimate accuracy." && false);
    return EXIT_FAILURE;
}

void Layer::BackwardCopy() {
    if (gradient_stop_) return;
    if (afterSplitLayer_) {
        checkCublasErrors(cublasSaxpy(cuda_->cublas(), input_.LengthNchw(),
                                      &cuda_->one, d_temp_grad_features_, 1,
                                      input_.CudaPtr(), 1));
    } else {
        checkCudaErrors(cudaMemcpy(input_.CudaPtr(), d_temp_grad_features_,
                                   input_.buf_size(),
                                   cudaMemcpyDeviceToDevice));
    }
}

void Layer::DescriptorsAndWorkSpace() { return; }
