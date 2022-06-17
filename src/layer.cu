/*
 * \file layer.cc
 */
#include <cassert>
#include <random>

#include "layer.h"

__global__ void InitiateZeros(float *d_one_vec, size_t length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    d_one_vec[i] = 0.f;
}

void Layer::InitiateWeightsAndBiases() {
    if (weights_.CudaPtr() == nullptr || biases_.CudaPtr() == nullptr) return;

    // Create random network
    std::random_device rd;
    std::mt19937 gen(0);

    float range;
    range = sqrt(6.f / input_.LengthChw());
    std::uniform_real_distribution<> dis(-range, range);

    std::vector<float> weights(weights_.LengthNchw(), 0.0);

    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = static_cast<float>(dis(gen));
    }

    weights_.ToDevice(weights.data(), weights.size());

    std::vector<float> biases(biases_.LengthNchw(), 0.0);
    for (size_t i = 0; i < biases.size(); i++) biases[i] = 0.f;

    biases_.ToDevice(biases.data(), biases.size());
}

int Layer::ObtainPredictionAccuracy(std::vector<label_t> const &labels,
                                    std::vector<int> &confusion_matrix) {
    assert("No Loss layer cannot estimate accuracy." && false);
    return EXIT_FAILURE;
}

void Layer::DescriptorsAndWorkSpace() { return; }
