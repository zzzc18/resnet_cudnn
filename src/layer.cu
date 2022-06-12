/*
 * \file layer.cc
 */
#include <cassert>
#include <random>

#include "layer.h"
#include "lenet_5_common.h"

void Layer::InitiateWeightsAndBiases() {
    if (weights_.CudaPtr() == nullptr || biases_.CudaPtr() == nullptr) return;

    // Create random network
    std::random_device rd;
    std::mt19937 gen(0);

    flt_type range;
    range = sqrt(6.f / input_.LengthChw());
    std::uniform_real_distribution<> dis(-range, range);

    std::vector<flt_type> weights(weights_.LengthNchw(), 0.0);

    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = static_cast<flt_type>(dis(gen));
    }

    weights_.ToDevice(weights.data(), weights.size());

    std::vector<flt_type> biases(biases_.LengthNchw(), 0.0);
    for (size_t i = 0; i < biases.size(); i++) biases[i] = 0.f;

    biases_.ToDevice(biases.data(), biases.size());
}

int Layer::ObtainPredictionAccuracy(std::vector<label_t> const &labels,
                                    std::vector<int> &confusion_matrix) {
    assert("No Loss layer cannot estimate accuracy." && false);
    return EXIT_FAILURE;
}

void Layer::DescriptorsAndWorkSpace() { return; }