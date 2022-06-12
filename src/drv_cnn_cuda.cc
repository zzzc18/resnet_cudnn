/*
 * \file drv_cnn_cuda.cc
 */

#include <iostream>

#include "ImageNetParser/ImageNetParser.h"
#include "lenet_5_common.h"
#include "mnist_parser.h"
#include "network.h"

using namespace LeNet_5;

int main(int argc, char *argv[]) {
    ImageNetDataset<dataType> trainDataset("ImageNet", "train");
    ImageNetDataset<dataType> valDataset("ImageNet", "val");

    int batch_size_train = 64;

    flt_type learning_rate = 0.01 * sqrt((flt_type)batch_size_train);
    std::cout << "Lr" << learning_rate << "\n";

    int batch_size_test = 64;

    Network net;
    net.AddLayers();

    const int epoch = 2;
    for (int i = 0; i < epoch; i++) {
        net.SetWorkloadType(WorkloadType::training);
        net.SetBatchSize(batch_size_train);
        net.AllocateMemoryForFeatures();
        net.InitWeights();
        net.DescriptorsAndWorkspace();

        Timer t;

        net.Train(&trainDataset, learning_rate);
        t.PrintDiff("训练时间...");

        net.SetWorkloadType(WorkloadType::inference);
        net.SetBatchSize(batch_size_test);
        net.AllocateMemoryForFeatures();
        net.DescriptorsAndWorkspace();
        net.Predict(&valDataset);

        t.PrintDiff("预测时间...");
        learning_rate *= 0.85;
        std::cout << "\n";
    }

    std::cout << "完成计算.\n";
    return 0;
}
