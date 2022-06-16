/*
 * \file drv_cnn_cuda.cc
 */

#include <iostream>

#include "ImageNetParser/ImageNetParser.h"
// #include "lenet_5_common.h"
#include "mnist_parser.h"
#include "network.h"

int main(int argc, char *argv[]) {
    ImageNetDataset<dataType> trainDataset("ImageNet", "train");
    ImageNetDataset<dataType> valDataset("ImageNet", "val");

    int batch_size_train = 64;

    float learning_rate = 0.01 * sqrt((float)batch_size_train);

    int batch_size_test = 64;

    Network net;
    net.AddLayers();

    const int epoch = 100;
    for (int i = 0; i < epoch; i++) {
        std::cout << "running at epoch: " << i << "\n";
        std::cout << "learning_rate: " << learning_rate << "\n";
        net.SetWorkloadType(WorkloadType::training);
        net.SetBatchSize(batch_size_train);
        net.AllocateMemoryForFeatures();
        net.InitWeights();
        net.DescriptorsAndWorkspace();
        Timer t;
        Log("loss.log", "INFO", "Train");
        net.Train(&trainDataset, learning_rate);

        t.PrintDiff("训练时间...");

        net.SetWorkloadType(WorkloadType::inference);
        net.SetBatchSize(batch_size_test);
        net.AllocateMemoryForFeatures();
        net.DescriptorsAndWorkspace();
        Log("loss.log", "INFO", "Predict");
        net.Predict(&valDataset);

        t.PrintDiff("预测时间...");
        // learning_rate *= 0.85;
        std::cout << "\n";
    }

    std::cout << "完成计算.\n";
    return 0;
}
