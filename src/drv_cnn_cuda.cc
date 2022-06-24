/*
 * \file drv_cnn_cuda.cc
 */

#include <cmath>
#include <iostream>

#include "ImageNetParser/ImageNetParser.h"
#include "network.h"
#include "resnet.h"

int main(int argc, char *argv[]) {
    ImageNetDataset<dataType> trainDataset("ImageNet", "train");
    ImageNetDataset<dataType> valDataset("ImageNet", "val");

    // float learning_rate = 0.01 * sqrt((float)batch_size_train);

    // Res18
    int batch_size_train = 64;
    int batch_size_test = 50;
    float learning_rate = 0.32;
    float learning_rate_lower_bound = 1E-4;
    float momentum = 0;
    float weightDecay = 0;
    // Res50
    // int batch_size_train = 32;
    // int batch_size_test = 32;
    // float learning_rate = 0.04;
    // float learning_rate_lower_bound = 1E-4;
    // float momentum = 0.875;
    // float weightDecay = 0;
    // float weightDecay = 1.0 / 32768;

    // Network net;
    ResNet net(18);
    net.AddLayers();

    const int epoch = 25;
    for (int i = 0; i < epoch; i++) {
        std::cout << "running at epoch: " << i << "\n";
        std::cout << "learning_rate: " << std::fixed << std::setprecision(5)
                  << learning_rate << "\n";
        net.SetWorkloadType(WorkloadType::training);
        net.SetBatchSize(batch_size_train);
        net.AllocateMemoryForFeatures();
        net.InitWeights();
        net.DescriptorsAndWorkspace();
        Timer t;
        Log("loss.log", "INFO", "Train@Epoch " + std::to_string(i));
        net.Train(&trainDataset, learning_rate, momentum, weightDecay);

        t.PrintDiff("训练时间...");

        net.SetWorkloadType(WorkloadType::inference);
        net.SetBatchSize(batch_size_test);
        net.AllocateMemoryForFeatures();
        net.DescriptorsAndWorkspace();
        Log("loss.log", "INFO", "Predict");
        net.Predict(&valDataset);

        t.PrintDiff("预测时间...");
        learning_rate *= 0.98;
        learning_rate = std::max(learning_rate_lower_bound, learning_rate);
        std::cout << "\n";
    }

    std::cout << "完成计算.\n";
    return 0;
}
