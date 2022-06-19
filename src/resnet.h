#pragma once

#include <string>

#include "network.h"

const float RELU_COEF = 5.0;

enum ResNetBlockType {
    BASIC_BLOCK,
    BOOTLENECK,
};

class ResNet : public Network {
   public:
    static const int BASIC_BLOCK_EXPANSION = 1;
    static const int BOOTLENECK_EXPANSION = 4;
    ResNet(int layer_nums, int groups = 1, int baseWidth = 64,
           std::array<bool, 3> replaceStrideWithDilation = std::array<bool, 3>{
               false, false, false});
    ~ResNet() {}

    virtual void AddLayers() override;
    Layer *AddBasicBlock(std::string blockName, Layer *lastLayer, int planes,
                         int stride = 1,
                         std::pair<Layer *, Layer *> downsampleLayer =
                             std::pair<Layer *, Layer *>{nullptr, nullptr},
                         int groups = 1, int baseWidth = 64, int dilation = 1);
    Layer *AddBottleneckBlock(std::string blockName, Layer *lastLayer,
                              int planes, int stride = 1,
                              std::pair<Layer *, Layer *> downsampleLayer =
                                  std::pair<Layer *, Layer *>{nullptr, nullptr},
                              int groups = 1, int baseWidth = 64,
                              int dilation = 1);
    Layer *MakeLayer(std::string layerName, Layer *lastLayer, int planes,
                     int blocks, int stride = 1, bool dilate = false);

    std::array<int, 4> blockNums_;
    std::array<bool, 3> replaceStrideWithDilation_;
    int layerNums_;
    ResNetBlockType blockType_;
    int dilation_;
    int inplanes_;
    int groups_;
    int baseWidth_;
};