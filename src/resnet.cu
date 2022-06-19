#include "resnet.h"

ResNet::ResNet(int layerNums, int groups, int baseWidth,
               std::array<bool, 3> replaceStrideWithDilation)
    : layerNums_(layerNums),
      groups_(groups),
      baseWidth_(baseWidth),
      replaceStrideWithDilation_(replaceStrideWithDilation) {
    assert(layerNums_ == 18 || layerNums_ == 34 || layerNums_ == 50 ||
           layerNums_ == 101 || layerNums_ == 152);
    if (layerNums_ == 18) {
        blockNums_ = std::array<int, 4>{2, 2, 2, 2};
        blockType_ = ResNetBlockType::BASIC_BLOCK;
    } else if (layerNums_ == 34) {
        blockNums_ = std::array<int, 4>{3, 4, 6, 3};
        blockType_ = ResNetBlockType::BASIC_BLOCK;
    } else if (layerNums_ == 50) {
        blockNums_ = std::array<int, 4>{3, 4, 6, 3};
        blockType_ = ResNetBlockType::BOOTLENECK;
    } else if (layerNums_ == 101) {
        blockNums_ = std::array<int, 4>{3, 4, 23, 3};
        blockType_ = ResNetBlockType::BOOTLENECK;
    } else if (layerNums_ == 152) {
        blockNums_ = std::array<int, 4>{3, 4, 36, 3};
        blockType_ = ResNetBlockType::BOOTLENECK;
    }
}

Layer *ResNet::AddBasicBlock(std::string blockName, Layer *lastLayer,
                             int planes, int stride,
                             std::pair<Layer *, Layer *> downsampleLayer,
                             int groups, int baseWidth, int dilation) {
    assert(groups == 1 and baseWidth == 64 and dilation == 1);

    Layer *indentity = lastLayer;
    Layer *conv1 = new Conv2D(blockName + "conv1", planes, 3, false, stride, 1);
    Layer *bn1 = new Batchnorm2D(blockName + "bn1");
    Layer *relu1 = new Activation(blockName + "relu1");
    Layer *conv2 = new Conv2D(blockName + "conv2", planes, 3, false, 1, 1);
    Layer *bn2 = new Batchnorm2D(blockName + "bn2", true);
    Layer *relu2 = new Activation(blockName + "relu2");

    if (downsampleLayer.first != nullptr) {
        layerGraph_.AddEdge(lastLayer, downsampleLayer.first);
        layerGraph_.AddEdge(downsampleLayer.first, downsampleLayer.second);
        indentity = downsampleLayer.second;
    }

    Layer *res = new Residual(blockName + "res", bn2, indentity);

    layerGraph_.AddEdge(lastLayer, conv1);
    layerGraph_.AddEdge(conv1, bn1);
    layerGraph_.AddEdge(bn1, relu1);
    layerGraph_.AddEdge(relu1, conv2);
    layerGraph_.AddEdge(conv2, bn2);

    layerGraph_.AddEdge(bn2, res);
    layerGraph_.AddEdge(indentity, res);
    layerGraph_.AddEdge(res, relu2);
    return relu2;
}

Layer *ResNet::AddBottleneckBlock(std::string blockName, Layer *lastLayer,
                                  int planes, int stride,
                                  std::pair<Layer *, Layer *> downsampleLayer,
                                  int groups, int baseWidth, int dilation) {
    int width = planes * (baseWidth / 64) * groups;

    Layer *indentity = lastLayer;
    Layer *conv1 = new Conv2D(blockName + "conv1", width, 1, false);
    Layer *bn1 = new Batchnorm2D(blockName + "bn1");
    Layer *relu1 = new Activation(blockName + "relu1");
    Layer *conv2 = new Conv2D(blockName + "conv2", width, 3, false, stride,
                              dilation, dilation);
    Layer *bn2 = new Batchnorm2D(blockName + "bn2");
    Layer *relu2 = new Activation(blockName + "relu2");
    Layer *conv3 =
        new Conv2D(blockName + "conv3", width * BOOTLENECK_EXPANSION, 1, false);
    Layer *bn3 = new Batchnorm2D(blockName + "bn3", true);
    Layer *relu3 = new Activation(blockName + "relu3");

    if (downsampleLayer.first != nullptr) {
        layerGraph_.AddEdge(lastLayer, downsampleLayer.first);
        layerGraph_.AddEdge(downsampleLayer.first, downsampleLayer.second);
        indentity = downsampleLayer.second;
    }

    Layer *res = new Residual(blockName + "res", bn3, indentity);

    layerGraph_.AddEdge(lastLayer, conv1);
    layerGraph_.AddEdge(conv1, bn1);
    layerGraph_.AddEdge(bn1, relu1);
    layerGraph_.AddEdge(relu1, conv2);
    layerGraph_.AddEdge(conv2, bn2);
    layerGraph_.AddEdge(bn2, relu2);
    layerGraph_.AddEdge(relu2, conv3);
    layerGraph_.AddEdge(conv3, bn3);

    layerGraph_.AddEdge(bn3, res);
    layerGraph_.AddEdge(indentity, res);
    layerGraph_.AddEdge(res, relu3);
    return relu3;
}

Layer *ResNet::MakeLayer(std::string layerName, Layer *lastLayer, int planes,
                         int blocks, int stride, bool dilate) {
    int previousDilation = dilation_;
    if (dilate) {
        dilation_ *= stride;
        stride = 1;
    }
    int blockExpansion = blockType_ == BASIC_BLOCK ? BASIC_BLOCK_EXPANSION
                                                   : BOOTLENECK_EXPANSION;
    std::pair<Layer *, Layer *> downsampleLayer{nullptr, nullptr};
    if (stride != 1 || inplanes_ != planes * blockExpansion) {
        downsampleLayer = std::make_pair(
            new Conv2D(layerName + "block1-" + "conv_ds",
                       planes * blockExpansion, 1, false, stride),
            new Batchnorm2D(layerName + "block1-" + "bn_ds"));
    }

    Layer *newLayer{nullptr};
    if (blockType_ == BASIC_BLOCK) {
        newLayer = this->AddBasicBlock(layerName + "block1-", lastLayer, planes,
                                       stride, downsampleLayer, groups_,
                                       baseWidth_, previousDilation);
        for (int i = 1; i < blocks; i++) {
            newLayer = this->AddBasicBlock(
                layerName + "block" + std::to_string(i + 1) + "-", newLayer,
                planes, 1, std::pair<Layer *, Layer *>{nullptr, nullptr},
                groups_, baseWidth_, dilation_);
        }
    } else {
        newLayer = this->AddBottleneckBlock(
            layerName + "block1-", lastLayer, planes, stride, downsampleLayer,
            groups_, baseWidth_, previousDilation);
        for (int i = 1; i < blocks; i++) {
            newLayer = this->AddBottleneckBlock(
                layerName + "block" + std::to_string(i + 1) + "-", newLayer,
                planes, 1, std::pair<Layer *, Layer *>{nullptr, nullptr},
                groups_, baseWidth_, dilation_);
        }
    }
    return newLayer;
}

void ResNet::AddLayers() {
    inplanes_ = 64;
    dilation_ = 1;

    Layer *conv1 = new Conv2D("conv1", 64, 7, false, 2, 3);
    conv1->SetGradientStop();
    Layer *bn1 = new Batchnorm2D("bn1");
    Layer *relu1 = new Activation("relu1");
    Layer *pool1 = new Pooling("pool1", 3, 1, 2, CUDNN_POOLING_MAX);

    layerGraph_.AddEdge(conv1, bn1);
    layerGraph_.AddEdge(bn1, relu1);
    layerGraph_.AddEdge(relu1, pool1);

    Layer *layer1 = this->MakeLayer("layer1-", pool1, 64, blockNums_[0]);
    Layer *layer2 = this->MakeLayer("layer2-", layer1, 128, blockNums_[1], 2,
                                    replaceStrideWithDilation_[0]);
    Layer *layer3 = this->MakeLayer("layer3-", layer2, 256, blockNums_[2], 2,
                                    replaceStrideWithDilation_[1]);
    Layer *layer4 = this->MakeLayer("layer4-", layer3, 512, blockNums_[3], 2,
                                    replaceStrideWithDilation_[2]);

    Layer *finalPool = new Pooling("finalPool", 7, 0, 1,
                                   CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
    Layer *fc = new Fully_connected("fc", 1000);
    Layer *softmax = new Softmax("softmax");
    layerGraph_.AddEdge(layer4, finalPool);
    layerGraph_.AddEdge(finalPool, fc);
    layerGraph_.AddEdge(fc, softmax);

    layerGraph_.TopoSort();
}
