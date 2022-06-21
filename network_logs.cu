void Network::AddLayers() {
    layers_.emplace_back(
        new Conv2D("conv0", 64, 7, 1, 3));  //[224x224x3]->[224x224x64]
    layers_.emplace_back(new Activation("tanh", CUDNN_ACTIVATION_TANH));
    layers_.emplace_back(new Pooling(
        "pool", 2, 0, 2, CUDNN_POOLING_MAX));  //[224x224x64]->[112x112x64]

    layers_.emplace_back(
        new Conv2D("conv1", 128, 5, 1, 2));  //[112x112x64]->[112x112x128]
    layers_.emplace_back(new Activation("tanh", CUDNN_ACTIVATION_TANH));
    layers_.emplace_back(new Pooling(
        "pool", 2, 0, 2, CUDNN_POOLING_MAX));  //[112x112x128]->[56x56x128]

    layers_.emplace_back(
        new Conv2D("conv2", 256, 3, 1, 1));  //[56x56x128]->[56x56x256]
    layers_.emplace_back(new Activation("tanh", CUDNN_ACTIVATION_TANH));
    layers_.emplace_back(
        new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));  //[28x28x256]

    layers_.emplace_back(
        new Conv2D("conv3", 256, 3, 1, 1));  //[28x28x256]->[28x28x256]
    layers_.emplace_back(new Activation("tanh", CUDNN_ACTIVATION_TANH));
    layers_.emplace_back(
        new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));  //[14x14x256]

    layers_.emplace_back(
        new Conv2D("conv4", 256, 3, 1, 1));  //[14x14x256]->[14x14x256]
    layers_.emplace_back(new Activation("tanh", CUDNN_ACTIVATION_TANH));
    layers_.emplace_back(
        new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));  //[7x7x256]

    layers_.emplace_back(
        new Conv2D("conv5", 512, 3, 1, 1));  //[7x7x256]->[7x7x512]
    layers_.emplace_back(new Activation("tanh", CUDNN_ACTIVATION_TANH));

    layers_.emplace_back(new Fully_connected("fully_connected", 1000));
    layers_.emplace_back(new Activation("tanh", CUDNN_ACTIVATION_TANH));
    layers_.emplace_back(new Softmax("softmax"));

    layers_[0]->SetGradientStop();
}

void Network::AddLayers() {
    layers_.emplace_back(
        new Conv2D("conv0", 64, 7, false, 1, 3));  //[224x224x3]->[224x224x64]
    layers_.emplace_back(new Batchnorm2D("bn0"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 1));
    layers_.emplace_back(new Pooling(
        "pool", 2, 0, 2, CUDNN_POOLING_MAX));  //[224x224x64]->[112x112x64]

    layers_.emplace_back(new Conv2D("conv1", 128, 5, false, 1,
                                    2));  //[112x112x64]->[112x112x128]
    layers_.emplace_back(new Batchnorm2D("bn1"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 1));
    layers_.emplace_back(new Pooling(
        "pool", 2, 0, 2, CUDNN_POOLING_MAX));  //[112x112x128]->[56x56x128]

    layers_.emplace_back(
        new Conv2D("conv2", 256, 3, false, 1, 1));  //[56x56x128]->[56x56x256]
    layers_.emplace_back(new Batchnorm2D("bn2"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 1));
    layers_.emplace_back(
        new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));  //[28x28x256]

    layers_.emplace_back(
        new Conv2D("conv3", 256, 3, false, 1, 1));  //[28x28x256]->[28x28x256]
    layers_.emplace_back(new Batchnorm2D("bn3"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 1));
    layers_.emplace_back(
        new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));  //[14x14x256]

    layers_.emplace_back(
        new Conv2D("conv4", 256, 3, false, 1, 1));  //[14x14x256]->[14x14x256]
    layers_.emplace_back(new Batchnorm2D("bn4"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 1));
    layers_.emplace_back(
        new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));  //[7x7x256]

    layers_.emplace_back(
        new Conv2D("conv5", 512, 3, false, 1, 1));  //[7x7x256]->[7x7x512]
    layers_.emplace_back(new Batchnorm2D("bn5"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 1));

    layers_.emplace_back(new Fully_connected("fully_connected", 1000));
    layers_.emplace_back(new Softmax("softmax"));

    layers_[0]->SetGradientStop();
}

void Network::AddLayers() {
    // AlexNet-BN
    layers_.emplace_back(new Conv2D("conv0", 64, 11, false, 4, 2));
    layers_.emplace_back(new Batchnorm2D("bn0"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));
    layers_.emplace_back(new Pooling("pool", 3, 0, 2, CUDNN_POOLING_MAX));

    layers_.emplace_back(new Conv2D("conv1", 192, 5, false, 1, 2));
    layers_.emplace_back(new Batchnorm2D("bn1"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));
    layers_.emplace_back(new Pooling("pool", 3, 0, 2, CUDNN_POOLING_MAX));

    layers_.emplace_back(new Conv2D("conv2", 384, 3, false, 1, 1));
    layers_.emplace_back(new Batchnorm2D("bn2"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));

    layers_.emplace_back(new Conv2D("conv3", 256, 3, false, 1, 1));
    layers_.emplace_back(new Batchnorm2D("bn3"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));

    layers_.emplace_back(new Conv2D("conv4", 256, 3, false, 1, 1));
    layers_.emplace_back(new Batchnorm2D("bn4"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));
    layers_.emplace_back(new Pooling("pool", 3, 0, 2, CUDNN_POOLING_MAX));

    layers_.emplace_back(new Fully_connected("fully_connected", 4096, false));
    layers_.emplace_back(new Batchnorm2D("bn5"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));
    layers_.emplace_back(new Fully_connected("fully_connected", 4096, false));
    layers_.emplace_back(new Batchnorm2D("bn6"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));
    layers_.emplace_back(new Fully_connected("fully_connected", 1000));
    layers_.emplace_back(new Softmax("softmax"));

    layers_[0]->SetGradientStop();
}

void Network::AddLayers() {
    // AlexNet-BN-residualFC
    layers_.emplace_back(new Conv2D("conv0", 64, 11, false, 4, 2));
    layers_.emplace_back(new Batchnorm2D("bn0"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));
    layers_.emplace_back(new Pooling("pool", 3, 0, 2, CUDNN_POOLING_MAX));

    layers_.emplace_back(new Conv2D("conv1", 192, 5, false, 1, 2));
    layers_.emplace_back(new Batchnorm2D("bn1"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));
    layers_.emplace_back(new Pooling("pool", 3, 0, 2, CUDNN_POOLING_MAX));

    layers_.emplace_back(new Conv2D("conv2", 384, 3, false, 1, 1));
    layers_.emplace_back(new Batchnorm2D("bn2"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));

    layers_.emplace_back(new Conv2D("conv3", 256, 3, false, 1, 1));
    layers_.emplace_back(new Batchnorm2D("bn3"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));

    layers_.emplace_back(new Conv2D("conv4", 256, 3, false, 1, 1));
    layers_.emplace_back(new Batchnorm2D("bn4"));
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));
    layers_.emplace_back(new Pooling("pool", 3, 0, 2, CUDNN_POOLING_MAX));

    layers_.emplace_back(new Fully_connected("fully_connected", 4096, false));
    layers_.emplace_back(new Batchnorm2D("bn5"));
    Layer *layer_fc1 = layers_.back();
    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));
    layers_.emplace_back(new Fully_connected("fully_connected", 4096, false));
    layers_.emplace_back(new Batchnorm2D("bn6"));
    Layer *layer_fc2 = layers_.back();

    // Residual
    layers_.emplace_back(new Residual("res0", layer_fc1, layer_fc2));

    layers_.emplace_back(new Activation("relu", CUDNN_ACTIVATION_RELU, 2));
    layers_.emplace_back(new Fully_connected("fully_connected", 1000));
    layers_.emplace_back(new Softmax("softmax"));

    layers_[0]->SetGradientStop();
}

void Network::AddLayers() {
    // AlexNet-BN-residualFC
    Layer *conv0 = new Conv2D("conv0", 64, 11, false, 4, 2);
    conv0->SetGradientStop();
    Layer *bn0 = new Batchnorm2D("bn0");
    Layer *relu0 = new Activation("relu0", CUDNN_ACTIVATION_RELU, 2);
    Layer *pool0 = new Pooling("pool0", 3, 0, 2, CUDNN_POOLING_MAX);
    layerGraph_.AddEdge(conv0, bn0);
    layerGraph_.AddEdge(bn0, relu0);
    layerGraph_.AddEdge(relu0, pool0);

    Layer *conv1 = new Conv2D("conv1", 192, 5, false, 1, 2);
    Layer *bn1 = new Batchnorm2D("bn1");
    Layer *relu1 = new Activation("relu1", CUDNN_ACTIVATION_RELU, 2);
    Layer *pool1 = new Pooling("pool1", 3, 0, 2, CUDNN_POOLING_MAX);
    layerGraph_.AddEdge(pool0, conv1);
    layerGraph_.AddEdge(conv1, bn1);
    layerGraph_.AddEdge(bn1, relu1);
    layerGraph_.AddEdge(relu1, pool1);

    Layer *conv2 = new Conv2D("conv2", 384, 3, false, 1, 1);
    Layer *bn2 = new Batchnorm2D("bn2");
    Layer *relu2 = new Activation("relu2", CUDNN_ACTIVATION_RELU, 2);
    layerGraph_.AddEdge(pool1, conv2);
    layerGraph_.AddEdge(conv2, bn2);
    layerGraph_.AddEdge(bn2, relu2);

    Layer *conv3 = new Conv2D("conv3", 256, 3, false, 1, 1);
    Layer *bn3 = new Batchnorm2D("bn3");
    Layer *relu3 = new Activation("relu3", CUDNN_ACTIVATION_RELU, 2);
    layerGraph_.AddEdge(relu2, conv3);
    layerGraph_.AddEdge(conv3, bn3);
    layerGraph_.AddEdge(bn3, relu3);

    Layer *conv4 = new Conv2D("conv4", 256, 3, false, 1, 1);
    Layer *bn4 = new Batchnorm2D("bn4");
    Layer *relu4 = new Activation("relu4", CUDNN_ACTIVATION_RELU, 2);
    Layer *pool4 = new Pooling("pool4", 3, 0, 2, CUDNN_POOLING_MAX);
    layerGraph_.AddEdge(relu3, conv4);
    layerGraph_.AddEdge(conv4, bn4);
    layerGraph_.AddEdge(bn4, relu4);
    layerGraph_.AddEdge(relu4, pool4);

    Layer *fc0 = new Fully_connected("fc0", 4096, false);
    Layer *fc0_bn = new Batchnorm2D("fc0_bn");
    Layer *fc0_relu = new Activation("fc0_relu", CUDNN_ACTIVATION_RELU, 2);
    layerGraph_.AddEdge(pool4, fc0);
    layerGraph_.AddEdge(fc0, fc0_bn);
    layerGraph_.AddEdge(fc0_bn, fc0_relu);

    Layer *fc1 = new Fully_connected("fc1", 4096, false);
    Layer *fc1_bn = new Batchnorm2D("fc1_bn");
    Layer *fc1_relu = new Activation("fc1_relu", CUDNN_ACTIVATION_RELU, 2);
    layerGraph_.AddEdge(fc0_relu, fc1);
    layerGraph_.AddEdge(fc1, fc1_bn);

    // Residual
    Layer *res0 = new Residual("res0", fc0_relu, fc1_bn);
    layerGraph_.AddEdge(fc0_relu, res0);
    layerGraph_.AddEdge(fc1_bn, res0);
    layerGraph_.AddEdge(res0, fc1_relu);

    Layer *fc2 = new Fully_connected("fc2", IMAGENET_CLASSES);
    Layer *softmax = new Softmax("softmax");
    layerGraph_.AddEdge(fc1_relu, fc2);
    layerGraph_.AddEdge(fc2, softmax);

    layerGraph_.TopoSort();
}