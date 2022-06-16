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