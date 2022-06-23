/*
 * \file network.cc
 */
#include <chrono>
#include <iostream>
#include <random>

#include "Dataset/Dataset.h"
#include "ImageNetParser/ImageNetParser.h"
#include "network.h"
#include "utilities_sc.h"

Network::Network() {
    // nothing
}

Network::~Network() {
    // destroy network
    for (auto layer : layerGraph_.layers_) delete layer;
    if (d_features_ != nullptr) checkCudaErrors(cudaFree(d_features_));
    // if (d_grad_features_ != nullptr)
    //     checkCudaErrors(cudaFree(d_grad_features_));
    if (d_grad_weights_ != nullptr) checkCudaErrors(cudaFree(d_grad_weights_));
    if (d_grad_biases_ != nullptr) checkCudaErrors(cudaFree(d_grad_biases_));
    if (d_grad_weights_history_ != nullptr)
        checkCudaErrors(cudaFree(d_grad_weights_history_));
    if (d_grad_biases_history_ != nullptr)
        checkCudaErrors(cudaFree(d_grad_biases_history_));

    if (d_weights_ != nullptr) checkCudaErrors(cudaFree(d_weights_));
    if (d_biases_ != nullptr) checkCudaErrors(cudaFree(d_biases_));
}

void Network::SetWorkloadType(WorkloadType const &in) {
    phase_ = in;
    for (auto layer : layerGraph_.layers_) {
        layer->SetWorkloadType(in);
    }
}

void Network::Forward() {
    for (auto layer : layerGraph_.layers_) {
        layer->Forward();

        // Save weights
        // layer->weights_.SaveAsNumpyArray("weights/" + layer->GetName() +
        //                                  "-weight.npy");
        // layer->biases_.SaveAsNumpyArray("weights/" + layer->GetName() +
        //                                 "-bias.npy");
        // if (layer->GetLayerType() == LayerType::Batchnorm2D) {
        //     Batchnorm2D *bn = (Batchnorm2D *)layer;
        //     BlobPointer<float> mean, var;
        //     mean.Initiate({1, bn->weights_.GetChannels(), 1, 1},
        //                   (float *)bn->resultRunningMean_);
        //     mean.SaveAsNumpyArray("weights/" + layer->GetName() +
        //                                     "-running_mean.npy");
        //     var.Initiate({1, bn->weights_.GetChannels(), 1, 1},
        //                  (float *)bn->resultRunningVariance_);
        //     var.SaveAsNumpyArray("weights/" + layer->GetName() +
        //                                     "-running_var.npy");
        // }

        // Debug
        // std::cout << "[[Forward ]][[ " << std::setw(7) << layer->GetName()
        //           << " ]]\t(" << layer->input_.GetChannels() << ", "
        //           << layer->input_.GetHeight() << ", "
        //           << layer->input_.GetWidth() << ")\t";
        // std::cout << "--> (" << layer->output_.GetChannels() << ", "
        //           << layer->output_.GetHeight() << ", "
        //           << layer->output_.GetWidth() << ")\n";
        // std::cout << "input_ " << layer->GetInput().CudaPtr() << "\n";
        // std::cout << "output_ " << layer->GetOutput().CudaPtr() << "\n";
    }
}

void Network::Backward(BlobPointer<float> const &labels) {
    if (phase_ == WorkloadType::inference) return;
    // clear split output_
    for (auto layer : layerGraph_.layers_) {
        if (layer->GetLayerType() == LayerType::Split) {
            checkCudaErrors(cudaMemset(layer->GetOutput().CudaPtr(), 0,
                                       layer->GetOutput().buf_size()));
        }
    }

    // back propagation.. update weights internally.....
    for (auto layer = layerGraph_.layers_.rbegin();
         layer != layerGraph_.layers_.rend(); layer++) {
        (*layer)->Backward(labels);
    }
}

void Network::Update(float const learning_rate, float const momentum,
                     float const weightDecay) {
    if (phase_ == WorkloadType::inference) return;

    float eta = -1.f * learning_rate;

    if (weightDecay > 0) {
        // dw = dw+weightDecay*w
        checkCublasErrors(cublasSaxpy(cuda_.cublas(), length_weights_,
                                      &weightDecay, d_weights_, 1,
                                      d_grad_weights_, 1));
    }

    if (momentum > 0) {
        // dw_t = momentum*dw_{t-1} + dw_t
        checkCublasErrors(cublasSaxpy(cuda_.cublas(), length_weights_,
                                      &momentum, d_grad_weights_history_, 1,
                                      d_grad_weights_, 1));
        // db_t = momentum*db_{t-1} + db_t
        checkCublasErrors(cublasSaxpy(cuda_.cublas(), length_biases_, &momentum,
                                      d_grad_biases_history_, 1, d_grad_biases_,
                                      1));
    }

    // w = w + eps * dw
    checkCublasErrors(cublasSaxpy(cuda_.cublas(), length_weights_, &eta,
                                  d_grad_weights_, 1, d_weights_, 1));
    // b = b + eps * db
    checkCublasErrors(cublasSaxpy(cuda_.cublas(), length_biases_, &eta,
                                  d_grad_biases_, 1, d_biases_, 1));

    // dw_t = dw
    checkCudaErrors(cudaMemcpy(d_grad_weights_history_, d_grad_weights_,
                               length_weights_, cudaMemcpyDeviceToDevice));
    // db_t = db
    checkCudaErrors(cudaMemcpy(d_grad_biases_history_, d_grad_biases_,
                               length_biases_, cudaMemcpyDeviceToDevice));
}

// 1. initialize cuda resource container
// 2. register the resource container to all the layers
void Network::SetCudaContext() {
    for (auto layer : layerGraph_.layers_) {
        layer->SetCudaContext(&cuda_);
    }
}

int Network::ObtainPredictionAccuracy(std::vector<label_t> const &target,
                                      std::vector<int> &confusion_matrix) {
    Layer *layer = layerGraph_.layers_.back();
    return layer->ObtainPredictionAccuracy(target, confusion_matrix);
}

void Network::AddLayers() {
    // AlexNet-BN
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
    layerGraph_.AddEdge(fc1_bn, fc1_relu);

    Layer *fc2 = new Fully_connected("fc2", IMAGENET_CLASSES);
    Layer *softmax = new Softmax("softmax");
    layerGraph_.AddEdge(fc1_relu, fc2);
    layerGraph_.AddEdge(fc2, softmax);

    layerGraph_.TopoSort();
}

void Network::Train(const Dataset<dataType> *datasetPtr,
                    float const learning_rate, float const momentum = 0,
                    float const weightDecay = 0) {
    std::vector<int> rand_perm(datasetPtr->Length());
    for (size_t i = 0; i < datasetPtr->Length(); i++) rand_perm[i] = i;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(rand_perm.begin(), rand_perm.end(),
                 std::default_random_engine(seed));

    int index = 0, progress = 0;
    int batch_size_train = this->GetBatchSize();
    int batch_count_train = datasetPtr->Length() / batch_size_train;
    int size_one_feature = 224 * 224 * 3;
    std::vector<float> samples(size_one_feature * batch_size_train);
    std::vector<float> labels(IMAGENET_CLASSES * batch_size_train, 0.0);

    int num_success = 0;
    std::vector<label_t> labelsPredict(IMAGENET_CLASSES * batch_size_train, 0);
    std::vector<int> confusion_matrix(IMAGENET_CLASSES * IMAGENET_CLASSES, 0);

    float *d_train_labels{nullptr};
    checkCudaErrors(
        cudaMalloc((void **)&d_train_labels,
                   sizeof(float) * batch_size_train * IMAGENET_CLASSES));
    BlobPointer<float> train_labels_ptr(batch_size_train, IMAGENET_CLASSES, 1,
                                        1, d_train_labels);

    std::cout << "训练 " << batch_count_train << "\n";
    for (int step = 0; step < batch_count_train; step++) {
#pragma omp parallel for num_threads(14)
        for (int i = 0; i < batch_size_train; i++) {
            dataType item =
                datasetPtr->GetItem(rand_perm[step * batch_size_train + i]);
            for (int k = 0; k < size_one_feature; k++) {
                samples[i * size_one_feature + k] = *(float *)(item.first + k);
            }
            labels[i * IMAGENET_CLASSES + item.second] = 1;
            labelsPredict[i * IMAGENET_CLASSES + item.second] = 1;
            free(item.first);
        }

        this->layerGraph_.layers_[0]->input_.ToDevice(samples);
        train_labels_ptr.ToDevice(labels);

        this->Forward();
        num_success +=
            this->ObtainPredictionAccuracy(labelsPredict, confusion_matrix);
        this->Backward(train_labels_ptr);
        this->Update(learning_rate, momentum, weightDecay);

        for (int i = 0; i < batch_size_train; i++) {
            labels[i * IMAGENET_CLASSES +
                   datasetPtr->GetLabel(
                       rand_perm[step * batch_size_train + i])] = 0;
            labelsPredict[i * IMAGENET_CLASSES +
                          datasetPtr->GetLabel(
                              rand_perm[step * batch_size_train + i])] = 0;
        }

        // fetch next data
        index += batch_size_train;
        int k = index * 100 / datasetPtr->Length();
        if (progress == 0 || k > progress + 1) {
            ProgressBar(k);
            progress = k;
        }
    }

    std::cout << "\nTrain精度: " << num_success << "/" << datasetPtr->Length()
              << "\n";
    std::cout << "Train_Acc: " << 100.0 * num_success / datasetPtr->Length()
              << "%\n";
    checkCudaErrors(cudaFree(d_train_labels));
}

void Network::Predict(const Dataset<dataType> *datasetPtr) {
    int num_success = 0;
    int batch_size_test = this->GetBatchSize();
    int batch_count_test = datasetPtr->Length() / batch_size_test;
    int size_one_feature = 224 * 224 * 3;

    std::vector<float> samples(size_one_feature * batch_size_test);
    std::vector<label_t> labels(IMAGENET_CLASSES * batch_size_test, 0);

    std::vector<int> confusion_matrix(IMAGENET_CLASSES * IMAGENET_CLASSES, 0);

    for (auto step = 0; step < batch_count_test; step++) {
#pragma omp parallel for num_threads(14)
        for (int i = 0; i < batch_size_test; i++) {
            dataType item = datasetPtr->GetItem(step * batch_size_test + i);
            for (int k = 0; k < size_one_feature; k++) {
                samples[i * size_one_feature + k] = *(item.first + k);
            }
            labels[i * IMAGENET_CLASSES + item.second] = 1;
            free(item.first);
        }

        this->layerGraph_.layers_[0]->input_.ToDevice(samples);
        this->Forward();

        num_success += this->ObtainPredictionAccuracy(labels, confusion_matrix);

        for (int i = 0; i < batch_size_test; i++) {
            labels[i * IMAGENET_CLASSES +
                   datasetPtr->GetLabel(step * batch_size_test + i)] = 0;
        }
    }

    // step 4. calculate loss and accuracy
    std::cout << "\nVal精度: " << num_success << "/" << datasetPtr->Length()
              << "\n";
    std::cout << "Val_Acc: " << 100.0 * num_success / datasetPtr->Length()
              << "%\n";
    // std::cout << "\n   *  ";
    // for (int i = 0; i < IMAGENET_CLASSES; i++)
    //     std::cout << std::setw(4) << i << "  ";
    // std::cout << "\n";
    // for (int i = 0; i < IMAGENET_CLASSES; i++) {
    //     std::cout << std::setw(4) << i << "  ";
    //     for (int j = 0; j < IMAGENET_CLASSES; j++) {
    //         std::cout << std::setw(4)
    //                   << confusion_matrix[i * IMAGENET_CLASSES + j] << "  ";
    //     }
    //     std::cout << "\n";
    // }
}

void Network::AllocateMemoryForFeatures() {
    if (d_features_ != nullptr) {
        checkCudaErrors(cudaFree(d_features_));
        d_features_ = nullptr;
    }
    if (d_temp_grad_features_ != nullptr) {
        checkCudaErrors(cudaFree(d_temp_grad_features_));
        d_temp_grad_features_ = nullptr;
    }

    std::array<int, 4> shape;

    shape[0] = batch_size_;
    shape[1] = 3;
    shape[2] = 224;
    shape[3] = 224;

    /// 1. set the shape of the feature at each layer
    /// 2. get the total length of features among all layers
    /// 3. init nextLayer->afterSplitLayer_ for BackwardCopy
    length_features_ = shape[0] * shape[1] * shape[2] * shape[3];

    layerGraph_.layers_[0]->in_shape_ = shape;
    max_layer_length_feature_ = length_features_;
    for (auto layer : layerGraph_.layers_) {
        layer->InitFeatureShape();
        for (auto nextLayer : layerGraph_.edgeGraph_[layer]) {
            if (layer->GetLayerType() == LayerType::Split) {
                nextLayer->afterSplitLayer_ = true;
            }
            nextLayer->in_shape_ = layer->out_shape_;
        }
        int layer_length_feature = layer->out_shape_[0] * layer->out_shape_[1] *
                                   layer->out_shape_[2] * layer->out_shape_[3];

        length_features_ += layer_length_feature;
        max_layer_length_feature_ =
            std::max(max_layer_length_feature_, layer_length_feature);
    }

    checkCudaErrors(
        cudaMalloc((void **)&d_features_, sizeof(float) * length_features_));
    checkCudaErrors(cudaMalloc((void **)&d_temp_grad_features_,
                               sizeof(float) * max_layer_length_feature_));

    float *d_ptr_feature = d_features_;
    layerGraph_.layers_[0]->input_.Initiate(layerGraph_.layers_[0]->in_shape_,
                                            d_ptr_feature);
    int in_length = layerGraph_.layers_[0]->in_shape_[0] *
                    layerGraph_.layers_[0]->in_shape_[1] *
                    layerGraph_.layers_[0]->in_shape_[2] *
                    layerGraph_.layers_[0]->in_shape_[3];
    d_ptr_feature += in_length;
    for (auto layer : layerGraph_.layers_) {
        // Swap for grad calc
        layer->d_temp_grad_features_ = d_temp_grad_features_;

        int out_length = layer->out_shape_[0] * layer->out_shape_[1] *
                         layer->out_shape_[2] * layer->out_shape_[3];

        // d_retain_output_
        if (layer->GetLayerType() == LayerType::Pooling) {
            if (layer->d_retain_output_ != nullptr) {
                checkCudaErrors(cudaFree(layer->d_retain_output_));
                layer->d_retain_output_ = nullptr;
            }
            checkCudaErrors(cudaMalloc((void **)&layer->d_retain_output_,
                                       sizeof(float) * out_length));
        }

        layer->output_.Initiate(layer->out_shape_, d_ptr_feature);
        for (auto nextLayer : layerGraph_.edgeGraph_[layer]) {
            nextLayer->input_.Initiate(layer->out_shape_, d_ptr_feature);
        }
        d_ptr_feature += out_length;
    }
    return;
}

void Network::DescriptorsAndWorkspace() {
    this->SetCudaContext();
    for (auto layer : layerGraph_.layers_) {
        layer->input_.CreateTensor();
        layer->output_.CreateTensor();
        layer->biases_.CreateTensor();
        layer->DescriptorsAndWorkSpace();
    }
}

void Network::InitWeights() {
    if (is_memory_for_weights_allocated_ == false) {
        is_memory_for_weights_allocated_ = true;
        if (d_weights_ != nullptr) checkCudaErrors(cudaFree(d_weights_));
        if (d_biases_ != nullptr) checkCudaErrors(cudaFree(d_biases_));

        if (d_grad_weights_ != nullptr)
            checkCudaErrors(cudaFree(d_grad_weights_));
        if (d_grad_biases_ != nullptr)
            checkCudaErrors(cudaFree(d_grad_biases_));
        if (d_grad_weights_history_ != nullptr)
            checkCudaErrors(cudaFree(d_grad_weights_history_));
        if (d_grad_biases_history_ != nullptr)
            checkCudaErrors(cudaFree(d_grad_biases_history_));

        std::vector<std::array<int, 4>> shape_of_weights, shape_of_biases;
        shape_of_weights.reserve(layerGraph_.layers_.size());
        shape_of_biases.reserve(layerGraph_.layers_.size());

        for (auto layer : layerGraph_.layers_) {
            layer->InitWeightsShape(shape_of_weights, shape_of_biases);
        }

        int length_weights = 0, length_biases = 0;
        std::vector<int> len_weight(layerGraph_.layers_.size(), 0),
            len_bias(layerGraph_.layers_.size(), 0);

        for (size_t i = 0; i < len_weight.size(); i++) {
            len_weight[i] = shape_of_weights[i][0] * shape_of_weights[i][1] *
                            shape_of_weights[i][2] * shape_of_weights[i][3];
            length_weights += len_weight[i];

            len_bias[i] = shape_of_biases[i][0] * shape_of_biases[i][1] *
                          shape_of_biases[i][2] * shape_of_biases[i][3];
            length_biases += len_bias[i];
        }

        length_weights_ = length_weights;
        length_biases_ = length_biases;

        checkCudaErrors(
            cudaMalloc((void **)&d_weights_, sizeof(float) * length_weights));
        checkCudaErrors(
            cudaMalloc((void **)&d_biases_, sizeof(float) * length_biases));

        float *d_ptr_grad_weights = d_weights_;
        float *d_ptr_grad_biases = d_biases_;

        int i = 0;
        for (auto layer : layerGraph_.layers_) {
            if (len_weight[i] != 0) {
                auto shape = shape_of_weights[i];
                layer->weights_.Initiate(shape, d_ptr_grad_weights);
                d_ptr_grad_weights += len_weight[i];

                shape = shape_of_biases[i];
                layer->biases_.Initiate(shape, d_ptr_grad_biases);
                d_ptr_grad_biases += len_bias[i];
            }
            i++;
        }

        for (auto layer : layerGraph_.layers_) {
            layer->InitiateWeightsAndBiases();
        }

        // gradients of weights and biases
        checkCudaErrors(cudaMalloc((void **)&d_grad_weights_,
                                   sizeof(float) * length_weights));
        checkCudaErrors(cudaMalloc((void **)&d_grad_biases_,
                                   sizeof(float) * length_biases));

        checkCudaErrors(cudaMalloc((void **)&d_grad_weights_history_,
                                   sizeof(float) * length_weights));
        checkCudaErrors(cudaMemset(d_grad_weights_history_, 0,
                                   sizeof(float) * length_weights));
        checkCudaErrors(cudaMalloc((void **)&d_grad_biases_history_,
                                   sizeof(float) * length_biases));
        checkCudaErrors(cudaMemset(d_grad_biases_history_, 0,
                                   sizeof(float) * length_biases));

        d_ptr_grad_weights = d_grad_weights_;
        d_ptr_grad_biases = d_grad_biases_;

        i = 0;
        for (auto layer : layerGraph_.layers_) {
            if (len_weight[i] != 0) {
                auto shape = shape_of_weights[i];
                layer->grad_weights_.Initiate(shape, d_ptr_grad_weights);
                d_ptr_grad_weights += len_weight[i];

                shape = shape_of_biases[i];
                layer->grad_biases_.Initiate(shape, d_ptr_grad_biases);
                d_ptr_grad_biases += len_bias[i];
            }
            i++;
        }
    } else {
        checkCudaErrors(cudaMemset(d_grad_weights_history_, 0,
                                   sizeof(float) * length_weights_));
        checkCudaErrors(cudaMemset(d_grad_biases_history_, 0,
                                   sizeof(float) * length_biases_));
    }
}