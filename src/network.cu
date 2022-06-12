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

__global__ void setZero(flt_type *d_arr, size_t N) { memset(d_arr, 0, N); }

Network::Network() {
    // nothing
}

Network::~Network() {
    // destroy network
    for (auto layer : layers_) delete layer;
    if (d_features_ != nullptr) checkCudaErrors(cudaFree(d_features_));
    if (d_grad_features_ != nullptr)
        checkCudaErrors(cudaFree(d_grad_features_));
    if (d_grad_weights_ != nullptr) checkCudaErrors(cudaFree(d_grad_weights_));
    if (d_grad_biases_ != nullptr) checkCudaErrors(cudaFree(d_grad_biases_));

    if (d_weights_ != nullptr) checkCudaErrors(cudaFree(d_weights_));
    if (d_biases_ != nullptr) checkCudaErrors(cudaFree(d_biases_));
}

void Network::Forward() {
    for (auto layer : layers_) {
        layer->Forward();

#if (DEBUG_FORWARD)
        std::cout << "[[Forward ]][[ " << std::setw(7) << layer->GetName()
                  << " ]]\t(" << layer->input_.GetChannels() << ", "
                  << layer->input_.GetHeight() << ", "
                  << layer->input_.GetWidth() << ")\t";
#endif

#if (DEBUG_FORWARD)
        std::cout << "--> (" << layer->output_.GetChannels() << ", "
                  << layer->output_.GetHeight() << ", "
                  << layer->output_.GetWidth() << ")\n";
        checkCudaErrors(cudaDeviceSynchronize());
#endif
    }
}

void Network::Backward(BlobPointer<flt_type> const &labels) {
    if (phase_ == WorkloadType::inference) return;

    // back propagation.. update weights internally.....
    for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++) {
        // getting back propagation status with gradient size
#if (DEBUG_BACKWARD)
        std::cout << "[[Backward]][[ " << std::setw(7) << (*layer)->GetName()
                  << " ]]\t(" << (*layer)->grad_input_.GetChannels() << ", "
                  << (*layer)->grad_input_.GetHeight() << ", "
                  << (*layer)->grad_input_.GetWidth() << ")\t";
        checkCudaErrors(cudaDeviceSynchronize());
#endif

        (*layer)->Backward(labels);

#if (DEBUG_BACKWARD)
        // and the gradient result
        std::cout << "--> ("
                  << ", " << (*layer)->grad_output_.GetChannels() << ", "
                  << (*layer)->grad_output_.GetHeight() << ", "
                  << (*layer)->grad_output_.GetWidth() << ")\n";
#endif
    }
}

void Network::Update(flt_type const learning_rate) {
    if (phase_ == WorkloadType::inference) return;

#if (DEBUG_UPDATE)
    std::cout << "Start update.. lr = " << learning_rate << "\n";
#endif

    flt_type eps = -1.f * learning_rate;
    // w = w + eps * dw
    checkCublasErrors(cublasSaxpy(cuda_.cublas(), length_weights_, &eps,
                                  d_grad_weights_, 1, d_weights_, 1));

    // b = b + eps * db
    checkCublasErrors(cublasSaxpy(cuda_.cublas(), length_biases_, &eps,
                                  d_grad_biases_, 1, d_biases_, 1));
}

// 1. initialize cuda resource container
// 2. register the resource container to all the layers
void Network::SetCudaContext() {
    for (auto layer : layers_) {
        layer->SetCudaContext(&cuda_);
    }
}

int Network::ObtainPredictionAccuracy(std::vector<label_t> const &target,
                                      std::vector<int> &confusion_matrix) {
    Layer *layer = layers_.back();
    return layer->ObtainPredictionAccuracy(target, confusion_matrix);
}

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
        new Conv2D("conv3", 256, 3, 1, 1));  //[14x14x256]->[14x14x256]
    layers_.emplace_back(new Activation("tanh", CUDNN_ACTIVATION_TANH));
    layers_.emplace_back(
        new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));  //[7x7x256]

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

void Network::Train(const Dataset<dataType> *datasetPtr,
                    flt_type const learning_rate) {
    std::vector<int> rand_perm(datasetPtr->Length());
    for (size_t i = 0; i < datasetPtr->Length(); i++) rand_perm[i] = i;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(rand_perm.begin(), rand_perm.end(),
                 std::default_random_engine(seed));

    int index = 0, progress = 0;
    int batch_size_train = this->GetBatchSize();
    int batch_count_train = datasetPtr->Length() / batch_size_train;
    int size_one_feature = 224 * 224 * 3;
    std::vector<flt_type> samples(size_one_feature * batch_size_train);
    std::vector<flt_type> labels(1000 * batch_size_train, 0.0);

    flt_type *d_train_labels{nullptr};
    checkCudaErrors(cudaMalloc((void **)&d_train_labels,
                               sizeof(flt_type) * batch_size_train * 1000));
    BlobPointer<flt_type> train_labels_ptr(batch_size_train, 1000, 1, 1,
                                           d_train_labels);

    std::cout << "训练 " << batch_count_train << "\n";
    for (int step = 0; step < batch_count_train; step++) {
#pragma omp parallel for num_threads(12)
        for (int i = 0; i < batch_size_train; i++) {
            dataType item =
                datasetPtr->GetItem(rand_perm[step * batch_size_train + i]);
            for (int k = 0; k < size_one_feature; k++) {
                samples[i * size_one_feature + k] = *(float *)(item.first + k);
            }
            labels[i * 1000 + item.second] = 1;
            free(item.first);
        }

        this->layers_[0]->input_.ToDevice(samples);
        train_labels_ptr.ToDevice(labels);

        for (int i = 0; i < batch_size_train; i++) {
            labels[i * 1000 + datasetPtr->GetLabel(
                                  rand_perm[step * batch_size_train + i])] = 0;
        }

        this->Forward();
        this->Backward(train_labels_ptr);
        this->Update(learning_rate);

        // fetch next data
        index += batch_size_train;
        int k = index * 100 / datasetPtr->Length();
        if (progress == 0 || k > progress + 4) {
            ProgressBar(k);
            progress = k;
        }
    }

    checkCudaErrors(cudaFree(d_train_labels));

    std::cout << "\n";
}

// void Network::Train(std::vector<one_image> const &train_sample,
//                     std::vector<label_t> const &train_label,
//                     flt_type const learning_rate) {
//     std::vector<int> rand_perm(train_sample.size());
//     for (size_t i = 0; i < train_sample.size(); i++) rand_perm[i] = i;

//     int index = 0, progress = 0;
//     int batch_size_train = this->GetBatchSize();
//     int batch_count_train = train_sample.size() / batch_size_train;
//     int size_one_feature =
//         LeNet_5::kLengthOfMapAtLayer0 * LeNet_5::kLengthOfMapAtLayer0;
//     std::vector<flt_type> samples(size_one_feature * batch_size_train);
//     std::vector<flt_type> labels(LeNet_5::kNumClasses * batch_size_train,
//     0.0);

//     flt_type *d_train_labels{nullptr};
//     checkCudaErrors(
//         cudaMalloc((void **)&d_train_labels,
//                    sizeof(flt_type) * batch_size_train *
//                    LeNet_5::kNumClasses));
//     BlobPointer<flt_type> train_labels_ptr(
//         batch_size_train, LeNet_5::kNumClasses, 1, 1, d_train_labels);

//     std::cout << "训练 " << batch_count_train << "\n";
//     for (int step = 0; step < batch_count_train; step++) {
//         //    for(int step = 0; step < 2; step++) {

//         for (int i = 0; i < batch_size_train; i++) {
//             // for(int i = 0; i < 2; i++)

//             std::copy(
//                 train_sample[rand_perm[step * batch_size_train + i]].begin(),
//                 train_sample[rand_perm[step * batch_size_train + i]].end(),
//                 (samples.begin() + i * size_one_feature));

//             labels[i * LeNet_5::kNumClasses +
//                    train_label[step * batch_size_train + i]] = 1;
//         }

//         this->layers_[0]->input_.ToDevice(samples);
//         train_labels_ptr.ToDevice(labels);

//         for (int i = 0; i < batch_size_train; i++) {
//             labels[i * LeNet_5::kNumClasses +
//                    train_label[step * batch_size_train + i]] = 0;
//         }

//         this->Forward();

//         this->Backward(train_labels_ptr);

//         this->Update(learning_rate);

//         // fetch next data
//         index += batch_size_train;
//         int k = index * 100 / train_sample.size();
//         if (progress == 0 || k > progress + 4) {
//             ProgressBar(k);
//             progress = k;
//         }
//     }

//     checkCudaErrors(cudaFree(d_train_labels));

//     std::cout << "\n";
// }

void Network::Predict(const Dataset<dataType> *datasetPtr) {
    int num_success = 0;
    int batch_size_test = this->GetBatchSize();
    int batch_count_test = datasetPtr->Length() / batch_size_test;
    int size_one_feature = 224 * 224 * 3;

    std::vector<flt_type> samples(size_one_feature * batch_size_test);
    std::vector<label_t> labels(1000 * batch_size_test, 0);

    std::vector<int> confusion_matrix(1000 * 1000, 0);

    for (auto step = 0; step < batch_count_test; step++) {
#pragma omp parallel for num_threads(12)
        for (int i = 0; i < batch_size_test; i++) {
            dataType item = datasetPtr->GetItem(step * batch_size_test + i);
            for (int k = 0; k < size_one_feature; k++) {
                samples[i * size_one_feature + k] = *(item.first + k);
            }
            labels[i * 1000 + item.second] = 1;
            free(item.first);
        }

        this->layers_[0]->input_.ToDevice(samples);
        this->Forward();

        num_success += this->ObtainPredictionAccuracy(labels, confusion_matrix);

        for (int i = 0; i < batch_size_test; i++) {
            labels[i * 1000 +
                   datasetPtr->GetLabel(step * batch_size_test + i)] = 0;
        }
    }

    // step 4. calculate loss and accuracy

    std::cout << "精度: " << num_success << "/" << datasetPtr->Length() << "\n";
    // std::cout << "\n   *  ";
    // for (int i = 0; i < LeNet_5::kNumClasses; i++)
    //     std::cout << std::setw(4) << i << "  ";
    // std::cout << "\n";
    // for (int i = 0; i < LeNet_5::kNumClasses; i++) {
    //     std::cout << std::setw(4) << i << "  ";
    //     for (int j = 0; j < LeNet_5::kNumClasses; j++) {
    //         std::cout << std::setw(4)
    //                   << confusion_matrix[i * LeNet_5::kNumClasses + j] << "
    //                   ";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "\n";
}
// void Network::Predict(std::vector<one_image> const &test_sample,
//                       std::vector<label_t> const &test_label) {
//     int num_success = 0;
//     int batch_size_test = this->GetBatchSize();
//     int batch_count_test = test_sample.size() / batch_size_test;
//     int size_one_feature =
//         LeNet_5::kLengthOfMapAtLayer0 * LeNet_5::kLengthOfMapAtLayer0;

//     std::vector<flt_type> samples(size_one_feature * batch_size_test);
//     std::vector<label_t> labels(LeNet_5::kNumClasses * batch_size_test, 0);

//     std::vector<int> confusion_matrix(
//         LeNet_5::kNumClasses * LeNet_5::kNumClasses, 0);

//     for (auto step = 0; step < batch_count_test; step++) {
//         //    for(auto step = 0; step < 2; step++){

//         for (int i = 0; i < batch_size_test; i++) {
//             // for(int i = 0; i < 2; i++)

//             std::copy(test_sample[step * batch_size_test + i].begin(),
//                       test_sample[step * batch_size_test + i].end(),
//                       (samples.begin() + i * size_one_feature));

//             labels[i * LeNet_5::kNumClasses +
//                    test_label[step * batch_size_test + i]] = 1;
//         }

//         this->layers_[0]->input_.ToDevice(samples);

//         this->Forward();

//         num_success += this->ObtainPredictionAccuracy(labels,
//         confusion_matrix);

//         for (int i = 0; i < batch_size_test; i++) {
//             labels[i * LeNet_5::kNumClasses +
//                    test_label[step * batch_size_test + i]] = 0;
//         }
//     }

//     // step 4. calculate loss and accuracy

//     std::cout << "精度: " << num_success << "/" << test_sample.size() <<
//     "\n"; std::cout << "\n   *  "; for (int i = 0; i < LeNet_5::kNumClasses;
//     i++)
//         std::cout << std::setw(4) << i << "  ";
//     std::cout << "\n";
//     for (int i = 0; i < LeNet_5::kNumClasses; i++) {
//         std::cout << std::setw(4) << i << "  ";
//         for (int j = 0; j < LeNet_5::kNumClasses; j++) {
//             std::cout << std::setw(4)
//                       << confusion_matrix[i * LeNet_5::kNumClasses + j] << "
//                       ";
//         }
//         std::cout << "\n";
//     }
//     std::cout << "\n";
// }

void Network::AllocateMemoryForFeatures() {
    if (d_features_ != nullptr) {
        checkCudaErrors(cudaFree(d_features_));
        d_features_ = nullptr;
    }
    if (d_grad_features_ != nullptr) {
        checkCudaErrors(cudaFree(d_grad_features_));
        d_grad_features_ = nullptr;
    }

    std::array<int, 4> shape, out_shape;

    shape[0] = batch_size_;
    shape[1] = 3;
    shape[2] = 224;
    shape[3] = 224;

    /// 1. set the shape of the feature at each layer
    /// 2. get the total length of features among all layers
    int length_max = shape[0] * shape[1] * shape[2] * shape[3];
    for (auto layer : layers_) {
        shape = layer->InitFeatureShape(shape);
        length_max += shape[0] * shape[1] * shape[2] * shape[3];
    }

    checkCudaErrors(
        cudaMalloc((void **)&d_features_, sizeof(flt_type) * length_max));
    checkCudaErrors(
        cudaMalloc((void **)&d_grad_features_, sizeof(flt_type) * length_max));

    flt_type *d_ptr_feature = d_features_;
    flt_type *d_ptr_grad_feature = d_grad_features_;

    for (auto layer : layers_) {
        shape = layer->in_shape_;
        out_shape = layer->out_shape_;
        int in_length = shape[0] * shape[1] * shape[2] * shape[3];

        layer->input_.Initiate(shape, d_ptr_feature);
        layer->grad_input_.Initiate(shape, d_ptr_grad_feature);

        layer->output_.Initiate(out_shape, d_ptr_feature + in_length);
        d_ptr_feature += in_length;

        layer->grad_output_.Initiate(out_shape, d_ptr_grad_feature + in_length);
        d_ptr_grad_feature += in_length;
    }
    return;
}

void Network::DescriptorsAndWorkspace() {
    this->SetCudaContext();
    for (auto layer : layers_) {
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

        std::vector<std::array<int, 4>> shape_of_weights, shape_of_biases;
        shape_of_weights.reserve(layers_.size());
        shape_of_biases.reserve(layers_.size());

        for (auto layer : layers_) {
            layer->InitWeightsShape(shape_of_weights, shape_of_biases);
        }

        int length_weights = 0, length_biases = 0;
        std::vector<int> len_weight(layers_.size(), 0),
            len_bias(layers_.size(), 0);

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

        // weights and biases
        checkCudaErrors(cudaMalloc((void **)&d_weights_,
                                   sizeof(flt_type) * length_weights));
        checkCudaErrors(
            cudaMalloc((void **)&d_biases_, sizeof(flt_type) * length_biases));

        flt_type *d_ptr_grad_weights = d_weights_;
        flt_type *d_ptr_grad_biases = d_biases_;

        int i = 0;
        for (auto layer : layers_) {
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

        for (auto layer : layers_) {
            layer->InitiateWeightsAndBiases();
        }

        // gradients of weights and biases
        checkCudaErrors(cudaMalloc((void **)&d_grad_weights_,
                                   sizeof(flt_type) * length_weights));
        checkCudaErrors(cudaMalloc((void **)&d_grad_biases_,
                                   sizeof(flt_type) * length_biases));

        d_ptr_grad_weights = d_grad_weights_;
        d_ptr_grad_biases = d_grad_biases_;

        i = 0;
        for (auto layer : layers_) {
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
        // dim3 grid(1, 1, 1), block(256, 1, 1);
        // size_t N = sizeof(flt_type) * length_weights_;
        // grid.x = N / block.x + 1;
        // setZero<<<grid, block>>>(d_grad_weights_, N);

        // N = sizeof(flt_type) * length_biases_;
        // grid.x = N / block.x + 1;
        // setZero<<<grid, block>>>(d_grad_biases_, N);

        // float sum = 0;
        // float *tmp = (float *)malloc(sizeof(flt_type) * length_weights_);
        // checkCudaErrors(cudaMemcpy(tmp, d_grad_weights_,
        //                            sizeof(flt_type) * length_weights_,
        //                            cudaMemcpyDeviceToHost));
        // checkCudaErrors(cudaDeviceSynchronize());
        // for (int i = 0; i < length_weights_; i++) {
        //     sum += tmp[i];
        // }
        // std::cout << "Before " << sum << "\n";

        checkCudaErrors(
            cudaMemset(d_grad_weights_, 0, sizeof(flt_type) * length_weights_));
        checkCudaErrors(
            cudaMemset(d_grad_biases_, 0, sizeof(flt_type) * length_biases_));

        // checkCudaErrors(cudaMemcpy(tmp, d_grad_weights_,
        //                            sizeof(flt_type) * length_weights_,
        //                            cudaMemcpyDeviceToHost));
        // checkCudaErrors(cudaDeviceSynchronize());
        // sum = 0;
        // for (int i = 0; i < length_weights_; i++) {
        //     sum += tmp[i];
        // }
        // std::cout << "After " << sum << "\n";
    }
}
