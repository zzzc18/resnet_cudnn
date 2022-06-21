/*
 * \file softmax_layer.cu
 */

#include "softmax_layer.h"
#include "utilities_sc.h"

void CrossEntropyLoss(int batch_size, float *output, float *labelsCPU) {
    float *losses = (float *)malloc(batch_size * sizeof(float));
    std::fill_n(losses, batch_size, 0);
    float loss_sum = 0;

#pragma omp parallel for num_threads(14)
    for (int n = 0; n < batch_size; n++) {
        for (int i = 0; i < IMAGENET_CLASSES; i++) {
            losses[n] -= log(output[i + n * IMAGENET_CLASSES]) *
                         labelsCPU[i + n * IMAGENET_CLASSES];
        }
    }
    for (int n = 0; n < batch_size; n++) {
        loss_sum += losses[n];
    }
    float loss_mean = loss_sum / batch_size;
    Log("loss.log", "DEBUG", std::to_string(loss_mean));
    free(output);
    free(labelsCPU);
    free(losses);
}

void CrossEntropyLoss(BlobPointer<float> &output_,
                      BlobPointer<float> const &labels) {
    float *output = (float *)malloc(output_.buf_size());
    float *labelsCPU = (float *)malloc(labels.LengthNchw() * sizeof(float));

    int length = output_.buf_size() / 4;
    checkCudaErrors(cudaMemcpy(labelsCPU, labels.CudaPtr(),
                               labels.LengthNchw() * sizeof(float),
                               cudaMemcpyDeviceToHost));
    output_.ToHost(output, length);
    CrossEntropyLoss(output_.get_n(), output, labelsCPU);
}

void CrossEntropyLoss(BlobPointer<float> &output_,
                      std::vector<label_t> labels) {
    float *output = (float *)malloc(output_.buf_size());
    float *labelsCPU = (float *)malloc(labels.size() * sizeof(float));

    int length = output_.buf_size() / 4;
    for (int i = 0; i < labels.size(); i++) {
        labelsCPU[i] = labels[i];
    }
    output_.ToHost(output, length);
    CrossEntropyLoss(output_.get_n(), output, labelsCPU);
}

void Softmax::InitFeatureShape() { out_shape_ = in_shape_; }

void Softmax::InitWeightsShape(std::vector<std::array<int, 4>> &w_l,
                               std::vector<std::array<int, 4>> &b_l) {
    w_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});
    b_l.emplace_back(std::array<int, 4>{0, 0, 0, 0});
    return;
}

void Softmax::Forward() {
    input_desc_ = input_.tensor();
    output_desc_ = output_.tensor();

#if (DEBUG_SOFTMAX & 0x01)
    std::cout << name_ << "[FORWARD]\n";
    input_.print(name_ + "::input", true);
#endif
    checkCudnnErrors(cudnnSoftmaxForward(
        cuda_->cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        &cuda_->one, input_desc_, input_.CudaPtr(), &cuda_->zero, output_desc_,
        output_.CudaPtr()));

#if (DEBUG_SOFTMAX & 0x01)
    output_.print(name_ + "::output", true);
#endif
    return;
}

void Softmax::Backward(BlobPointer<float> const &labels) {
#ifdef LOSS_LOG
    CrossEntropyLoss(output_, labels);
#endif
    checkCudaErrors(cudaMemcpy(d_temp_grad_features_, output_.CudaPtr(),
                               output_.buf_size(), cudaMemcpyDeviceToDevice));

    checkCublasErrors(cublasSaxpy(cuda_->cublas(), labels.LengthNchw(),
                                  &cuda_->minus_one, labels.CudaPtr(), 1,
                                  d_temp_grad_features_, 1));

    // normalize the grad-output by the batch size
    float scale = 1.f / input_.get_n();
    checkCublasErrors(cublasSscal(cuda_->cublas(), labels.LengthNchw(), &scale,
                                  d_temp_grad_features_, 1));
    this->BackwardCopy();

    return;
}

int Softmax::ObtainPredictionAccuracy(std::vector<label_t> const &labels,
                                      std::vector<int> &confusion_matrix) {
#ifdef LOSS_LOG
    CrossEntropyLoss(output_, labels);
#endif
    int batch_size = output_.get_n();
    int output_size = output_.LengthChw();

    std::vector<float> h_output(batch_size * output_size, 0);

    // get prediction results
    output_.ToHost(h_output.data(), h_output.size());

    int hit_count{0}, idx_output, idx_target;
    for (int b = 0; b < batch_size; b++) {
        idx_output = 0;
        idx_target = 0;

        for (int i = 1; i < IMAGENET_CLASSES; i++) {
            if (h_output[b * output_size + i] >
                h_output[b * output_size + idx_output])
                idx_output = i;
            if (labels[b * output_size + i] >
                labels[b * output_size + idx_target])
                idx_target = i;
        }

        if (idx_output == idx_target) hit_count++;
        confusion_matrix[idx_output * IMAGENET_CLASSES + idx_target]++;
    }

    return hit_count;
}
