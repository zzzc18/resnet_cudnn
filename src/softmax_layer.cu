/*
 * \file softmax_layer.cu
 */

#include "softmax_layer.h"
#include "utilities_sc.h"

void CrossEntropyLoss(BlobPointer<flt_type> &output_,
                      BlobPointer<flt_type> const &labels) {
    // Debug loss
    // std::cout << output_.buf_size() << "\n";
    // std::cout << labels.LengthNchw() << "\n";
    float *output = (float *)malloc(output_.buf_size());
    float *labelsCPU = (float *)malloc(labels.LengthNchw() * sizeof(float));

    int length = output_.buf_size() / 4;
    checkCudaErrors(cudaMemcpy(labelsCPU, labels.CudaPtr(),
                               labels.LengthNchw() * sizeof(float),
                               cudaMemcpyDeviceToHost));
    output_.ToHost(output, length);
    checkCudaErrors(cudaDeviceSynchronize());
    float *losses = (float *)malloc(labels.get_n() * sizeof(float));
    std::fill_n(losses, labels.get_n(), 0);
    float loss_sum = 0;

    // #pragma omp parallel for num_threads(12)
    for (int n = 0; n < labels.get_n(); n++) {
        for (int i = 0; i < 1000; i++) {
            losses[n] -= log(output[i + n * 1000]) * labelsCPU[i + n * 1000];
            if (labelsCPU[i + n * 1000] > 0) {
            }
        }
        // if (n < 3) {
        //     std::cout << "loss " << n << " " << losses[n] << "\n";
        // }
    }
    for (int n = 0; n < labels.get_n(); n++) {
        loss_sum += losses[n];
    }
    float loss_mean = loss_sum / labels.get_n();
    // std::cout << "loss_mean: " << loss_mean << "\n";
    Log("loss.log", "DEBUG", std::to_string(loss_mean));
    free(output);
    free(labelsCPU);
    free(losses);
}

std::array<int, 4> Softmax::InitFeatureShape(
    std::array<int, 4> const &in_shape) {
    in_shape_ = in_shape;
    out_shape_ = in_shape;

    return out_shape_;
}

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

void Softmax::Backward(BlobPointer<flt_type> const &labels) {
#ifdef ZDEBUG
    CrossEntropyLoss(output_, labels);
#endif
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpyAsync(grad_input_.CudaPtr(), output_.CudaPtr(),
                                    output_.buf_size(),
                                    cudaMemcpyDeviceToDevice));

    checkCublasErrors(cublasSaxpy(cuda_->cublas(), labels.LengthNchw(),
                                  &cuda_->minus_one, labels.CudaPtr(), 1,
                                  grad_input_.CudaPtr(), 1));

    // normalize the grad-output by the batch size
    int grad_output_size = labels.LengthNchw();

    flt_type scale = 1.f / input_.get_n();

    checkCublasErrors(cublasSscal(cuda_->cublas(), grad_output_size, &scale,
                                  grad_input_.CudaPtr(), 1));

    return;
}

int Softmax::ObtainPredictionAccuracy(std::vector<label_t> const &labels,
                                      std::vector<int> &confusion_matrix) {
    int batch_size = output_.get_n();
    int output_size = output_.LengthChw();

    std::vector<flt_type> h_output(batch_size * output_size, 0);

    // get prediction results
    output_.ToHost(h_output.data(), h_output.size());

    int hit_count{0}, idx_output, idx_target;
    for (int b = 0; b < batch_size; b++) {
        idx_output = 0;
        idx_target = 0;

        for (int i = 1; i < LeNet_5::kNumClasses; i++) {
            if (h_output[b * output_size + i] >
                h_output[b * output_size + idx_output])
                idx_output = i;
            if (labels[b * output_size + i] >
                labels[b * output_size + idx_target])
                idx_target = i;
        }

        if (idx_output == idx_target) hit_count++;
        confusion_matrix[idx_output * LeNet_5::kNumClasses + idx_target]++;
    }

    return hit_count;
}
