/*
 * \file blob.h
 */

#pragma once

#include <fstream>
#include <string>

#include "cnpy.h"
#include "cuda_helper.h"
#include "utilities_sc.h"

template <typename T>
class BlobPointer {
   public:
    BlobPointer(int const n = 1, int const c = 1, int const h = 1,
                int const w = 1, T *d_ptr = nullptr)
        : n_(n), channels_(c), height_(h), width_(w), d_ptr_(d_ptr) {
        checkCudnnErrors(cudnnCreateTensorDescriptor(&tensor_desc_));
    }

    BlobPointer(std::array<int, 4> const &size, T *d_ptr = nullptr)
        : n_(size[0]),
          channels_(size[1]),
          height_(size[2]),
          width_(size[3]),
          d_ptr_(d_ptr) {
        checkCudnnErrors(cudnnCreateTensorDescriptor(&tensor_desc_));
    }

    BlobPointer(BlobPointer const &) = delete;

    ~BlobPointer() {
        d_ptr_ = nullptr;
        if (tensor_desc_ != nullptr)
            checkCudnnErrors(cudnnDestroyTensorDescriptor(tensor_desc_));
    }

    BlobPointer &operator=(BlobPointer const &) = delete;

    int LengthChw() const { return channels_ * height_ * width_; }
    int LengthNchw() const { return n_ * channels_ * height_ * width_; }

    int buf_size() { return sizeof(T) * LengthNchw(); }

    int get_n() const { return n_; }
    int GetChannels() const { return channels_; }
    int GetHeight() const { return height_; }
    int GetWidth() const { return width_; }

    T *CudaPtr() const { return d_ptr_; }

    void Initiate(std::array<int, 4> const &size, T *d_ptr) {
        n_ = size[0];
        channels_ = size[1];
        height_ = size[2];
        width_ = size[3];
        d_ptr_ = d_ptr;
    }

    cudnnTensorDescriptor_t tensor() const { return tensor_desc_; }
    cudnnTensorDescriptor_t CreateTensor() {
        checkCudnnErrors(cudnnSetTensor4dDescriptor(
            tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, channels_,
            height_, width_));

        return tensor_desc_;
    }

    cudaError_t ToDevice(T const *h_ptr, size_t const len) {
        return (cudaMemcpy(this->CudaPtr(), h_ptr, sizeof(T) * len,
                           cudaMemcpyHostToDevice));
    }

    cudaError_t ToDevice(std::vector<T> const &h_ptr) {
        return (cudaMemcpy(this->CudaPtr(), h_ptr.data(),
                           sizeof(T) * h_ptr.size(), cudaMemcpyHostToDevice));
    }

    cudaError_t ToHost(std::vector<T> &h_ptr) {
        return (cudaMemcpy(h_ptr.data(), this->CudaPtr(),
                           sizeof(T) * h_ptr.size(), cudaMemcpyDeviceToHost));
    }

    cudaError_t ToHost(T *h_ptr, size_t const len) {
        return (cudaMemcpy(h_ptr, this->CudaPtr(), sizeof(T) * len,
                           cudaMemcpyDeviceToHost));
    }

    void SaveAsNumpyArray(std::string fileName) {
        std::vector<T> dataCPU(this->LengthNchw());
        this->ToHost(dataCPU);
        cnpy::npy_save<T>(
            fileName, &dataCPU[0],
            {(size_t)n_, (size_t)channels_, (size_t)height_, (size_t)width_},
            "w");
    }

    void print(std::string name, bool view_param = false, int width = 16) {
        std::vector<T> h_ptr(this->LengthNchw());
        ToHost(h_ptr);
        std::cout << "**" << name << "\t: (" << LengthChw() << ")\t";
        std::cout << ".n: " << n_ << ", .c: " << channels_
                  << ", .h: " << height_ << ", .w: " << width_;
        std::cout << std::hex << "\t(h:" << h_ptr.data() << ", d:" << d_ptr_
                  << ")" << std::dec << "\n";

        if (view_param) {
            std::cout << std::fixed;
            std::cout.precision(6);
            int max_print_line;
            if (width == 32) {
                std::cout.precision(4);
                max_print_line = 32;
            }
            if (width == 28) {
                std::cout.precision(4);
                max_print_line = 28;
            }
            if (width == 0) {
                max_print_line = 16;
                width = 16;
            } else {
                max_print_line = width;
            }

            int offset = 0;
            int num_batch = 1;

            for (int n = 0; n < num_batch; n++) {
                if (num_batch > 1)
                    std::cout << "<--- batch[" << n << "] ---> \n";
                int count = 0;
                int print_line_count = 0;
                while (count < LengthChw() &&
                       print_line_count < max_print_line) {
                    std::cout << "\t";
                    for (int s = 0; s < width && count < LengthChw(); s++) {
                        std::cout << h_ptr[LengthChw() * n + count + offset]
                                  << "\t";
                        count++;
                    }
                    std::cout << "\n";
                    print_line_count++;
                }
            }
            std::cout.unsetf(std::ios::fixed);
        }
    }

   private:
    int n_{1};
    int channels_{1};
    int height_{1};
    int width_{1};
    T *d_ptr_{nullptr};
    cudnnTensorDescriptor_t tensor_desc_{nullptr};
};
