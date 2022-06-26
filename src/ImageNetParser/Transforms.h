#pragma once

#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

class RandomGenerator final {
   public:
    RandomGenerator(const RandomGenerator &) = delete;
    RandomGenerator &operator=(const RandomGenerator &) = delete;
    static RandomGenerator &Singleton() {
        static RandomGenerator singleton;
        return singleton;
    }

    std::mt19937 gen_;

   private:
    RandomGenerator() {
        unsigned seed =
            std::chrono::system_clock::now().time_since_epoch().count();
        gen_.seed(seed);
    };
    ~RandomGenerator() = default;
};

class Transform {
   public:
    Transform() { probDis_ = std::uniform_real_distribution<>(0.0, 1.0); }
    virtual cv::Mat Forward(cv::Mat img) = 0;

    std::uniform_real_distribution<> probDis_;
    float GetRandomValue() {
        return static_cast<float>(probDis_(RandomGenerator::Singleton().gen_));
    }
};

class RandomHorizontalFlip : public Transform {
   public:
    RandomHorizontalFlip(float p) : p_(p) { assert(p_ >= 0 && p <= 1); }

    virtual cv::Mat Forward(cv::Mat img) override {
        if (this->GetRandomValue() > p_) {
            return img;
        }
        cv::flip(img, img, 1);
        return img;
    }

    float p_;
};

class RandomResizedCrop : public Transform {
   public:
    RandomResizedCrop(std::pair<float, float> size = {224, 224},
                      std::pair<float, float> scale = {0.08, 1.0},
                      std::pair<float, float> ratio = {3.0 / 4, 4.0 / 3})
        : size_(size), scale_(scale), ratio_(ratio) {
        scaleDis_ =
            std::uniform_real_distribution<>(scale_.first, scale_.second);
        ratioDis_ =
            std::uniform_real_distribution<>(ratio_.first, ratio_.second);
    }

    std::array<int, 4> GetCropBox(const cv::Mat &img) {  // x,y,h,w
        for (int i = 0; i < 10; i++) {
            float area = img.size().width * img.size().height;
            float targetArea = static_cast<float>(scaleDis_(
                                   RandomGenerator::Singleton().gen_)) *
                               area;
            float aspectRatio = static_cast<float>(
                ratioDis_(RandomGenerator::Singleton().gen_));
            int w = round(sqrt(targetArea * aspectRatio));
            int h = round(sqrt(targetArea / aspectRatio));
            if (this->GetRandomValue() > 0.5) {
                std::swap(w, h);
            }
            if (w <= img.size().width && h <= img.size().height) {
                int x = (img.size().height - h) * this->GetRandomValue();
                int y = (img.size().width - w) * this->GetRandomValue();
                return std::array<int, 4>{x, y, h, w};
            }
        }
        // Fallback
        return std::array<int, 4>{0, 0, img.size().height, img.size().width};
    }

    virtual cv::Mat Forward(cv::Mat img) override {
        std::array<int, 4> cropBox = this->GetCropBox(img);  // x,y,h,w
        assert(cropBox[0] >= 0 && cropBox[1] >= 0 &&
               cropBox[2] <= img.size().height &&
               cropBox[3] <= img.size().width);
        img = img(cv::Range(cropBox[0], cropBox[0] + cropBox[2]),
                  cv::Range(cropBox[1], cropBox[1] + cropBox[3]));
        cv::resize(img, img, cv::Size(size_.first, size_.second), 0.0, 0.0,
                   cv::INTER_LINEAR);
        return img;
    }

    std::pair<float, float> size_, scale_, ratio_;
    std::uniform_real_distribution<> scaleDis_, ratioDis_;
};

class Resize : public Transform {
   public:
    Resize(std::pair<float, float> size = {256, 256}) : size_(size) {}

    virtual cv::Mat Forward(cv::Mat img) override {
        cv::resize(img, img, cv::Size(size_.first, size_.second), 0.0, 0.0,
                   cv::INTER_LINEAR);
        return img;
    }

    std::pair<float, float> size_;
};

class CenterCrop : public Transform {
   public:
    CenterCrop(std::pair<float, float> size = {256, 256}) : size_(size) {}

    virtual cv::Mat Forward(cv::Mat img) override {
        int x = round((img.size().height - size_.first) * 0.5);
        int y = round((img.size().width - size_.second) * 0.5);
        img =
            img(cv::Range(x, x + size_.first), cv::Range(y, y + size_.second));
        return img;
    }

    std::pair<float, float> size_;
};

class TrainTransform : public Transform {
   public:
    TrainTransform() {
        transforms.emplace_back(new RandomResizedCrop);
        transforms.emplace_back(new RandomHorizontalFlip(0.5));
    }

    virtual cv::Mat Forward(cv::Mat img) override {
        for (auto transform : transforms) {
            img = transform->Forward(img);
        }
        return img;
    }

    std::vector<Transform *> transforms;
};

class TestTransform : public Transform {
   public:
    TestTransform() {
        transforms.emplace_back(new Resize({224, 224}));
        // transforms.emplace_back(new CenterCrop({224, 224}));
    }

    virtual cv::Mat Forward(cv::Mat img) override {
        for (auto transform : transforms) {
            img = transform->Forward(img);
        }
        return img;
    }

    std::vector<Transform *> transforms;
};
