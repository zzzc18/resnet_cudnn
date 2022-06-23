#pragma once

#include <cassert>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <utility>
#include <vector>

#include "Dataset/Dataset.h"
#include "utilities_sc.h"

using dataType = std::pair<float *, int>;

inline float *HWC2CHW(cv::Mat img) {
    float mean_rgb[3] = {0.485, 0.456, 0.406};
    float std_rgb[3] = {0.229, 0.224, 0.225};
    uint8_t *imgPtr = img.ptr<uint8_t>(0);
    int area = img.size().width * img.size().height;
    float *chwMat = (float *)malloc(area * 3 * sizeof(float));
    memset(chwMat, 0, area * 3 * sizeof(float));

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < img.size().height; ++h) {
            for (int w = 0; w < img.size().width; ++w) {
                int divider = h * img.size().width + w;
                int srcIdx = divider * 3 + c;
                chwMat[divider + c * area] = static_cast<float>(
                    (imgPtr[srcIdx] * 1.0f / 255.0f - mean_rgb[c]) * 1.0f /
                    std_rgb[c]);
            }
        }
    }
    return chwMat;
}

template <class DataType>
class ImageNetDataset : public Dataset<DataType> {
   private:
    std::vector<std::pair<std::string, int>> fileNameLabelPair;
    std::map<std::string, int> labelName2Num;
    std::string root;
    std::string type;

   public:
    ImageNetDataset(std::string _root, std::string _type)
        : Dataset<DataType>(), root(_root), type(_type) {
        assert(type == "train" or type == "val");

        FILE *fp = fopen((root + "/index2folder.txt").c_str(), "r");
        int totalClasses;
        fscanf(fp, "%d", &totalClasses);
        while (totalClasses--) {
            char labelName[10];
            int idx;
            char newlineFlag;
            fscanf(fp, "%d%s%*[^\n]", &idx, labelName);
            labelName2Num[labelName] = idx;
        }

        std::string typeRoot = root + "/" + type;
        std::vector<std::string> folderNames = this->Glob(typeRoot);
        for (std::string folderName : folderNames) {
            std::string folderPath = typeRoot + "/" + folderName;
            std::vector<std::string> imagePaths = this->Glob(folderPath);
            for (auto iter = imagePaths.begin(); iter != imagePaths.end();
                 iter++) {
                // if (labelName2Num[folderName] % 100 == 0)
                //     fileNameLabelPair.push_back(
                //         std::make_pair(folderPath + "/" + *iter,
                //                        labelName2Num[folderName] / 100));
                fileNameLabelPair.push_back(std::make_pair(
                    folderPath + "/" + *iter, labelName2Num[folderName]));
            }
        }
        std::sort(fileNameLabelPair.begin(), fileNameLabelPair.end());
    }
    ~ImageNetDataset() {}

    int Length() const { return fileNameLabelPair.size(); }

    DataType GetItem(int index) const {
        cv::Mat img;
        img = cv::imread(fileNameLabelPair[index].first);
        if (img.data == nullptr) {
            throw "图片文件不存在: " + fileNameLabelPair[index].first;
        }
        cv::resize(img, img, cv::Size(224, 224), 0.0, 0.0, cv::INTER_LINEAR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        float *imgCHW{nullptr};
        imgCHW = ::HWC2CHW(img);
        return std::make_pair(imgCHW, fileNameLabelPair[index].second);
    }

    int GetLabel(int index) const { return fileNameLabelPair[index].second; }
};

// template <class DataType>
// class ImageNetDatasetTiny : public Dataset<DataType> {
//    private:
//     std::vector<std::pair<std::string, int>> fileNameLabelPair;
//     std::map<std::string, int> labelName2Num;
//     std::string root;
//     std::string type;

//    public:
//     ImageNetDatasetTiny(std::string _root, std::string _type)
//         : Dataset<DataType>(), root(_root), type(_type) {
//         assert(type == "train" or type == "val");

//         FILE *fp = fopen((root + "/mapping.txt").c_str(), "r");
//         int totalClasses;
//         fscanf(fp, "%d", &totalClasses);
//         while (totalClasses--) {
//             char labelName[10];
//             int idx;
//             fscanf(fp, "%s%d", labelName, &idx);
//             labelName2Num[labelName] = idx;
//         }

//         std::string typeRoot = root + "/" + type;
//         std::vector<std::string> folderNames = this->Glob(typeRoot);
//         for (std::string folderName : folderNames) {
//             std::string folderPath = typeRoot + "/" + folderName;
//             std::vector<std::string> imagePaths = this->Glob(folderPath);
//             for (auto iter = imagePaths.begin(); iter != imagePaths.end();
//                  iter++) {
//                 fileNameLabelPair.push_back(std::make_pair(
//                     folderPath + "/" + *iter, labelName2Num[folderName]));
//             }
//         }
//     }
//     ~ImageNetDatasetTiny() {}

//     int Length() const { return fileNameLabelPair.size(); }

//     DataType GetItem(int index) const {
//         cv::Mat img
//         img = cv::imread(fileNameLabelPair[index].first);
//         if (img.data == nullptr) {
//             throw "图片文件不存在: " + fileNameLabelPair[index].first;
//         }
//         cv::resize(img, img, cv::Size(224, 224));
//         cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//         float *imgCHW{nullptr};
//         imgCHW = ::HWC2CHW(img);
//         return std::make_pair(imgCHW, fileNameLabelPair[index].second);
//     }

//     int GetLabel(int index) const { return fileNameLabelPair[index].second; }
// };