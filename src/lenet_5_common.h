/**
 * \file lenet_5_common.h
 * \brief 定义LeNet-5网络的一些常数
 */
#pragma once

#include "utilities_sc.h"

// using uchar = unsigned char;
// using flt_type = float;
// using label_t = uint32_t;
// using one_image = std::vector<flt_type>;

namespace LeNet_5 {
int const kNumClasses = 10;  // 不同的数字个数
int const kPadding = 2;      // 卷积填白
int const kLengthOfKernel =
    5;  // 卷积核一个维度的大小，本示例网络使用方形卷积核

// 各层输出的特征图个数
int const kNumOfInputMap = 1;
int const kNumOfMapAtLayer1 = 6;
int const kNumOfMapAtLayer2 = 6;
int const kNumOfMapAtLayer3 = 16;
int const kNumOfMapAtLayer4 = 16;
int const kNumOfMapAtLayer5 = 120;
int const kNumOfOutputMap = 10;

// 各层特征图的尺寸；用方形矩阵来描述，只需给出一个维度即可
int const kLengthOfMapAtLayer0 = (28 + 2 * kPadding);
int const kLengthOfMapAtLayer1 = (kLengthOfMapAtLayer0 - kLengthOfKernel + 1);
int const kLengthOfMapAtLayer2 = (kLengthOfMapAtLayer1 >> 1);
int const kLengthOfMapAtLayer3 = (kLengthOfMapAtLayer2 - kLengthOfKernel + 1);
int const kLengthOfMapAtLayer4 = (kLengthOfMapAtLayer3 >> 1);
int const kLengthOfMapAtLayer5 = (kLengthOfMapAtLayer4 - kLengthOfKernel + 1);
};  // namespace LeNet_5
