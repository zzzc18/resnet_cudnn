/**
 * @file DataLoader.h
 * @author 张维璞 (zhangweipu1118@outlook.com)
 * @brief 负责具体进行每次迭代的数据获取（通过dataset的路径和读取方法）
 * @version 0.1
 * @date 2022-06-11
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include "Dataset/Dataset.h"

template <class DataType, class TrainDataType>
class DataLoader {
   private:
    Dataset &dataset;
    int batchSize;

   public:
    DataLoader(Dataset<DataType> &_dataset, int _batchSize)
        : dataset(_dataset), batchSize(_batchSize) {}
    TrainDataType FetchData() {}
    ~DataLoader() {}
};
