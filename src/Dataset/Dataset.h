/**
 * @file Dataset.h
 * @author 张维璞 (zhangweipu1118@outlook.com)
 * @brief Dataset头文件，Dataset负责获取数据集的文件路径及对应label获取方式
 * @version 0.1
 * @date 2022-06-11
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include <dirent.h>

#include <string>
#include <vector>

template <class DataType>
class Dataset {
   protected:
    /**
     * @brief 读取某文件夹下所有内容
     *
     * @param root
     * @return std::vector<std::string>
     */
    std::vector<std::string> Glob(std::string root) {
        std::vector<std::string> file_names;
        DIR *dp;  //创建一个指向root路径下每个文件的指针
        struct dirent *dirp;
        if ((dp = opendir(root.c_str())) == NULL) {
            throw "Cannot open dir: " + root;
        }
        while ((dirp = readdir(dp)) != NULL) {
            std::string str(dirp->d_name);
            if (str == "." || str == "..") continue;
            file_names.push_back(str);
        }
        return file_names;
    }

   public:
    Dataset() {}
    ~Dataset() {}

    virtual int Length() const = 0;
    virtual DataType GetItem(int index) const = 0;
    virtual int GetLabel(int index) const = 0;
};
