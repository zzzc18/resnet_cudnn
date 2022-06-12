/**
 * \file utilities_sc.h
 */
#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

class Timer {
   private:
    std::chrono::steady_clock::time_point last;

   public:
    Timer() : last{std::chrono::steady_clock::now()} {}
    double PrintDiff(const std::string& msg = "Timer diff: ") {
        auto now{std::chrono::steady_clock::now()};
        std::chrono::duration<double, std::milli> diff{now - last};
        std::cout << msg << std::fixed << std::setprecision(2) << diff.count()
                  << " ms\n";
        last = std::chrono::steady_clock::now();

        return diff.count();
    }
};

inline std::vector<std::string> Arguments(int argc, char* argv[]) {
    std::vector<std::string> res;
    for (int i = 0; i != argc; ++i) res.push_back(argv[i]);
    return res;
}

// 输出进度条
inline void ProgressBar(int n) {
    if (n == 0)
        return;
    else if (n == 1) {
        std::cout << "\n ";
    } else if (n < 11) {
        std::cout << "\b\b\b";
    } else {
        std::cout << "\b\b\b\b";
    }
    std::cout << "> " << n << "%";

    std::cout.flush();
}

inline void Log(std::string filePath, std::string priority, std::string text) {
    std::ofstream logfile;
    logfile.open(filePath, std::ios::app);
    if (logfile.is_open()) {
        logfile << "[" << priority << "] " << text << std::endl;
        logfile.close();
    }
}
