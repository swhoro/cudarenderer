#pragma once

#include <iostream>

#define CHECK_CUDA(func) \
    do { \
        cudaError_t status = (func); \
        if (status != cudaSuccess) { \
            std::cout << "Cuda failure " << status << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)
