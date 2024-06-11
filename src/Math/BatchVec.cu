#include "Math/Vec.hpp"
#include "Math/BatchVec.hpp"
#include "utils.hpp"
#include <cuda_runtime_api.h>

__global__ void VecAdd(const float *const a, const float *const b, float *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    result[index] = a[index] + b[index];
}

template <int dim>
std::vector<Vec<dim>> bacthVecAdd(std::vector<Vec<dim>> &a, std::vector<Vec<dim>> &b) {
    const int numVec        = a.size();
    const int allDataLength = numVec * sizeof(Vec<dim>);

    float *dev_a, *dev_b, *dev_result;
    CHECK_CUDA(cudaMalloc(&dev_a, allDataLength));
    CHECK_CUDA(cudaMalloc(&dev_b, allDataLength));
    CHECK_CUDA(cudaMalloc(&dev_result, allDataLength));

    CHECK_CUDA(cudaMemcpy(dev_a, a.data(), allDataLength, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, b.data(), allDataLength, cudaMemcpyHostToDevice));

    const int threadsPerBlock = dim;
    const int blocksPerGrid   = (numVec * dim + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_result);

    std::vector<Vec<dim>> result(numVec);
    CHECK_CUDA(cudaMemcpy(result.data(), dev_result, allDataLength, cudaMemcpyDeviceToHost));

    return std::move(result);
}

template std::vector<Vec2> bacthVecAdd<2>(std::vector<Vec2> &a, std::vector<Vec2> &b);
template std::vector<Vec3> bacthVecAdd<3>(std::vector<Vec3> &a, std::vector<Vec3> &b);


__global__ void VecMul(float *a, float *b, float *result, int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    result[index] = a[index] * b[index];
    __syncthreads();
    if (threadIdx.x % length == 0) [[unlikely]] {}
}

template <int dim>
std::vector<Vec<dim>> bacthVecMul(std::vector<Vec<dim>> &a, std::vector<Vec<dim>> &b) {
    const int numVec        = a.size();
    const int allDataLength = numVec * sizeof(Vec<dim>);

    const int threadsPerBlock = dim;
    const int blocksPerGrid   = (numVec * dim + threadsPerBlock - 1) / threadsPerBlock;

    float *dev_a, *dev_b, *dev_result;
    CHECK_CUDA(cudaMalloc(&dev_a, allDataLength));
    CHECK_CUDA(cudaMalloc(&dev_b, allDataLength));
    CHECK_CUDA(cudaMalloc(&dev_result, allDataLength));

    CHECK_CUDA(cudaMemcpy(dev_a, a.data(), allDataLength, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, b.data(), allDataLength, cudaMemcpyHostToDevice));

    VecMul<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_result, dim);

    std::vector<Vec<dim>> result(numVec);
    CHECK_CUDA(cudaMemcpy(result.data(), dev_result, allDataLength, cudaMemcpyDeviceToHost));

    return std::move(result);
}

template std::vector<Vec2> bacthVecMul<2>(std::vector<Vec2> &a, std::vector<Vec2> &b);
template std::vector<Vec3> bacthVecMul<3>(std::vector<Vec3> &a, std::vector<Vec3> &b);