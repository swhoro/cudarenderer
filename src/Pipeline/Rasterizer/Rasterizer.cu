#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include "Math/Mat.hpp"
#include "Math/Vec.hpp"
#include "Pipeline/Rasterizer/Rasterizer.hpp"
#include "Pipeline/Rasterizer/kernels.hpp"
#include "utils.hpp"


Rasterizer::Rasterizer(const int w, const int h) : Pipeline(w, h), nrPixels(width * height) {
    allocatePixelData();
}

int Rasterizer::loadModel(std::string path) {
    models.push_back(Model(path));
    return models.size() - 1;
}

int Rasterizer::loadModel(Model &model) {
    models.push_back(std::move(model));
    return models.size() - 1;
}

const Vec3 *Rasterizer::render(const Camera &camera, const int modelIdx) {
    if (modelIdx < 0 || modelIdx >= models.size()) { return nullptr; }

    setViewMatrix(camera.getViewMatrix());
    setProjectionMatrix(camera.getProjectionMatrix());

    int threadsPerBlock;
    int blocksPerGrid;
    for (auto &mesh : models[modelIdx].getMeshes()) {
        Vec4 *dev_ndc_positions;
        CHECK_CUDA(cudaMalloc(&dev_ndc_positions, mesh.nrVertices * sizeof(Vec4)));
        // vertex shader
        threadsPerBlock = 32;
        blocksPerGrid   = (mesh.nrIndices + threadsPerBlock - 1) / threadsPerBlock;
        vertexShader<<<blocksPerGrid, threadsPerBlock>>>(mesh.dev_verticies,
                                                         mesh.dev_indices,
                                                         mesh.nrIndices,
                                                         dev_ndc_positions);

        CHECK_CUDA(cudaDeviceSynchronize());

        // rasterize
        threadsPerBlock = 1;
        blocksPerGrid   = 1;
        rasterize<<<blocksPerGrid, threadsPerBlock>>>(dev_ndc_positions,
                                                      mesh.dev_verticies,
                                                      mesh.dev_indices,
                                                      mesh.nrIndices,
                                                      width,
                                                      height,
                                                      MSAA,
                                                      dev_fragmentDatas);
        cudaFree(dev_ndc_positions);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // fragment shader
    threadsPerBlock = 640;
    blocksPerGrid   = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    fragmentShader<<<blocksPerGrid, threadsPerBlock>>>(dev_fragmentDatas,
                                                       width,
                                                       height,
                                                       MSAA,
                                                       dev_pixelColor);

    CHECK_CUDA(
        cudaMemcpy(pixelColor, dev_pixelColor, nrPixels * sizeof(Vec3), cudaMemcpyDeviceToHost));
    return pixelColor;
}

const Vec3 *Rasterizer::getDevPixelColor() const {
    CHECK_CUDA(
        cudaMemcpy(pixelColor, dev_pixelColor, nrPixels * sizeof(Vec3), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    return pixelColor;
}

void Rasterizer::resize(const int width, const int height) {
    this->width    = width;
    this->height   = height;
    this->nrPixels = width * height;

    freePixelData();
    allocatePixelData();
}

void Rasterizer::setMSAA(const unsigned int msaa) {
    MSAA = msaa;
    freePixelData();
    allocatePixelData();
}

void Rasterizer::setBackground(const Vec3 &color) {
    const unsigned int threadsPerBlock = 64;
    const unsigned int blocksPerGrid   = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    setFragmentData<<<blocksPerGrid, threadsPerBlock>>>(dev_pixelColor,
                                                        nrPixels,
                                                        dev_fragmentDatas,
                                                        MSAA,
                                                        color);
    CHECK_CUDA(cudaDeviceSynchronize());
}

void Rasterizer::setViewMatrix(const Mat4 &viewMatrix) {
    CHECK_CUDA(cudaMemcpyToSymbol(dev_viewMatrix, &viewMatrix.data, sizeof(PureMat4)));
}

void Rasterizer::setProjectionMatrix(const Mat4 &projectionMatrix) {
    CHECK_CUDA(cudaMemcpyToSymbol(dev_projectionMatrix, &projectionMatrix.data, sizeof(PureMat4)));
}

void Rasterizer::allocatePixelData() {
    CHECK_CUDA(cudaMalloc(&dev_fragmentDatas, nrPixels * MSAA * MSAA * sizeof(FragmentData)));
    pixelColor = new Vec3[nrPixels];
    CHECK_CUDA(cudaMalloc(&dev_pixelColor, nrPixels * sizeof(Vec3)));
}

void Rasterizer::freePixelData() {
    CHECK_CUDA(cudaFree(dev_fragmentDatas));
    delete[] pixelColor;
    CHECK_CUDA(cudaFree(dev_pixelColor));
}