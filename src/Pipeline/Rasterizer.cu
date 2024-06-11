#include <cmath>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include "Math/Mat.hpp"
#include "Math/Vec.hpp"
#include "Pipeline/Rasterizer.hpp"
#include "utils.hpp"

__constant__ static PureMat4 dev_viewMatrix;
__constant__ static PureMat4 dev_projectionMatrix;


/* one thread per indice
 *
 * map all verticies into [-1, 1]^3
 */
__global__ void vertexShader(const Vertex *const       verticies,
                             const unsigned int *const indices,
                             const int                 nrIndices,
                             Vec4 *const               afterVertexShaderPositions) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrIndices) { return; }

    const unsigned int  vertexIdx = indices[idx];
    const Vertex *const vertex    = &verticies[vertexIdx];

    // view projection
    Vec4 afterVertexShaderPosition = dev_projectionMatrix * (dev_viewMatrix * vertex->postition);
    afterVertexShaderPosition /= afterVertexShaderPosition.w;
    afterVertexShaderPositions[vertexIdx] = afterVertexShaderPosition;
}

/*
 * a,b,c are in counter clockwise order
 *             b
 *
 *                        a
 *              p
 *
 *      c
 */
__device__ bool isInTriangle(const Vec2 &p, const Vec2 &a, const Vec2 &b, const Vec2 &c) {
    Vec2 t;

    t = b - a;
    const Vec3 ab{t.x, t.y, 0};
    t = c - b;
    const Vec3 bc{t.x, t.y, 0};
    t = a - c;
    const Vec3 ca{t.x, t.y, 0};

    t = p - a;
    const Vec3 ap{t.x, t.y, 0};
    t = p - b;
    const Vec3 bp{t.x, t.y, 0};
    t = p - c;
    const Vec3 cp{t.x, t.y, 0};

    const Vec3 aa = ab.cross(ap);
    const Vec3 bb = bc.cross(bp);
    const Vec3 cc = ca.cross(cp);
    if (aa.z < 0 && bb.z < 0 && cc.z < 0) {
        return true;
    } else if (aa.z > 0 && bb.z > 0 && cc.z > 0) {
        return true;
    }
    return false;
}

// return {minx, miny, maxx, maxy}
__device__ Vec4 getBoundingBox(const Vec4 &a, const Vec4 &b, const Vec4 &c) {
    Vec4 result;
    result.x = fminf(a.x, fminf(b.x, c.x));
    result.y = fminf(a.y, fminf(b.y, c.y));
    result.z = fmaxf(a.x, fmaxf(b.x, c.x));
    result.w = fmaxf(a.y, fmaxf(b.y, c.y));
    return result;
}

__device__ Vec3 calBarycentric(const Vec2 &p, const Vec4 &a, const Vec4 &b, const Vec4 &c) {
    Vec3 result;

    float ta = (p.y - a.y) * (c.x - a.x) - (p.x - a.x) * (c.y - a.y);
    float tb = (b.y - a.y) * (c.x - a.x) - (b.x - a.x) * (c.y - a.y);
    result.y = ta / tb;

    ta       = (p.y - a.y) * (b.x - a.x) - (p.x - a.x) * (b.y - a.y);
    tb       = (c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y);
    result.z = ta / tb;

    result.x = 1 - result.y - result.z;
    return result;
}

__device__ __host__ unsigned int upleftIdx2bottomLeftIdx(const unsigned int &idx,
                                                         const int          &width,
                                                         const int          &height,
                                                         const unsigned int &msaa) {
    const unsigned int fragmentsPerRow = width * msaa * msaa;
    // calculate which row it is from the top
    unsigned int       rowIdx          = idx / fragmentsPerRow;
    // calculate which column it is from the left
    unsigned int       colIdx          = idx % fragmentsPerRow;
    // calculate which row it is from the bottom
    rowIdx                             = height - 1 - rowIdx;
    return rowIdx * fragmentsPerRow + colIdx;
}

/* rasterize
 *
 * one thread per fragment
 *
 *        ^ y
 *        |
 *        |
 *        |
 *        |
 *        |
 *        |------------> x
 *       /
 *      /
 *     /
 *    /
 *   z
 */
__global__ void rasterize(const Vec4 *const         afterVertexShaderPositions,
                          const Vertex *const       verticies,
                          const unsigned int *const indices,
                          const int                 nrIndices,
                          const int                 width,
                          const int                 height,
                          const unsigned int        msaa,
                          FragmentData *const       fragmentDatas,
                          bool                      shouldprint) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height * msaa * msaa) { return; }
    const unsigned int blIdx    = upleftIdx2bottomLeftIdx(idx, width, height, msaa);
    fragmentDatas[blIdx].inMesh = false;

    const int fragmentIdx = idx % (msaa * msaa);
    const int pixelIdx    = idx / (msaa * msaa);
    const int pixelX      = pixelIdx % width;
    const int pixelY      = pixelIdx / width;
    float     x           = pixelX + 1.0f / (msaa * 2) + (1.0f / msaa) * (int)(fragmentIdx % msaa);
    float     y           = pixelY + 1.0f / (msaa * 2) + (1.0f / msaa) * (int)(fragmentIdx / msaa);
    // map x, y to [-1, 1]^2
    x                     = 2.0f * x / width - 1.0f;
    y                     = 1.0f - 2.0f * y / height;
    const Vec2 p{x, y};

    for (int i = 0; i < nrIndices; i += 3) {
        const unsigned int ia = indices[i];
        const unsigned int ib = indices[i + 1];
        const unsigned int ic = indices[i + 2];

        // verticies after MVP
        const Vec4 a = afterVertexShaderPositions[ia];
        const Vec4 b = afterVertexShaderPositions[ib];
        const Vec4 c = afterVertexShaderPositions[ic];

        if (isInTriangle(p, a, b, c)) {
            // original verticies
            // const Vertex &oa = verticies[ia];
            // const Vertex &ob = verticies[ib];
            // const Vertex &oc = verticies[ic];

            const Vec3 barycentric      = calBarycentric(p, a, b, c);
            fragmentDatas[blIdx].inMesh = true;
            const float alpha           = barycentric.x / a.z;
            const float beta            = barycentric.y / b.z;
            const float gamma           = barycentric.z / c.z;
            const float tempz           = 1 / (alpha + beta + gamma);
            // if (shouldprint) { printf("fragment idx:%d\r\n", idx); }
            if (tempz > fragmentDatas[blIdx].z) {
                // fragmentDatas[idx].z     = tempz;
                fragmentDatas[blIdx].color = Vec3{1.0f, 0, 0};
            }
        }
    }
}


__device__ static inline float circle(float cx, float cy, float r, float x, float y) {
    const float len = __fsqrt_rn(powf(x - cx, 2) + powf(y - cy, 2));
    return len - r;
}

// one thread per pixel
__global__ void fragmentShader(const FragmentData *const fragmentData,
                               const int                 width,
                               const int                 height,
                               const int                 msaa,
                               Vec3 *const               pixelColor) {
    const unsigned int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixelIdx >= width * height) { return; }
    const unsigned int blPixelIdx = upleftIdx2bottomLeftIdx(pixelIdx, width, height, 1);

    // all varaiables which can be used in the shader
    const int x = pixelIdx % width;
    const int y = pixelIdx / width;

    // fragment shader
    Vec3               color{0.0f, 0.0f, 0.0f};
    const unsigned int firstFragmentIdx = pixelIdx * msaa * msaa;
    unsigned int       nrFragments      = 0;
    for (int i = 0; i < msaa * msaa; i++) {
        const unsigned int fragmentIdx = firstFragmentIdx + i;
        const unsigned int blFragmentIdx =
            upleftIdx2bottomLeftIdx(fragmentIdx, width, height, msaa);
        if (fragmentData[blFragmentIdx].inMesh) {
            color.x += fragmentData[blFragmentIdx].color.x;
            color.y += fragmentData[blFragmentIdx].color.y;
            color.z += fragmentData[blFragmentIdx].color.z;
            nrFragments++;
        }
    }
    if (nrFragments == 0) { return; }
    color /= nrFragments;
    pixelColor[blPixelIdx] = color;
}


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
    bool shouldprint = true;

    int threadsPerBlock;
    int blocksPerGrid;
    for (auto &mesh : models[modelIdx].getMeshes()) {
        Vec4 *dev_afterVertexShaderPositions;
        CHECK_CUDA(cudaMalloc(&dev_afterVertexShaderPositions, mesh.nrVertices * sizeof(Vec4)));
        // vertex shader
        threadsPerBlock = 32;
        blocksPerGrid   = (mesh.nrIndices + threadsPerBlock - 1) / threadsPerBlock;
        vertexShader<<<blocksPerGrid, threadsPerBlock>>>(mesh.dev_verticies,
                                                         mesh.dev_indices,
                                                         mesh.nrIndices,
                                                         dev_afterVertexShaderPositions);

        CHECK_CUDA(cudaDeviceSynchronize());

        // rasterize
        threadsPerBlock                = 640;
        const unsigned int nrFragments = width * height * MSAA * MSAA;
        blocksPerGrid                  = (nrFragments + threadsPerBlock - 1) / threadsPerBlock;

        rasterize<<<blocksPerGrid, threadsPerBlock>>>(dev_afterVertexShaderPositions,
                                                      mesh.dev_verticies,
                                                      mesh.dev_indices,
                                                      mesh.nrIndices,
                                                      width,
                                                      height,
                                                      MSAA,
                                                      dev_fragmentData,
                                                      shouldprint);
        shouldprint = false;

        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // fragment shader
    threadsPerBlock = 640;
    blocksPerGrid   = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    fragmentShader<<<blocksPerGrid, threadsPerBlock>>>(dev_fragmentData,
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

// one thread per pixel
__global__ void setFragmentData(Vec3 *const         pixelColor,
                                const unsigned int  nrPixels,
                                FragmentData *const fragmentDatas,
                                const unsigned int  msaa,
                                const Vec3          color) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nrPixels) { return; }
    pixelColor[tid] = color;

    for (int i = 0; i < msaa * msaa; i++) {
        fragmentDatas[tid * 3 + i].inMesh = false;
        fragmentDatas[tid * 3 + i].z      = -1e10f;
        fragmentDatas[tid * 3 + i].color  = color;
    }
}
void Rasterizer::setBackground(const Vec3 &color) {
    const unsigned int threadsPerBlock = 64;
    const unsigned int blocksPerGrid   = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    setFragmentData<<<blocksPerGrid, threadsPerBlock>>>(dev_pixelColor,
                                                        nrPixels,
                                                        dev_fragmentData,
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
    CHECK_CUDA(cudaMalloc(&dev_fragmentData, nrPixels * MSAA * MSAA * sizeof(FragmentData)));
    pixelColor = new Vec3[nrPixels];
    CHECK_CUDA(cudaMalloc(&dev_pixelColor, nrPixels * sizeof(Vec3)));
}

void Rasterizer::freePixelData() {
    CHECK_CUDA(cudaFree(dev_fragmentData));
    delete[] pixelColor;
    CHECK_CUDA(cudaFree(dev_pixelColor));
}