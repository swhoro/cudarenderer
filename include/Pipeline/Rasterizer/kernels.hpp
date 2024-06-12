#pragma once

#include <cuda_runtime.h>
#include "Asset/Vertex.hpp"
#include "Pipeline/Rasterizer/Rasterizer.hpp"

extern __constant__ PureMat4 dev_viewMatrix;
extern __constant__ PureMat4 dev_projectionMatrix;

__global__ void vertexShader(const Vertex *const       verticies,
                             const unsigned int *const indices,
                             const int                 nrIndices,
                             Vec4 *const               ndc_positions);

__device__ bool         isInTriangle(const Vec2 &p, const Vec2 &a, const Vec2 &b, const Vec2 &c);
__device__ Vec4         getBoundingBox(const Vec4 &a, const Vec4 &b, const Vec4 &c);
__device__ Vec3         calBarycentric(const Vec2 &p, const Vec4 &a, const Vec4 &b, const Vec4 &c);
__device__ unsigned int upLeftIdx2bottomLeftIdx(const unsigned int &idx,
                                                const int          &width,
                                                const int          &height,
                                                const unsigned int &msaa);
__global__ void         rasterize(const Vec4 *const         ndc_positions,
                                  const Vertex *const       verticies,
                                  const unsigned int *const indices,
                                  const unsigned int        nrIndices,
                                  const int                 width,
                                  const int                 height,
                                  const unsigned int        msaa,
                                  FragmentData *const       fragmentDatas);
__global__ void         fragmentShader(const FragmentData *const fragmentData,
                                       const int                 width,
                                       const int                 height,
                                       const int                 msaa,
                                       Vec3 *const               pixelColor);

__global__ void setFragmentData(Vec3 *const         pixelColor,
                                const unsigned int  nrPixels,
                                FragmentData *const fragmentDatas,
                                const unsigned int  msaa,
                                const Vec3          color);
