#include "Pipeline/Rasterizer/Rasterizer.hpp"
#include "Pipeline/Rasterizer/kernels.hpp"
#include "Math/Mat.hpp"
#include "Asset/Vertex.hpp"
#include <cuda_runtime_api.h>


__constant__ PureMat4 dev_viewMatrix;
__constant__ PureMat4 dev_projectionMatrix;

/* one thread per indice
 *
 * map all verticies into [-1, 1]^3
 */
__global__ void vertexShader(const Vertex *const       verticies,
                             const unsigned int *const indices,
                             const int                 nrIndices,
                             Vec4 *const               ndc_positions) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrIndices) { return; }

    const unsigned int vertexIdx = indices[idx];
    const Vertex       vertex    = verticies[vertexIdx];

    // view projection
    Vec4 ndc_position = dev_projectionMatrix * (dev_viewMatrix * vertex.postition);
    ndc_position /= ndc_position.w;
    ndc_positions[vertexIdx] = ndc_position;
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
__device__ static inline bool isInTriangle(const Vec2 &p, const Vec2 &a, const Vec2 &b, const Vec2 &c) {
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
__device__ static inline Vec4 getBoundingBox(const Vec4 &a, const Vec4 &b, const Vec4 &c) {
    Vec4 result;
    result.x = fminf(a.x, fminf(b.x, c.x));
    result.y = fminf(a.y, fminf(b.y, c.y));
    result.z = fmaxf(a.x, fmaxf(b.x, c.x));
    result.w = fmaxf(a.y, fmaxf(b.y, c.y));
    return result;
}

__device__ static inline Vec3 calBarycentric(const Vec2 &p, const Vec4 &a, const Vec4 &b, const Vec4 &c) {
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

__device__ static inline unsigned int upLeftIdx2bottomLeftIdx(const unsigned int &idx,
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

/* the returned Vec2 is in screen space,
 * x is the index of point from top to bottom,
 * y is the index of point from left to right,
 * both x and y are always positive integers.
 * if less is true, the returned Vec2 is floored,
 * or the returned Vec2 is ceiled
 */
__device__ static inline Vec2i ndc2screen(const Vec2         &p,
                            const int          &width,
                            const int          &height,
                            const unsigned int &msaa,
                            bool                less) {
    int         resultx;
    int         resulty;
    const float w = width * msaa;
    const float h = height * msaa;

    resultx = w * p.x / 2 + w / 2;
    resulty = h / 2 - h * p.y / 2;
    if (!less) {
        resultx += 1;
        resulty += 1;
    }

    return Vec2i{resultx, resulty};
}

/* p.x is the index of point from top to bottom,
 * p.y is the index of point from left to right,
 * x and y in p are always positive integers.
 */
__device__ static inline Vec2 screen2ndc(const Vec2i        &p,
                           const int          &width,
                           const int          &height,
                           const unsigned int &msaa) {
    Vec2      result;
    const int w = width * msaa;
    const int h = height * msaa;
    result.x    = 2.0f * p.x / w - 1.0f;
    result.y    = 1.0f - 2.0f * p.y / h;
    return result;
}

/*
 * convert screen space coordicate into
 * the top left index of the fragment
 */
__device__ static inline unsigned int
scrFragment2idx(const Vec2i &p, const int &width, const int &height, const unsigned int &msaa) {
    Vec2i scr_pixel;
    scr_pixel.x = p.x / msaa;
    scr_pixel.y = p.y / msaa;
    Vec2i fragmentInPixel;
    fragmentInPixel.x         = p.x % msaa;
    fragmentInPixel.y         = p.y % msaa;
    unsigned int indexInPixel = fragmentInPixel.y * msaa + fragmentInPixel.x;

    return scr_pixel.y * width * msaa * msaa + scr_pixel.x * msaa * msaa + indexInPixel;
}

__device__ static inline float intersect(const Vec3  &barycentric,
                           const float &a,
                           const float &b,
                           const float &c,
                           bool         perspective = true) {
    if (perspective) {
        const float ta = barycentric.x / a;
        const float tb = barycentric.y / b;
        const float tc = barycentric.z / c;
        return 1 / (ta + tb + tc);
    } else {
        return barycentric.x * a + barycentric.y * b + barycentric.z * c;
    }
}

// one thread for one fragment
__global__ void rasterizeFragmentOnTriangle(const Vec2i         scr_min,
                                            const Vec2i         scr_max,
                                            const unsigned int  nrFragments,
                                            // a, b, c is the vertices of the triangle
                                            const Vec4          ndc_a,
                                            const Vec4          ndc_b,
                                            const Vec4          ndc_c,
                                            const int           width,
                                            const int           height,
                                            const unsigned int  msaa,
                                            FragmentData *const fragmentDatas) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrFragments) { return; }

    Vec2i scr_fragment;
    scr_fragment.x = scr_min.x + idx % (scr_max.x - scr_min.x);
    scr_fragment.y = scr_min.y + idx / (scr_max.x - scr_min.x);

    const Vec2 ndc_p = screen2ndc(scr_fragment, width, height, msaa);

    if (isInTriangle(ndc_p, ndc_a, ndc_b, ndc_c)) {
        const unsigned int fragmentIdx = scrFragment2idx(scr_fragment, width, height, msaa);
        const unsigned int bl_fragmentIdx =
            upLeftIdx2bottomLeftIdx(fragmentIdx, width, height, msaa);
        const Vec3 &barycentric = calBarycentric(ndc_p, ndc_a, ndc_b, ndc_c);
        const float tempz       = intersect(barycentric, ndc_a.z, ndc_b.z, ndc_c.z);

        fragmentDatas[bl_fragmentIdx].inMesh = true;

        if (tempz > fragmentDatas[bl_fragmentIdx].z) {
            fragmentDatas[bl_fragmentIdx].z     = tempz;
            fragmentDatas[bl_fragmentIdx].color = Vec3{1, 0, 0};
        }
    }
}

/* rasterize
 *
 * only one thread testing each triangle
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
__global__ void rasterize(const Vec4 *const         ndc_positions,
                          const Vertex *const       verticies,
                          const unsigned int *const indices,
                          const unsigned int        nrIndices,
                          const int                 width,
                          const int                 height,
                          const unsigned int        msaa,
                          FragmentData *const       fragmentDatas) {
    for (int i = 0; i < nrIndices / 3; i++) {
        const unsigned int ia = indices[i * 3];
        const unsigned int ib = indices[i * 3 + 1];
        const unsigned int ic = indices[i * 3 + 2];

        const Vec4 &a           = ndc_positions[ia];
        const Vec4 &b           = ndc_positions[ib];
        const Vec4 &c           = ndc_positions[ic];
        const Vec4  boundingbox = getBoundingBox(a, b, c);

        Vec2i scr_min = ndc2screen(Vec2{boundingbox.x, boundingbox.w}, width, height, msaa, true);
        Vec2i scr_max = ndc2screen(Vec2{boundingbox.z, boundingbox.y}, width, height, msaa, false);
        
        // check if the triangle is in the screen
        if (scr_min.x >= width * msaa || scr_min.y >= height * msaa) { continue; }
        if (scr_max.x < 0 || scr_max.y < 0) { continue; }
        if (scr_max.x > width * msaa) { scr_max.x = width * msaa; }
        if (scr_max.y > height * msaa) { scr_max.y = height * msaa; }
        if (scr_min.x < 0 || scr_min.y < 0) { scr_min.x = 0; }
        if (scr_min.y < 0) { scr_min.y = 0; }

        const unsigned int nrFragmentsWidth  = scr_max.x - scr_min.x;
        const unsigned int nrFragmentsHeight = scr_max.y - scr_min.y;
        const unsigned int nrFragments     = nrFragmentsWidth * nrFragmentsHeight;
        const unsigned int threadsPerBlock = 768;
        const unsigned int blocksPerGrid   = (nrFragments + threadsPerBlock - 1) / threadsPerBlock;
        rasterizeFragmentOnTriangle<<<blocksPerGrid, threadsPerBlock>>>(scr_min,
                                                                        scr_max,
                                                                        nrFragments,
                                                                        a,
                                                                        b,
                                                                        c,
                                                                        width,
                                                                        height,
                                                                        msaa,
                                                                        fragmentDatas);
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
    const unsigned int bl_pixelIdx = upLeftIdx2bottomLeftIdx(pixelIdx, width, height, 1);

    // all varaiables which can be used in the shader
    const int x = pixelIdx % width;
    const int y = pixelIdx / width;

    // fragment shader
    Vec3               color{0.0f, 0.0f, 0.0f};
    const unsigned int firstFragmentIdx = pixelIdx * msaa * msaa;
    unsigned int       usedFragments    = 0;
    for (int i = 0; i < msaa * msaa; i++) {
        const unsigned int fragmentIdx = firstFragmentIdx + i;
        const unsigned int bl_fragmentIdx =
            upLeftIdx2bottomLeftIdx(fragmentIdx, width, height, msaa);
        if (fragmentData[bl_fragmentIdx].inMesh) {
            color += fragmentData[bl_fragmentIdx].color;
            usedFragments++;
        }
    }
    if (usedFragments == 0) { return; }
    color /= usedFragments;
    pixelColor[bl_pixelIdx] = color;
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
        fragmentDatas[tid * msaa * msaa + i].inMesh = false;
        fragmentDatas[tid * msaa * msaa + i].z      = -1e10f;
        fragmentDatas[tid * msaa * msaa + i].color  = color;
    }
}
