#pragma once

#include <iostream>
#include <cuda_runtime.h>


template <int dim>
struct Vec;

template <>
struct Vec<2> {
    float x;
    float y;

    __device__ __host__ Vec<2>();
    __device__ __host__ Vec<2>(float x, float y);
    __device__ __host__ Vec<2>(const Vec<4> &o);

    __device__ __host__ Vec<2> operator+(const Vec<2> &o) const;
    __device__ __host__ Vec<2> &operator+=(const Vec<2> &o);

    __device__ __host__ Vec<2> operator-(const Vec<2> &o) const;
    __device__ __host__ Vec<2> &operator-=(const Vec<2> &o);

    // dot product
    __device__ __host__ float operator*(const Vec<2> &o) const;

    __device__ __host__ Vec<2> operator+(const float &o) const;
    __device__ __host__ Vec<2> &operator+=(const float &o);

    __device__ __host__ Vec<2> operator-(const float &o) const;
    __device__ __host__ Vec<2> &operator-=(const float &o);


    __device__ __host__ Vec<2> operator*(const float &o) const;
    __device__ __host__ Vec<2> &operator*=(const float &o);

    __device__ __host__ Vec<2> operator/(const float &o) const;
    __device__ __host__ Vec<2> &operator/=(const float &o);

    friend std::ostream &operator<<(std::ostream &os, const Vec<2> &vec);
};

template <>
struct Vec<3> {
    float x;
    float y;
    float z;

    __device__ __host__ Vec<3>();
    __device__ __host__ Vec<3>(float x, float y, float z);

    __device__ __host__ Vec<3> operator+(const Vec<3> &o) const;
    __device__ __host__ Vec<3> &operator+=(const Vec<3> &o);

    __device__ __host__ Vec<3> operator-(const Vec<3> &o) const;
    __device__ __host__ Vec<3> &operator-=(const Vec<3> &o);

    // dot product
    __device__ __host__ float operator*(const Vec<3> &o) const;

    // cross product
    __device__ __host__ Vec<3> cross(Vec<3> o) const;

    __device__ __host__ Vec<3> operator+(const float &o) const;
    __device__ __host__ Vec<3> &operator+=(const float &o);

    __device__ __host__ Vec<3> operator-(const float &o) const;
    __device__ __host__ Vec<3> &operator-=(const float &o);

    __device__ __host__ Vec<3> operator*(const float &o) const;
    __device__ __host__ Vec<3> &operator*=(const float o);

    __device__ __host__ Vec<3> operator/(const float &o) const;
    __device__ __host__ Vec<3> &operator/=(const float &o);

    friend std::ostream &operator<<(std::ostream &os, const Vec<3> &vec);
};

template <>
struct Vec<4> {
    float x;
    float y;
    float z;
    float w;

    __device__ __host__ Vec<4>();
    __device__ __host__ Vec<4>(float x, float y, float z, float w);

    __device__ __host__ Vec<4> operator+(const Vec<4> &o) const;
    __device__ __host__ Vec<4> &operator+=(const Vec<4> &o);

    __device__ __host__ Vec<4> operator-(const Vec<4> &o) const;
    __device__ __host__ Vec<4> &operator-=(const Vec<4> &o);

    // dot product
    __device__ __host__ float operator*(const Vec<4> &o) const;

    __device__ __host__ Vec<4> operator/(const float &o) const;
    __device__ __host__ Vec<4> &operator/=(const float &o);

    friend std::ostream &operator<<(std::ostream &os, const Vec<4> &vec);
};


using Vec2 = Vec<2>;
using Vec3 = Vec<3>;
using Vec4 = Vec<4>;


struct Vec2i {
    int x;
    int y;
};


template <int dim>
__device__ __host__ inline Vec<dim> normalize(Vec<dim> vec) {
#ifdef __CUDA_ARCH__
    return vec / __fsqrt_rn(vec * vec);
#else
    return vec / std::sqrt(vec * vec);
#endif
}
