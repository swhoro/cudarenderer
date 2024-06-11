#include "Math/Vec.hpp"

// 2D vector
__device__ __host__ Vec<2>::Vec() : x(0), y(0) {
}
__device__ __host__ Vec<2>::Vec(float x, float y) : x(x), y(y) {
}
__device__ __host__ Vec<2>::Vec(const Vec<4> &o) : x(o.x), y(o.y) {
}

__device__ __host__ Vec<2> Vec<2>::operator+(const Vec<2> &o) const {
    return Vec<2>(x + o.x, y + o.y);
}
__device__ __host__ Vec<2> &Vec<2>::operator+=(const Vec<2> &o) {
    *this = *this + o;
    return *this;
}

__device__ __host__ Vec<2> Vec<2>::operator-(const Vec<2> &o) const {
    return Vec<2>(x - o.x, y - o.y);
}
__device__ __host__ Vec<2> &Vec<2>::operator-=(const Vec<2> &o) {
    *this = *this - o;
    return *this;
}

// dot product
__device__ __host__ float Vec<2>::operator*(const Vec<2> &o) const {
    return x * o.x + y * o.y;
}

__device__ __host__ Vec<2> Vec<2>::operator+(const float &o) const {
    return Vec<2>(x + o, y + o);
}
__device__ __host__ Vec<2> &Vec<2>::operator+=(const float &o) {
    *this = *this + o;
    return *this;
}

__device__ __host__ Vec<2> Vec<2>::operator-(const float &o) const {
    return Vec<2>(x - o, y - o);
}
__device__ __host__ Vec<2> &Vec<2>::operator-=(const float &o) {
    *this = *this - o;
    return *this;
}

__device__ __host__ Vec<2> Vec<2>::operator*(const float &o) const {
    return Vec<2>(x * o, y * o);
}
__device__ __host__ Vec<2> &Vec<2>::operator*=(const float &o) {
    *this = *this * o;
    return *this;
}

__device__ __host__ Vec<2> Vec<2>::operator/(const float &o) const {
    return Vec<2>(x / o, y / o);
}
__device__ __host__ Vec<2> &Vec<2>::operator/=(const float &o) {
    *this = *this / o;
    return *this;
}

std::ostream &operator<<(std::ostream &os, const Vec<2> &vec) {
    os << "Vec<2>: (" << vec.x << ", " << vec.y << ")";
    return os;
}


// 3D vector
__device__ __host__ Vec<3>::Vec() : x(0), y(0), z(0) {
}
__device__ __host__ Vec<3>::Vec(float x, float y, float z) : x(x), y(y), z(z) {
}

__device__ __host__ Vec<3> Vec<3>::operator+(const Vec<3> &o) const {
    return Vec<3>(x + o.x, y + o.y, z + o.z);
}
__device__ __host__ Vec<3> &Vec<3>::operator+=(const Vec<3> &o) {
    *this = *this + o;
    return *this;
}

__device__ __host__ Vec<3> Vec<3>::operator-(const Vec<3> &o) const {
    return Vec<3>(x - o.x, y - o.y, z - o.z);
}
__device__ __host__ Vec<3> &Vec<3>::operator-=(const Vec<3> &o) {
    *this = *this - o;
    return *this;
}

// dot product
__device__ __host__ float Vec<3>::operator*(const Vec<3> &o) const {
    return x * o.x + y * o.y + z * o.z;
}

// cross product
__device__ __host__ Vec<3> Vec<3>::cross(Vec<3> o) const {
    return Vec<3>(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
}

__device__ __host__ Vec<3> Vec<3>::operator+(const float &o) const {
    return Vec<3>(x + o, y + o, z + o);
}
__device__ __host__ Vec<3> &Vec<3>::operator+=(const float &o) {
    *this = *this + o;
    return *this;
}

__device__ __host__ Vec<3> Vec<3>::operator-(const float &o) const {
    return Vec<3>(x - o, y - o, z - o);
}
__device__ __host__ Vec<3> &Vec<3>::operator-=(const float &o) {
    *this = *this - o;
    return *this;
}

__device__ __host__ Vec<3> Vec<3>::operator*(const float &o) const {
    return Vec<3>(x * o, y * o, z * o);
}
__device__ __host__ Vec<3> &Vec<3>::operator*=(float o) {
    *this = *this * o;
    return *this;
}

__device__ __host__ Vec<3> Vec<3>::operator/(const float &o) const {
    return Vec<3>(x / o, y / o, z / o);
}
__device__ __host__ Vec<3> &Vec<3>::operator/=(const float &o) {
    *this = *this / o;
    return *this;
}

std::ostream &operator<<(std::ostream &os, const Vec<3> &vec) {
    os << "Vec<3>: (" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}


// 4D vector
__device__ __host__ Vec<4>::Vec() : x(0), y(0), z(0), w(0) {
}
__device__ __host__ Vec<4>::Vec(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {
}

__device__ __host__ Vec<4> Vec<4>::operator+(const Vec<4> &o) const {
    return Vec<4>(x + o.x, y + o.y, z + o.z, w + o.w);
}
__device__ __host__ Vec<4> &Vec<4>::operator+=(const Vec<4> &o) {
    *this = *this + o;
    return *this;
}

__device__ __host__ Vec<4> Vec<4>::operator-(const Vec<4> &o) const {
    return Vec<4>(x - o.x, y - o.y, z - o.z, w - o.w);
}
__device__ __host__ Vec<4> &Vec<4>::operator-=(const Vec<4> &o) {
    *this = *this - o;
    return *this;
}

__device__ __host__ float Vec<4>::operator*(const Vec<4> &o) const {
    return x * o.x + y * o.y + z * o.z + w;
}

__device__ __host__ Vec<4> Vec<4>::operator/(const float &o) const {
    return Vec<4>{x / o, y / o, z / o, w / o};
}
__device__ __host__ Vec<4> &Vec<4>::operator/=(const float &o) {
    *this = *this / o;
    return *this;
}

std::ostream &operator<<(std::ostream &os, const Vec<4> &vec) {
    os << "Vec<4>: (" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << ")";
    return os;
}
