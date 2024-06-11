#pragma once

#include "Vec.hpp"


template <int dim>
struct PureMatBase {
    float data[dim][dim];

    PureMatBase() = default;
    PureMatBase(const float d);

    void transpose();

    PureMatBase<dim> operator*(const PureMatBase<dim> &o) const;
    PureMatBase<dim> &operator<<(float o);
};
template <int dim>
std::ostream &operator<<(std::ostream &os, const PureMatBase<dim> &mat);

struct PureMat4 : public PureMatBase<4> {
    __device__ Vec4 operator*(const Vec3 &vec) const;
    __device__ Vec4 operator*(const Vec4 &vec) const;
};

template <int dim>
struct MatBase : public PureMatBase<dim> {
    using PureMatBase<dim>::PureMatBase;
    // using PureMatBase<dim>::operator*;

    virtual float det() const;
    bool          inv(MatBase<dim> &result) const;
};

template <int dim>
struct Mat : public MatBase<dim> {};

template <>
struct Mat<3> : public MatBase<3> {
    using MatBase<3>::MatBase;

    Mat(const Mat<4> &o, const Vec2 &cross);

    float det() const override;
};

template <>
struct Mat<4> : public MatBase<4> {
    using MatBase<4>::MatBase;
    using MatBase<4>::operator*;

    Mat(PureMatBase<4> o);

    float det() const override;

    Vec4 operator*(const Vec3 &vec) const;
    Vec4 operator*(const Vec4 &vec) const;
};

using Mat3 = Mat<3>;
using Mat4 = Mat<4>;
