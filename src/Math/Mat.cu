#include "Math/Mat.hpp"
#include "Math/Vec.hpp"
#include <iostream>


// PureMat
template <int dim>
PureMatBase<dim>::PureMatBase(const float d) {
#pragma unroll
    for (int i = 0; i < dim; i++) {
#pragma unroll
        for (int j = 0; j < dim; j++) {
            if (i == j) {
                data[i][j] = d;
            } else {
                data[i][j] = 0;
            }
        }
    }
}

template <int dim>
void PureMatBase<dim>::transpose() {
    PureMatBase<dim> copy = *this;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            this->data[i][j] = copy.data[j][i];
        }
    }
}
template void PureMatBase<3>::transpose();
template void PureMatBase<4>::transpose();

template <int dim>
PureMatBase<dim> PureMatBase<dim>::operator*(const PureMatBase<dim> &o) const {
    PureMatBase<dim> result{0};
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                result.data[i][j] += this->data[i][k] * o.data[k][j];
            }
        }
    }
    return result;
}
template PureMatBase<3> PureMatBase<3>::operator*(const PureMatBase<3> &o) const;
template PureMatBase<4> PureMatBase<4>::operator*(const PureMatBase<4> &o) const;

template <int dim>
PureMatBase<dim> &PureMatBase<dim>::operator<<(float o) {
    static unsigned int loadIndex = 0;

    int x            = loadIndex / dim;
    int y            = loadIndex % dim;
    this->data[x][y] = o;
    loadIndex        = (loadIndex + 1) % (dim * dim);

    return *this;
}

template <int dim>
std::ostream &operator<<(std::ostream &os, const PureMatBase<dim> &mat) {
    for (int i = 0; i < dim; i++) {
        os << "| ";
        for (int j = 0; j < dim; j++) {
            os << mat.data[i][j] << " ";
        }
        os << "|\n";
    }
    return os;
}
template std::ostream &operator<<(std::ostream &os, const PureMatBase<3> &mat);
template std::ostream &operator<<(std::ostream &os, const PureMatBase<4> &mat);

__device__ Vec4 PureMat4::operator*(const Vec3 &vec) const {
    Vec4 result;
    result.x = data[0][0] * vec.x + data[0][1] * vec.y + data[0][2] * vec.z + data[0][3] * 1;
    result.y = data[1][0] * vec.x + data[1][1] * vec.y + data[1][2] * vec.z + data[1][3] * 1;
    result.z = data[2][0] * vec.x + data[2][1] * vec.y + data[2][2] * vec.z + data[2][3] * 1;
    result.w = data[3][0] * vec.x + data[3][1] * vec.y + data[3][2] * vec.z + data[3][3] * 1;
    return result;
};

__device__ Vec4 PureMat4::operator*(const Vec4 &vec) const {
    Vec4 result;
    result.x = data[0][0] * vec.x + data[0][1] * vec.y + data[0][2] * vec.z + data[0][3] * vec.w;
    result.y = data[1][0] * vec.x + data[1][1] * vec.y + data[1][2] * vec.z + data[1][3] * vec.w;
    result.z = data[2][0] * vec.x + data[2][1] * vec.y + data[2][2] * vec.z + data[2][3] * vec.w;
    result.w = data[3][0] * vec.x + data[3][1] * vec.y + data[3][2] * vec.z + data[3][3] * vec.w;
    return result;
}


// MatBase
template <int dim>
float MatBase<dim>::det() const {
    std::cout << "MatBase::det() not implemented!" << std::endl;
    return 0;
}

template <int dim>
bool MatBase<dim>::inv(MatBase<dim> &result) const {
    MatBase<dim> copy = *this;
    MatBase<dim> eye{1};

    if (fabs(this->det()) < 1e-10) { return false; }

    // from up to top
    for (int i = 0; i < dim; i++) {
        // 寻找第 i 列不为零的元素
        int k;
        for (k = i; k < dim; k++) {
            if (fabs(copy.data[k][i]) > 1e-10) { break; }
        }

        if (k != dim) {
            if (k != i) {
                // 第 i 行 第 i 列元素为零，需要和其他行交换
                // 需从第一个元素交换，注意与之前化上三角矩阵不同
                // 使用mat[0][j]作为中间变量交换元素,两个矩阵都要交换
                for (int j = 0; j < dim; j++) {
                    copy.data[0][j] = copy.data[i][j];
                    copy.data[i][j] = copy.data[k][j];
                    copy.data[k][j] = copy.data[0][j];
                    eye.data[0][j]  = eye.data[i][j];
                    eye.data[i][j]  = eye.data[k][j];
                    eye.data[k][j]  = eye.data[0][j];
                }
            }
            float b = copy.data[i][i];
            for (int j = 0; j < dim; j++) {
                copy.data[i][j] /= b;
                eye.data[i][j] /= b;
            }
            for (int j = i + 1; j < dim; j++) {
                // 注意本来为 -a.mat[j][i]/a.mat[i][i],因为a.mat[i][i]等于 1，则不需要除它
                b = -copy.data[j][i];
                for (k = 0; k < dim; k++) {
                    copy.data[j][k] += b * copy.data[i][k]; // 第 i 行 b 倍加到第 j 行
                    eye.data[j][k] += b * eye.data[i][k];
                }
            }
        } else {
            // 找不到不为零的元素，矩阵不可逆
            return false;
        }
    }

    // from bottom to top
    for (int i = dim - 1; i >= 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            double b        = -copy.data[j][i];
            copy.data[j][i] = 0; // 实际上是通过初等行变换将这个元素化为 0,
            for (int k = 0; k < dim; k++) { // 通过相同的初等行变换来变换右边矩阵
                eye.data[j][k] += b * eye.data[i][k];
            }
        }
    }

    result = eye;
    return true;
}
template bool MatBase<3>::inv(MatBase<3> &result) const;
template bool MatBase<4>::inv(MatBase<4> &result) const;


// Mat3
Mat<3>::Mat(const Mat<4> &o, const Vec2 &cross) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i != cross.x && j != cross.y) { *this << o.data[i][j]; }
        }
    }
}

float Mat<3>::det() const {
    float result = 0;
    for (int i = 0; i < 3; i++) {
        result += data[0][i] * data[1][(i + 1) % 3] * data[2][(i + 2) % 3];
    }
    for (int i = 0; i < 3; i++) {
        result -= data[2][i] * data[1][(i + 1) % 3] * data[0][(i + 2) % 3];
    }
    return result;
}


// Mat4
Mat<4>::Mat(PureMatBase<4> o) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            *this << o.data[i][j];
        }
    }
}

float Mat<4>::det() const {
    float result = 0;
    for (int i = 0; i < 4; i++) {
        float  mul = (i % 2 == 0) ? 1 : -1;
        Mat<3> m{*this, {0, static_cast<float>(i)}};
        result += data[0][i] * m.det() * mul;
    }
    return result;
}

Vec4 Mat4::operator*(const Vec3 &vec) const {
    Vec4 result;
    result.x = data[0][0] * vec.x + data[0][1] * vec.y + data[0][2] * vec.z + data[0][3] * 1;
    result.y = data[1][0] * vec.x + data[1][1] * vec.y + data[1][2] * vec.z + data[1][3] * 1;
    result.z = data[2][0] * vec.x + data[2][1] * vec.y + data[2][2] * vec.z + data[2][3] * 1;
    result.w = data[3][0] * vec.x + data[3][1] * vec.y + data[3][2] * vec.z + data[3][3] * 1;
    return result;
};

Vec4 Mat4::operator*(const Vec4 &vec) const {
    Vec4 result;
    result.x = data[0][0] * vec.x + data[0][1] * vec.y + data[0][2] * vec.z + data[0][3] * vec.w;
    result.y = data[1][0] * vec.x + data[1][1] * vec.y + data[1][2] * vec.z + data[1][3] * vec.w;
    result.z = data[2][0] * vec.x + data[2][1] * vec.y + data[2][2] * vec.z + data[2][3] * vec.w;
    result.w = data[3][0] * vec.x + data[3][1] * vec.y + data[3][2] * vec.z + data[3][3] * vec.w;
    return result;
}