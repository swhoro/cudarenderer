#include "Camera.hpp"
#include "Math/Vec.hpp"
#include "Math/utils.hpp"

Camera::Camera(Vec3         postition,
               Vec3         cameraFront,
               Vec3         cameraUp,
               int          width,
               int          height,
               float        zNear,
               float        zFar,
               unsigned int fov)
    : postition(postition), cameraFront(normalize(cameraFront)), cameraUp(normalize(cameraUp)),
      cameraRight(cameraFront.cross(cameraUp)), width(width), height(height), fov(fov), yaw(0.0f),
      pitch(0.0f), zNear(zNear), zFar(zFar) {
}

void Camera::move(Vec3 offset) {
    postition += offset;
}

void Camera::rotate(const float yawOffset, const float pitchOffset) {
    this->pitch += pitchOffset;
    this->yaw += yawOffset;
    if (this->pitch > 89.0f) { this->pitch = 89.0f; }
    if (this->pitch < -89.0f) { this->pitch = -89.0f; }
    float r_pitch       = toRadians(this->pitch);
    float r_yaw         = toRadians(this->yaw);
    this->cameraFront.x = cos(r_pitch) * sin(r_yaw);
    this->cameraFront.y = sin(r_pitch);
    this->cameraFront.z = -cos(r_pitch) * cos(r_yaw);
    this->cameraFront   = normalize(this->cameraFront);

    this->cameraRight = this->cameraFront.cross(this->cameraUp);
}

Vec3 Camera::getFront() const {
    return cameraFront;
}

Vec3 Camera::getUp() const {
    return cameraUp;
}

Vec3 Camera::getRight() const {
    return cameraRight;
}

Mat4 Camera::getViewMatrix() const {
    /*
     * | 1    0    0    -x |
     * | 0    1    0    -y |
     * | 0    0    1    -z |
     * | 0    0    0     1 |
     */
    Mat4 translate{1.0f};
    translate.data[0][3] = -postition.x;
    translate.data[1][3] = -postition.y;
    translate.data[2][3] = -postition.z;

    /*
     * |  cr.x      cr.y     cr.z   0  |
     * |  cu.x      cu.y     cu.z   0  |
     * | -cf.x    -cf.y    -cf.z    0  |
     * |   0        0        0      1  |
     */
    Mat4 rotate{1.0f};
    rotate.data[0][0] = cameraRight.x;
    rotate.data[0][1] = cameraRight.y;
    rotate.data[0][2] = cameraRight.z;
    rotate.data[1][0] = cameraUp.x;
    rotate.data[1][1] = cameraUp.y;
    rotate.data[1][2] = cameraUp.z;
    rotate.data[2][0] = -cameraFront.x;
    rotate.data[2][1] = -cameraFront.y;
    rotate.data[2][2] = -cameraFront.z;

    return rotate * translate;
}

Mat4 Camera::getProjectionMatrix() const {
    /*
     * |-n  0    0    0 |
     * | 0 -n    0    0 |
     * | 0  0  -n-f  -nf|
     * | 0  0    0    1 |
     */
    Mat4 perspective{0.0f};
    perspective.data[0][0] = -zNear;
    perspective.data[1][1] = -zNear;
    perspective.data[2][2] = -zNear - zFar;
    perspective.data[2][3] = -zNear * zFar;
    perspective.data[3][2] = 1.0f;

    /*
     * | 0    0    0      0     |
     * | 0    0    0      0     |
     * | 0    0    0   -(n+f)/2 |
     * | 0    0    0      1     |
     */
    Mat4 translate{1.0f};
    translate.data[0][3] = 0;
    translate.data[1][3] = 0;
    translate.data[2][3] = (zNear + zFar) / 2.0f;

    /*
     * | 1/r   0      0       0 |
     * | 0     1/t    0       0 |
     * | 0     0    2/(n+f)   0 |
     * | 0     0      0       1 |
     */
    Mat4        scale{1.0f};
    const float top   = zNear * tan(toRadians(fov / 2.0f));
    const float right = top * width / height;
    scale.data[0][0]  = 1.0f / right;
    scale.data[1][1]  = 1.0f / top;
    scale.data[2][2]  = 2.0f / (zNear + zFar);
    scale.data[3][3]  = 1.0f;

    return scale * translate * perspective;
}
