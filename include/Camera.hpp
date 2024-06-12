#pragma once

#include "Math/Mat.hpp"
#include "Math/Vec.hpp"


class Camera {
public:
    Camera(Vec3         postition,
           Vec3         cameraFront,
           Vec3         cameraUp,
           int          width,
           int          height,
           float        zNear,
           float        zFar,
           unsigned int fov = 90);
    void move(Vec3 offset);
    void rotate(const float yaw, const float pitch);
    Vec3 getFront() const;
    Vec3 getUp() const;
    Vec3 getRight() const;
    Mat4 getViewMatrix() const;
    Mat4 getProjectionMatrix() const;

private:
    Vec3         postition;
    Vec3         cameraFront;
    Vec3         cameraUp;
    Vec3         cameraRight;
    int          width, height;
    unsigned int fov;
    // see negative -z is pitch 0, yaw 0
    float        yaw, pitch;
    // zNear and zFar are always positive
    float        zNear, zFar;
};
