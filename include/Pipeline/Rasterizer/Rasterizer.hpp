#pragma once

#include <vector>
#include "Asset/Model.hpp"
#include "Camera.hpp"
#include "Math/Mat.hpp"
#include "Pipeline/Pipeline.hpp"


class FragmentData {
public:
    bool  inMesh;
    float z;
    Vec3  color;
    // Vec2         uv;
    // unsigned int textureIdx = 0;
};

class Rasterizer : virtual public Pipeline {
public:
    Rasterizer(const int w, const int h);
    // return the index of loaded model
    int         loadModel(std::string path);
    int         loadModel(Model &model);
    /* get rendered pixel color
     *
     * return nullptr if model is not exist
     */
    const Vec3 *render(const Camera &camera, const int modelIdx);
    const Vec3 *getDevPixelColor() const;
    void        resize(const int width, const int height);
    void        setMSAA(const unsigned int msaa);
    void        setBackground(const Vec3 &color);
    void        setViewMatrix(const Mat4 &viewMatrix);
    void        setProjectionMatrix(const Mat4 &projectionMatrix);

private:
    std::vector<Model> models;
    /* fragment data for each fragment
     *
     * all fragments in one pixel will be stored in adjacent memory
     */
    FragmentData      *dev_fragmentDatas;
    // red, green, blue
    Vec3              *pixelColor;
    Vec3              *dev_pixelColor;
    // how many pixel data, calculate from width * height
    int                nrPixels;
    unsigned int       MSAA = 1;

    void freePixelData();
    void allocatePixelData();
};