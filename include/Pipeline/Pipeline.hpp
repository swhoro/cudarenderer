#pragma once

#include <string>

class Pipeline {
public:
    Pipeline(int w, int h) : width(w), height(h) {}
    // return the index of loaded model
    virtual int  loadModel(std::string path)   = 0;
    // virtual float *render(Camera &camera)        = 0;
    virtual void resize(int width, int height) = 0;

protected:
    int width;
    int height;
};