#include "Asset/Texture.hpp"
#include "utils.hpp"


Texture::Texture(unsigned char *dev_data, std::string type, int width, int height, int nrChannels)
    : dev_data(dev_data), type(type), width(width), height(height), nrChannels(nrChannels) {
}

Texture::Texture(Texture &&o) {
    dev_data   = o.dev_data;
    type       = o.type;
    width      = o.width;
    height     = o.height;
    o.dev_data = nullptr;
    o.type     = "";
    o.width    = 0;
    o.height   = 0;
}

Texture &Texture::operator=(Texture &&o) {
    dev_data   = o.dev_data;
    type       = o.type;
    width      = o.width;
    height     = o.height;
    o.dev_data = nullptr;
    o.type     = "";
    o.width    = 0;
    o.height   = 0;
    return *this;
}

Texture::~Texture() {
    if (dev_data) { CHECK_CUDA(cudaFree(dev_data)); };
}
