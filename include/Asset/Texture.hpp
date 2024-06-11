#pragma once

#include <string>
#include <assimp/types.h>

class Texture {
public:
    unsigned char *dev_data;
    std::string    type;
    int            width, height;
    int            nrChannels;

    Texture(unsigned char *dev_data, std::string type, int width, int height, int nrChannels);

    Texture(Texture &)            = delete;
    Texture &operator=(Texture &) = delete;

    Texture(Texture &&o);
    Texture &operator=(Texture &&o);

    ~Texture();
};