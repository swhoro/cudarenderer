#pragma once

#include <vector>
#include "Asset/Texture.hpp"
#include "Asset/Vertex.hpp"


class Mesh {
public:
    Vertex              *dev_verticies;
    unsigned int         nrVertices;
    unsigned int        *dev_indices;
    unsigned int         nrIndices;
    std::vector<Texture> textures = {};

    Mesh(std::vector<Vertex>       vertices,
         std::vector<unsigned int> indices,
         std::vector<Texture>      textures);

    Mesh(Mesh &)            = delete;
    Mesh &operator=(Mesh &) = delete;

    Mesh(Mesh &&o);
    Mesh &operator=(Mesh &&o);

    ~Mesh();
};