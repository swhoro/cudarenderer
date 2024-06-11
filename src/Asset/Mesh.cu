#include "Asset/Mesh.hpp"
#include "utils.hpp"


Mesh::Mesh(std::vector<Vertex>       vertices,
           std::vector<unsigned int> indices,
           std::vector<Texture>      textures) {
    nrVertices = vertices.size();
    CHECK_CUDA(cudaMalloc(&dev_verticies, nrVertices * sizeof(Vertex)));
    CHECK_CUDA(cudaMemcpy(dev_verticies,
                          vertices.data(),
                          nrVertices * sizeof(Vertex),
                          cudaMemcpyHostToDevice));

    nrIndices = indices.size();
    CHECK_CUDA(cudaMalloc(&dev_indices, indices.size() * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemcpy(dev_indices,
                          indices.data(),
                          indices.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    textures = std::move(textures);
};

Mesh::Mesh(Mesh &&o) {
    dev_verticies   = o.dev_verticies;
    nrVertices      = o.nrVertices;
    dev_indices     = o.dev_indices;
    nrIndices       = o.nrIndices;
    textures        = std::move(o.textures);
    o.dev_verticies = nullptr;
    o.nrVertices    = 0;
    o.dev_indices   = nullptr;
    o.nrIndices     = 0;
};

Mesh &Mesh::operator=(Mesh &&o) {
    dev_verticies   = o.dev_verticies;
    nrVertices      = o.nrVertices;
    dev_indices     = o.dev_indices;
    nrIndices       = o.nrIndices;
    textures        = std::move(o.textures);
    o.dev_verticies = nullptr;
    o.nrVertices    = 0;
    o.dev_indices   = nullptr;
    o.nrIndices     = 0;
    return *this;
};

Mesh::~Mesh() {
    if (dev_verticies) { CHECK_CUDA(cudaFree(dev_verticies)); }
    if (dev_indices) { CHECK_CUDA(cudaFree(dev_indices)); }
}