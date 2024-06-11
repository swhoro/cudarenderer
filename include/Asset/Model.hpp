#pragma once

#include <vector>
#include <string>
#include <assimp/scene.h>
#include "Mesh.hpp"

class Model {
public:
    Model() = default;
    Model(std::string path);

    void                     pushMesh(Mesh &mesh);
    const std::vector<Mesh> &getMeshes() const;

private:
    std::vector<Mesh> meshes = {};
    // std::vector<Texture> trextures;
    std::string       directory;

    void processNode(aiNode *node, const aiScene *scene);
    Mesh processMesh(aiMesh *mesh, const aiScene *scene);
    std::vector<Texture>
    loadMaterialTextures(aiMaterial *material, aiTextureType type, std::string typeName);
};