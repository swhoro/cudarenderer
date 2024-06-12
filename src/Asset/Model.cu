#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iterator>
#include <string>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <stb_image.h>
#include "Asset/Model.hpp"
#include "utils.hpp"

Model::Model(std::string path) {
    Assimp::Importer importer;
    const aiScene   *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return;
    }

    directory = path.substr(0, path.find_last_of('/'));
    processNode(scene->mRootNode, scene);
}

void Model::pushMesh(Mesh &mesh) {
    meshes.push_back(std::move(mesh));
}

const std::vector<Mesh> &Model::getMeshes() const {
    return meshes;
}

void Model::processNode(aiNode *node, const aiScene *scene) {
    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene));
    }

    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene);
    }
}

Mesh Model::processMesh(aiMesh *mesh, const aiScene *scene) {
    std::vector<Vertex>       vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture>      textures;

    // load vertices
    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        Vec3   vector;

        vector.x         = mesh->mVertices[i].x;
        vector.y         = mesh->mVertices[i].y;
        vector.z         = mesh->mVertices[i].z;
        vertex.postition = vector;

        // vector.x      = mesh->mNormals[i].x;
        // vector.y      = mesh->mNormals[i].y;
        // vector.z      = mesh->mNormals[i].z;
        vertex.normal = vector;

        // if (mesh->mTextureCoords[0]) {
        //     Vec2 vec;
        //     vec.x            = mesh->mTextureCoords[0][i].x;
        //     vec.y            = mesh->mTextureCoords[0][i].y;
        //     vertex.texCoords = vec;
        // } else {
        //     vertex.texCoords = Vec2(0.0f, 0.0f);
        // }

        vertices.push_back(vertex);
    }

    int nrindices = 0;
    // load indices
    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        nrindices += face.mNumIndices;
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }
    std::cout << nrindices << std::endl;

    // load textures
    if (mesh->mMaterialIndex > 0) {
        aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];

        std::vector<Texture> diffuseMaps =
            loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse");
        textures.insert(textures.end(),
                        std::make_move_iterator(diffuseMaps.begin()),
                        std::make_move_iterator(diffuseMaps.end()));

        std::vector<Texture> specularMaps =
            loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
        textures.insert(textures.end(),
                        std::make_move_iterator(specularMaps.begin()),
                        std::make_move_iterator(specularMaps.end()));
    }

    return Mesh(vertices, indices, std::move(textures));
}

Texture loadTextureFromFile(const std::string path, const std::string typeName) {
    int            width, height, nrChannels;
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
    unsigned char *dev_data;
    CHECK_CUDA(cudaMalloc(&dev_data, width * height * nrChannels * sizeof(unsigned char)));
    CHECK_CUDA(cudaMemcpy(dev_data,
                          data,
                          width * height * nrChannels * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    stbi_image_free(data);
    return Texture{dev_data, typeName, width, height, nrChannels};
}

std::vector<Texture>
Model::loadMaterialTextures(aiMaterial *material, aiTextureType type, std::string typeName) {
    std::vector<Texture> textures;

    unsigned int textCount = material->GetTextureCount(type);
    for (unsigned int i = 0; i < textCount; i++) {
        aiString str;
        material->GetTexture(type, i, &str);
        Texture texture = loadTextureFromFile(directory + "/" + str.C_Str(), typeName);
        textures.push_back(std::move(texture));
    }

    return textures;
}