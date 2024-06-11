#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <iostream>
#include <chrono>

#include "Asset/Mesh.hpp"
#include "Asset/Model.hpp"
#include "Camera.hpp"
#include "Math/Vec.hpp"
#include "Pipeline/Rasterizer.hpp"
#include "Controller.hpp"

int main() {
    if (!glfwInit()) return -1;
    GLFWwindow *window = glfwCreateWindow(1920, 1080, "cudarenderer", nullptr, nullptr);
    if (window == nullptr) {
        std::cout << "failed to crete window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "failed to initialize glad!" << std::endl;
        return -1;
    }

    Rasterizer r{1920, 1080};
    r.setMSAA(2);
    Mesh  mesh{{Vertex{Vec3(-1.0f, 0.0f, -5.0f), Vec3(0.0f, 0.0f, 0.0f), Vec2(0.0f, 0.0f)},
                Vertex{Vec3(1.0f, 0.0f, -5.0f), Vec3(0.0f, 0.0f, 0.0f), Vec2(0.0f, 0.0f)},
                Vertex{Vec3(0.0f, 1.0f, -5.0f), Vec3(0.0f, 0.0f, 0.0f), Vec2(0.0f, 0.0f)}},
               {0, 1, 2},
               {}};
    Model model;
    model.pushMesh(mesh);
    // int modelIdx = r.loadModel("models/nanosuit/nanosuit.obj");
    int modelIdx = r.loadModel(model);
    r.setBackground({1.0f, 1.0f, 1.0f});

    Camera camera{Vec3(0.0f, 0.0f, 0.0f),
                  Vec3(0.0f, 0.0f, -1.0f),
                  Vec3(0.0f, 1.0f, 0.0f),
                  1920,
                  1080,
                  0.1f,
                  100.0f};

    std::chrono::steady_clock::time_point begin  = std::chrono::steady_clock::now();
    unsigned int                          frames = 0;
    while (!glfwWindowShouldClose(window)) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() >= 1.0) {
            std::cout << "FPS: " << frames << std::endl;
            frames = 0;
            begin  = end;
        }

        glfwPollEvents();
        key_input(window);

        const Vec3 *pixelData = r.render(camera, modelIdx);
        /*
         * pixelData mem layout
         * |             |
         * |             |
         * | 6 ...       |
         * | 0 1 2 3 4 5 |
         */
        glDrawPixels(1920, 1080, GL_RGB, GL_FLOAT, pixelData);
        glfwSwapBuffers(window);

        frames++;
    }

    glfwTerminate();

    return 0;
}