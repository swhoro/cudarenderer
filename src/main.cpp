#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <iostream>
#include <chrono>

#include "Camera.hpp"
#include "Math/Vec.hpp"
#include "Pipeline/Rasterizer/Rasterizer.hpp"
#include "Controller.hpp"

#define WIDTH 800
#define HEIGHT 600

int main() {
    if (!glfwInit()) return -1;
    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "cudarenderer", nullptr, nullptr);
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

    Rasterizer r{WIDTH, HEIGHT};
    // r.setMSAA(2);
    // Mesh  triangle{{Vertex{Vec3(-1.0f, 0.0f, -5.0f), Vec3(), Vec2()},
    //                 Vertex{Vec3(1.0f, 0.0f, -5.0f), Vec3(), Vec2()},
    //                 Vertex{Vec3(0.0f, 1.0f, -5.0f), Vec3(), Vec2()}},
    //                {0, 1, 2},
    //                {}};
    // Mesh  square{{
    //                 Vertex{Vec3(-1.0f, -1.0f, -5.0f), Vec3(), Vec2()},
    //                 Vertex{Vec3(-1.0f, -3.0f, -5.0f), Vec3(), Vec2()},
    //                 Vertex{Vec3(1.0f, -3.0f, -5.0f), Vec3(), Vec2()},
    //                 Vertex{Vec3(1.0f, -1.0f, -5.0f), Vec3(), Vec2()},
    //             },
    //              {0, 1, 2, 0, 2, 3},
    //              {}};
    // Model model;
    // model.pushMesh(triangle);
    // model.pushMesh(square);
    int modelIdx = r.loadModel("models/nanosuit/nanosuit.obj");
    // int modelIdx = r.loadModel(model);
    
    Camera camera{Vec3(0.0f, 0.0f, 0.0f),
                  Vec3(0.0f, 0.0f, -1.0f),
                  Vec3(0.0f, 1.0f, 0.0f),
                  WIDTH,
                  HEIGHT,
                  0.1f,
                  100.0f};

    std::chrono::steady_clock::time_point begin  = std::chrono::steady_clock::now();
    unsigned int                          frames = 0;
    while (!glfwWindowShouldClose(window)) {
        r.setBackground({1.0f, 1.0f, 1.0f});

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() >= 1.0) {
            std::cout << "FPS: " << frames << std::endl;
            frames = 0;
            begin  = end;
        }

        glfwPollEvents();
        key_input(window, camera);

        const Vec3 *pixelData = r.render(camera, modelIdx);
        /*
         * pixelData mem layout
         * |             |
         * |             |
         * | 6 ...       |
         * | 0 1 2 3 4 5 |
         */
        glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, pixelData);
        glfwSwapBuffers(window);

        frames++;
    }

    glfwTerminate();

    return 0;
}