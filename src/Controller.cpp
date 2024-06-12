#include "Controller.hpp"
#include "Camera.hpp"

void key_input(GLFWwindow *window, Camera &camera) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    float cameraSpeed = 0.05f;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) { cameraSpeed *= 0.1f; }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        Vec3 offset = camera.getFront() * cameraSpeed;
        camera.move(offset);
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        Vec3 offset = camera.getFront() * -cameraSpeed;
        camera.move(offset);
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        Vec3 offset = camera.getRight() * -cameraSpeed;
        camera.move(offset);
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        Vec3 offset = camera.getRight() * cameraSpeed;
        camera.move(offset);
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        Vec3 offset = camera.getUp() * cameraSpeed;
        camera.move(offset);
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
        Vec3 offset = camera.getUp() * -cameraSpeed;
        camera.move(offset);
    }
}