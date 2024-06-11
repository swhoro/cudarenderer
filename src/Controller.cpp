#include "Controller.hpp"

void key_input(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) { glfwSetWindowShouldClose(window, true); }
    // float cameraSpeed = 0.05f;
    // if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) { cameraSpeed *= 0.1f; }
    // if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    //     glm::vec3 offset = cameraSpeed * camera.cameraFront;
    //     camera.move(offset);
    // }
    // if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    //     glm::vec3 offset = -cameraSpeed * camera.cameraFront;
    //     camera.move(offset);
    // }
    // if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    //     glm::vec3 offset = -camera.getRight() * cameraSpeed;
    //     camera.move(offset);
    // }
    // if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    //     glm::vec3 offset = camera.getRight() * cameraSpeed;
    //     camera.move(offset);
    // }
    // if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
    //     glm::vec3 offset = camera.cameraUp * cameraSpeed;
    //     camera.move(offset);
    // }
    // if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
    //     glm::vec3 offset = -camera.cameraUp * cameraSpeed;
    //     camera.move(offset);
    // }
}