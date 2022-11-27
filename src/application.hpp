#pragma once

// Standart headers:
#include <chrono>
// Library headers:
// Project headers:
#include "sandbox.hpp"
#include "window.hpp"
#include "rendering/renderer.hpp"
#include "scene/scene.hpp"

struct CameraController {
    void process_input(Window& window, f32 dt);
    void update_matrices(Window& window);

    CameraInfo cam_info;
    
    bool bZoom = false; 
    f32 fov = 74.0f;
    f32 near = 0.01f;
    f32 far = 2000.0f;
    f32 cameraSwaySpeed = 0.05f;
    f32 translationSpeed = 0.01f;
    glm::vec3 up = { 0.f, 0.f, 1.0f };
    glm::vec3 forward = { 0.f, 0.f, 0.f };
    glm::vec3 position = { 0.f, 1.f, 0.f };
    f32 yaw = 0.0f;
    f32 pitch = 0.0f;
};

struct Application
{
    Application();
    ~Application();
    auto run() -> i32;
    void update();

    Window window;
    Renderer renderer;
    AssetManager asset_manager;
    Scene scene;
    SceneLoader scene_loader;
    CameraController camera_controller;

    bool keep_running{ true };
    f32 delta_time{ 0.016666f };
    std::chrono::time_point<std::chrono::steady_clock> last_time_point = {};
};