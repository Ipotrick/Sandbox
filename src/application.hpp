#pragma once

// Standart headers:
#include <chrono>
// Library headers:
// Project headers:
#include "sandbox.hpp"
#include "window.hpp"
#include "rendering/renderer.hpp"
#include "scene/scene.hpp"
#include "scene/asset_processor.hpp"
#include "ui.hpp"

struct CameraController
{
    void process_input(Window &window, f32 dt);
    void update_matrices(Window &window);

    CameraInfo cam_info;

    bool bZoom = false;
    f32 fov = 70.0f;
    f32 near = 0.1f;
    f32 cameraSwaySpeed = 0.05f;
    f32 translationSpeed = 10.0f;
    glm::vec3 up = {0.f, 0.f, 1.0f};
    glm::vec3 forward = {0.f, 0.f, 0.f};
    glm::vec3 position = {0.f, 1.f, 0.f};
    f32 yaw = 0.0f;
    f32 pitch = 0.0f;
};

struct Application
{
    Application();
    ~Application();

    auto run() -> i32;
    void update();

    /**
     * EXPLANATION: Why do we use unique pointers here?
     * Many of these members are non-movable.
     * They can NOT be made movable easily!
     * They require dependency injection between each other!
     * A pattern that solves these problems (non-movable + dep injection), is to wrap these structs in heap allocations and refer to them only with pointers.
     * We do NOT need shared_ptr here, as we know the lifetime of these members beforehand! The Application controls their lifetime!
     * If member B referes to member A, it will be below it in the struct delaration. This means it will be destroyed before A. This way we wont have dangling pointers.
     * There are no performance implications of this as these structs are VERY low frequency creation/deletion and are never copied.
     * This allows the construction of Application to be much simpler, less bug prone and makes Application movable!
     * A good rule of thumb is to have structs movable. If you need members that are not movable, simply wrap them in pointers.
     * WARNING: THIS CAN ONLY BE APPLIED LIKE THIS FOR LOW FREQUENCY TYPES, AS IT MIGHT INCUR PERFORMANCE PROBLEMS OTHERWISE!
    */
    std::unique_ptr<Window> _window = {};
    std::unique_ptr<GPUContext> _gpu_context = {};
    std::unique_ptr<Scene> _scene = {};
    std::unique_ptr<AssetProcessor> _asset_manager = {};
    std::unique_ptr<UIEngine> _ui_engine = {};
    std::unique_ptr<Renderer> _renderer = {};
    CameraController camera_controller = {};
    CameraController observer_camera_controller = {};
    bool control_observer = false;
    bool keep_running = true;
    f32 delta_time = 0.016666f;
    std::chrono::time_point<std::chrono::steady_clock> last_time_point = {};
};