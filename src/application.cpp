#include "application.hpp"


void CameraController::process_input(Window& window, f32 dt) {
    f32 speed = window.key_pressed(GLFW_KEY_LEFT_SHIFT) ? translationSpeed * 4.0f : translationSpeed;
    if (window.is_cursor_captured()) {
        if (window.key_just_pressed(GLFW_KEY_ESCAPE)) {
            window.release_cursor();
        }
    } else {
        if (window.button_just_pressed(GLFW_MOUSE_BUTTON_LEFT) && window.is_cursor_over_window()) {
            window.capture_cursor();
        }
    }

    if (window.key_pressed(GLFW_KEY_2)) {
        printf("dingus\n");
    }

    auto cameraSwaySpeed = this->cameraSwaySpeed;
    if (window.key_pressed(GLFW_KEY_C)) {
        cameraSwaySpeed *= 0.25;
        bZoom = true;
    } else {
        bZoom = false;
    }

    auto yawRotaAroundUp = glm::rotate(glm::mat4(1.0f), yaw, {0.f,1.f,0.f});
    auto pitchRotation = glm::rotate(glm::mat4(1.0f), pitch, glm::vec3{1.f,0.f,0.f});
    glm::vec4 translation = {};
    if (window.is_cursor_captured()) {
        if (window.key_pressed(GLFW_KEY_W)) {
            glm::vec4 direction = { 0.0f, 0.0f, -1.0f, 0.0f };
            translation += yawRotaAroundUp * pitchRotation * direction * dt * speed;
        }
        if (window.key_pressed(GLFW_KEY_S)) {
            glm::vec4 direction = { 0.0f, 0.0f, 1.0f, 0.0f };
            translation += yawRotaAroundUp * pitchRotation * direction * dt * speed;
        }
        if (window.key_pressed(GLFW_KEY_A)) {
            glm::vec4 direction = { 1.0f, 0.0f, 0.0f, 0.0f };
            translation += yawRotaAroundUp * direction * dt * speed;
        }
        if (window.key_pressed(GLFW_KEY_D)) {
            glm::vec4 direction = { -1.0f, 0.0f, 0.0f, 0.0f };
            translation += yawRotaAroundUp * direction * dt * speed;
        }
        if (window.key_pressed(GLFW_KEY_SPACE)) {
            translation += yawRotaAroundUp * pitchRotation * glm::vec4{ 0.f,  1.f, 0.f, 0.f } * dt * speed;
        }
        if (window.key_pressed(GLFW_KEY_LEFT_CONTROL)) {
            translation += yawRotaAroundUp * pitchRotation * glm::vec4{ 0.f, -1.f,  0.f, 0.f } * dt * speed;
        }
        pitch -= window.get_cursor_change_y() * cameraSwaySpeed;
        pitch = std::clamp(pitch, -0.5f * glm::pi<f32>(), 0.5f * glm::pi<f32>());
        yaw += window.get_cursor_change_x() * cameraSwaySpeed;
    }
    position += translation;
}

void CameraController::update_matrices(Window& window) {
    auto fov = this->fov;
    if (bZoom) {
        fov *= 0.25f;
    }
    auto yawRotaAroundUp = glm::rotate(glm::mat4(1.0f), yaw, {0.f,1.f,0.f});
    auto pitchRotation = glm::rotate(glm::mat4(1.0f), pitch, glm::vec3{1.f,0.f,0.f});
    glm::mat4 prespective = glm::perspective(fov, (f32)window.get_width()/(f32)window.get_height(), near, far);
    auto rota = yawRotaAroundUp * pitchRotation;
    auto cameraModelMat = glm::translate(glm::mat4(1.0f), {position.x, position.y, position.z}) * rota;
    glm::mat4 view = glm::inverse(cameraModelMat);
    this->cam_info.proj = prespective;
    this->cam_info.view = view;
    this->cam_info.vp = this->cam_info.proj * this->cam_info.view;
}

Application::Application()
    : window{ 400, 300, "sandbox" }
    , renderer{ window }
    , asset_manager{ renderer.context.device }
{
    std::cout << "Application::Application" << std::endl;
    this->renderer.compile_pipelines();
    this->scene_loader = SceneLoader{ "./assets/" };
    scene_loader.load_entities_from_fbx(this->scene, this->asset_manager, "Bistro_v5_2/BistroExterior.fbx");
    last_time_point = std::chrono::steady_clock::now();
}
using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;

auto Application::run() -> i32
{
    std::cout << "Application::run" << std::endl;
    while(keep_running)
    {
        auto new_time_point = std::chrono::steady_clock::now();
        this->delta_time = std::chrono::duration_cast<FpMilliseconds>(new_time_point - this->last_time_point).count();
        this->last_time_point = new_time_point;
        std::cout << "Application::run::loop" << std::endl;
        glfwPollEvents();
        keep_running &= !static_cast<bool>(glfwWindowShouldClose(this->window.glfw_handle));
        i32vec2 new_window_size;
        glfwGetWindowSize(this->window.glfw_handle, &new_window_size.x, &new_window_size.y);
        if (this->window.size.x != new_window_size.x || this->window.size.y != new_window_size.y)
        {
            this->window.size = new_window_size;
            renderer.window_resized(this->window);
        }
        this->update();
        this->renderer.hotload_pipelines();
        this->renderer.render_frame(this->window, this->camera_controller.cam_info);
    }
    return 0;
}

void Application::update()
{
    std::cout << "update: delta time: " << this->delta_time << std::endl;
    if (this->window.size.x == 0 || this->window.size.y == 0)
    {
        return;
    }
    camera_controller.process_input(this->window, this->delta_time);
    camera_controller.update_matrices(this->window);
}

Application::~Application()
{
    std::cout << "Application::~Application" << std::endl;
}