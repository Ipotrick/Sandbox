#include "application.hpp"

void CameraController::process_input(Window &window, f32 dt)
{
    f32 speed = window.key_pressed(GLFW_KEY_LEFT_SHIFT) ? translationSpeed * 4.0f : translationSpeed;
    speed = window.key_pressed(GLFW_KEY_LEFT_CONTROL) ? speed * 0.25f : speed;

    if (window.is_focused())
    {
        if (window.key_just_pressed(GLFW_KEY_ESCAPE))
        {
            if (window.is_cursor_captured())
            {
                window.release_cursor();
            }
            else
            {
                window.capture_cursor();
            }
        }
    }
    else if (window.is_cursor_captured())
    {
        window.release_cursor();
    }

    auto cameraSwaySpeed = this->cameraSwaySpeed;
    if (window.key_pressed(GLFW_KEY_C))
    {
        cameraSwaySpeed *= 0.25;
        bZoom = true;
    }
    else
    {
        bZoom = false;
    }

    glm::vec3 right = glm::cross(forward, up);
    glm::vec3 fake_up = glm::cross(right, forward);
    if (window.is_cursor_captured())
    {
        if (window.key_pressed(GLFW_KEY_W))
        {
            position += forward * speed * dt;
        }
        if (window.key_pressed(GLFW_KEY_S))
        {
            position -= forward * speed * dt;
        }
        if (window.key_pressed(GLFW_KEY_A))
        {
            position -= glm::normalize(glm::cross(forward, up)) * speed * dt;
        }
        if (window.key_pressed(GLFW_KEY_D))
        {
            position += glm::normalize(glm::cross(forward, up)) * speed * dt;
        }
        if (window.key_pressed(GLFW_KEY_SPACE))
        {
            position += fake_up * speed * dt;
        }
        if (window.key_pressed(GLFW_KEY_LEFT_ALT))
        {
            position -= fake_up * speed * dt;
        }
        pitch += window.get_cursor_change_y() * cameraSwaySpeed;
        pitch = std::clamp(pitch, -85.0f, 85.0f);
        yaw += window.get_cursor_change_x() * cameraSwaySpeed;
    }
    forward.x = -glm::cos(glm::radians(yaw - 90.0f)) * glm::cos(glm::radians(pitch));
    forward.y = glm::sin(glm::radians(yaw - 90.0f)) * glm::cos(glm::radians(pitch));
    forward.z = -glm::sin(glm::radians(pitch));
}

void CameraController::update_matrices(Window &window)
{
    auto fov = this->fov;
    if (bZoom)
    {
        fov *= 0.25f;
    }
    glm::mat4 prespective = glm::perspective(glm::radians(fov), (f32)window.get_width() / (f32)window.get_height(), near, far);
    prespective[1][1] *= -1.0f;
    this->cam_info.proj = prespective;
    this->cam_info.view = glm::lookAt(position, position + forward, up);
    this->cam_info.vp = this->cam_info.proj * this->cam_info.view;
}

Application::Application()
    : window{400, 300, "sandbox"}, renderer{window}, asset_manager{renderer.context.device},
      scene{}
{
    this->renderer.compile_pipelines();
    this->scene_loader = SceneLoader{"./assets/"};
    this->scene_loader.load_entities_from_fbx(this->scene, this->asset_manager, "Bistro_v5_2/BistroExterior.fbx");
    //this->scene.set_combined_transforms();
    auto cmd = this->asset_manager.get_update_commands().value();
    auto cmd2 = this->renderer.context.device.create_command_list({});
    this->scene.record_full_entity_update(this->renderer.context.device, cmd2, this->scene, this->renderer.context.entity_data_buffer.id);
    cmd2.pipeline_barrier({
        .awaited_pipeline_access = daxa::AccessConsts::TRANSFER_WRITE,
        .waiting_pipeline_access = daxa::AccessConsts::READ,
    });
    cmd2.complete();
    this->renderer.context.device.submit_commands({
        .command_lists = {std::move(cmd), std::move(cmd2)},
    });
    last_time_point = std::chrono::steady_clock::now();
}
using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;

auto Application::run() -> i32
{
    while (keep_running)
    {
        auto new_time_point = std::chrono::steady_clock::now();
        this->delta_time = std::chrono::duration_cast<FpMilliseconds>(new_time_point - this->last_time_point).count();
        this->last_time_point = new_time_point;
        window.update(delta_time);
        keep_running &= !static_cast<bool>(glfwWindowShouldClose(this->window.glfw_handle));
        i32vec2 new_window_size;
        glfwGetWindowSize(this->window.glfw_handle, &new_window_size.x, &new_window_size.y);
        if (this->window.size.x != new_window_size.x || this->window.size.y != new_window_size.y)
        {
            this->window.size = new_window_size;
            renderer.window_resized(this->window);
        }
        this->update();
        this->renderer.context.pipeline_manager.reload_all();
        this->renderer.render_frame(this->window, this->camera_controller.cam_info);
    }
    return 0;
}

void Application::update()
{
    if (this->window.size.x == 0 || this->window.size.y == 0)
    {
        return;
    }
    camera_controller.process_input(this->window, this->delta_time);
    camera_controller.update_matrices(this->window);
}

Application::~Application()
{
}