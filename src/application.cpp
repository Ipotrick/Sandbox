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
    auto inf_depth_reverse_z_perspective = [](auto fov_rads, auto aspect, auto zNear)
    {
		assert(abs(aspect - std::numeric_limits<f32>::epsilon()) > 0.0f);

		f32 const tanHalfFovy = 1.0f / std::tan(fov_rads * 0.5f);

		glm::mat4x4 ret(0.0f);
		ret[0][0] = tanHalfFovy / aspect;
		ret[1][1] = tanHalfFovy;
        ret[2][2] = 0.0f;
        ret[2][3] = -1.0f;
        ret[3][2] = zNear;
		return ret;
    };
    glm::mat4 prespective = inf_depth_reverse_z_perspective(glm::radians(fov), f32(window.get_width()) / f32(window.get_height()), near);
    prespective[1][1] *= -1.0f;
    this->cam_info.proj = prespective;
    this->cam_info.view = glm::lookAt(position, position + forward, up);
    this->cam_info.vp = this->cam_info.proj * this->cam_info.view;
    this->cam_info.pos = this->position;
    this->cam_info.up = this->up;
    glm::vec3 ws_ndc_corners[2][2][2];
    glm::mat4 inv_view_proj = glm::inverse(this->cam_info.proj * this->cam_info.view);
    for (u32 z = 0; z < 2; ++z)
    {
        for (u32 y = 0; y < 2; ++y)
        {
            for (u32 x = 0; x < 2; ++x)
            {
                glm::vec3 corner = glm::vec3((glm::vec2(x,y) - 0.5f) * 2.0f, 1.0f - z * 0.5f);
                glm::vec4 proj_corner = inv_view_proj * glm::vec4(corner, 1);
                ws_ndc_corners[x][y][z] = glm::vec3(proj_corner) / proj_corner.w;
            }
        }
    }
    this->cam_info.camera_near_plane_normal = glm::normalize(glm::cross(ws_ndc_corners[0][1][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0]));
    this->cam_info.camera_right_plane_normal = glm::normalize(glm::cross(ws_ndc_corners[1][1][0] - ws_ndc_corners[1][0][0], ws_ndc_corners[1][0][1] - ws_ndc_corners[1][0][0]));
    this->cam_info.camera_left_plane_normal = glm::normalize(glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][0][1], ws_ndc_corners[0][0][0] - ws_ndc_corners[0][0][1]));
    this->cam_info.camera_top_plane_normal = glm::normalize(glm::cross(ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[0][0][1] - ws_ndc_corners[0][0][0]));
    this->cam_info.camera_bottom_plane_normal = glm::normalize(glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][1][0], ws_ndc_corners[1][1][0] - ws_ndc_corners[0][1][0]));
    int i = 0;
}
 
Application::Application()
    : window{3840, 2160, "sandbox"},
      gpu_context{this->window},
      asset_manager{this->gpu_context.device},
      scene{},
      renderer{&(this->window), &(this->gpu_context), &(this->scene), &(this->asset_manager)}
{
    this->scene_loader = SceneLoader{"./assets/"};
    this->scene_loader.load_entities_from_fbx(this->scene, this->asset_manager, "Bistro_v5_2/BistroExterior.fbx"); // "Bistro_v5_2/BistroExterior.fbx" "small_city.glb"
    this->scene.process_transforms();
    auto cmd = this->asset_manager.get_update_commands().value();
    auto cmd2 = this->gpu_context.device.create_command_list({});
    this->scene.record_full_entity_update(
        this->gpu_context.device, 
        cmd2, 
        this->scene, 
        this->renderer.entity_meta.get_state().buffers[0],
        this->renderer.entity_transforms.get_state().buffers[0],
        this->renderer.entity_combined_transforms.get_state().buffers[0],
        this->renderer.entity_first_children.get_state().buffers[0],
        this->renderer.entity_next_silbings.get_state().buffers[0],
        this->renderer.entity_parents.get_state().buffers[0],
        this->renderer.entity_meshlists.get_state().buffers[0]);
    cmd2.pipeline_barrier({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::READ,
    });
    cmd2.complete();
    this->gpu_context.device.submit_commands({
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
        this->delta_time = std::chrono::duration_cast<FpMilliseconds>(new_time_point - this->last_time_point).count() * 0.001f;
        this->last_time_point = new_time_point;
        window.update(delta_time);
        keep_running &= !static_cast<bool>(glfwWindowShouldClose(this->window.glfw_handle));
        i32vec2 new_window_size;
        glfwGetWindowSize(this->window.glfw_handle, &new_window_size.x, &new_window_size.y);
        if (this->window.size.x != new_window_size.x || this->window.size.y != new_window_size.y)
        {
            this->window.size = new_window_size;
            renderer.window_resized();
        }
        this->update();
        this->renderer.render_frame(this->camera_controller.cam_info, this->observer_camera_controller.cam_info, delta_time);
    }
    return 0;
}

void Application::update()
{
    if (this->window.size.x == 0 || this->window.size.y == 0)
    {
        return;
    }
    if (control_observer)
    {
        observer_camera_controller.process_input(this->window, this->delta_time);
        observer_camera_controller.update_matrices(this->window);
    }
    else
    {
        camera_controller.process_input(this->window, this->delta_time);
        camera_controller.update_matrices(this->window);
    }
    if (window.key_just_pressed(GLFW_KEY_H))
    {
        std::cout << "switched enable_observer from " << renderer.context->settings.enable_observer << " to " << !(renderer.context->settings.enable_observer) << std::endl;
        renderer.context->settings.enable_observer = !renderer.context->settings.enable_observer;
    }
    if (window.key_just_pressed(GLFW_KEY_J))
    {
        std::cout << "switched control_observer from " << control_observer << " to " << !(control_observer) << std::endl;
        control_observer = !control_observer;
    }
    if (window.key_just_pressed(GLFW_KEY_K))
    {
        std::cout << "reset observer" << std::endl;
        control_observer = false;
        renderer.context->settings.enable_observer = false;
        observer_camera_controller = camera_controller;
    }
    #if COMPILE_IN_MESH_SHADER
    if (window.key_just_pressed(GLFW_KEY_M))
    {
        std::cout << "switched enable_mesh_shader from " << renderer.context->settings.enable_mesh_shader << " to " << !(renderer.context->settings.enable_mesh_shader) << std::endl;
        renderer.context->settings.enable_mesh_shader = !renderer.context->settings.enable_mesh_shader;
    }
    #endif
    if (window.key_just_pressed(GLFW_KEY_O))
    {
        std::cout << "switched observer_show_pass from " << renderer.context->settings.observer_show_pass << " to " << ((renderer.context->settings.observer_show_pass + 1) % 3) << std::endl;
        renderer.context->settings.observer_show_pass = (renderer.context->settings.observer_show_pass + 1) % 3;
    }
}

Application::~Application()
{
}