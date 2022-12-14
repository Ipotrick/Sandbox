#pragma once

// Standart headers:
// Library headers:
// Project headers:
#include "gpu_context.hpp"
#include "../window.hpp"
#include "tasks/triangle.hpp"
#include "../mesh/mesh.inl"

struct CameraInfo
{
    glm::mat4 view = {};
    glm::mat4 proj = {};
    glm::mat4 vp = {};
};

// Renderer struct.
// This should idealy handle all rendering related information and functionality.
struct Renderer
{
    Renderer(Window const& window);
    ~Renderer();

    void compile_pipelines();
    void recreate_resizable_images(Window const& window);
    void window_resized(Window const& window);
    auto create_main_task_list() -> daxa::TaskList;
    void render_frame(Window const& window, CameraInfo const& camera_info);

    RenderContext context;
    daxa::TaskList main_task_list;
};