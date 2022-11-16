#pragma once

// Standart headers:
// Library headers:
// Project headers:
#include "../gpu_context.hpp"
#include "../window.hpp"
#include "tasks/triangle.hpp"
#include "../mesh/mesh.inl"

// Renderer struct.
// This should idealy handle all rendering related information and functionality.
struct Renderer
{
    Renderer(Window const& window);
    ~Renderer();

    void compile_pipelines();
    void window_resized(Window const& window);
    auto create_main_task_list() -> daxa::TaskList;
    void render_frame(Window const& window);

    RenderContext context;
    daxa::TaskList main_task_list;
};