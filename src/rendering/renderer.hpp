#pragma once

// Standart headers:
// Library headers:
// Project headers:
#include "render_context.hpp"
#include "../window.hpp"
#include "tasks/triangle.hpp"

// Renderer struct.
// This should idealy handle all rendering related information and functionality.
struct Renderer
{
    Renderer(Window const& window);
    ~Renderer();

    void compile_pipelines();
    auto create_main_task_list() -> daxa::TaskList;

    RenderContext context;
    daxa::TaskList main_task_list;
};