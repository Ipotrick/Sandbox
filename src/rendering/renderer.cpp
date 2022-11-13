#include "renderer.hpp"

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_NATIVE_INCLUDE_NONE
using HWND = void *;
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#include <GLFW/glfw3native.h>

Renderer::Renderer(Window const &window)
{
    this->context.context = daxa::create_context({});
    this->context.device = this->context.context.create_device({
        .debug_name = "Sandbox Device",
    });
    this->context.swapchain = this->context.device.create_swapchain({
        .native_window = glfwGetWin32Window(window.glfw_handle),
        .native_window_platform = daxa::NativeWindowPlatform::WIN32_API,
        .debug_name = "Sandbox Swapchain",
    });
    this->context.pipeline_compiler = this->context.device.create_pipeline_compiler({
        .shader_compile_options = daxa::ShaderCompileOptions{
            .root_paths = {
                std::filesystem::path{"./shaders"},
            },
            .language = daxa::ShaderLanguage::GLSL,
        },
        .debug_name = "Sandbox PipelineCompiler",
    });
}

Renderer::~Renderer()
{
}

void Renderer::compile_pipelines()
{
    this->context.triangle_pipe =
        this->context.pipeline_compiler.create_raster_pipeline(TRIANGLE_TASK_RASTER_PIPE_INFO).value();
}

auto Renderer::create_main_task_list() -> daxa::TaskList
{
    using namespace daxa;
    TaskList task_list{{
        .device = this->context.device,
        .swapchain = this->context.swapchain,
        .debug_name = "Sandbox main TaskList",
    }};

    TaskImageId t_swapchain_image = task_list.create_task_image({
        .swapchain_image = true,
        .debug_name = "Sandbox main Tasklist Swapchain Task Image",
    });

    t_draw_triangle({
        .task_list = task_list,
        .context = this->context,
        .t_swapchain_image = t_swapchain_image,
    });

    task_list.submit({});
    task_list.present({});
    return task_list;
}