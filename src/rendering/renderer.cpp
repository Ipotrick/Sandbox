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
                "./shaders",
                DAXA_SHADER_INCLUDE_DIR,
            },
            .language = daxa::ShaderLanguage::GLSL,
        },
        .debug_name = "Sandbox PipelineCompiler",
    });

    this->main_task_list =  this->create_main_task_list();
}

Renderer::~Renderer()
{
}

void Renderer::compile_pipelines()
{
    auto compilation_result = this->context.pipeline_compiler.create_raster_pipeline(TRIANGLE_TASK_RASTER_PIPE_INFO);
    std::cout << compilation_result.to_string() << std::endl;
    this->context.triangle_pipe = compilation_result.value();
}

void Renderer::window_resized(Window const& window)
{
    if (window.size.x == 0 || window.size.y == 0)
    {
        return;
    }
    this->context.swapchain.resize();
}

auto Renderer::create_main_task_list() -> daxa::TaskList
{
    using namespace daxa;
    TaskList task_list{{
        .device = this->context.device,
        .swapchain = this->context.swapchain,
        .debug_name = "Sandbox main TaskList",
    }};

    this->context.t_swapchain_image = task_list.create_task_image({
        .swapchain_image = true,
        .debug_name = "Sandbox main Tasklist Swapchain Task Image",
    });
    task_list.add_runtime_image(context.t_swapchain_image, context.swapchain_image);

    t_draw_triangle({
        .task_list = task_list,
        .context = this->context,
        .t_swapchain_image = this->context.t_swapchain_image,
    });

    task_list.submit({});
    task_list.present({});
    return task_list;
}

void Renderer::render_frame(Window const& window)
{
    if (window.size.x == 0 || window.size.y == 0)
    {
        return;
    }

    main_task_list.remove_runtime_image(context.t_swapchain_image, context.swapchain_image);
    context.swapchain_image = context.swapchain.acquire_next_image();
    if (context.swapchain_image.is_empty())
    {
        return;
    }
    main_task_list.add_runtime_image(context.t_swapchain_image, context.swapchain_image);

    main_task_list.execute();
}