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

// Not needed, this is set by cmake.
// Intellisense doesnt get it, so this prevents it from complaining.
#if !defined(DAXA_SHADER_INCLUDE_DIR)
#define DAXA_SHADER_INCLUDE_DIR "."
#endif

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

    this->context.globals_buffer.id = this->context.device.create_buffer({
        .memory_flags = daxa::MemoryFlagBits::DEDICATED_MEMORY,
        .size = sizeof(ShaderGlobals),
        .debug_name = "globals",
    });

    this->main_task_list = this->create_main_task_list();
}

Renderer::~Renderer()
{
    this->context.device.wait_idle();
    this->context.device.collect_garbage();
}

void Renderer::compile_pipelines()
{
    auto compilation_result = this->context.pipeline_compiler.create_raster_pipeline(TRIANGLE_TASK_RASTER_PIPE_INFO);
    std::cout << compilation_result.to_string() << std::endl;
    this->context.triangle_pipe = compilation_result.value();
}

void Renderer::hotload_pipelines()
{
    if (this->context.pipeline_compiler.check_if_sources_changed(this->context.triangle_pipe))
    {
        auto result = this->context.pipeline_compiler.recreate_raster_pipeline(this->context.triangle_pipe);
        std::cout << result.to_string() << std::endl;
        if (result.is_ok())
        {
            this->context.triangle_pipe = result.value();
        }
    }
}

void Renderer::window_resized(Window const &window)
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

    this->context.swapchain_image.t_id = task_list.create_task_image({
        .swapchain_image = true,
        .debug_name = "Sandbox main Tasklist Swapchain Task Image",
    });
    task_list.add_runtime_image(context.swapchain_image.t_id, context.swapchain_image.id);

    this->context.globals_buffer.t_id = task_list.create_task_buffer({
        .debug_name = "Shader Globals TaskBuffer",
    });
    task_list.add_runtime_buffer(this->context.globals_buffer.t_id, this->context.globals_buffer.id);

    //task_list.add_task({
    //    .used_buffers = {
    //        {this->context.globals_buffer.t_id, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
    //    },
    //    .task = [&](daxa::TaskRuntime const &runtime)
    //    {
    //        auto cmd = runtime.get_command_list();
    //        auto staging_buffer = runtime.get_device().create_buffer({
    //            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
    //            .size = sizeof(ShaderGlobals),
    //            .debug_name = "ShaderGlobals staging buffer",
    //        });
    //        cmd.destroy_buffer_deferred(staging_buffer);
    //        *runtime.get_device().get_host_address_as<ShaderGlobals>(staging_buffer) = context.shader_globals;
    //        cmd.copy_buffer_to_buffer({
    //            .src_buffer = staging_buffer,
    //            .dst_buffer = this->context.globals_buffer.id,
    //            .size = sizeof(ShaderGlobals),
    //        });
    //    },
    //});

    t_draw_triangle({
        .task_list = task_list,
        .context = this->context,
        .t_swapchain_image = this->context.swapchain_image.t_id,
        .t_shader_globals = this->context.globals_buffer.t_id,
    });

    task_list.submit({});
    task_list.present({});
    return task_list;
}

void Renderer::render_frame(Window const &window, CameraInfo const & camera_info)
{
    if (window.size.x == 0 || window.size.y == 0)
    {
        return;
    }
    this->context.shader_globals.camera_view = *reinterpret_cast<f32mat4x4 const*>(&camera_info.view);
    this->context.shader_globals.camera_projection = *reinterpret_cast<f32mat4x4 const*>(&camera_info.proj);
    this->context.shader_globals.camera_view_projection = *reinterpret_cast<f32mat4x4 const*>(&camera_info.vp);

    main_task_list.remove_runtime_image(context.swapchain_image.t_id, context.swapchain_image.id);
    context.swapchain_image.id = context.swapchain.acquire_next_image();
    if (context.swapchain_image.id.is_empty())
    {
        return;
    }
    main_task_list.add_runtime_image(context.swapchain_image.t_id, context.swapchain_image.id);

    main_task_list.execute();
}