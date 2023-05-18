#include "gpu_context.hpp"

#include "../mesh/mesh.inl"

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_NATIVE_INCLUDE_NONE
using HWND = void *;
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#include <GLFW/glfw3native.h>

#include "../scene/scene.inl"

// Not needed, this is set by cmake.
// Intellisense doesnt get it, so this prevents it from complaining.
#if !defined(DAXA_SHADER_INCLUDE_DIR)
#define DAXA_SHADER_INCLUDE_DIR "."
#endif

GPUContext::GPUContext(Window const &window)
    : context{daxa::create_context({.enable_validation = false})},
      device{this->context.create_device({.name = "Sandbox Device"})},
      swapchain{this->device.create_swapchain({
          .native_window = glfwGetWin32Window(window.glfw_handle),
          .native_window_platform = daxa::NativeWindowPlatform::WIN32_API,
          .surface_format_selector = [&](daxa::Format format) -> i32
          {
                switch (format)
                {
                case daxa::Format::R8G8B8A8_UNORM: return 80;
                case daxa::Format::B8G8R8A8_UNORM: return 60;
                default: return 0;
                }
          },
          .image_usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE,
          .present_mode = daxa::PresentMode::IMMEDIATE,
          .name = "Sandbox Swapchain",
      })},
      pipeline_manager{daxa::PipelineManager{{
          .device = this->device,
          .shader_compile_options = daxa::ShaderCompileOptions{
              .root_paths = {
                  "./shaders",
                  DAXA_SHADER_INCLUDE_DIR,
              },
              .write_out_preprocessed_code = "./preproc",
              .language = daxa::ShaderLanguage::GLSL,
              .enable_debug_info = true,
          },
          .name = "Sandbox PipelineCompiler",
      }}},
      transient_mem{{
          .device = this->device,
          .capacity = 4096,
          .name = "transient memory pool",
      }},
      shader_globals_buffer{this->device.create_buffer({
        .size = (sizeof(ShaderGlobalsBlock) * 4 + 64 - 1) / 64 * 64,
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE | daxa::MemoryFlagBits::DEDICATED_MEMORY,
        .name = "globals",
      })}
{
}

GPUContext::~GPUContext()
{
    device.destroy_buffer(shader_globals_buffer);
}