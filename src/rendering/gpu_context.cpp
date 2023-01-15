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
    : context{daxa::create_context({})},
      device{this->context.create_device({.debug_name = "Sandbox Device"})},
      swapchain{this->device.create_swapchain({
          .native_window = glfwGetWin32Window(window.glfw_handle),
          .native_window_platform = daxa::NativeWindowPlatform::WIN32_API,
          .debug_name = "Sandbox Swapchain",
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
          .debug_name = "Sandbox PipelineCompiler",
      }}},
      transient_mem{{
          .device = this->device,
          .debug_name = "transient memory pool",
      }},
      globals_buffer{.id = this->device.create_buffer({
                         .size = sizeof(ShaderGlobals),
                         .debug_name = "globals_buffer",
                     })},
      entity_data_buffer{.id = this->device.create_buffer({
                             .size = sizeof(EntityData),
                             .debug_name = "entity_data_buffer",
                         })},
      instanciated_meshlets{.id = this->device.create_buffer({
                                .size = sizeof(MeshletDrawInfo) * MAX_DRAWN_MESHLETS + /*reserved space for a counter*/ 16,
                                .debug_name = "instanciated_meshlets",
                            })},
      index_buffer{.id = this->device.create_buffer({
                       .size = TRIANGLE_SIZE * MAX_DRAWN_TRIANGLES + /*reserved space for a counter*/ 16,
                       .debug_name = "index_buffer",
                   })},
      ent_meshlet_count_prefix_sum_buffer{.id = this->device.create_buffer({
                                              .size = static_cast<u32>(sizeof(u32)) * round_up_to_multiple(MAX_ENTITY_COUNT, 1024),
                                              .debug_name = "ent_meshlet_count_prefix_sum_buffer",
                                          })},
      ent_meshlet_count_partial_sum_buffer{.id = this->device.create_buffer({
                                               .size = round_up_to_multiple(round_up_div((sizeof(u32) * MAX_ENTITY_COUNT), 1024), 1024),
                                               .debug_name = "ent_meshlet_count_prefix_sum_buffer",
                                           })}
{
}

GPUContext::~GPUContext()
{
    this->device.destroy_buffer(this->ent_meshlet_count_partial_sum_buffer.id);
    this->device.destroy_buffer(this->ent_meshlet_count_prefix_sum_buffer.id);
    this->device.destroy_buffer(this->globals_buffer.id);
    this->device.destroy_buffer(this->entity_data_buffer.id);
    this->device.destroy_buffer(this->instanciated_meshlets.id);
    this->device.destroy_buffer(this->index_buffer.id);
    this->device.destroy_image(this->depth_image.id);
}