#include "gpu_context.hpp"

#include "../mesh/mesh.inl"
#include "../../shaders/util.inl"

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
      globals_buffer{.id = this->device.create_buffer({
                         .size = sizeof(ShaderGlobals),
                         .name = "globals_buffer",
                     })},
      entity_meta_data{.id = this->device.create_buffer({
                           .size = sizeof(EntityMetaData),
                           .name = "entity_meta_data",
                       })},
      entity_transforms{.id = this->device.create_buffer({
                            .size = sizeof(daxa_f32mat4x4) * MAX_ENTITY_COUNT,
                            .name = "entity_transforms",
                        })},
      entity_combined_transforms{.id = this->device.create_buffer({
                                     .size = sizeof(daxa_f32mat4x4) * MAX_ENTITY_COUNT,
                                     .name = "entity_combined_transforms",
                                 })},
      entity_first_children{.id = this->device.create_buffer({
                                .size = sizeof(EntityId) * MAX_ENTITY_COUNT,
                                .name = "entity_first_children",
                            })},
      entity_next_silbings{.id = this->device.create_buffer({
                               .size = sizeof(EntityId) * MAX_ENTITY_COUNT,
                               .name = "entity_next_silbings",
                           })},
      entity_parents{.id = this->device.create_buffer({
                         .size = sizeof(EntityId) * MAX_ENTITY_COUNT,
                         .name = "entity_parents",
                     })},
      entity_meshlists{.id = this->device.create_buffer({
                           .size = sizeof(MeshList) * MAX_ENTITY_COUNT,
                           .name = "entity_meshlists",
                       })},
      instanciated_meshlets{.id = this->device.create_buffer({
                                .size = sizeof(MeshletDrawInfo) * MAX_DRAWN_MESHLETS + /*reserved space for a counter*/ 16,
                                .name = "instanciated_meshlets",
                            })},
      index_buffer{.id = this->device.create_buffer({
                       .size = TRIANGLE_SIZE * MAX_DRAWN_TRIANGLES + /*reserved space for a counter*/ 16,
                       .name = "index_buffer",
                   })},
      ent_meshlet_count_prefix_sum_buffer{.id = this->device.create_buffer({
                                              .size = static_cast<u32>(sizeof(u32)) *
                                                      round_up_to_multiple(MAX_ENTITY_COUNT, PREFIX_SUM_WORKGROUP_SIZE),
                                              .name = "ent_meshlet_count_prefix_sum_buffer",
                                          })},
      ent_meshlet_count_partial_sum_buffer{.id = this->device.create_buffer({
                                               .size = round_up_to_multiple(round_up_div((sizeof(u32) * MAX_ENTITY_COUNT), PREFIX_SUM_WORKGROUP_SIZE), PREFIX_SUM_WORKGROUP_SIZE),
                                               .name = "ent_meshlet_count_prefix_sum_buffer",
                                           })},
        draw_opaque_id_info_buffer{

        }
{
}

GPUContext::~GPUContext()
{
    this->device.destroy_buffer(this->ent_meshlet_count_partial_sum_buffer.id);
    this->device.destroy_buffer(this->ent_meshlet_count_prefix_sum_buffer.id);
    this->device.destroy_buffer(this->globals_buffer.id);
    this->device.destroy_buffer(this->entity_meta_data.id);
    this->device.destroy_buffer(this->entity_transforms.id);
    this->device.destroy_buffer(this->entity_combined_transforms.id);
    this->device.destroy_buffer(this->entity_first_children.id);
    this->device.destroy_buffer(this->entity_next_silbings.id);
    this->device.destroy_buffer(this->entity_parents.id);
    this->device.destroy_buffer(this->entity_meshlists.id);
    this->device.destroy_buffer(this->instanciated_meshlets.id);
    this->device.destroy_buffer(this->index_buffer.id);
    this->device.destroy_image(this->depth_image.id);
}