#pragma once

#include "../window.hpp"
#include "../mesh/mesh.inl"
#include "../scene/scene.hpp"

#include "gpu_context.hpp"

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
    Renderer(Window *window, GPUContext *context, Scene *scene, AssetManager *asset_manager);
    ~Renderer();

    void compile_pipelines();
    void recreate_resizable_images();
    void window_resized();
    auto create_main_task_list() -> daxa::TaskList;
    void render_frame(CameraInfo const &camera_info);

    daxa::TaskBuffer entity_meta = {};
    daxa::TaskBuffer entity_transforms = {};
    daxa::TaskBuffer entity_combined_transforms = {};
    daxa::TaskBuffer entity_first_children = {};
    daxa::TaskBuffer entity_next_silbings = {};
    daxa::TaskBuffer entity_parents = {};
    daxa::TaskBuffer entity_meshlists = {};

    daxa::TaskBuffer globals = {};
    daxa::TaskBuffer instanciated_meshlets = {};
    daxa::TaskBuffer index_buffer = {};
    daxa::TaskBuffer ent_meshlet_count_prefix_sum_buffer = {};
    daxa::TaskBuffer ent_meshlet_count_partial_sum_buffer = {};
    daxa::TaskBuffer draw_opaque_id_info_buffer = {};

    std::vector<daxa::TaskBuffer> buffers = {};

    // Render Targets:
    daxa::TaskImage swapchain_image = {};
    daxa::TaskImage depth = {};

    std::vector<daxa::TaskImage> images = {};

    Window *window = {};
    GPUContext *context = {};
    Scene *scene = {};
    AssetManager *asset_manager = {};
    daxa::TaskList main_task_list;
    daxa::CommandSubmitInfo submit_info = {};
};