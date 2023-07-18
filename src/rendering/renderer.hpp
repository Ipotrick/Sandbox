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
    void recreate_framebuffer();
    void clear_select_buffers();
    void window_resized();
    auto create_main_task_list() -> daxa::TaskGraph;
    void render_frame(CameraInfo const &camera_info, f32 const delta_time);

    daxa::TaskBuffer zero_buffer = {};

    daxa::TaskBuffer entity_meta = {};
    daxa::TaskBuffer entity_transforms = {};
    daxa::TaskBuffer entity_combined_transforms = {};
    daxa::TaskBuffer entity_first_children = {};
    daxa::TaskBuffer entity_next_silbings = {};
    daxa::TaskBuffer entity_parents = {};
    daxa::TaskBuffer entity_meshlists = {};
    // Each entity has up 7 offsets into the visiblility bitfield. 
    daxa::TaskBuffer entity_visibility_bitfield_offsets = {};
    // The visibility bitfield is a list of per meshlet bitfields.
    // The bitfield is segmented into uvec4's. Each uvec4 stores 128 bits.
    // Each bit represents the visibility of a triagle, each uvec4 is a segment representing a meshlet.
    // The bitfield segments are per entity.
    daxa::TaskBuffer entity_visibility_bitfield = {};
    daxa::TaskBuffer entity_debug = {};

    // We need the prev frame meshlets in order to efficiently access the entity visibility bitmasks.
    daxa::TaskBuffer instantiated_meshlets_last_frame = {};
    daxa::TaskBuffer initial_pass_triangles = {};
    daxa::TaskBuffer mesh_draw_list = {};
    daxa::TaskBuffer instantiated_meshlets = {};
    daxa::TaskBuffer triangle_draw_list = {};
    daxa::TaskBuffer visible_triangles = {};

    std::vector<daxa::TaskBuffer> buffers = {};

    // Render Targets:
    daxa::TaskImage swapchain_image = {};
    daxa::TaskImage depth = {};
    daxa::TaskImage visbuffer = {};
    daxa::TaskImage debug_image = {};

    std::vector<daxa::TaskImage> images = {};
    std::vector<std::pair<daxa::ImageInfo, daxa::TaskImage>> frame_buffer_images = {};

    Window *window = {};
    GPUContext *context = {};
    Scene *scene = {};
    AssetManager *asset_manager = {};
    daxa::TaskGraph main_task_list;
    daxa::CommandSubmitInfo submit_info = {};
};