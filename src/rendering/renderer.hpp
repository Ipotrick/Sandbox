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
    glm::vec3 pos = {};
    glm::vec3 up = {};
    glm::vec3 camera_near_plane_normal = {};
    glm::vec3 camera_left_plane_normal = {};
    glm::vec3 camera_right_plane_normal = {};
    glm::vec3 camera_top_plane_normal = {};
    glm::vec3 camera_bottom_plane_normal = {};
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
    void update_settings();
    void render_frame(CameraInfo const &camera_info, f32 const delta_time);

    daxa::TaskBuffer zero_buffer = {};

    daxa::TaskBuffer entity_meta = {};
    daxa::TaskBuffer entity_transforms = {};
    daxa::TaskBuffer entity_combined_transforms = {};
    daxa::TaskBuffer entity_first_children = {};
    daxa::TaskBuffer entity_next_silbings = {};
    daxa::TaskBuffer entity_parents = {};
    daxa::TaskBuffer entity_meshlists = {};

    daxa::TaskBuffer instantiated_meshlets = {};
    daxa::TaskBuffer instantiated_meshlets_prev = {};
    daxa::TaskBuffer visible_meshlets = {};

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