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
    Renderer(Window* window, GPUContext* context, Scene* scene, AssetManager* asset_manager);
    ~Renderer();

    void compile_pipelines();
    void recreate_resizable_images();
    void window_resized();
    auto create_main_task_list() -> daxa::TaskList;
    void render_frame(CameraInfo const& camera_info);

    Window* window = {};
    GPUContext* context = {};
    Scene* scene = {};
    AssetManager* asset_manager = {};
    daxa::TaskList main_task_list;
};