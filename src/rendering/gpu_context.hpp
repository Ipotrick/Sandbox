#pragma once

#include "../sandbox.hpp"
#include "../window.hpp"

struct BufferIdCombo
{
    daxa::BufferId id = {};
    daxa::TaskBufferId t_id = {};
};

struct ImageIdCombo
{
    daxa::ImageId id = {};
    daxa::TaskImageId t_id = {};
};

struct RenderContext
{
    RenderContext(Window const& window);
    ~RenderContext();

    // common unique:
    daxa::Context context = {};
    daxa::Device device = {};
    daxa::Swapchain swapchain = {};
    daxa::PipelineManager pipeline_manager = {};
    ShaderGlobals shader_globals = {};
    daxa::TransferMemoryPool transient_mem;

    // Buffers:
    BufferIdCombo globals_buffer = {};
    BufferIdCombo entity_data_buffer = {};
    // First 16 bytes are reserved for a counter variable.
    BufferIdCombo instanciated_meshlets = {};
    // First 16 bytes are reserved for a counter variable.
    BufferIdCombo index_buffer = {};

    // Render Targets:
    ImageIdCombo swapchain_image = {};
    ImageIdCombo depth_image = {};

    // Pipelines:
    std::shared_ptr<daxa::RasterPipeline> triangle_pipe = {};
    std::shared_ptr<daxa::RasterPipeline> vis_prepass = {};
    std::shared_ptr<daxa::ComputePipeline> transform_pipeline = {};
    std::shared_ptr<daxa::ComputePipeline> meshlet_culling_pipeline = {};
    std::shared_ptr<daxa::ComputePipeline> vertex_id_writeout_pipeline = {};
};