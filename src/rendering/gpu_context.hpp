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

struct GPUContext
{
    GPUContext(Window const& window);
    ~GPUContext();

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
    BufferIdCombo ent_meshlet_count_prefix_sum_buffer = {};
    // First 16 bytes are reserved for a counter variable.
    BufferIdCombo instanciated_meshlets = {};
    // First 16 bytes are reserved for a counter variable.
    BufferIdCombo index_buffer = {};

    // Render Targets:
    ImageIdCombo swapchain_image = {};
    ImageIdCombo depth_image = {};

    // Pipelines:
    std::unordered_map<std::string_view, std::shared_ptr<daxa::RasterPipeline>> raster_pipelines = {};
    std::unordered_map<std::string_view, std::shared_ptr<daxa::ComputePipeline>> compute_pipelines = {};
};