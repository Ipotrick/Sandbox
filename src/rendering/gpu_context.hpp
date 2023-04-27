#pragma once

#include "../sandbox.hpp"
#include "../window.hpp"

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
    daxa::types::BufferDeviceAddress shader_globals_ptr = {};
    daxa::TransferMemoryPool transient_mem;

    // Pipelines:
    std::unordered_map<std::string_view, std::shared_ptr<daxa::RasterPipeline>> raster_pipelines = {};
    std::unordered_map<std::string_view, std::shared_ptr<daxa::ComputePipeline>> compute_pipelines = {};

    // Data
    usize total_meshlet_count = {};

    u32 meshlet_sums_step2_dispatch_size = {}; // (scene->entity_meta.entity_count + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE)
};