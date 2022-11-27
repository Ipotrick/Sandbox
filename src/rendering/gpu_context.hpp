#pragma once

#include "../sandbox.hpp"

struct BufferIdCombo
{
    daxa::TaskBufferId t_id = {};
    daxa::BufferId id = {};
};

struct ImageIdCombo
{
    daxa::TaskImageId t_id = {};
    daxa::ImageId id = {};
};

struct RenderContext
{
    daxa::Context context = {};
    daxa::Device device = {};
    daxa::Swapchain swapchain = {};
    daxa::PipelineCompiler pipeline_compiler = {};

    // Global resources:
    ShaderGlobals shader_globals = {};
    BufferIdCombo globals_buffer = {};
    BufferIdCombo index_buffer = {};
    ImageIdCombo swapchain_image = {};
    ImageIdCombo depth_image = {};

    // Pipelines:
    daxa::RasterPipeline triangle_pipe = {};
    daxa::RasterPipeline vis_prepass = {};
};