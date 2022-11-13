#pragma once

#include "../sandbox.hpp"

struct RenderContext
{
    daxa::Context context;
    daxa::Device device;
    daxa::Swapchain swapchain;
    daxa::PipelineCompiler pipeline_compiler;

    // Pipelines:
    daxa::RasterPipeline triangle_pipe;
};