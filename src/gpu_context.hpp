#pragma once

#include "sandbox.hpp"

struct RenderContext
{
    daxa::Context context = {};
    daxa::Device device = {};
    daxa::Swapchain swapchain = {};
    daxa::PipelineCompiler pipeline_compiler = {};

    // Global resources:
    daxa::TaskImageId t_swapchain_image = {};
    daxa::ImageId swapchain_image = {};

    // Pipelines:
    daxa::RasterPipeline triangle_pipe = {};
};