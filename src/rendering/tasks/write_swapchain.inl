#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

DAXA_DECL_TASK_USES_BEGIN(WriteSwapchain, 1)
DAXA_TASK_USE_IMAGE(swapchain, daxa_RWImage2Df32, COMPUTE_SHADER_WRITE)
DAXA_TASK_USE_IMAGE(debug_image, daxa_Image2Df32, COMPUTE_SHADER_READ)
DAXA_DECL_TASK_USES_END()

struct WriteSwapchainPush
{
    daxa_u32 width;
    daxa_u32 height;
};

#define WRITE_SWAPCHAIN_WG_X 16
#define WRITE_SWAPCHAIN_WG_Y 8

#if __cplusplus

#include "../gpu_context.hpp"

static const daxa::ComputePipelineCompileInfo WRITE_SWAPCHAIN_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/tasks/write_swapchain.glsl"}},
    .push_constant_size = sizeof(WriteSwapchainPush),
    .name = std::string{WriteSwapchain::NAME},
};

struct WriteSwapchainTask : WriteSwapchain
{
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*pipeline);
        u32 const dispatch_x = round_up_div(ti.get_device().info_image(uses.swapchain.image()).size.x, WRITE_SWAPCHAIN_WG_X);
        u32 const dispatch_y = round_up_div(ti.get_device().info_image(uses.swapchain.image()).size.y, WRITE_SWAPCHAIN_WG_Y);
        cmd.push_constant(WriteSwapchainPush{
            .width = ti.get_device().info_image(uses.swapchain.image()).size.x,
            .height = ti.get_device().info_image(uses.swapchain.image()).size.y,
        });
        cmd.dispatch(dispatch_x, dispatch_y, 1);
    }
};
#endif