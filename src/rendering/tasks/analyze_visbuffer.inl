#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../mesh/mesh.inl"

#define ANALYZE_VIS_BUFFER_WORKGROUP_X 16
#define ANALYZE_VIS_BUFFER_WORKGROUP_Y 8

DAXA_DECL_TASK_USES_BEGIN(AnalyzeVisbufferBase, 1)
DAXA_TASK_USE_IMAGE(u_visbuffer, daxa_Image2Du32, COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(InstantiatedMeshlet), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_visibility_bitfield_offsets, daxa_BufferPtr(EntityVisibilityBitfieldOffsets), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlet_counters, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(u_meshlet_visibility_bitfield, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets_last_frame, daxa_RWBufferPtr(InstantiatedMeshlet), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()

struct AnalyzeVisbufferPush
{
    daxa_u32 width;
    daxa_u32 height;
};

#if __cplusplus

#include "../gpu_context.hpp"

static const daxa::ComputePipelineCompileInfo ANALYZE_VISBUFFER_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/tasks/analyze_visbuffer.glsl"}},
    .push_constant_size = sizeof(AnalyzeVisbufferPush),
    .name = std::string{AnalyzeVisbufferBase::NAME},
};

struct AnalyzeVisbufferTask : AnalyzeVisbufferBase
{
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*pipeline);
        auto const x = ti.get_device().info_image(uses.u_visbuffer.image()).size.x;
        auto const y = ti.get_device().info_image(uses.u_visbuffer.image()).size.y;
        cmd.push_constant(AnalyzeVisbufferPush{
            .width = x,
            .height = y,
        });
        auto const dispatch_x = round_up_div(x, ANALYZE_VIS_BUFFER_WORKGROUP_X * 2);
        auto const dispatch_y = round_up_div(y, ANALYZE_VIS_BUFFER_WORKGROUP_Y * 2);
        cmd.dispatch(dispatch_x, dispatch_y, 1);
    }
};
#endif