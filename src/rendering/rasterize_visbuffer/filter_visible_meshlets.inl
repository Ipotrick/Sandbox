#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_list.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../mesh/mesh.inl"
#include "../../mesh/visbuffer_meshlet_util.inl"

#define FILTER_VISIBLE_MESHLETS_DISPATCH_X MAX_TRIANGLES_PER_MESHLET

DAXA_INL_TASK_USE_BEGIN(FilterVisibleMeshletsBase, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_BUFFER(u_src_instantiated_meshlets, daxa_BufferPtr(InstantiatedMeshlets), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_meshlet_visibility_bitmasks, daxa_BufferPtr(daxa_uvec4), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_filtered_meshlets, daxa_RWBufferPtr(InstantiatedMeshlets), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_BUFFER(u_filtered_triangles, daxa_RWBufferPtr(TriangleDrawList), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_END()

#if __cplusplus
#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const FILTER_VISIBLE_MESHLETS_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/filter_visible_meshlets.glsl";

struct FilterVisibleMeshlets : FilterVisibleMeshletsBase
{
    daxa::ComputePipelineCompileInfo pipe_info = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{FILTER_VISIBLE_MESHLETS_SHADER_PATH},
        },
        .name = std::string{FilterVisibleMeshletsBase::NAME},
    };
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    GPUContext *context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_constant_buffer(context->shader_globals_set_info);
        cmd.set_constant_buffer(ti.uses.constant_buffer_set_info());
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_src_instantiated_meshlets.buffer(),
            .offset = offsetof(InstantiatedMeshlets, total_count),
        });
    }
};

#endif