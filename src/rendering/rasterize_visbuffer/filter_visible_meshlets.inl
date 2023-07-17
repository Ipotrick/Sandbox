#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shaders/shared.inl"
#include "../../mesh/mesh.inl"
#include "../../mesh/visbuffer_meshlet_util.inl"

#define FILTER_VISIBLE_MESHLETS_DISPATCH_X MAX_TRIANGLES_PER_MESHLET

#if __cplusplus || defined(FilterVisibleMeshletsCommandWriteBase_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(FilterVisibleMeshletsCommandWriteBase, 1)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets_prev, daxa_BufferPtr(InstantiatedMeshlets), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_RWBufferPtr(InstantiatedMeshlets), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(u_command, daxa_RWBufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_WRITE)
DAXA_DECL_TASK_USES_END()
#endif

#if __cplusplus || !defined(FilterVisibleMeshletsCommandWriteBase_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(FilterVisibleMeshletsBase, 1)
DAXA_TASK_USE_BUFFER(u_command, daxa_BufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_visibility_bitfield_offsets_prev, daxa_BufferPtr(EntityVisibilityBitfieldOffsets), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_visibility_bitfield_prev, daxa_BufferPtr(daxa_u32vec4), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets_prev, daxa_BufferPtr(InstantiatedMeshlets), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_RWBufferPtr(InstantiatedMeshlets), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(u_triangle_draw_list, daxa_RWBufferPtr(TriangleDrawList), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()
#endif

#if __cplusplus
#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const FILTER_VISIBLE_MESHLETS_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/filter_visible_meshlets.glsl";

using FilterVisibleMeshletsCommandWrite = WriteIndirectDispatchArgsBaseTask<
    FilterVisibleMeshletsCommandWriteBase,
    FILTER_VISIBLE_MESHLETS_SHADER_PATH
>;

struct FilterVisibleMeshlets : FilterVisibleMeshletsBase
{
    static const inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{FILTER_VISIBLE_MESHLETS_SHADER_PATH},
        },
        .name = std::string{FilterVisibleMeshletsBase::NAME},
    };
    GPUContext *context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(FilterVisibleMeshlets::NAME));
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_command.buffer(),
        });
    }
};

void task_filter_visible_meshlets(GPUContext* context, daxa::TaskGraph& task_list, FilterVisibleMeshlets::Uses uses)
{
    auto command = task_list.create_transient_buffer({sizeof(DispatchIndirectStruct), "filter_visible_meshlets_command"});
    task_list.add_task(FilterVisibleMeshletsCommandWrite{
        {.uses={
            .u_instantiated_meshlets_prev = uses.u_instantiated_meshlets_prev,
            .u_command = command,
            .u_instantiated_meshlets = uses.u_instantiated_meshlets,
        }},
        context,
    });
    uses.u_command.handle = command;
    task_list.add_task(FilterVisibleMeshlets{
        {.uses=uses},
        context,
    });
}

#endif