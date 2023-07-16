#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"
#include "../../mesh/visbuffer_meshlet_util.inl"

#define CULL_MESHLETS_WORKGROUP_X 128

#if __cplusplus || defined(CullMeshletsCommandWriteBase_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(CullMeshletsCommandWriteBase, 1)
    DAXA_TASK_USE_BUFFER(u_mesh_draw_list, daxa_BufferPtr(MeshDrawList), COMPUTE_SHADER_READ)
    DAXA_TASK_USE_BUFFER(u_meshlet_count_prefix_sum, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
    DAXA_TASK_USE_BUFFER(u_command, daxa_RWBufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_WRITE)
DAXA_DECL_TASK_USES_END()
#endif
#if __cplusplus || !defined(CullMeshletsCommandWriteBase_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(CullMeshletsBase, 1)
    DAXA_TASK_USE_BUFFER(u_command, daxa_BufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_READ)
    // Used to know the number of drawn meshes.
    // Used to know mesh draw infos.
    DAXA_TASK_USE_BUFFER(u_mesh_draw_list, daxa_BufferPtr(MeshDrawList), COMPUTE_SHADER_READ)
    // Used to match flat thread index to meshlet and entity index.
    // This is done by binary searching the thread index in the prefix sum array.
    DAXA_TASK_USE_BUFFER(u_meshlet_count_prefix_sum, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
    DAXA_TASK_USE_BUFFER(u_entity_meta_data, daxa_BufferPtr(EntityMetaData), COMPUTE_SHADER_READ)
    DAXA_TASK_USE_BUFFER(u_entity_meshlists, daxa_BufferPtr(MeshList), COMPUTE_SHADER_READ)
    DAXA_TASK_USE_BUFFER(u_entity_visibility_bitfield_offsets, daxa_BufferPtr(EntityVisibilityBitfieldOffsets), COMPUTE_SHADER_READ)
    DAXA_TASK_USE_BUFFER(u_meshlet_visibility_bitfield, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
    DAXA_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), COMPUTE_SHADER_READ)
    DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_RWBufferPtr(InstantiatedMeshlets), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()
#endif

#if __cplusplus

#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"

inline static constexpr char const CULL_MESHLETS_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/cull_meshlets.glsl";

using CullMeshletsCommandWrite = WriteIndirectDispatchArgsBaseTask<
    CullMeshletsCommandWriteBase,
    CULL_MESHLETS_SHADER_PATH
>;

struct CullMeshlets : CullMeshletsBase
{
    inline static const daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{CULL_MESHLETS_SHADER_PATH},
        },
        .name = std::string{CullMeshletsBase::NAME},
    };
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::CommandList cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(CullMeshletsBase::NAME));
        cmd.dispatch_indirect({.indirect_buffer = uses.u_command.buffer()});
    }
};

void task_cull_meshlets(GPUContext * context, daxa::TaskGraph & task_list, CullMeshlets::Uses uses)
{
    auto cull_meshlets_command = task_list.create_transient_buffer({sizeof(DispatchIndirectStruct), "cull meshlets command"});
    task_list.add_task(CullMeshletsCommandWrite{
        {.uses={
            .u_mesh_draw_list = uses.u_mesh_draw_list, 
            .u_meshlet_count_prefix_sum = uses.u_meshlet_count_prefix_sum,
            .u_command = cull_meshlets_command}
        },
        context
    });
    uses.u_command.handle = cull_meshlets_command;
    task_list.add_task(CullMeshlets{{.uses = uses}, context});
}

#endif