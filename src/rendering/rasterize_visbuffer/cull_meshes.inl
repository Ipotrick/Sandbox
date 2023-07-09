#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shaders/shared.inl"
#include "../../mesh/mesh.inl"
#include "../../mesh/visbuffer_meshlet_util.inl"
#include "../../scene/scene.inl"

/// 
/// CullMeshes goes through all entities and their meshlists.
/// It checks if the meshes are visible and if they are they get inserted into a visible meshlist.
/// It also generates a list of meshlet counts for each mesh, that the following meshlet culling uses.
///

#define CULL_MESHES_WORKGROUP_X 128

#if __cplusplus || defined(CullMeshesCommandBase_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(CullMeshesCommandBase, 1)
DAXA_TASK_USE_BUFFER(u_entity_meta, daxa_BufferPtr(EntityMetaData), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_command, daxa_RWBufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_WRITE)
DAXA_DECL_TASK_USES_END()
#endif
#if __cplusplus || !defined(CullMeshesCommandBase_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(CullMeshesBase, 1)
DAXA_TASK_USE_BUFFER(u_command, daxa_BufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_meta, daxa_BufferPtr(EntityMetaData), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_meshlists, daxa_BufferPtr(MeshList), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_transforms, daxa_BufferPtr(daxa_f32mat4x4), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_combined_transforms, daxa_BufferPtr(daxa_f32mat4x4), COMPUTE_SHADER_READ)
BUFFER_COMPUTE_WRITE(u_mesh_draw_list, MeshDrawList)
DAXA_DECL_TASK_USES_END()
#endif

#if __cplusplus
#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const CULL_MESHES_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/cull_meshes.glsl";

using CullMeshesCommandWrite = WriteIndirectDispatchArgsBaseTask<
    CullMeshesCommandBase,
    CULL_MESHES_SHADER_PATH>;

struct CullMeshes : CullMeshesBase
{
    static const inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{CULL_MESHES_SHADER_PATH},
        },
        .name = std::string{CullMeshesBase::NAME},
    };
    GPUContext *context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(CullMeshesBase::NAME));
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_command.buffer(),
        });
    }
};

void tasks_cull_meshes(GPUContext * context, daxa::TaskGraph& task_list, CullMeshesBase::Uses uses)
{
    task_list.add_task({
        .uses = {
            daxa::BufferTransferWrite{uses.u_mesh_draw_list.handle},
        },
        .task = [=](daxa::TaskInterface ti){
            auto cmd = ti.get_command_list();
            auto alloc = ti.get_allocator().allocate(sizeof(DispatchIndirectStruct)).value();
            *reinterpret_cast<DispatchIndirectStruct*>(alloc.host_address) = {0,1,1};
            cmd.copy_buffer_to_buffer({
                .src_buffer = ti.get_allocator().get_buffer(),
                .src_offset = alloc.buffer_offset,
                .dst_buffer = ti.uses[uses.u_mesh_draw_list.handle].buffer(),
                .dst_offset = offsetof(MeshDrawList, count),
            });
        },
        .name = "clear u_mesh_draw_list",
    });

    auto command_buffer = task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct),
        .name = "CullMeshesCommand",
    });

    task_list.add_task(CullMeshesCommandWrite{
        {.uses={
            .u_entity_meta = uses.u_entity_meta,
            .u_command = command_buffer,
        }},
        .context = context,
    });

    uses.u_command.handle = command_buffer;

    task_list.add_task(CullMeshes{
        {.uses={uses}},
        .context = context,
    });
}

#endif