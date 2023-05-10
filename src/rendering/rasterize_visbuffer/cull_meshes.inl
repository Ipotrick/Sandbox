#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_list.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../mesh/mesh.inl"
#include "../../mesh/visbuffer_meshlet_util.inl"

#define CULL_MESHES_WORKGROUP_X 128

#if __cplusplus || defined(WRITE_COMMAND)
DAXA_INL_TASK_USE_BEGIN(CullMeshesCommandBase, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_BUFFER(u_entity_meta, daxa_BufferPtr(EntityMetaData), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_command, daxa_RWBufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_WRITE)
DAXA_INL_TASK_USE_END()
#endif
#if __cplusplus || !defined(WRITE_COMMAND)
DAXA_INL_TASK_USE_BEGIN(CullMeshesBase, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_BUFFER(u_command, daxa_RWBuffedaxa_BufferPtrrPtr(DispatchIndirectStruct), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_entity_meta, daxa_BufferPtr(EntityMetaData), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_entity_meshlists, daxa_BufferPtr(EntityMeshlist), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_entity_transforms, daxa_BufferPtr(daxa_mat4x4f32), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_entity_combined_transforms, daxa_BufferPtr(daxa_mat4x4f32), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_mesh_draw_list, daxa_RWBufferPtr(MeshDrawList), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_BUFFER(u_mesh_draw_meshlet_counts, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_WRITE)
DAXA_INL_TASK_USE_END()
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
    static const inline daxa::ComputePipelineCompileInfo COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{CULL_MESHES_SHADER_PATH},
        },
        .name = std::string{CullMeshesBase::NAME},
    };
    GPUContext *context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_constant_buffer(context->shader_globals_set_info);
        cmd.set_constant_buffer(ti.uses.constant_buffer_set_info());
        cmd.set_pipeline(*context->compute_pipelines.at(CullMeshesBase::NAME));
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_command.buffer(),
        });
    }
};

#endif