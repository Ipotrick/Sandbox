#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

#define WRITE_DRAW_OPAQUE_INFO_BUFFER_WORKGROUP_X 128

DAXA_DECL_TASK_USES_BEGIN(WriteDrawOpaqueIndexBufferBase, 1)
DAXA_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshlet_list, daxa_BufferPtr(InstantiatedMeshlet), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_index_buffer_and_count, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()

#if __cplusplus

#include "../gpu_context.hpp"

static const daxa::ComputePipelineCompileInfo WRITE_DRAW_OPAQUE_INDEX_BUFFER_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/tasks/write_draw_opaque_index_buffer.glsl"}},
    .name = std::string{WriteDrawOpaqueIndexBufferBase::NAME},
};

struct WriteDrawOpaqueIndexBufferTask : WriteDrawOpaqueIndexBufferBase
{
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(WriteDrawOpaqueIndexBufferBase::NAME));
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_meshlet_list.buffer(),
            .offset = 0,
        });
    }
};
#endif