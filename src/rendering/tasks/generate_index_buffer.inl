#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_list.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

#define GENERATE_INDEX_BUFFER_WORKGROUP_X MAX_TRIANGLES_PER_MESHLET

DAXA_INL_TASK_USE_BEGIN(GenDrawInfoBase, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_instanciated_meshlets, daxa_BufferPtr(InstanciatedMeshlet), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_index_buffer_and_count, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_END()

#if __cplusplus

#include "../gpu_context.hpp"

static constexpr std::string_view GENERATE_INDEX_BUFFER_NAME = "generate_index_buffer";

static const daxa::ComputePipelineCompileInfo GENERATE_INDEX_BUFFER_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/tasks/generate_index_buffer.glsl"}},
    .name = std::string{GENERATE_INDEX_BUFFER_NAME},
};

struct GenDrawInfoTask : GenDrawInfoBase
{
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_constant_buffer(context->shader_globals_set_info);
        cmd.set_constant_buffer(ti.uses.constant_buffer_set_info());
        cmd.set_pipeline(*context->compute_pipelines.at(GenDrawInfoBase::NAME));
        if (context->settings.indexed_id_rendering)
        {
            cmd.dispatch_indirect({
                .indirect_buffer = uses.u_instanciated_meshlets.buffer(),
                .offset = 0,
            });
        }
        else
        {
            cmd.dispatch(1,1,1);
        }
    }
};
#endif