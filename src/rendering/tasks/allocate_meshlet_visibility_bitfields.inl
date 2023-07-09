#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

DAXA_DECL_TASK_USES_BEGIN(AllocateMeshletVisibilityBase, 1)
DAXA_TASK_USE_BUFFER(u_meshlists, daxa_BufferPtr(MeshList), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_meta, daxa_BufferPtr(EntityMetaData), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_visibility_bitfield_sratch, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(u_meshlet_visibilities, daxa_RWBufferPtr(EntityVisibilityBitfieldOffsets), COMPUTE_SHADER_WRITE)
DAXA_DECL_TASK_USES_END()

#define ALLOCATE_MESHLET_VISIBILITIES_WORKGROUP_X 128

#if __cplusplus

#include "../gpu_context.hpp"
#include "../../scene/scene.hpp"

static const daxa::ComputePipelineCompileInfo ALLOCATE_MESHLET_VISIBILITY_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/tasks/allocate_meshlet_visibility_bitfields.glsl"}},
    .name = std::string{AllocateMeshletVisibilityBase::NAME},
};

struct AllocateMeshletVisibilityTask : AllocateMeshletVisibilityBase
{
    GPUContext * context = {};
    Scene * scene = {};
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*pipeline);
        auto const x = round_up_div(scene->entity_meta.entity_count, ALLOCATE_MESHLET_VISIBILITIES_WORKGROUP_X);
        cmd.dispatch(x,1,1);
    }
};
#endif