#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

DAXA_DECL_TASK_USES_BEGIN(PatchDrawOpaqueIndirectBase, 1)
DAXA_TASK_USE_BUFFER(u_meshlets, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()

#if __cplusplus

#include "../gpu_context.hpp"

static const daxa::ComputePipelineCompileInfo PATCH_DRAW_OPAQUE_INDIRECT_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/tasks/patch_draw_opaque_indirect.glsl"}},
    .name = std::string{WriteDrawOpaqueIndexBufferBase::NAME},
};

struct PatchDrawOpaqueIndirectTask : PatchDrawOpaqueIndirectBase
{
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(PatchDrawOpaqueIndirectBase::NAME));
        cmd.dispatch(1,1,1);
    }
};
#endif