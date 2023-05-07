#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_list.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

DAXA_INL_TASK_USE_BEGIN(PatchDrawOpaqueIndirectBase, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_BUFFER(u_meshlets, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_END()

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
        cmd.set_constant_buffer(context->shader_globals_set_info);
        cmd.set_constant_buffer(ti.uses.constant_buffer_set_info());
        cmd.set_pipeline(*context->compute_pipelines.at(PatchDrawOpaqueIndirectBase::NAME));
        cmd.dispatch(1,1,1);
    }
};
#endif