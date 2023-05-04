#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_list.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

#define FILL_MESHLET_BUFFER_WORKGROUP_X 128

DAXA_INL_TASK_USE_BEGIN(FillMeshletBufferBase, DAXA_CBUFFER_SLOT1)
    DAXA_INL_TASK_USE_BUFFER(u_prefix_sum_mehslet_counts, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
    DAXA_INL_TASK_USE_BUFFER(u_entity_meta_data, daxa_BufferPtr(EntityMetaData), COMPUTE_SHADER_READ)
    DAXA_INL_TASK_USE_BUFFER(u_entity_meshlists, daxa_BufferPtr(MeshList), COMPUTE_SHADER_READ)
    DAXA_INL_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), COMPUTE_SHADER_READ)
    DAXA_INL_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_RWBufferPtr(InstanciatedMeshlet), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_END()

struct FillMeshletBufferPush
{
    daxa_u32 meshlet_count;
};
DAXA_ENABLE_BUFFER_PTR(FillMeshletBufferPush)

#if __cplusplus

#include "../gpu_context.hpp"

inline static constexpr std::string_view FILL_MESHLET_BUFFER_PIPELINE_NAME = "fill_meshlet_buffer";

inline static const daxa::ComputePipelineCompileInfo FILL_MESHLET_BUFFER_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"./src/rendering/tasks/fill_meshlet_buffer.glsl"},
        .compile_options = {
            .defines = {{"d"}},
        },
    },
    .push_constant_size = sizeof(FillMeshletBufferPush),
    .name = std::string{FILL_MESHLET_BUFFER_PIPELINE_NAME},
};

struct FillMeshletBufferTask : FillMeshletBufferBase
{
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    GPUContext * context = {};
    usize * meshlet_count = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::CommandList cmd = ti.get_command_list();
        cmd.set_constant_buffer(context->shader_globals_set_info);
        cmd.set_constant_buffer(ti.uses.constant_buffer_set_info());
        cmd.set_pipeline(*pipeline);
        cmd.push_constant(FillMeshletBufferPush{
            .meshlet_count = static_cast<u32>(*meshlet_count),
        });
        cmd.dispatch(round_up_div(*meshlet_count, FILL_MESHLET_BUFFER_WORKGROUP_X), 1, 1);
    }
};

#endif