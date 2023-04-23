#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_list.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

#define GENERATE_INDEX_BUFFER_WORKGROUP_X MAX_TRIANGLES_PER_MESHLET

DAXA_INL_TASK_USE_BEGIN(GenIndexBufferBase, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_BUFFER(meshes, daxa_BufferPtr(Mesh), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(instanciated_meshlets, daxa_BufferPtr(InstanciatedMeshlet), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(index_buffer, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
DAXA_INL_TASK_USE_END()

#if __cplusplus

#include "../gpu_context.hpp"
#include "../../../shaders/util.inl"

static constexpr std::string_view GENERATE_INDEX_BUFFER_NAME = "generate_index_buffer";

static const daxa::ComputePipelineCompileInfo GENERATE_INDEX_BUFFER_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/tasks/generate_index_buffer.inl"}},
    .name = std::string{GENERATE_INDEX_BUFFER_NAME},
};

struct GenIndexBufferTask : GenIndexBufferBase
{
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        auto value_count = context->total_meshlet_count;
        cmd.set_pipeline(*context->compute_pipelines.at(GENERATE_INDEX_BUFFER_NAME));
        cmd.dispatch(round_up_div(value_count * MAX_TRIANGLES_PER_MESHLET, GENERATE_INDEX_BUFFER_WORKGROUP_X), 1, 1);
    }
};

#elif DAXA_SHADER
#extension GL_EXT_debug_printf : enable
#include "../../../shaders/util.glsl"
DEFINE_PUSHCONSTANT(GenerateIndexBufferPush, push)
shared uint group_global_triangle_offset;
layout(local_size_x = GENERATE_INDEX_BUFFER_WORKGROUP_X) in;
void main()
{
    const daxa_u32 instanced_meshlet_index = gl_WorkGroupID.x;
    const daxa_u32 meshlet_triangle_index = gl_LocalInvocationID.x;

    InstanciatedMeshlet instanced_meshlet = deref(push.instanciated_meshlets[instanced_meshlet_index]);
    Meshlet meshlet = push.meshes[instanced_meshlet.mesh_id].value.meshlets[instanced_meshlet.meshlet_index].value;
    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref(push.meshes[instanced_meshlet.mesh_id]).micro_indices;
    daxa_BufferPtr(daxa_u32) indirect_vertices = deref(push.meshes[instanced_meshlet.mesh_id]).indirect_vertices;
    const uint triangle_count = meshlet.triangle_count;
    const bool is_active = meshlet_triangle_index < meshlet.triangle_count;
    if (is_active)
    {
        const uint index_buffer_offset = atomicAdd(deref(push.global_triangle_count), 1);
        const uint mesh_triangle_offset = meshlet.triangle_offset + meshlet_triangle_index;
        const uint mesh_index_offset = mesh_triangle_offset * 3;
        uint triangle_id[3] = {0, 0, 0};
        for (uint tri_index = 0; tri_index < 3; ++tri_index)
        {
            const uint micro_index = get_micro_index(micro_index_buffer, mesh_index_offset + tri_index);
            uint vertex_id = 0;
            encode_vertex_id(instanced_meshlet_index, micro_index, vertex_id);
            triangle_id[tri_index] = vertex_id;
        }
        push.index_buffer[(index_buffer_offset + meshlet_triangle_index) * 3 + 0].value = (index_buffer_offset + meshlet_triangle_index) * 3 + 0;
        push.index_buffer[(index_buffer_offset + meshlet_triangle_index) * 3 + 1].value = (index_buffer_offset + meshlet_triangle_index) * 3 + 1;
        push.index_buffer[(index_buffer_offset + meshlet_triangle_index) * 3 + 2].value = (index_buffer_offset + meshlet_triangle_index) * 3 + 2;
    }
}
#endif