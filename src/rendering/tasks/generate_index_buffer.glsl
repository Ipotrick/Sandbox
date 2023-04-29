#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "generate_index_buffer.inl"
#include "../../../shaders/util.glsl"

shared uint index_buffer_offset;
layout(local_size_x = GENERATE_INDEX_BUFFER_WORKGROUP_X) in;
void main()
{
    const daxa_u32 instanced_meshlet_index = gl_WorkGroupID.x;
    const daxa_u32 meshlet_triangle_index = gl_LocalInvocationID.x;
    daxa_RWBufferPtr(DrawIndexedIndirectInfo) draw_info = daxa_RWBufferPtrDrawIndexedIndirectInfo(daxa_u64(u_index_buffer_and_count));
    daxa_RWBufferPtr(daxa_u32) index_buffer = u_index_buffer_and_count + 8;

    daxa_BufferPtr(InstanciatedMeshlet) instanciated_meshlets = 
        daxa_BufferPtr(InstanciatedMeshlet)(daxa_u64(u_instanciated_meshlets) + 32);
    InstanciatedMeshlet instanced_meshlet = deref(instanciated_meshlets[instanced_meshlet_index]);
    Meshlet meshlet = u_meshes[instanced_meshlet.mesh_id].value.meshlets[instanced_meshlet.meshlet_index].value;
    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref(u_meshes[instanced_meshlet.mesh_id]).micro_indices;
    daxa_BufferPtr(daxa_u32) indirect_vertices = deref(u_meshes[instanced_meshlet.mesh_id]).indirect_vertices;
    const uint triangle_count = meshlet.triangle_count;
    const bool is_active = meshlet_triangle_index < meshlet.triangle_count;
    if (gl_LocalInvocationID.x == 0)
    {
        index_buffer_offset = atomicAdd(deref(draw_info).index_count, meshlet.triangle_count * 3);
    }
    memoryBarrierShared();
    barrier();
    if (is_active)
    {
        const uint mesh_index_offset = meshlet.micro_indices_offset + meshlet_triangle_index * 3;
        uint triangle_id[3] = {0, 0, 0};
        for (uint tri_index = 0; tri_index < 3; ++tri_index)
        {
            const uint mesh_local_index_buffer_index = meshlet_triangle_index * 3 + tri_index;
            const uint micro_index = get_micro_index(micro_index_buffer, mesh_index_offset + tri_index);
            uint vertex_id = 0;
            encode_vertex_id(instanced_meshlet_index, micro_index, vertex_id);
            triangle_id[tri_index] = vertex_id;
        }
        index_buffer[index_buffer_offset + meshlet_triangle_index * 3 + 0].value = triangle_id[0];
        index_buffer[index_buffer_offset + meshlet_triangle_index * 3 + 1].value = triangle_id[1];
        index_buffer[index_buffer_offset + meshlet_triangle_index * 3 + 2].value = triangle_id[2];
    }
}