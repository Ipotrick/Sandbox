#include <daxa/daxa.glsl>

#include "../../../shaders/util.glsl"

#include "generate_index_buffer.inl"

DEFINE_PUSHCONSTANT(GenerateIndexBufferPush, push)
shared uint group_global_triangle_offset;
layout(local_size_x = PREFIX_SUM_WORKGROUP_SIZE) in;
void main()
{
    uint instanciated_meshlet_index = gl_GlobalInvocationID.x;
    const uint warp_id = gl_SubgroupID;
    const uint warp_index = gl_SubgroupInvocationID;
    InstanciatedMeshlet instanciated_meshlet;
    uint triangle_count = 0;
    daxa_BufferPtr(Meshlet) meshlets;
    if (instanciated_meshlet_index < push.instanciated_meshlet_count)
    {
        instanciated_meshlet = deref(push.instanciated_meshlets[instanciated_meshlet_index])
        meshlets = deref(push.meshes[instanciated_meshlet.mesh_index]).meshlets;
        triangle_count = deref(meshlets[instanciated_meshlet.meshlet_index]).triangle_count;
    }    
    uint triangle_sum_to_thread = triangle_count;
    prefix_sum(
        warp_index,
        warp_id,
        triangle_sum_to_thread);
    if (gl_GroupInvocationId.x == (PREFIX_SUM_WORKGROUP_SIZE - 1))
    {
        group_global_triangle_offset = atomicAdd(deref(push.global_triangle_count), triangle_sum_to_thread);
    }
    barrier();
    memoryBarrierShared();
    
    if (instanciated_meshlet_index < push.instanciated_meshlet_count)
    {
        uint thread_global_triangle_offset = group_global_triangle_offset + triangle_sum_to_thread;
        uint thread_global_index_offset = thread_global_triangle_offset * 3;
        daxa_BufferPtr(IndexType) micro_indices = deref(push.meshes[instanciated_meshlet.mesh_index]).micro_indices;
        uint micro_index_offset = deref(meshlets[instanciated_meshlet.meshlet_index]).triangle_offset * 3;

        for (uint triangle_i = 0; triangle_i < triangle_count; ++triangle_count)
        {
            uint index0 = thread_global_index_offset + triangle_i * 3;
            uint index1 = index0 + 1;
            uint index2 = index0 + 2;
            uint vertex_index = deref()
            deref(push.index_buffer[index0]) = encode_vertex_id(instanciated_meshlet_index, get_micro_index(micro_indices, micro_index_offset + triangle_i * 3 + 0));
            deref(push.index_buffer[index1]) = encode_vertex_id(instanciated_meshlet_index, get_micro_index(micro_indices, micro_index_offset + triangle_i * 3 + 1));
            deref(push.index_buffer[index2]) = encode_vertex_id(instanciated_meshlet_index, get_micro_index(micro_indices, micro_index_offset + triangle_i * 3 + 2));
        }
    }
}