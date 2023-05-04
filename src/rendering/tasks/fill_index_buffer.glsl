#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "fill_index_buffer.inl"
#include "../../../shaders/util.glsl"
#include "../../../shaders/cull_util.glsl"

shared vec3 transformed_vertex_positions[128];
shared uint post_cull_triangle_indices[128];
shared uint post_cull_triangle_count;
shared uint index_buffer_offset;
layout(local_size_x = FILL_INDEX_BUFFER_WORKGROUP_X) in;
void main()
{
    const bool indexed_id_rendering = globals.settings.indexed_id_rendering == 1;
    if (!indexed_id_rendering)
    {
        if (gl_GlobalInvocationID.x == 0)
        {
            daxa_RWBufferPtr(DrawIndirectStruct) draw_info = daxa_RWBufferPtr(DrawIndirectStruct)(daxa_u64(u_index_buffer_and_count));
            daxa_BufferPtr(DispatchIndirectStruct) meshlet_dispatch = daxa_BufferPtr(DispatchIndirectStruct)(daxa_u64(u_instantiated_meshlets));
            deref(draw_info).vertex_count = 128 * 3 * deref(meshlet_dispatch).x;
        }
        return;
    }
    const daxa_u32 instanced_meshlet_index = gl_WorkGroupID.x;
    daxa_u32 meshlet_triangle_index = gl_LocalInvocationID.x;
    daxa_RWBufferPtr(DrawIndexedIndirectStruct) draw_info = daxa_RWBufferPtr(DrawIndexedIndirectStruct)(daxa_u64(u_index_buffer_and_count));
    daxa_RWBufferPtr(daxa_u32) index_buffer = u_index_buffer_and_count + 8;

    daxa_BufferPtr(InstanciatedMeshlet) instantiated_meshlets = 
        daxa_BufferPtr(InstanciatedMeshlet)(daxa_u64(u_instantiated_meshlets) + 32);
    InstanciatedMeshlet instanced_meshlet = deref(instantiated_meshlets[instanced_meshlet_index]);
    Meshlet meshlet = u_meshes[instanced_meshlet.mesh_id].value.meshlets[instanced_meshlet.meshlet_index].value;
    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref(u_meshes[instanced_meshlet.mesh_id]).micro_indices;
    daxa_BufferPtr(daxa_u32) indirect_vertices = deref(u_meshes[instanced_meshlet.mesh_id]).indirect_vertices;
    uint triangle_count = meshlet.triangle_count;
#if ENABLE_TRIANGLE_CULLING
    BoundingSphere meshlet_bounds = deref(deref(u_meshes[instanced_meshlet.mesh_id]).meshlet_bounds[instanced_meshlet.meshlet_index]);
    const float threshhold = 150.0f;
    const bool cull_triangles = meshlet_bounds.radius > threshhold;
#else
    const bool cull_triangles = false;
#endif
    if (cull_triangles && bool(ENABLE_TRIANGLE_CULLING))
    {
        Mesh mesh = deref(u_meshes[instanced_meshlet.mesh_id]);
        [[unroll]]
        for (uint loop_offset = 0; loop_offset < MAX_VERTICES_PER_MESHLET; loop_offset += FILL_INDEX_BUFFER_WORKGROUP_X)
        {
            const uint thread_index = gl_LocalInvocationID.x + loop_offset;
            if (thread_index < meshlet.vertex_count)
            {
                const uint vertex_index = mesh.indirect_vertices[meshlet.indirect_vertex_offset + gl_LocalInvocationID.x].value;
                vec3 ws_vertex_pos = mesh.vertex_positions[vertex_index].value;
                vec4 projected_vertex_pos = globals.cull_camera_view_projection * vec4(ws_vertex_pos,1);
                transformed_vertex_positions[thread_index] = projected_vertex_pos.xyz / projected_vertex_pos.w;
            }
        }
        if (gl_LocalInvocationID.x == 0)
        {
            post_cull_triangle_count = 0;
        }
        barrier();
        memoryBarrierShared();

        [[unroll]]
        for (uint loop_offset = 0; loop_offset < MAX_VERTICES_PER_MESHLET; loop_offset += FILL_INDEX_BUFFER_WORKGROUP_X)
        {
            const uint thread_index = gl_LocalInvocationID.x + loop_offset;
            if (thread_index < meshlet.triangle_count)
            {
                const uint mesh_index_offset = meshlet.micro_indices_offset + thread_index * 3;
                NdcBounds ndc_bounds;
                init_ndc_bounds(ndc_bounds);
                // construct bounding box from bounding sphere,
                // project each vertex of the box to ndc, min and max the coordinates.
                for (uint tri_index = 0; tri_index < 3; ++tri_index)
                {
                    const uint micro_index = get_micro_index(micro_index_buffer, mesh_index_offset + tri_index);
                    vec3 ndc_vertex_pos = transformed_vertex_positions[micro_index];
                    add_vertex_to_ndc_bounds(ndc_bounds, ndc_vertex_pos);
                }

                const bool out_of_frustum = !is_in_frustum(ndc_bounds);

                const bool triangle_visible = !out_of_frustum;
                if (triangle_visible)
                {
                    const uint culled_tri_list_index = atomicAdd(post_cull_triangle_count, 1);
                    post_cull_triangle_indices[culled_tri_list_index] = thread_index;
                }
            }
        }
        memoryBarrierShared();
        barrier();
        if (gl_LocalInvocationID.x == 0)
        {
            index_buffer_offset = atomicAdd(deref(draw_info).index_count, post_cull_triangle_count * 3);
        }
        memoryBarrierShared();
        barrier();
        triangle_count = post_cull_triangle_count;
        // Reorder threads to match compact culled triangle list.
        meshlet_triangle_index = post_cull_triangle_indices[gl_LocalInvocationID.x];
    }
    else
    {
        if (gl_LocalInvocationID.x == 0)
        {
            index_buffer_offset = atomicAdd(deref(draw_info).index_count, meshlet.triangle_count * 3);
        }
        memoryBarrierShared();
        barrier();
    }
    [[unroll]]
    for (uint loop_offset = 0; loop_offset < MAX_TRIANGLES_PER_MESHLET; loop_offset += FILL_INDEX_BUFFER_WORKGROUP_X)
    {
        const uint thread_index = gl_LocalInvocationID.x + loop_offset;
        if (thread_index < triangle_count)
        {
            const uint triangle_index = cull_triangles ? post_cull_triangle_indices[thread_index] : thread_index;
            const uint mesh_index_offset = meshlet.micro_indices_offset + triangle_index * 3;
            uint triangle_id[3] = {0, 0, 0};
            [[unroll]]
            for (uint corner_index = 0; corner_index < 3; ++corner_index)
            {
                const uint micro_index = get_micro_index(micro_index_buffer, mesh_index_offset + corner_index);
                uint vertex_id = 0;
                encode_vertex_id(instanced_meshlet_index, triangle_index, corner_index, vertex_id);
                triangle_id[corner_index] = vertex_id;
            }
            index_buffer[index_buffer_offset + thread_index * 3 + 0].value = triangle_id[0];
            index_buffer[index_buffer_offset + thread_index * 3 + 1].value = triangle_id[1];
            index_buffer[index_buffer_offset + thread_index * 3 + 2].value = triangle_id[2];
        }
    }
}