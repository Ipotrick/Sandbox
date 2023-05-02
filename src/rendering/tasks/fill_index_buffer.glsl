#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "fill_index_buffer.inl"
#include "../../../shaders/util.glsl"

shared uint post_cull_triangle_count;
shared uint index_buffer_offset;
layout(local_size_x = FILL_INDEX_BUFFER_WORKGROUP_X) in;
void main()
{
    // TODO add tringle visibility tests
    const bool indexed_id_rendering = globals.settings.indexed_id_rendering == 1;
    if (!indexed_id_rendering)
    {
        if (gl_GlobalInvocationID.x == 0)
        {
            daxa_RWBufferPtr(DrawIndirectStruct) draw_info = daxa_RWBufferPtr(DrawIndirectStruct)(daxa_u64(u_index_buffer_and_count));
            daxa_BufferPtr(DispatchIndirectStruct) meshlet_dispatch = daxa_BufferPtr(DispatchIndirectStruct)(daxa_u64(u_instanciated_meshlets));
            deref(draw_info).vertex_count = 128 * 3 * deref(meshlet_dispatch).x;
        }
        return;
    }
    const daxa_u32 instanced_meshlet_index = gl_WorkGroupID.x;
    daxa_u32 meshlet_triangle_index = gl_LocalInvocationID.x;
    daxa_RWBufferPtr(DrawIndexedIndirectStruct) draw_info = daxa_RWBufferPtr(DrawIndexedIndirectStruct)(daxa_u64(u_index_buffer_and_count));
    daxa_RWBufferPtr(daxa_u32) index_buffer = u_index_buffer_and_count + 8;

    daxa_BufferPtr(InstanciatedMeshlet) instanciated_meshlets = 
        daxa_BufferPtr(InstanciatedMeshlet)(daxa_u64(u_instanciated_meshlets) + 32);
    InstanciatedMeshlet instanced_meshlet = deref(instanciated_meshlets[instanced_meshlet_index]);
    Meshlet meshlet = u_meshes[instanced_meshlet.mesh_id].value.meshlets[instanced_meshlet.meshlet_index].value;
    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref(u_meshes[instanced_meshlet.mesh_id]).micro_indices;
    daxa_BufferPtr(daxa_u32) indirect_vertices = deref(u_meshes[instanced_meshlet.mesh_id]).indirect_vertices;
    const uint triangle_count = meshlet.triangle_count;
    bool is_active = meshlet_triangle_index < meshlet.triangle_count;

    BoundingSphere meshlet_bounds = deref(deref(u_meshes[instanced_meshlet.mesh_id]).meshlet_bounds[instanced_meshlet.meshlet_index]);

    const float threshhold = 400.0f;
    const bool cull_triangles = true;//meshlet_bounds.radius < threshhold;
    if (cull_triangles)
    {
        if (gl_LocalInvocationID.x == 0)
        {
            post_cull_triangle_count = 0;
        }
        memoryBarrierShared();
        barrier();

        if (is_active)
        {
            const uint mesh_index_offset = meshlet.micro_indices_offset + meshlet_triangle_index * 3;
            Mesh mesh = deref(u_meshes[instanced_meshlet.mesh_id]);
            vec3 ndc_min;
            vec3 ndc_max;
            // construct bounding box from bounding sphere,
            // project each vertex of the box to ndc, min and max the coordinates.
            for (uint tri_index = 0; tri_index < 3; ++tri_index)
            {
                const uint micro_index = get_micro_index(micro_index_buffer, mesh_index_offset + tri_index);
                const uint vertex_index = mesh.indirect_vertices[meshlet.indirect_vertex_offset + micro_index].value;
                const vec3 vertex_position_ws = mesh.vertex_positions[vertex_index].value;
                const vec4 vertex_position_proj = globals.cull_camera_view_projection * vec4(vertex_position_ws,1);
                const vec3 vertex_position = vertex_position_proj.xyz / vertex_position_proj.w;
                if (tri_index == 0)
                {
                    ndc_min = vertex_position;
                    ndc_max = vertex_position;
                }
                else
                {
                    ndc_min = vec3(
                        min(vertex_position.x, ndc_min.x),
                        min(vertex_position.y, ndc_min.y),
                        min(vertex_position.z, ndc_min.z)
                    );
                    ndc_max = vec3(
                        max(vertex_position.x, ndc_max.x),
                        max(vertex_position.y, ndc_max.y),
                        max(vertex_position.z, ndc_max.z)
                    );
                }
            }

            const bool out_of_frustum = ndc_max.z < 0.0f ||
                                        ndc_min.x > 1.0f ||
                                        ndc_min.y > 1.0f ||
                                        ndc_max.x < -1.0f ||
                                        ndc_max.y < -1.0f;

            const bool triangle_visible = !out_of_frustum;
            if (triangle_visible)
            {
                meshlet_triangle_index = atomicAdd(post_cull_triangle_count, 1);
            }
            // Deactivate threads with culled triangles.
            is_active = is_active && triangle_visible;
        }
        memoryBarrierShared();
        barrier();
        if (gl_LocalInvocationID.x == 0)
        {
            index_buffer_offset = atomicAdd(deref(draw_info).index_count, post_cull_triangle_count * 3);
        }
        memoryBarrierShared();
        barrier();
    }
    else
    {
        if (gl_LocalInvocationID.x == 0)
        {
            post_cull_triangle_count = 0;
            index_buffer_offset = atomicAdd(deref(draw_info).index_count, meshlet.triangle_count * 3);
        }
        memoryBarrierShared();
        barrier();
    }

    if (is_active)
    {
        const uint mesh_index_offset = meshlet.micro_indices_offset + meshlet_triangle_index * 3;
        uint triangle_id[3] = {0, 0, 0};
        for (uint tri_index = 0; tri_index < 3; ++tri_index)
        {
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