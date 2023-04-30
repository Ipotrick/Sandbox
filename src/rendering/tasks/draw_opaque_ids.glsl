#include <daxa/daxa.inl>
#include "draw_opaque_ids.inl"

#extension GL_EXT_debug_printf : enable
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

layout(location = 3) flat out uint vout_instanciated_meshlet_index;
layout(location = 4) flat out uint vout_entity_index;
layout(location = 5) flat out uint vout_meshlet_index;

void main()
{
    const bool indexed_id_rendering = globals.settings.indexed_id_rendering == 1;

    uint instanciated_meshlet_index = 0;
    uint micro_index = 0;

    if (indexed_id_rendering)
    {
        const uint vertex_id = gl_VertexIndex;
        instanciated_meshlet_index = 0;
        micro_index = 0;
        decode_vertex_id(vertex_id, instanciated_meshlet_index, micro_index);
        vout_instanciated_meshlet_index = instanciated_meshlet_index;
    }
    else
    {
        instanciated_meshlet_index = gl_VertexIndex / (128 * 3);
        InstanciatedMeshlet instanciated_meshlet = 
            deref((daxa_BufferPtr(InstanciatedMeshlet)(daxa_u64(u_instanciated_meshlets) + 32) + instanciated_meshlet_index));
        Mesh mesh = deref((u_meshes + instanciated_meshlet.mesh_id));
        Meshlet meshlet = mesh.meshlets[instanciated_meshlet.meshlet_index].value;
        const uint meshlet_local_triangle_corner = gl_VertexIndex % (128 * 3);
        if (meshlet_local_triangle_corner >= meshlet.triangle_count * 3)
        {
            gl_Position = vec4(2,2,2,1);
            return;
        }
        daxa_BufferPtr(daxa_u32) micro_index_buffer = deref(u_meshes[instanciated_meshlet.mesh_id]).micro_indices;
        micro_index = get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + meshlet_local_triangle_corner);
        const uint vertex_index = mesh.indirect_vertices[meshlet.indirect_vertex_offset + micro_index].value;
        const vec4 vertex_position = vec4(mesh.vertex_positions[vertex_index].value, 1);
        const vec4 pos = globals.camera_view_projection * vertex_position;
        vout_entity_index = instanciated_meshlet.entity_index;
        vout_meshlet_index = instanciated_meshlet.meshlet_index;
        gl_Position = pos.xyzw;
        return;
    }

    // InstanciatedMeshlet:
    // daxa_u32 entity_index;
    // daxa_u32 mesh_id;
    // daxa_u32 mesh_index;
    // daxa_u32 meshlet_index;
    InstanciatedMeshlet instanciated_meshlet = 
        deref((daxa_BufferPtr(InstanciatedMeshlet)(daxa_u64(u_instanciated_meshlets) + 32) + instanciated_meshlet_index));
    vout_entity_index = instanciated_meshlet.entity_index;
    vout_meshlet_index = instanciated_meshlet.meshlet_index;

    // Mesh:
    // daxa_BufferId mesh_buffer;
    // daxa_u32 meshlet_count;
    // daxa_BufferPtr(Meshlet) meshlets;
    // daxa_BufferPtr(BoundingSphere) meshlet_bounds;
    // daxa_BufferPtr(daxa_u32) micro_indices;
    // daxa_BufferPtr(daxa_u32) indirect_vertices;
    // daxa_BufferPtr(daxa_f32vec3) vertex_positions;
    Mesh mesh = deref((u_meshes + instanciated_meshlet.mesh_id));
    
    // Meshlet:
    // daxa_u32 indirect_vertex_offset;
    // daxa_u32 micro_indices_offset;
    // daxa_u32 vertex_count;
    // daxa_u32 triangle_count;
    Meshlet meshlet = mesh.meshlets[instanciated_meshlet.meshlet_index].value;

    const uint vertex_index = mesh.indirect_vertices[meshlet.indirect_vertex_offset + micro_index].value;
    const vec4 vertex_position = vec4(mesh.vertex_positions[vertex_index].value, 1);
    const vec4 pos = globals.camera_view_projection * vertex_position;
    gl_Position = pos.xyzw;
}
#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT
layout(location = 3) flat in uint vout_instanciated_meshlet_index;
layout(location = 4) flat in uint vout_entity_index;
layout(location = 5) flat in uint vout_meshlet_index;

layout(location = 0) out uint visibility_id;
layout(location = 1) out vec4 debug_color;
void main()
{
    float f = float(vout_entity_index * 100 + vout_meshlet_index) * 0.093213213232;
    vec3 color = vec3(cos(f), cos(f+2), cos(f+4));
    color = color * 0.5 + 0.5;
    const float originalZ = gl_FragCoord.z * gl_FragCoord.w;
    //color = color * vec3(originalZ,originalZ,originalZ) * 10000;
    visibility_id = 1;
    debug_color = vec4(color,1);
}
#endif