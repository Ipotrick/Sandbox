#include <daxa/daxa.inl>
#include "draw_opaque_ids.inl"

#extension GL_EXT_debug_printf : enable
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

layout(location = 3) flat out uint vout_instanciated_meshlet_index;
layout(location = 4) flat out uint vout_entity_index;
layout(location = 5) flat out uint vout_meshlet_index;

void main()
{
    uint vertex_id = gl_VertexIndex;
    uint instanciated_meshlet_index = 0;
    uint micro_index = 0;
    decode_vertex_id(vertex_id, instanciated_meshlet_index, micro_index);
    vout_instanciated_meshlet_index = instanciated_meshlet_index;

    // InstanciatedMeshlet:
    // daxa_u32 entity_index;
    // daxa_u32 mesh_id;
    // daxa_u32 mesh_index;
    // daxa_u32 meshlet_index;
    InstanciatedMeshlet instanciated_meshlet = 
        deref((daxa_BufferPtr(InstanciatedMeshlet)(daxa_u64(u_instanciated_meshlets) + 32) + instanciated_meshlet_index));

    // Mesh:
    // daxa_BufferId mesh_buffer;
    // daxa_u32 meshlet_count;
    // daxa_BufferPtr(Meshlet) meshlets;
    // daxa_BufferPtr(BoundingSphere) meshlet_bounds;
    // daxa_BufferPtr(daxa_u32) micro_indices;
    // daxa_BufferPtr(daxa_u32) indirect_vertices;
    // daxa_BufferPtr(daxa_f32vec3) vertex_positions;
    daxa_BufferPtr(Mesh) mesh = u_meshes + instanciated_meshlet.mesh_id;
    
    // Meshlet:
    // daxa_u32 indirect_vertex_offset;
    // daxa_u32 micro_indices_offset;
    // daxa_u32 vertex_count;
    // daxa_u32 triangle_count;
    Meshlet meshlet = mesh.value.meshlets[instanciated_meshlet.meshlet_index].value;
    const uint vertex_index = mesh.value.indirect_vertices[meshlet.indirect_vertex_offset + micro_index].value;
    // This read on the next line causes a false validation message, claiming an out of bounds access!
    const vec4 vertex_position = vec4(mesh.value.vertex_positions[vertex_index].value, 1);
    vout_entity_index = instanciated_meshlet.entity_index;
    vout_meshlet_index = instanciated_meshlet.meshlet_index;
    daxa_u64 addr = daxa_u64(mesh.value.vertex_positions + vertex_index);

    // This is the last valid address when reading vec3's in scalar format
    daxa_u64 last_valud_addr = mesh.value.end_ptr - 12;
    // Manual bounds check does not find an error
    if (addr > last_valud_addr)
    {
        debugPrintfEXT("address: %u%u end address:%u%u\n", uint(addr << 32), uint(addr), int(mesh.value.end_ptr << 32), uint(mesh.value.end_ptr));
    }
    gl_Position = vertex_position + vec4(instanciated_meshlet.meshlet_index + mesh.value.meshlet_count + meshlet.indirect_vertex_offset + vertex_index,1,1,1);//pos.xyzw;
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
    visibility_id = 1;
    debug_color = vec4(color,1);
}
#endif