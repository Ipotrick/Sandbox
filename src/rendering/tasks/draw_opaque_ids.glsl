#include <daxa/daxa.inl>
#include "draw_opaque_ids.inl"

#extension GL_EXT_debug_printf : enable
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

layout(location = 1) out float kill;
layout(location = 2) out vec3 wPos;
layout(location = 3) flat out uint vout_instanciated_meshlet_index;

void main()
{
    uint vertex_id = gl_VertexIndex;
    uint instanciated_meshlet_index = 0;
    uint micro_index = 0;
    decode_vertex_id(vertex_id, instanciated_meshlet_index, micro_index);
    vout_instanciated_meshlet_index = instanciated_meshlet_index;

    // daxa_u32 entity_index;
    // daxa_u32 mesh_id;
    // daxa_u32 mesh_index;
    // daxa_u32 meshlet_index;
    InstanciatedMeshlet instanciated_meshlet = instanciated_meshlets[instanciated_meshlet_index].value;

    // daxa_BufferId mesh_buffer;
    // daxa_u32 meshlet_count;
    // daxa_BufferPtr(Meshlet) meshlets;
    // daxa_BufferPtr(BoundingSphere) meshlet_bounds;
    // daxa_BufferPtr(daxa_u32) micro_indices;
    // daxa_BufferPtr(daxa_u32) indirect_vertices;
    // daxa_BufferPtr(daxa_f32vec3) vertex_positions;
    daxa_BufferPtr(Mesh) mesh = meshes + instanciated_meshlet.mesh_id;
    
    // daxa_u32 indirect_vertex_offset;
    // daxa_u32 micro_indices_offset;
    // daxa_u32 vertex_count;
    // daxa_u32 triangle_count;
    Meshlet meshlet = mesh.value.meshlets[instanciated_meshlet.meshlet_index].value;
    const uint vertex_index = mesh.value.indirect_vertices[meshlet.indirect_vertex_offset + micro_index].value;
    const vec4 vertex_position = vec4(mesh.value.vertex_positions[vertex_index].value, 1);
    mat4 model_matrix = combined_transforms[instanciated_meshlet.entity_index].value;

    vec4 worldPos = model_matrix * vertex_position.xzyw;
    vec4 pos = globals.value.camera_view_projection * vertex_position.xyzw;

    kill = 0.0f;

    wPos = vertex_position.xyz;
    gl_Position = pos.xyzw;
}
#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT
layout(location = 1) in float kill;
layout(location = 2) in vec3 wPos;
layout(location = 3) flat in uint vout_instanciated_meshlet_index;

layout(location = 0) out uint visibility_id;
layout(location = 1) out vec4 debug_color;
void main()
{
    float f = float(vout_instanciated_meshlet_index) * 0.13213213232;
    vec3 color = vec3(cos(f), cos(f+2), cos(f+4));
    color = color * 0.5 + 0.5;
    visibility_id = 1;
    debug_color = vec4(color,1);
}
#endif