#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "draw_opaque_ids.inl"
#include "visbuffer.glsl"
#include "depth_util.glsl"
#include "../../mesh/visbuffer_meshlet_util.glsl"

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
#define VERTEX_OUT out
#else
#define VERTEX_OUT in
#endif

layout(location = 0) flat VERTEX_OUT uint vout_triangle_index;
layout(location = 1) flat VERTEX_OUT uint vout_instantiated_meshlet_index;
layout(location = 2) flat VERTEX_OUT uint vout_meshlet_index;
layout(location = 3) flat VERTEX_OUT uint vout_entity_index;

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX

void main()
{
    const bool indexed_id_rendering = globals.settings.indexed_id_rendering == 1;

    uint instantiated_meshlet_index = 0;
    uint triangle_index = 0;
    uint corner_index = 0;

    if (indexed_id_rendering)
    {
        const uint vertex_id = gl_VertexIndex;
        decode_vertex_id(vertex_id, instantiated_meshlet_index, triangle_index, corner_index);
    }
    else
    {
        instantiated_meshlet_index = gl_InstanceIndex;
        const uint meshlet_local_triangle_corner = gl_VertexIndex;
        triangle_index = meshlet_local_triangle_corner / 3; // gl_PrimitiveID
        corner_index = meshlet_local_triangle_corner % 3; // gl_PrimitiveIDIn
    }

    // InstantiatedMeshlet:
    // daxa_u32 entity_index;
    // daxa_u32 mesh_id;
    // daxa_u32 mesh_index;
    // daxa_u32 meshlet_index;
    InstantiatedMeshlet instantiated_meshlet = InstantiatedMeshletsView(u_instantiated_meshlets).meshlets[instantiated_meshlet_index];

    // Mesh:
    // daxa_BufferId mesh_buffer;
    // daxa_u32 meshlet_count;
    // daxa_BufferPtr(Meshlet) meshlets;
    // daxa_BufferPtr(BoundingSphere) meshlet_bounds;
    // daxa_BufferPtr(daxa_u32) micro_indices;
    // daxa_BufferPtr(daxa_u32) indirect_vertices;
    // daxa_BufferPtr(daxa_f32vec3) vertex_positions;
    Mesh mesh = deref((u_meshes + instantiated_meshlet.mesh_id));
    
    // Meshlet:
    // daxa_u32 indirect_vertex_offset;
    // daxa_u32 micro_indices_offset;
    // daxa_u32 vertex_count;
    // daxa_u32 triangle_count;
    Meshlet meshlet = mesh.meshlets[instantiated_meshlet.meshlet_index].value;

    if (triangle_index >= meshlet.triangle_count)
    {
        gl_Position = vec4(2,2,2,1);
        return;
    }

    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref(u_meshes[instantiated_meshlet.mesh_id]).micro_indices;
    const uint micro_index = get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + triangle_index * 3 + corner_index);
    const uint vertex_index = mesh.indirect_vertices[meshlet.indirect_vertex_offset + micro_index].value;
    const vec4 vertex_position = vec4(mesh.vertex_positions[vertex_index].value, 1);
    const vec4 pos = globals.camera_view_projection * vertex_position;

    vout_triangle_index = triangle_index;
    vout_entity_index = instantiated_meshlet.entity_index;
    vout_instantiated_meshlet_index = instantiated_meshlet_index;
    vout_meshlet_index = instantiated_meshlet.meshlet_index;
    gl_Position = pos.xyzw;
}
#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT

layout(location = 0) out uint visibility_id;
layout(location = 1) out vec4 debug_color;
void main()
{
    float f = float(vout_entity_index * 100 + vout_meshlet_index) * 0.093213213232;
    vec3 color = vec3(cos(f), cos(f+2), cos(f+4));
    color = color * 0.5 + 0.5;
    const float near = 20.0f;
    const float far = 8000.0f;
    // color = unband_depth_color(int(gl_FragCoord.x), int(gl_FragCoord.y), gl_FragCoord.z, near, far);
    uint vis_id_out;
    encode_pixel_id(vout_instantiated_meshlet_index, vout_triangle_index, vis_id_out);
    visibility_id = vis_id_out;
    debug_color = vec4(color,1);
}
#endif