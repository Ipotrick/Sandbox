#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "draw_visbuffer.inl"
#include "depth_util.glsl"
#include "../../mesh/visbuffer_meshlet_util.inl"


#if defined(DrawVisbufferWriteCommand_COMMAND)
DAXA_DECL_PUSH_CONSTANT(DrawVisbufferWriteCommandPush, push)
layout(local_size_x = 1) in;
void main()
{
    const uint index = gl_LocalInvocationID.x;
    uint meshlets_to_draw = 0;
    switch (push.pass)
    {
        case DRAW_VISBUFFER_PASS_ONE:
        {
            meshlets_to_draw = deref(u_instantiated_meshlets).first_count;
            break;
        }
        case DRAW_VISBUFFER_PASS_TWO:
        {
            meshlets_to_draw = deref(u_instantiated_meshlets).second_count;
            break;
        }
        case DRAW_VISBUFFER_PASS_OBSERVER:
        {
            meshlets_to_draw = deref(u_instantiated_meshlets).first_count + deref(u_instantiated_meshlets).second_count;
            break;
        }
        default: break;
    }
    if (push.mesh_shader == 1)
    {
        DispatchIndirectStruct command;
        command.x = meshlets_to_draw;
        command.y = 1;
        command.z = 1;
        deref((daxa_RWBufferPtr(DispatchIndirectStruct)(u_command))) = command;
    }
    else
    {
        DrawIndirectStruct command;
        command.vertex_count = MAX_TRIANGLES_PER_MESHLET * 3;
        command.instance_count = meshlets_to_draw;
        command.first_vertex = 0;
        command.first_instance = 0;
        deref((daxa_RWBufferPtr(DrawIndirectStruct)(u_command))) = command;
    }
}
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
#define VERTEX_OUT out 
#endif
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT
#define VERTEX_OUT in
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX || DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT
layout(location = 0) flat VERTEX_OUT uint vout_triangle_index;
layout(location = 1) flat VERTEX_OUT uint vout_instantiated_meshlet_index;
layout(location = 2) flat VERTEX_OUT uint vout_meshlet_index;
layout(location = 3) flat VERTEX_OUT uint vout_entity_index;
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
DAXA_DECL_PUSH_CONSTANT(DrawVisbufferPush, push)
void main()
{
    const uint triangle_corner_index = gl_VertexIndex % 3;
    uint inst_meshlet_index;
    uint triangle_index;
    
    const uint meshlet_offset = (push.pass == DRAW_VISBUFFER_PASS_ONE || push.pass == DRAW_VISBUFFER_PASS_OBSERVER) ? 0 : deref(u_instantiated_meshlets).first_count;
    inst_meshlet_index = gl_InstanceIndex + meshlet_offset;
    triangle_index = gl_VertexIndex / 3;

    // InstantiatedMeshlet:
    // daxa_u32 entity_index;
    // daxa_u32 mesh_id;
    // daxa_u32 mesh_index;
    // daxa_u32 meshlet_index;
    InstantiatedMeshlet instantiated_meshlet = deref(u_instantiated_meshlets).meshlets[inst_meshlet_index];

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

    // Discard triangle indices that are out of bounds of the meshlets triangle list.
    if (triangle_index >= meshlet.triangle_count)
    {
        gl_Position = vec4(2,2,2,1);
        return;
    }

    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref(u_meshes[instantiated_meshlet.mesh_id]).micro_indices;
    const uint micro_index = get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + triangle_index * 3 + triangle_corner_index);
    uint vertex_index = mesh.indirect_vertices[meshlet.indirect_vertex_offset + micro_index].value;
    vertex_index = min(vertex_index, mesh.vertex_count - 1);
    const vec4 vertex_position = vec4(mesh.vertex_positions[vertex_index].value, 1);
    const mat4x4 view_proj = (push.pass == DRAW_VISBUFFER_PASS_OBSERVER) ? globals.observer_camera_view_projection : globals.camera_view_projection;
    const vec4 pos = view_proj * deref(u_combined_transforms[instantiated_meshlet.entity_index]) * vertex_position;

    vout_triangle_index = triangle_index;
    vout_entity_index = instantiated_meshlet.entity_index;
    vout_instantiated_meshlet_index = inst_meshlet_index;
    vout_meshlet_index = instantiated_meshlet.meshlet_index;
    gl_Position = pos.xyzw;
}
#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT
DAXA_DECL_PUSH_CONSTANT(DrawVisbufferPush, push)
layout(location = 0) out uint visibility_id;
//layout(location = 1) out vec4 debug_color;
void main()
{
    float f = float(vout_entity_index * 100 + vout_meshlet_index) * 0.093213213232;
    vec3 color = vec3(cos(f), cos(f+2), cos(f+4));
    color = color * 0.5 + 0.5;
    const float near = 20.0f;
    const float far = 8000.0f;
    // color = unband_depth_color(int(gl_FragCoord.x), int(gl_FragCoord.y), gl_FragCoord.z, near, far);
    uint vis_id_out;
    encode_triangle_id(vout_instantiated_meshlet_index, vout_triangle_index, vis_id_out);
    visibility_id = vis_id_out;
    //debug_color = vec4(color,1);
}
#endif

#if MESH_SHADER
#extension GL_EXT_mesh_shader : require
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_TASK
layout(local_size_x = 32) in;
void main()
{

}
#endif 

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_MESH
DAXA_DECL_PUSH_CONSTANT(DrawVisbufferPush, push)
layout(local_size_x = MESH_SHADER_WORKGROUP_X) in;
layout(triangles) out;
layout(max_vertices = MAX_VERTICES_PER_MESHLET, max_primitives = MAX_TRIANGLES_PER_MESHLET) out;
struct Vertex
{
    vec4 position;
};
shared Vertex s_vertices[MAX_VERTICES_PER_MESHLET];
layout(location = 0) perprimitiveEXT out uint fin_triangle_index[];
layout(location = 1) perprimitiveEXT out uint fin_instantiated_meshlet_index[];
layout(location = 2) perprimitiveEXT out uint fin_meshlet_index[];
layout(location = 3) perprimitiveEXT out uint fin_entity_index[];
void main()
{
    const uint meshlet_offset = (push.pass == DRAW_VISBUFFER_PASS_ONE || push.pass == DRAW_VISBUFFER_PASS_OBSERVER) ? 0 : deref(u_instantiated_meshlets).first_count;
    const uint inst_meshlet_index = gl_WorkGroupID.x + meshlet_offset;

    // InstantiatedMeshlet:
    // daxa_u32 entity_index;
    // daxa_u32 mesh_id;
    // daxa_u32 mesh_index;
    // daxa_u32 meshlet_index;
    InstantiatedMeshlet instantiated_meshlet = deref(u_instantiated_meshlets).meshlets[inst_meshlet_index];

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

    // Transform vertices:
    const mat4 model_matrix = deref(u_combined_transforms[instantiated_meshlet.entity_index]);
    const mat4 view_proj_matrix = (push.pass == DRAW_VISBUFFER_PASS_OBSERVER) ? globals.observer_camera_view_projection : globals.camera_view_projection;
    for (uint offset = 0; offset < meshlet.vertex_count; offset += MESH_SHADER_WORKGROUP_X)
    {
        const uint meshlet_local_vertex_index = gl_LocalInvocationID.x + offset;
        if (meshlet_local_vertex_index >= meshlet.vertex_count)
        { 
            break;
        }
        const uint vertex_index = mesh.indirect_vertices[meshlet.indirect_vertex_offset + meshlet_local_vertex_index].value;
        Vertex vertex;
        vertex.position = vec4(mesh.vertex_positions[vertex_index].value, 1);
        vertex.position = view_proj_matrix * model_matrix * vertex.position;
        s_vertices[meshlet_local_vertex_index] = vertex;
    } 
    // TODO: Cull triangles
    SetMeshOutputsEXT(meshlet.vertex_count,meshlet.triangle_count);
    // Write out vertices:
    for (uint offset = 0; offset < meshlet.vertex_count; offset += MESH_SHADER_WORKGROUP_X)
    {
        const uint meshlet_local_vertex_index = gl_LocalInvocationID.x + offset;
        if (meshlet_local_vertex_index >= meshlet.vertex_count) 
        { 
            break;
        }
        gl_MeshVerticesEXT[meshlet_local_vertex_index].gl_Position = s_vertices[meshlet_local_vertex_index].position;
    }
    // Write traingle indices:
    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref(u_meshes[instantiated_meshlet.mesh_id]).micro_indices;
    for (uint offset = 0; offset < meshlet.triangle_count; offset += MESH_SHADER_WORKGROUP_X)
    {
        const uint triangle_index = gl_LocalInvocationID.x + offset;
        if (triangle_index >= meshlet.triangle_count)
        { 
            break;
        }
        const uvec3 triangle_micro_indices = uvec3(
            get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + triangle_index * 3 + 0),
            get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + triangle_index * 3 + 1),
            get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + triangle_index * 3 + 2)
        );
        gl_PrimitiveTriangleIndicesEXT[triangle_index] = triangle_micro_indices;
        fin_triangle_index[triangle_index] = triangle_index;
        fin_instantiated_meshlet_index[triangle_index] = inst_meshlet_index;
        fin_meshlet_index[triangle_index] = instantiated_meshlet.meshlet_index;
        fin_entity_index[triangle_index] = instantiated_meshlet.entity_index;
    }
}
#endif 
#endif