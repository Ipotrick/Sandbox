#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#if DAXA_SHADER_STAGE != DAXA_SHADER_STAGE_FRAGMENT
#include "draw_visbuffer.inl"
#include "../../../shader_shared/visbuffer.glsl"
#include "depth_util.glsl"
#include "cull_util.glsl"
#endif


#if defined(DrawVisbufferWriteCommand_COMMAND) || !defined(DAXA_SHADER)
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
layout(location = 0) flat VERTEX_OUT uint vout_triange_id;
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX || !defined(DAXA_SHADER)
DAXA_DECL_PUSH_CONSTANT(DrawVisbufferPush, push)
void main()
{
    const uint triangle_corner_index = gl_VertexIndex % 3;
    uint inst_meshlet_index;
    uint triangle_index;
    
    const uint meshlet_offset = (push.pass == DRAW_VISBUFFER_PASS_ONE || push.pass == DRAW_VISBUFFER_PASS_OBSERVER) ? 0 : deref(u_instantiated_meshlets).first_count;
    inst_meshlet_index = gl_InstanceIndex + meshlet_offset;
    triangle_index = gl_VertexIndex / 3;

    // MeshletInstance:
    // daxa_u32 entity_index;
    // daxa_u32 mesh_id;
    // daxa_u32 mesh_index;
    // daxa_u32 entity_meshlist_index;
    MeshletInstance instantiated_meshlet = deref(u_instantiated_meshlets).meshlets[inst_meshlet_index];

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
    Meshlet meshlet = mesh.meshlets[instantiated_meshlet.entity_meshlist_index].value;

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
    const vec4 pos = view_proj * deref(u_entity_combined_transforms[instantiated_meshlet.entity_index]) * vertex_position;

    uint triangle_id;
    encode_triangle_id(inst_meshlet_index, triangle_index, triangle_id);
    vout_triange_id = triangle_id;
    gl_Position = pos.xyzw;
}
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT || !defined(DAXA_SHADER)
layout(location = 0) out uint visibility_id;
void main()
{
    visibility_id = vout_triange_id;
}
#endif

#if (MESH_SHADER || MESH_SHADER_CULL_AND_DRAW) && ((DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_TASK) || (DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_MESH))
#extension GL_EXT_mesh_shader : enable
#endif

#if (MESH_SHADER_CULL_AND_DRAW) && ((DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_TASK) || (DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_MESH))
struct TaskPayload{
    MeshletInstance meshlet_instance;
    uint meshlet_instance_index;
};
taskPayloadSharedEXT TaskPayload tps[128];
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_TASK || !defined(DAXA_SHADER)
DAXA_DECL_PUSH_CONSTANT(DrawVisbufferCullAndDrawPush, push)
layout(local_size_x = 128) in;
shared uint s_meshlet_count; 
void main()
{
    MeshletInstance instanced_meshlet;
    bool active_thread = get_meshlet_instance_from_arg(gl_GlobalInvocationID.x, push.bucket_index, u_meshlet_cull_indirect_args, instanced_meshlet);
    if (!active_thread)
    {
        return;
    }
#if ENABLE_MESHLET_CULLING
    if (active_thread)
    {
        active_thread = active_thread && !is_meshlet_occluded(
            instanced_meshlet,
            u_entity_meshlet_visibility_bitfield_offsets,
            u_entity_meshlet_visibility_bitfield_arena,
            u_entity_combined_transforms,
            u_meshes,
            u_hiz);
    }
#endif
    uint meshlet_instance_index = ~0; 
    if (active_thread)
    {
        const uint out_index = atomicAdd(deref(u_instantiated_meshlets).second_count, 1);
        const uint offset = deref(u_instantiated_meshlets).first_count;
        meshlet_instance_index = out_index + offset;
        deref(u_instantiated_meshlets).meshlets[meshlet_instance_index] = instanced_meshlet;
    }

    if (gl_LocalInvocationID.x == 0)
    {
        s_meshlet_count = 0;
    }
    barrier();
    if (active_thread)
    {
        const uint out_index = atomicAdd(s_meshlet_count, 1);
        tps[out_index].meshlet_instance = instanced_meshlet;
        tps[out_index].meshlet_instance_index = meshlet_instance_index;
    }
    barrier();
    
    EmitMeshTasksEXT(s_meshlet_count,1,1);
}
#endif 

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_MESH || !defined(DAXA_SHADER)
#if MESH_SHADER_CULL_AND_DRAW
DAXA_DECL_PUSH_CONSTANT(DrawVisbufferCullAndDrawPush, push)
#else
DAXA_DECL_PUSH_CONSTANT(DrawVisbufferPush, push)
#endif
layout(local_size_x = MESH_SHADER_WORKGROUP_X) in;
layout(triangles) out;
layout(max_vertices = MAX_VERTICES_PER_MESHLET, max_primitives = MAX_TRIANGLES_PER_MESHLET) out;
struct Vertex
{
    vec4 position;
    vec3 ws_position;
    uint vis_count;
    uint post_cull_index;
};
shared Vertex s_vertices[MAX_VERTICES_PER_MESHLET];
shared uvec3 s_surviving_triangles[MAX_TRIANGLES_PER_MESHLET];
shared uint s_surviving_triangle_count;
shared uint s_surviving_vertex_count;
layout(location = 0) perprimitiveEXT out uint fin_triangle_id[];
layout(location = 1) perprimitiveEXT out uint fin_instantiated_meshlet_index[];
void main()
{
    #if MESH_SHADER_CULL_AND_DRAW
    const uint inst_meshlet_index = tps[gl_WorkGroupID.x].meshlet_instance_index;
    MeshletInstance instantiated_meshlet = tps[gl_WorkGroupID.x].meshlet_instance;
    #else
    const uint meshlet_offset = (push.pass == DRAW_VISBUFFER_PASS_ONE || push.pass == DRAW_VISBUFFER_PASS_OBSERVER) ? 0 : deref(u_instantiated_meshlets).first_count;
    const uint inst_meshlet_index = gl_WorkGroupID.x + meshlet_offset;
    MeshletInstance instantiated_meshlet = deref(u_instantiated_meshlets).meshlets[inst_meshlet_index];
    #endif

    if (gl_LocalInvocationID.x == 0)
    {
        s_surviving_triangle_count = 0;
        s_surviving_vertex_count = 0;
    }
    barrier();

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
    Meshlet meshlet = mesh.meshlets[instantiated_meshlet.entity_meshlist_index].value;
    
    const vec3 frustum_planes[5] = {
        globals.camera_right_plane_normal,
        globals.camera_left_plane_normal,
        globals.camera_top_plane_normal,
        globals.camera_bottom_plane_normal,
        globals.camera_near_plane_normal,
    };

    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref(u_meshes[instantiated_meshlet.mesh_id]).micro_indices;

    // Transform vertices:
    const mat4 model_matrix = deref(u_entity_combined_transforms[instantiated_meshlet.entity_index]);
    #if MESH_SHADER_CULL_AND_DRAW
    const mat4 view_proj_matrix = globals.camera_view_projection;
    #else
    const mat4 view_proj_matrix = (push.pass == DRAW_VISBUFFER_PASS_OBSERVER) ? globals.observer_camera_view_projection : globals.camera_view_projection;
    #endif
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
        vertex.ws_position = (model_matrix * vertex.position).xyz;
        vertex.position = view_proj_matrix * vec4(vertex.ws_position,1);
        vertex.vis_count = 0;
        s_vertices[meshlet_local_vertex_index] = vertex;
    } 
    barrier();
    // Cull triangles:
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
        bool out_of_frustum = false;
        for (uint i = 0; i < 5; ++i)
        {
            bool tri_out_of_plane = true;
            for (uint tri_i = 0; tri_i < 3; ++tri_i)
            {
                const bool vertex_out = dot((s_vertices[triangle_micro_indices[tri_i]].ws_position - globals.camera_pos), frustum_planes[i]) > 0.0f;
                tri_out_of_plane = tri_out_of_plane && vertex_out;
            }
            out_of_frustum = out_of_frustum || tri_out_of_plane;
        }
        bool culled = out_of_frustum;
        if (!culled)
        {
            uint offset = atomicAdd(s_surviving_triangle_count, 1);
            s_surviving_triangles[offset] = triangle_micro_indices;
            atomicAdd(s_vertices[triangle_micro_indices[0]].vis_count, 1);
            atomicAdd(s_vertices[triangle_micro_indices[1]].vis_count, 1);
            atomicAdd(s_vertices[triangle_micro_indices[2]].vis_count, 1);
        }
    }
    // Cull vertices:
    for (uint offset = 0; offset < meshlet.vertex_count; offset += MESH_SHADER_WORKGROUP_X)
    {
        const uint meshlet_local_vertex_index = gl_LocalInvocationID.x + offset;
        if (meshlet_local_vertex_index >= meshlet.vertex_count) 
        { 
            break;
        }

        if (s_vertices[meshlet_local_vertex_index].vis_count > 0)
        {
            const uint offset = atomicAdd(s_surviving_vertex_count, 1);
            s_vertices[meshlet_local_vertex_index].post_cull_index = offset;
        }
    }
    barrier();
    SetMeshOutputsEXT(s_surviving_vertex_count,s_surviving_triangle_count);
    // Write vertices:
    for (uint offset = 0; offset < meshlet.vertex_count; offset += MESH_SHADER_WORKGROUP_X)
    {
        const uint meshlet_local_vertex_index = gl_LocalInvocationID.x + offset;
        if (meshlet_local_vertex_index >= meshlet.vertex_count) 
        { 
            break;
        }

        if (s_vertices[meshlet_local_vertex_index].vis_count > 0)
        {
            uint out_index = s_vertices[meshlet_local_vertex_index].post_cull_index;
            gl_MeshVerticesEXT[out_index].gl_Position = s_vertices[meshlet_local_vertex_index].position;
        }
    }
    // Write triangles:
    for (uint offset = 0; offset < s_surviving_triangle_count; offset += MESH_SHADER_WORKGROUP_X)
    {
        const uint triangle_index = gl_LocalInvocationID.x + offset;
        if (triangle_index >= s_surviving_triangle_count)
        { 
            break;
        }
        const uvec3 triangle_micro_indices = s_surviving_triangles[triangle_index];
        uvec3 post_cull_triangle_micro_indices;
        // Update indices to culled vertices:
        for (uint tri_i = 0; tri_i < 3; ++tri_i)
        {
            post_cull_triangle_micro_indices[tri_i] = s_vertices[triangle_micro_indices[tri_i]].post_cull_index;
        }
        gl_PrimitiveTriangleIndicesEXT[triangle_index] = post_cull_triangle_micro_indices;
        uint triangle_id;
        encode_triangle_id(inst_meshlet_index, triangle_index, triangle_id);
        fin_triangle_id[triangle_index] = triangle_id;
    }
}
#endif 