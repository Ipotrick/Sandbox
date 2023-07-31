#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "draw_visbuffer.inl"
#include "../../../shader_shared/visbuffer.glsl"
#include "depth_util.glsl"
#include "cull_util.glsl"
#include "cull_util.inl"
#include "observer.glsl"

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
            meshlets_to_draw = observer_get_meshlet_instance_draw_count(u_instantiated_meshlets);
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

uint get_meshlet_draw_offset_from_pass(daxa_BufferPtr(InstantiatedMeshlets) meshlet_instances, uint pass)
{
    switch (pass)
    {
        case DRAW_VISBUFFER_PASS_ONE: return 0;
        case DRAW_VISBUFFER_PASS_TWO: return deref(meshlet_instances).first_count;
        case DRAW_VISBUFFER_PASS_OBSERVER: return observer_get_meshlet_instance_draw_offset(meshlet_instances);
        default: return 0;
    }
}

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
    
    const uint meshlet_offset = get_meshlet_draw_offset_from_pass(u_instantiated_meshlets, push.pass);
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
struct NewTaskPayload
{
    uint global_meshlet_args_offset;
    uint global_meshlet_instances_offset;
    uint local_surviving_meshlet_args_mask;
};
taskPayloadSharedEXT NewTaskPayload tps;
#endif

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_TASK || !defined(DAXA_SHADER)
DAXA_DECL_PUSH_CONSTANT(DrawVisbufferCullAndDrawPush, push)
layout(local_size_x = TASK_SHADER_WORKGROUP_X) in;
void main()
{
    MeshletInstance meshlet_instance;
    bool active_thread = get_meshlet_instance_from_arg(gl_GlobalInvocationID.x, push.bucket_index, u_meshlet_cull_indirect_args, meshlet_instance);
#if ENABLE_MESHLET_CULLING
    if (active_thread)
    {
        active_thread = active_thread && !is_meshlet_occluded(
            meshlet_instance,
            u_entity_meshlet_visibility_bitfield_offsets,
            u_entity_meshlet_visibility_bitfield_arena,
            u_entity_combined_transforms,
            u_meshes,
            u_hiz);
    }
#endif
    const uint local_arg_offset = gl_SubgroupInvocationID.x;
    const uint local_surviving_meshlet_count = subgroupBallotBitCount(subgroupBallot(active_thread));
    const uint local_meshlet_instances_offset = subgroupExclusiveAdd(active_thread ? 1 : 0);
    const uint local_surviving_meshlet_args_mask = subgroupBallot(active_thread).x;
    uint global_meshlet_instances_offset;
    if (subgroupElect())
    {
        global_meshlet_instances_offset = atomicAdd(deref(u_instantiated_meshlets).second_count, local_surviving_meshlet_count) + deref(u_instantiated_meshlets).first_count;
        tps.global_meshlet_instances_offset = global_meshlet_instances_offset;
        tps.global_meshlet_args_offset = gl_GlobalInvocationID.x;
        tps.local_surviving_meshlet_args_mask = local_surviving_meshlet_args_mask;
    }
    global_meshlet_instances_offset = subgroupBroadcastFirst(global_meshlet_instances_offset);
    if (active_thread)
    {
        const uint meshlet_instance_index = global_meshlet_instances_offset + local_meshlet_instances_offset;
        deref(u_instantiated_meshlets).meshlets[meshlet_instance_index] = meshlet_instance;
    }
    EmitMeshTasksEXT(local_surviving_meshlet_count,1,1);
}
#endif 

// Very big problems with mesh shaders is that they take A LOT of shared memory space.
// For culling we must be very smart about using as little as possible.
#define MESH_SHADER_TRIANGLE_CULL 0

// Big problems with culling here:
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_MESH || !defined(DAXA_SHADER)
#if MESH_SHADER_CULL_AND_DRAW
DAXA_DECL_PUSH_CONSTANT(DrawVisbufferCullAndDrawPush, push)
#else
DAXA_DECL_PUSH_CONSTANT(DrawVisbufferPush, push)
#endif
layout(local_size_x = MESH_SHADER_WORKGROUP_X) in;
layout(triangles) out;
layout(max_vertices = MAX_VERTICES_PER_MESHLET, max_primitives = MAX_TRIANGLES_PER_MESHLET) out;
shared uint s_local_meshlet_arg_offset;
layout(location = 0) perprimitiveEXT out uint fin_triangle_id[];
layout(location = 1) perprimitiveEXT out uint fin_instantiated_meshlet_index[];
void main()
{
    #if MESH_SHADER_CULL_AND_DRAW
    const uint local_meshlet_instances_offset = gl_WorkGroupID.x;
    const uint test_thread_local_meshlet_arg_offset = gl_SubgroupInvocationID.x;
    const uint set_bits_prefix_sum = subgroupInclusiveAdd(((tps.local_surviving_meshlet_args_mask & (1u << test_thread_local_meshlet_arg_offset)) != 0) ? 1 : 0);
    if (set_bits_prefix_sum == (local_meshlet_instances_offset + 1))
    {
        if(subgroupElect())
        {
            s_local_meshlet_arg_offset = test_thread_local_meshlet_arg_offset;
        }
    }
    barrier();
    const uint arg_index = tps.global_meshlet_args_offset + s_local_meshlet_arg_offset;
    const uint meshlet_instance_index = tps.global_meshlet_instances_offset + local_meshlet_instances_offset;
    MeshletInstance instantiated_meshlet;
    bool active_thread = get_meshlet_instance_from_arg(arg_index, push.bucket_index, u_meshlet_cull_indirect_args, instantiated_meshlet);
    #else
    const uint meshlet_offset = get_meshlet_draw_offset_from_pass(u_instantiated_meshlets, push.pass);
    const uint meshlet_instance_index = gl_WorkGroupID.x + meshlet_offset;
    MeshletInstance instantiated_meshlet = deref(u_instantiated_meshlets).meshlets[meshlet_instance_index];
    #endif


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

    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref(u_meshes[instantiated_meshlet.mesh_id]).micro_indices;

    // Transform vertices:
    const mat4 model_matrix = deref(u_entity_combined_transforms[instantiated_meshlet.entity_index]);
    #if MESH_SHADER_CULL_AND_DRAW
    const mat4 view_proj_matrix = globals.camera_view_projection;
    #else

    const mat4 view_proj_matrix = (push.pass == DRAW_VISBUFFER_PASS_OBSERVER) ? globals.observer_camera_view_projection : globals.camera_view_projection;
    #endif
    SetMeshOutputsEXT(meshlet.vertex_count,meshlet.triangle_count);
    // Write Vertices:
    for (uint offset = 0; offset < meshlet.vertex_count; offset += MESH_SHADER_WORKGROUP_X)
    {
        const uint meshlet_local_vertex_index = gl_LocalInvocationID.x + offset;
        if (meshlet_local_vertex_index >= meshlet.vertex_count)
        { 
            break;
        }
        const uint vertex_index = mesh.indirect_vertices[meshlet.indirect_vertex_offset + meshlet_local_vertex_index].value;
        const vec4 vertex_pos = vec4(mesh.vertex_positions[vertex_index].value, 1);
        const vec4 vertex_pos_ws = model_matrix * vertex_pos;
        const vec4 vertex_pos_cs = view_proj_matrix * vertex_pos_ws;
        gl_MeshVerticesEXT[meshlet_local_vertex_index].gl_Position = vertex_pos_cs;
    } 
    // Write Triangles:
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
            get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + triangle_index * 3 + 2));
        gl_PrimitiveTriangleIndicesEXT[triangle_index] = triangle_micro_indices;
        uint triangle_id;
        encode_triangle_id(meshlet_instance_index, triangle_index, triangle_id);
        fin_triangle_id[triangle_index] = triangle_id;
    }
}
#endif 