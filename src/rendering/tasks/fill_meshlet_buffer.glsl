#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "../../../shaders/cull_util.glsl"
#include "fill_meshlet_buffer.inl"

DEFINE_PUSHCONSTANT(FillMeshletBufferPush, push)
layout(local_size_x = FILL_MESHLET_BUFFER_WORKGROUP_X) in;
void main()
{
    const int test_meshlet_instance_index = int(gl_GlobalInvocationID.x);
    const int entity_count = int(deref(u_entity_meta_data).entity_count);

    daxa_RWBufferPtr(daxa_u32) instantiated_meshlet_counter = daxa_RWBufferPtr(daxa_u32)(daxa_u64(u_instantiated_meshlets));
    daxa_RWBufferPtr(InstanciatedMeshlet) instantiated_meshlets = 
        daxa_RWBufferPtr(InstanciatedMeshlet)(daxa_u64(u_instantiated_meshlets) + 32);

    // Binary Serarch the entity the meshlet id belongs to.
    InstanciatedMeshlet instanced_meshlet;
    instanced_meshlet.entity_index = 0xFFFFFFFF;
    instanced_meshlet.mesh_index = 0xFFFFFFFF;
    instanced_meshlet.meshlet_index = 0xFFFFFFFF;
    int in_entity_meshlet_index = 0xFFFFFFFF;
    if (test_meshlet_instance_index >= int(push.meshlet_count))
    {
        return;
    }
    int first = 0;
    int last = entity_count - 1;
    int middle = (first + last) / 2;
    int up_count = 0;
    int down_count = 0;
    int iter = 0;
    while(true)
    {
        ++iter;
        const int meshlet_sum_for_entity = int(deref(u_prefix_sum_mehslet_counts[middle]));
        int meshlet_sum_prev_entity = 0;
        if (middle != 0)
        {
            const uint index = middle - 1;
            meshlet_sum_prev_entity = int(deref(u_prefix_sum_mehslet_counts[index]));
        }

        if (last < first)
        {
            instanced_meshlet.entity_index = up_count;
            instanced_meshlet.mesh_index = down_count;
            instanced_meshlet.meshlet_index = iter;
            deref(instantiated_meshlets[test_meshlet_instance_index]) = instanced_meshlet;
            return;
        }
        if (test_meshlet_instance_index < meshlet_sum_prev_entity)
        {
            last = middle -1;
            down_count++;
        }
        else if (test_meshlet_instance_index >= meshlet_sum_for_entity)
        {
            first = middle + 1;
            up_count++;
        }
        else
        {
            // Found ranage.
            in_entity_meshlet_index = test_meshlet_instance_index - meshlet_sum_prev_entity;
            instanced_meshlet.entity_index = middle;
            break;
        }

        middle = (first + last) / 2;
    }
    // TODO(pahrens): there is a bug here causing the last few threads to think they belong to entity index 11, meshlet 12...
    // middle is now the entity the meshlet id belongs to.
    // Now find the mesh, the meshlet belongs to within the entity.
    const MeshList mesh_list = deref(u_entity_meshlists[instanced_meshlet.entity_index]);
    int entity_meshlet_sum = 0;
    for (int mesh_i = 0; mesh_i < mesh_list.count; ++mesh_i)
    {
        const uint mesh_id = mesh_list.mesh_indices[mesh_i];
        int meshlet_count_range_begin = entity_meshlet_sum;
        entity_meshlet_sum += int(deref(u_meshes[mesh_id]).meshlet_count);
        int meshlet_count_range_end = entity_meshlet_sum;

        if (in_entity_meshlet_index >= meshlet_count_range_begin && in_entity_meshlet_index < meshlet_count_range_end)
        {
            instanced_meshlet.meshlet_index = in_entity_meshlet_index - meshlet_count_range_begin;
            instanced_meshlet.mesh_index = mesh_i;
            instanced_meshlet.mesh_id = mesh_id;
            break;
        }
    }

#if ENABLE_MESHLET_CULLING
    Mesh mesh_data = deref(u_meshes[instanced_meshlet.mesh_id]);
    BoundingSphere bounds = deref(mesh_data.meshlet_bounds[instanced_meshlet.meshlet_index]);

    NdcBounds ndc_bounds;
    init_ndc_bounds(ndc_bounds);
    // construct bounding box from bounding sphere,
    // project each vertex of the box to ndc, min and max the coordinates.
    for (int z = -1; z <= 1; z += 2)
    {
        for (int y = -1; y <= 1; y += 2)
        {
            for (int x = -1; x <= 1; x += 2)
            {
                const vec3 bounding_box_corner_ws = bounds.center + bounds.radius * vec3(x,y,z);
                const vec4 projected_pos = globals.cull_camera_view_projection * vec4(bounding_box_corner_ws, 1);
                const vec3 ndc_pos = projected_pos.xyz / projected_pos.w;
                add_vertex_to_ndc_bounds(ndc_bounds, ndc_pos);
            }
        }
    }

    bool culled = !is_in_frustum(ndc_bounds);

    if (push.cull_alredy_visible_meshlets != 0)
    {
        EntityVisibilityBitfieldOffsets offsets = deref(u_entity_visibility_bitfield_offsets[instanced_meshlet.entity_index]);
        const uint uint_base_offset = offsets.mesh_bitfield_offset[instanced_meshlet.mesh_index];
        const uint uint_offset = uint_base_offset + (instanced_meshlet.meshlet_index / 32);
        uint mask = 1 << instanced_meshlet.meshlet_index % 32;
        const uint bitfield_section = deref(u_meshlet_visibility_bitfield[uint_offset]);
        bool visible = (mask & bitfield_section) != 0;
        // When visible set to be culled.
        culled = !visible;
    }

    if (!culled)
#endif
    {
        uint out_index = atomicAdd(deref(instantiated_meshlet_counter), 1);
        deref(instantiated_meshlets[out_index]) = instanced_meshlet;
    }
}