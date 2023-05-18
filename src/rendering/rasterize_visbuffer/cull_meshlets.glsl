#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "../../../shaders/cull_util.glsl"
#include "cull_meshlets.inl"
#include "../../mesh/visbuffer_meshlet_util.glsl"

#if defined(CullMeshletsCommandWriteBase)
layout(local_size_x = 1) in;
void main()
{
    const uint count = deref(u_mesh_draw_list).count;
    DispatchIndirectStruct command;
    command.x = (count + CULL_MESHLETS_WORKGROUP_X - 1) / CULL_MESHLETS_WORKGROUP_X;
    command.y = 1;
    command.z = 1;
}
#else
uint get_meshlet_count(uint index)
{
    return deref(u_mesh_draw_list).mesh_dispatch_indirects[index].x;
}
layout(local_size_x = CULL_MESHLETS_WORKGROUP_X) in;
void main()
{
    const int test_meshlet_instance_index = int(gl_GlobalInvocationID.x);
    const int mesh_count = int(deref(u_mesh_draw_list).count);

    InstantiatedMeshletsView instantiated_meshlets_view = InstantiatedMeshletsView(u_instantiated_meshlets);

    // Binary Serarch the entity the meshlet id belongs to.
    int mesh_draw_index = -1;
    int meshlet_sum = -1;
    if (test_meshlet_instance_index >= int(mesh_count))
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
        const int meshlet_sum_for_entity = int(get_meshlet_count(middle));
        int meshlet_sum_prev_entity = 0;
        if (middle != 0)
        {
            const uint index = middle - 1;
            meshlet_sum_prev_entity = int(get_meshlet_count(index));
        }

        if (last < first)
        {
            // ERROR CASE
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
            mesh_draw_index = middle;
            meshlet_sum = get_meshlet_count(mesh_draw_index);
            break;
        }

        middle = (first + last) / 2;
    }
    if (mesh_draw_index == -1)
    {
        // Should not happen.
        return;
    }
    if (mesh_draw_index >= MAX_INSTANTIATED_MESHES)
    {
        // Should not happen.
        return;
    }
    const uint meshlet_index = test_meshlet_instance_index - meshlet_sum;
    
    MeshDrawInfo draw_mesh_info = deref(u_mesh_draw_list).mesh_infos[mesh_draw_index];

    InstantiatedMeshlet inst_meshlet;
    inst_meshlet.entity_index = draw_mesh_info.entity_id;
    inst_meshlet.mesh_id = draw_mesh_info.mesh_id;
    inst_meshlet.mesh_index = draw_mesh_info.mesh_index;
    inst_meshlet.meshlet_index = meshlet_index;

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

    if (true)
    {
        EntityVisibilityBitfieldOffsets offsets = deref(u_entity_visibility_bitfield_offsets[instanced_meshlet.entity_index]);
        const uint uint_base_offset = offsets.mesh_bitfield_offset[instanced_meshlet.mesh_index];
        if (uint_base_offset != (~0))
        {
            const uint uint_offset = uint_base_offset + (instanced_meshlet.meshlet_index / 32);
            uint mask = 1 << instanced_meshlet.meshlet_index % 32;
            const uint bitfield_section = deref(u_meshlet_visibility_bitfield[uint_offset]);
            bool visible = (mask & bitfield_section) != 0;
            // When visible set to be culled.
            culled = culled || visible;
        }
    }

    if (!culled)
#endif
    {
        const uint out_index = atomicAdd(instantiated_meshlets_view.second_pass_meshlet_count, 1) + instantiated_meshlets_view.first_pass_meshlet_count;
        instantiated_meshlets_view.meshlets[out_index] = instanced_meshlet;
    }
}
#endif