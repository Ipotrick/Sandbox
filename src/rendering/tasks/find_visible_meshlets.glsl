#include <daxa/daxa.glsl>

#include "find_visible_meshlets.inl"

DEFINE_PUSHCONSTANT(FindVisibleMeshletsPush, push)
layout(local_size_x = 96) in;
void main()
{
    const int global_invocation_id = int(gl_GlobalInvocationID.x);
    const int meshlet_id = global_invocation_id;
    const int entity_count = int(deref(push.entities).entity_count);

    // Binary Serarch the entity the meshlet id belongs to.
    int in_entity_meshlet_id = 0;
    int entity_index = 0;
    int in_entity_mesh_index = 0;
    int in_entity_in_mesh_meshlet_index = 0;
    if (global_invocation_id >= int(push.meshlet_count))
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
        const int meshlet_sum_for_entity = int(deref(push.prefix_sum_mehslet_counts[middle]));
        int meshlet_sum_prev_entity = 0;
        if (middle != 0)
        {
            const uint index = middle - 1;
            meshlet_sum_prev_entity = int(deref(push.prefix_sum_mehslet_counts[index]));
        }

        if (last < first)
        {
            InstanciatedMeshlet instanced_meshlet;
            instanced_meshlet.entity_index = up_count;
            instanced_meshlet.mesh_index = down_count;
            instanced_meshlet.meshlet_index = iter;
            instanced_meshlet.pad = meshlet_id;
            deref(push.instanciated_meshlets[meshlet_id]) = instanced_meshlet;
            return;
        }
        if (meshlet_id < meshlet_sum_prev_entity)
        {
            last = middle -1;
            down_count++;
        }
        else if (meshlet_id > meshlet_sum_for_entity)
        {
            first = middle + 1;
            up_count++;
        }
        else
        {
            // Found ranage.
            in_entity_meshlet_id = meshlet_id - meshlet_sum_prev_entity;
            entity_index = middle;
            break;
        }

        middle = (first + last) / 2;
    }
    // middle is now the entity the meshlet id belongs to.
    // Now find the mesh, the meshlet belongs to within the entity.
    const MeshList mesh_list = deref(push.entities).meshes[entity_index];
    int entity_meshlet_sum = 0;
    for (int mesh_i = 0; mesh_i < mesh_list.count; ++mesh_i)
    {
        const uint mesh_id = mesh_list.mesh_indices[mesh_i];
        int meshlet_count_range_begin = entity_meshlet_sum;
        entity_meshlet_sum += int(deref(push.meshes[mesh_id]).meshlet_count);
        int meshlet_count_range_end = entity_meshlet_sum;

        if (in_entity_meshlet_id >= meshlet_count_range_begin && in_entity_meshlet_id < meshlet_count_range_end)
        {
            in_entity_in_mesh_meshlet_index = in_entity_meshlet_id - meshlet_count_range_begin;
            in_entity_mesh_index = mesh_i;
            break;
        }
    }

    InstanciatedMeshlet instanced_meshlet;
    instanced_meshlet.entity_index = entity_index;
    instanced_meshlet.mesh_index = in_entity_mesh_index;
    instanced_meshlet.meshlet_index = in_entity_in_mesh_meshlet_index;
    deref(push.instanciated_meshlets[meshlet_id]) = instanced_meshlet;
}