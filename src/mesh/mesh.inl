#pragma once

#include <daxa/daxa.inl>

struct DrawIndexedIndirectCommand
{
    daxa_u32 index_count;
    daxa_u32 instance_count;
    daxa_u32 first_index;
    daxa_u32 vertex_offset;
    daxa_u32 first_instance;
};

struct Model
{
    daxa_f32mat4x4 matrix;
    daxa_u32 first_mesh_index;
    daxa_u32 mesh_count;
};

struct Mesh
{
    daxa_u32 first_meshlet;
    daxa_u32 meshlet_count;
    daxa_ImageId texture_albedo;
    daxa_ImageId texture_normals;
    daxa_f32vec2 aabb_min;
    daxa_f32vec2 aabb_max;
    daxa_u32 vertex_offset_position;
    daxa_u32 vertex_offset_uv;
    daxa_u32 vertex_offset_normal;
};

struct Meshlet
{
    daxa_f32vec2 aabb_min;
    daxa_f32vec2 aabb_max;
};

struct InstanciatedMeshlet
{
    daxa_u32 object_index;
    daxa_u32 mesh_index;
    daxa_u32 meshlet_index;
};

struct TriangleId
{
    // Bits 7-31: instanciated meshlet id;
    // Bits 0-6: in meshlet id;
    daxa_u32 value;
};

DAXA_DECL_BUFFER(
    InstanciatedMeshlets,
    {
        daxa_u32 count;
        InstanciatedMeshlet meshlets[];
    }
);

DAXA_DECL_BUFFER(
    IndexBuffer,
    {
        daxa_u32 count;
        daxa_u32 indices[];
    }
);

// Meshlet rendering strategy:
// There will be the following persistent buffers:
// 0. start a list of drawn meshlets
///   * first content is just the visible meshlets from last frame
// 1. draw the current list of drawn meshlets (currently only visible meshlets from last frame)
// 2. generate HIZ buffer from current depth
// 3. cull vs HIZ
//    * cull frustum first
//    * cull object
//    * cull mesh
//    * cull meshlets (ignore meshlets that have already been drawn, need to store a list of already drawn meshlets per mesh for that)
//    * write result out to another indirect dispatch buffer for full draw pass
// 4. draw culled results
// 5. visbuffer scan pass:
//    * compute pass
//    * large workgroup, 16x16 (maybe smaller or bigger, depends on how well it runs)
//    * scan for visible meshlets by looking at all object and meshlet ids in the visbuffer, mark visible meshlets in a per mesh list
//    * draw to a fake depth texture what material what pixel is (one depth value per material 16bit)
//    * mark tiles (16x16) on features like if they contain material X at all into a bitfield (16bit)
//    * fill indirect dispatch buffer for material passes, only queue a material pass for the tile if it contains material X
// 6. opaque material pass
//    * 16x16 tile quads indirect dispatches for all materials for the whole screen.
//    * now use material depth texture as real depth texture with depth equal. Render each material quad at its fake depth material. This abuses the early z test hardware to pack threads into efficient waves.
// 7. to prefix sum on all objects meshes and meshlets for visibility
// 8. write out new visible meshlet buffer, using the prefix sum values per meshlet.
