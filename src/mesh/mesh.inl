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

struct Object
{
    daxa_f32mat4x4 matrix;
    daxa_u32 model_index;
};

/// 
/// A Model is a collection of meshes with some meta information.
/// Each model can point to up to 8 meshes.
///
struct Model
{
    daxa_u32 mesh_indices[8];
    daxa_u32 mesh_count;
};

/// An array of draw infos can be indexed with draw indirect
struct MeshDrawInfo
{
    daxa_u32 object_id;
    daxa_u32 mesh_id;
};

#define MAX_TRIANGLES_PER_MESHLET 124
#define MAX_VERTICES_PER_MESHLET 64

// !!NEEDS TO BE ABI COMPATIBLE WITH meshopt_Meshlet!!
struct Meshlet
{
    daxa_u32 indirect_vertex_offset;
    daxa_u32 triangle_offset;
    daxa_u32 vertex_count;
    daxa_u32 triangle_count;
};
DAXA_ENABLE_BUFFER_PTR(Meshlet)

/// Can be indexed when drawing meshlets via draw indirect.
struct MeshletDrawInfo
{
    daxa_u32 object_index;
    daxa_u32 mesh_index;
    daxa_u32 meshlet_index;
    daxa_u32 pad;          // Alignment 16 access optimization.
};
DAXA_ENABLE_BUFFER_PTR(MeshletDrawInfo)

struct InstanciatedMeshlet
{
    daxa_u32 entity_index;
    daxa_u32 mesh_id;
    daxa_u32 mesh_index;
    daxa_u32 meshlet_index;
};
DAXA_ENABLE_BUFFER_PTR(InstanciatedMeshlet)

struct TriangleId
{
    // Bits 7-31: instanciated meshlet id;
    // Bits 0-6: in meshlet id;
    daxa_u32 value;
};
DAXA_ENABLE_BUFFER_PTR(TriangleId)

struct BoundingSphere{
    daxa_f32vec3 center;
    daxa_f32 radius;
};
DAXA_ENABLE_BUFFER_PTR(BoundingSphere)

#if defined(DAXA_SHADER)

void encode_vertex_id(daxa_u32 instanciated_meshlet_index, daxa_u32 micro_index, out daxa_u32 vertex_id)
{
    vertex_id = (instanciated_meshlet_index << 6) | micro_index;
}

void decode_vertex_id(daxa_u32 vertex_id, out daxa_u32 instanciated_meshlet_index, out daxa_u32 micro_index)
{
    instanciated_meshlet_index = vertex_id >> 6;
    micro_index = vertex_id & 0x3F;
}

#endif // #if defined(DAXA_SHADER)

/// 
/// A Mesh is a piece of a model.
/// All triangles within a Mesh have the same material and textures.
/// A mesh is allocated into one chunk of memory, in one buffer. That is the mesh_buffer.
/// A mesh has at least one:
/// * meshlet buffer
/// * meshlet micro index buffer
/// * meshlet indirect vertex buffer
/// * vertex position buffer
///
/// To access any of these buffers, get the buffer device address of the mesh_buffer and add on it the relevant offset.
/// The address can then be casted to a daxa_RWBuffer(TYPE), to access the data efficiently. 
/// 
struct Mesh
{
    daxa_BufferId mesh_buffer;
    daxa_u32 meshlet_count;
    daxa_BufferPtr(Meshlet) meshlets;
    daxa_BufferPtr(BoundingSphere) meshlet_bounds;
    daxa_BufferPtr(daxa_u32) micro_indices;
    daxa_BufferPtr(daxa_u32) indirect_vertices;
    daxa_BufferPtr(daxa_f32vec3) vertex_positions;
};
DAXA_ENABLE_BUFFER_PTR(Mesh)

#if defined(DAXA_SHADER)
uint get_micro_index(daxa_BufferPtr(daxa_u32) micro_indices, daxa_u32 index_offset)
{
    uint pack_index = index_offset / 4;
    uint index_pack = deref(micro_indices[pack_index]);
    uint in_pack_offset = index_offset % 4;
    uint in_pack_shift = in_pack_offset * 8;
    return (index_pack >> in_pack_shift) & 0xFF;
}
#endif // #if defined(DAXA_SHADER)

// mesh.meshlets.get[index]

// Meshlet rendering strategy:
// There will be the following persistent buffers:
// 0. start a list of instanciated meshlets
///   * first content is just the visible meshlets from last frame
// 1. draw the current list of instanciated meshlets (currently only visible meshlets from last frame)
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

// There are 4 renderpaths planned, and will be implemented in the following order:
// 1. multi draw indirect, one indirect draw enty per meshlet
// 2. index buffer expansion, one draw for everything
// 3. non indexed, one draw indirect for everything
// 3. mesh shaders (way later)

// There are 2 shading pipelines planned:
// 1. Culled Indirect vis buffer shading (aka Unreal 5 rearly depth test material trick)
// 2. Culled forward
// 3. Brute force forward

// Dynamic Ligghts will first be bruteforced then later 3d volume cluster culled.