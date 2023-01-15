#include <daxa/daxa.inl>

#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

struct PrefixSumMeshletCountPush
{
    daxa_BufferPtr(EntityData) entities;
    daxa_BufferPtr(Mesh) meshes;
    daxa_RWBufferPtr(daxa_u32) dst;
    // Need to send raw address because of renderdoc/ nvidia driver bugs.
    // daxa_u64 entities;
    // daxa_u64 meshes;
    // daxa_u64 dst;
};