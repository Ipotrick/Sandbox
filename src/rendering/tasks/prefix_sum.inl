#pragma once

#include <daxa/daxa.inl>

#include "../../mesh/mesh.inl"
#include "../../scene/scene.inl"
#include "../../../shaders/util.inl"

struct PrefixSumMeshletCountPush
{
    daxa_BufferPtr(EntityMetaData) entity_meta_data;
    daxa_BufferPtr(MeshList) entity_meshlists;
    daxa_BufferPtr(Mesh) meshes;
    daxa_RWBufferPtr(daxa_u32) dst;
};

#if __cplusplus

#include "../gpu_context.hpp"

static constexpr std::string_view PREFIX_SUM_NAME = "prefix sum";
inline static const daxa::ComputePipelineCompileInfo PREFIX_SUM_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"util.glsl"},
        .compile_options = {
            .defines = {{"ENTRY_PREFIX_SUM"}},
        },
    },
    .push_constant_size = sizeof(PrefixSumPush),
    .name = std::string{PREFIX_SUM_NAME},
};

static constexpr std::string_view PREFIX_SUM_TWO_PASS_FINALIZE_NAME = "prefix sum two pass finalize";
inline static const daxa::ComputePipelineCompileInfo PREFIX_SUM_TWO_PASS_FINALIZE_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"util.glsl"},
        .compile_options = {
            .defines = {{"ENTRY_PREFIX_SUM_TWO_PASS_FINALIZE"}},
        },
    },
    .push_constant_size = sizeof(PrefixSumTwoPassFinalizePush),
    .name = std::string{PREFIX_SUM_TWO_PASS_FINALIZE_NAME},
};

static constexpr std::string_view PREFIX_SUM_MESHLETS_NAME = "prefix sum meshlets";
inline static const daxa::ComputePipelineCompileInfo PREFIX_SUM_MESHLETS_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"./src/rendering/tasks/prefix_sum.inl"},
        .compile_options = {
            .defines = {{"ENTRY_PREFIX_SUM_MESHLETS"}},
        },
    },
    .push_constant_size = sizeof(PrefixSumMeshletCountPush),
    .name = std::string{PREFIX_SUM_MESHLETS_NAME},
};

struct PrefixSumTask
{
    struct Uses {
        daxa::BufferComputeShaderRead src{};
        daxa::BufferComputeShaderWrite dst{};
    } uses = {};
    std::string_view name = PREFIX_SUM_NAME;
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    struct Config
    {
        u32 const * src_stride = {};
        u32 const * src_offset = {};
        u32 const * value_count = {};
    } config = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::CommandList cmd = ti.get_command_list();
        cmd.set_pipeline(*pipeline);
        cmd.push_constant(PrefixSumPush{
            .src = ti.get_device().get_device_address(uses.src.buffer()),
            .dst = ti.get_device().get_device_address(uses.dst.buffer()),
            .src_stride = *config.src_stride,
            .src_offset = *config.src_offset,
            .value_count = *config.value_count,
        });
        cmd.dispatch((*config.value_count + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE, 1, 1);
    }
};

struct PrefixSumFinalizeTask
{
    struct Uses {
        daxa::BufferComputeShaderRead partial_sums{};
        daxa::BufferComputeShaderWrite values{};
    } uses = {};
    std::string_view name = PREFIX_SUM_TWO_PASS_FINALIZE_NAME;
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    struct Config
    {
        u32 const * value_count = {};
    } config = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::CommandList cmd = ti.get_command_list();
        cmd.set_pipeline(*pipeline);
        cmd.push_constant(PrefixSumTwoPassFinalizePush{
            .partial_sums = ti.get_device().get_device_address(uses.partial_sums.buffer()),
            .values = ti.get_device().get_device_address(uses.values.buffer()),
        });
        const u32 workgroups = static_cast<u32>(std::max(0, static_cast<i32>(round_up_div(*config.value_count, PREFIX_SUM_WORKGROUP_SIZE)) - 1));
        const u32 dispatch_x = workgroups * (PREFIX_SUM_WORKGROUP_SIZE / PREFIX_SUM_TWO_PASS_FINALIZE_WORKGROUP_SIZE);
        cmd.dispatch(dispatch_x, 1, 1);
    }
};

struct PrefixSumMeshletTask
{
    struct Uses {
        daxa::BufferComputeShaderRead entity_meta{};
        daxa::BufferComputeShaderRead entity_meshlists{};
        daxa::BufferComputeShaderWrite ent_meshlet_count_prefix_sum_buffer{};
    } uses = {};
    std::string_view name = PREFIX_SUM_MESHLETS_NAME;
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    struct Config
    {
        u32 * entity_count = {};
        BufferId * meshes = {};
    } config = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::CommandList cmd = ti.get_command_list();
        cmd.set_pipeline(*pipeline);
        cmd.push_constant(PrefixSumMeshletCountPush{
            .entity_meta_data = ti.get_device().get_device_address(uses.entity_meta.buffer()),
            .entity_meshlists = ti.get_device().get_device_address(uses.entity_meshlists.buffer()),
            .meshes = ti.get_device().get_device_address(*config.meshes),
            .dst = ti.get_device().get_device_address(uses.ent_meshlet_count_prefix_sum_buffer.buffer()),
        });
        cmd.dispatch((*config.entity_count + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE, 1, 1);
    }
};

#elif DAXA_SHADER && defined(ENTRY_PREFIX_SUM_MESHLETS)

#include "../../../shaders/util.glsl"

DEFINE_PUSHCONSTANT(PrefixSumMeshletCountPush, push)
layout(local_size_x = PREFIX_SUM_WORKGROUP_SIZE) in;
void main()
{
    const uint entity_index = gl_GlobalInvocationID.x;
    const uint warp_id = gl_SubgroupID;
    const uint warp_index = gl_SubgroupInvocationID;

    daxa_BufferPtr(EntityMetaData) entities = daxa_BufferPtr(EntityMetaData)(push.entity_meta_data);
    daxa_BufferPtr(MeshList) entity_meshlists = daxa_BufferPtr(MeshList)(push.entity_meshlists);
    daxa_BufferPtr(Mesh) meshes = daxa_BufferPtr(Mesh)(push.meshes);
    daxa_RWBufferPtr(daxa_u32) dst = daxa_RWBufferPtr(daxa_u32)(push.dst);

    uint meshlets = 0;
    if (entity_index < deref(entities).entity_count)
    {
        const MeshList meshlist = deref(entity_meshlists[entity_index]);
        for (uint mesh_i = 0; mesh_i < meshlist.count; ++mesh_i)
        {
            const uint mesh_index = meshlist.mesh_indices[mesh_i];
            meshlets += deref(meshes[mesh_index]).meshlet_count;
        }
    }
    prefix_sum(
        warp_index,
        warp_id,
        meshlets);
    deref(dst[entity_index]) = meshlets;
}

#endif