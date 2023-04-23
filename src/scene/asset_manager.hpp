#pragma once

#include <assimp/Importer.hpp>
#include <assimp/mesh.h>
#include <assimp/scene.h>
#include <assimp/IOStream.hpp>
#include <assimp/IOSystem.hpp>

#include <meshoptimizer.h>

#include "../sandbox.hpp"
#include "../rendering/gpu_context.hpp"
#include "../mesh/mesh.inl"

using MeshIndex = size_t;
using ImageIndex = size_t;

#define MAX_MESHES 10000

inline std::string generate_mesh_name(aiMesh* mesh)
{
    return
        std::string(mesh->mName.C_Str()) + std::string(" m:") + std::to_string(mesh->mMaterialIndex);
}

struct AssetManager
{
    daxa::Device device = {};
    std::optional<daxa::CommandList> asset_update_cmd_list = {};
    daxa::BufferId meshes_buffer = {};
    daxa::TaskBuffer tmeshes = {};
    
    std::vector<Mesh> meshes = {};
    std::vector<std::string> mesh_names = {};
    std::vector<daxa::ImageId> images = {};
    std::vector<std::string> image_names = {};
    std::unordered_map<std::string_view, u32> mesh_lut = {};
    std::unordered_map<std::string_view, u32> image_lut = {};
    usize total_meshlet_count = {};

    AssetManager(daxa::Device device);
    ~AssetManager();

    auto create_mesh(std::string_view unique_name, aiMesh *aimesh) -> std::pair<u32, Mesh const *>
    {
        ASSERT_M(meshes.size() + 1 < MAX_MESHES, "Exceeded max mesh count!");
        // Create entry of mesh in fields.
        ASSERT_M(!mesh_lut.contains(unique_name), "All meshes MUST have unique names!");
        u32 mesh_index = static_cast<u32>(this->meshes.size());
        this->meshes.push_back({});
        this->mesh_names.push_back(std::string{unique_name});
        this->mesh_lut[this->mesh_names[mesh_index]] = mesh_index;
        Mesh &mesh = this->meshes.back();
        // Create standart index buffer.
        std::vector<u32> index_buffer(aimesh->mNumFaces * 3);
        for (usize face_i = 0; face_i < aimesh->mNumFaces; ++face_i)
        {
            for (usize index = 0; index < 3; ++index)
            {
                index_buffer[face_i * 3 + index] = aimesh->mFaces[face_i].mIndices[index];
            }
        }
        // Nvidia optimal numbers.
        // Should work well on amd as well.
        constexpr usize MAX_VERTICES = 64;
        constexpr usize MAX_TRIANGLES = 124;
        // No clue what cone culling is.
        constexpr float CONE_WEIGHT = 0.0f;
        size_t max_meshlets = meshopt_buildMeshletsBound(index_buffer.size(), MAX_VERTICES, MAX_TRIANGLES);
        std::vector<meshopt_Meshlet> meshlets(max_meshlets);
        std::vector<u32> meshlet_indirect_vertices(max_meshlets * MAX_VERTICES);
        std::vector<u8> meshlet_micro_indices(max_meshlets * MAX_TRIANGLES * 3);
        size_t meshlet_count = meshopt_buildMeshlets(
            meshlets.data(),
            meshlet_indirect_vertices.data(),
            meshlet_micro_indices.data(),
            index_buffer.data(),
            index_buffer.size(),
            reinterpret_cast<float *>(aimesh->mVertices),
            static_cast<usize>(aimesh->mNumVertices),
            sizeof(f32vec3),
            MAX_VERTICES,
            MAX_TRIANGLES,
            CONE_WEIGHT);
        std::vector<BoundingSphere> meshlet_bounds(meshlet_count);
        for (size_t meshlet_i = 0; meshlet_i < meshlet_count; ++meshlet_i)
        {
            meshopt_Bounds raw_bounds = meshopt_computeMeshletBounds(
                &meshlet_indirect_vertices[meshlets[meshlet_i].vertex_offset],
                &meshlet_micro_indices[meshlets[meshlet_i].triangle_offset],
                meshlets[meshlet_i].triangle_count,
                &aimesh->mVertices[0].x,
                static_cast<usize>(aimesh->mNumVertices),
                sizeof(f32vec3));
            meshlet_bounds[meshlet_i].center[0] = raw_bounds.center[0];
            meshlet_bounds[meshlet_i].center[1] = raw_bounds.center[1];
            meshlet_bounds[meshlet_i].center[2] = raw_bounds.center[2];
            meshlet_bounds[meshlet_i].radius = raw_bounds.radius;
        }
        // Trimm array sizes.
        const meshopt_Meshlet &last = meshlets[meshlet_count - 1];
        meshlet_indirect_vertices.resize(last.vertex_offset + last.vertex_count);
        meshlet_micro_indices.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
        meshlets.resize(meshlet_count);
        total_meshlet_count += meshlet_count;
        // Determine offsets and size of the buffer containing all mesh data.
        u32 allocation_size = 0;

        u32 const meshlet_array_offset = allocation_size;
        usize const meshlet_array_bytesize = meshlets.size() * sizeof(Meshlet);
        allocation_size += meshlet_array_bytesize;

        u32 const meshlet_bounds_array_offset = allocation_size;
        usize const meshlet_bounds_array_bytesize = meshlet_bounds.size() * sizeof(BoundingSphere);
        allocation_size += meshlet_bounds_array_bytesize;

        u32 const micro_index_array_offset = allocation_size;
        usize const micro_index_array_bytesize = ((meshlet_micro_indices.size() * sizeof(u8)) + 3) & ~0x03;
        allocation_size += micro_index_array_bytesize;

        u32 const indirect_vertex_array_offset = allocation_size;
        usize const indirect_vertex_array_bytesize = meshlet_indirect_vertices.size() * sizeof(u32);
        allocation_size += indirect_vertex_array_bytesize;

        u32 const vertex_positions_array_offset = allocation_size;
        usize const vertex_positions_array_bytesize = static_cast<usize>(aimesh->mNumVertices) * sizeof(f32vec3);
        allocation_size += vertex_positions_array_bytesize;
        // Create mesh.
        mesh.mesh_buffer = device.create_buffer({
            // .memory_flags = daxa::MemoryFlagBits::DEDICATED_MEMORY,
            .size = allocation_size,
            .name = std::string("Mesh Buffer of mesh \"") + std::string(unique_name) + "\"",
        });
        mesh.meshlet_count = meshlets.size();
        mesh.meshlets = device.get_device_address(mesh.mesh_buffer) + meshlet_array_offset;
        mesh.meshlet_bounds = device.get_device_address(mesh.mesh_buffer) + meshlet_bounds_array_offset;
        mesh.micro_indices = device.get_device_address(mesh.mesh_buffer) + micro_index_array_offset;
        mesh.indirect_vertices = device.get_device_address(mesh.mesh_buffer) + indirect_vertex_array_offset;
        mesh.vertex_positions = device.get_device_address(mesh.mesh_buffer) + vertex_positions_array_offset;
        // Stage buffer upload.
        daxa::BufferId staging_buffer = device.create_buffer({
            .size = allocation_size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = std::string("Staging buffer for mesh \"") + std::string(unique_name) + "\"",
        });
        void *staging_buffer_ptr = device.get_host_address(staging_buffer);
//
        void *staging_meshlets = reinterpret_cast<u8 *>(staging_buffer_ptr) + meshlet_array_offset;
        std::memcpy(staging_meshlets, meshlets.data(), meshlet_array_bytesize);

        void *staging_meshlet_bounds = reinterpret_cast<u8 *>(staging_buffer_ptr) + meshlet_bounds_array_offset;
        std::memcpy(staging_meshlet_bounds, meshlet_bounds.data(), meshlet_bounds_array_bytesize);

        void *staging_micro_indices = reinterpret_cast<u8 *>(staging_buffer_ptr) + micro_index_array_offset;
        std::memcpy(staging_micro_indices, meshlet_micro_indices.data(), micro_index_array_bytesize);

        void *staging_indirect_vertices = reinterpret_cast<u8 *>(staging_buffer_ptr) + indirect_vertex_array_offset;
        std::memcpy(staging_indirect_vertices, meshlet_indirect_vertices.data(), indirect_vertex_array_bytesize);

        void *staging_vertex_positions = reinterpret_cast<u8 *>(staging_buffer_ptr) + vertex_positions_array_offset;
        std::memcpy(staging_vertex_positions, aimesh->mVertices, vertex_positions_array_bytesize);
        // Record mesh buffer update calls.
        if (!this->asset_update_cmd_list.has_value())
        {
            this->asset_update_cmd_list = this->device.create_command_list({.name = "asset update cmd list"});
            this->asset_update_cmd_list.value().pipeline_barrier({
                .awaited_pipeline_access = daxa::AccessConsts::HOST_WRITE,
                .waiting_pipeline_access = daxa::AccessConsts::TRANSFER_READ,
            });
        }
        auto &cmd = this->asset_update_cmd_list.value();
        cmd.destroy_buffer_deferred(staging_buffer);
        cmd.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = mesh.mesh_buffer,
            .size = allocation_size,
        });
        std::cout << "mesh \"" << unique_name << "\" has " << meshlets.size() << " meshlets" << std::endl;
        return { mesh_index, &mesh };
    }

    auto get_mesh_if_present(std::string_view const &mesh_name) -> std::optional<std::pair<u32, Mesh const *>>
    {
        if (mesh_lut.contains(mesh_name))
        {
            return {{mesh_lut[mesh_name], &this->meshes[mesh_lut[mesh_name]]}};
        }
        return std::nullopt;
    }

    auto get_or_create_mesh(aiMesh * aimesh) -> std::pair<u32, Mesh const *>
    {
        auto const unique_name = generate_mesh_name(aimesh);
        auto ret = get_mesh_if_present(unique_name);
        if (!ret.has_value())
        {
            ret = create_mesh(unique_name, aimesh);
        }
        return ret.value();
    }

    auto get_update_commands() -> std::optional<daxa::CommandList>
    {
        if (this->asset_update_cmd_list.has_value())
        {
            daxa::CommandList cmd = std::move(this->asset_update_cmd_list.value());
            this->asset_update_cmd_list.reset();
            auto staging_buffer = device.create_buffer({ .size = sizeof(Mesh) * MAX_MESHES, .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM, .name = "mesh buffer staging upload buffer" });
            cmd.destroy_buffer_deferred(staging_buffer);
            auto host_ptr = device.get_host_address_as<Mesh>(staging_buffer);
            for (usize mesh_i = 0; mesh_i < meshes.size(); ++mesh_i)
            {
                host_ptr[mesh_i] = meshes[mesh_i];
            }
            cmd.pipeline_barrier({
                .awaited_pipeline_access = daxa::AccessConsts::HOST_WRITE,
                .waiting_pipeline_access = daxa::AccessConsts::TRANSFER_READ,
            });
            cmd.copy_buffer_to_buffer({
                .src_buffer = staging_buffer,
                .dst_buffer = meshes_buffer,
                .size = sizeof(Mesh) * meshes.size(),
            });
            cmd.pipeline_barrier({
                .awaited_pipeline_access = daxa::AccessConsts::TRANSFER_WRITE,
                .waiting_pipeline_access = daxa::AccessConsts::READ,
            });
            cmd.complete();
            return cmd;
        }
        return {};
    }
};