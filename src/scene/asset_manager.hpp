#pragma once

#include "../sandbox.hpp"
#include "../gpu_context.hpp"
#include "../mesh/mesh.inl"

#include <meshoptimizer.h>

using MeshIndex = size_t;
using ImageIndex = size_t;

struct AssetManager
{ 
    daxa::Device device = {};

    daxa::BufferId meshlet_index_buffer = {};
    daxa::BufferId meshlet_vertex_positions_buffer = {};
    u32 index_buffer_back_index = {};
    u32 vertex_buffer_back_index = {};

    daxa::TaskBufferId t_meshlet_index_buffer = {};
    daxa::TaskBufferId t_meshlet_vertex_positions_buffer = {};

    std::vector<Meshlet> meshlets = {};
    std::vector<Mesh> meshes = {};
    std::vector<std::string> mesh_names = {};
    std::vector<daxa::ImageId> images = {};
    std::vector<std::string> image_names = {};
    std::unordered_map<std::string_view, u32> mesh_lut = {};
    std::unordered_map<std::string_view, u32> image_lut = {};
 
    AssetManager(daxa::Device device);

    struct AllocateMeshInfo
    {
        std::span<u32> indices = {};
        std::span<f32vec3> vertex_positions = {};
        std::string name = {};
    };
    auto create_mesh(AllocateMeshInfo const & alloc_info) -> std::pair<u32, Mesh>&
    {
        ASSERT_M(!mesh_lut.contains(alloc_info.name), "All meshes MUST have unique names!");

        u32 mesh_index = static_cast<u32>(this->meshes.size());
        this->meshes.push_back({});
        Mesh& mesh = this->meshes.back();
        
        // Optimize mesh, and generate meshlets.
    }

    auto get_mesh_if_present(std::string_view const& mesh_name) -> std::optional<std::pair<u32, Mesh&>>
    {
        if (mesh_lut.contains(mesh_name))
        {
            return {{ mesh_lut[mesh_name], this->meshes[mesh_lut[mesh_name]] }};
        }
        return std::nullopt;
    }
};