#pragma once

#include <assimp/Importer.hpp>
#include <assimp/mesh.h>
#include <assimp/scene.h>
#include <assimp/IOStream.hpp>
#include <assimp/IOSystem.hpp>

#include <meshoptimizer.h>

#include "../sandbox.hpp"
#include "../gpu_context.hpp"
#include "../mesh/mesh.inl"

using MeshIndex = size_t;
using ImageIndex = size_t;

struct AssetManager
{
    daxa::Device device = {};

    std::vector<Mesh> meshes = {};
    std::vector<std::string> mesh_names = {};
    std::vector<daxa::ImageId> images = {};
    std::vector<std::string> image_names = {};
    std::unordered_map<std::string_view, u32> mesh_lut = {};
    std::unordered_map<std::string_view, u32> image_lut = {};

    AssetManager(daxa::Device device);

    auto create_mesh(aiMesh *aimesh) -> std::pair<u32, Mesh> &
    {
        ASSERT_M(!mesh_lut.contains(aimesh->mName.C_Str()), "All meshes MUST have unique names!");

        u32 mesh_index = static_cast<u32>(this->meshes.size());
        this->meshes.push_back({});
        Mesh &mesh = this->meshes.back();

        // Nvidia optimal numbers.
        // Should work well on amd as well.
        const size_t max_vertices = 64;
        const size_t max_triangles = 126;
        // No clue what cone culling is.
        const float cone_weight = 0.0f;
        size_t max_meshlets = meshopt_buildMeshletsBound(aimesh->mFaces->mNumIndices, max_vertices, max_triangles);
        std::vector<meshopt_Meshlet> meshlets(max_meshlets);
        std::vector<u32> meshlet_indirect_vertices(max_meshlets * max_vertices);
        std::vector<u8> meshlet_micro_indices(max_meshlets * max_triangles * 3);
        size_t meshlet_count = meshopt_buildMeshlets(
            meshlets.data(),
            meshlet_indirect_vertices.data(),
            meshlet_micro_indices.data(),
            aimesh->mFaces->mIndices,
            static_cast<usize>(aimesh->mFaces->mNumIndices),
            reinterpret_cast<float*>(aimesh->mVertices), 
            static_cast<usize>(aimesh->mNumVertices),
            sizeof(f32vec3),
            max_vertices,
            max_triangles,
            cone_weight
        );
        const meshopt_Meshlet& last = meshlets[meshlet_count - 1];

        meshlet_indirect_vertices.resize(last.vertex_offset + last.vertex_count);
        meshlet_micro_indices.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
        meshlets.resize(meshlet_count);

        // Allocate ONE big buffer to contain all mesh information, including indices, meshlets, vertices and texture ids.
    }

    auto get_mesh_if_present(std::string_view const &mesh_name) -> std::optional<std::pair<u32, Mesh &>>
    {
        if (mesh_lut.contains(mesh_name))
        {
            return {{mesh_lut[mesh_name], this->meshes[mesh_lut[mesh_name]]}};
        }
        return std::nullopt;
    }
};