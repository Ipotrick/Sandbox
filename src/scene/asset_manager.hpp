#pragma once

#include <assimp/Importer.hpp>
#include <assimp/mesh.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/IOStream.hpp>
#include <assimp/IOSystem.hpp>

#include <meshoptimizer.h>

#include "../sandbox.hpp"
#include "../rendering/gpu_context.hpp"
#include "../../shader_shared/mesh.inl"

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

    auto create_mesh(std::string_view unique_name, aiMesh *aimesh) -> std::pair<u32, Mesh const *>;

    auto get_mesh_if_present(std::string_view const &mesh_name) -> std::optional<std::pair<u32, Mesh const *>>;

    auto get_or_create_mesh(aiMesh * aimesh) -> std::pair<u32, Mesh const *>;

    auto get_update_commands() -> std::optional<daxa::CommandList>;
};