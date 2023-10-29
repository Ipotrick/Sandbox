#pragma once

// #include <assimp/Importer.hpp>
// #include <assimp/mesh.h>
// #include <assimp/scene.h>
// #include <assimp/postprocess.h>
// #include <assimp/IOStream.hpp>
// #include <assimp/IOSystem.hpp>

#include <meshoptimizer.h>

#include "../sandbox.hpp"
#include "../rendering/gpu_context.hpp"
#include "../../shader_shared/asset.inl"

using MeshIndex = size_t;
using ImageIndex = size_t;

#define MAX_MESHES 10000

// inline std::string generate_mesh_name(aiMesh* mesh)
// {
//     return
//         std::string(mesh->mName.C_Str()) + std::string(" m:") + std::to_string(mesh->mMaterialIndex);
// }
// 
// inline std::string generate_texture_name(aiMaterial * material, aiTextureType type)
// {
//     return std::string(material->GetName().C_Str()) + aiTextureTypeToString(type);
// }

struct AssetManager
{
    static inline constexpr daxa::ImageUsageFlags TEXTURE_USE_FLAGS = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::TRANSFER_DST;
    daxa::Device device = {};
    std::optional<daxa::CommandRecorder> asset_update_cmd_list = {};
    std::vector<std::pair<daxa::BufferId, daxa::ImageId>> texture_upload_list = {};
    daxa::BufferId meshes_buffer = {};
    daxa::TaskBuffer tmeshes = {};
    
    std::vector<GPUMesh> meshes = {};
    std::vector<std::string> mesh_names = {};
    std::vector<daxa::ImageId> textures = {};
    std::vector<std::string> texture_names = {};
    std::unordered_map<std::string_view, u32> mesh_lut = {};
    std::unordered_map<std::string_view, u32> texture_lut = {};
    usize total_meshlet_count = {};

    AssetManager(daxa::Device device);
    AssetManager(AssetManager&&) = default;
    ~AssetManager();

    // auto get_texture_if_present(std::string_view unique_name) -> std::optional<std::pair<u32, daxa::ImageId>>;

    // auto create_texture(std::string_view unique_name, aiScene const*scene, aiMaterial *aimaterial, aiTextureType type) -> std::pair<u32, daxa::ImageId>;

    // auto get_or_create_texture(aiScene const*scene, aiMaterial *aimaterial, aiTextureType type) -> std::pair<u32, daxa::ImageId>;

    // auto create_mesh(std::string_view unique_name, aiScene const*scene, aiMesh *aimesh) -> std::pair<u32, GPUMesh const *>;

    // auto get_mesh_if_present(std::string_view const &mesh_name) -> std::optional<std::pair<u32, GPUMesh const *>>;

    // auto get_or_create_mesh(aiScene const*scene, aiMesh * aimesh) -> std::pair<u32, GPUMesh const *>;

    // auto get_update_commands() -> std::optional<daxa::ExecutableCommandList>;
};