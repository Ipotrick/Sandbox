#pragma once

#include "fastgltf/parser.hpp"

#include "../sandbox.hpp"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/scene.inl"
#include "asset_manager.hpp"

using AssetName = std::array<char, 32>;

struct TextureManifestEntry
{
    enum class Status
    {
        UNLOADED = 0,
        LOADING_REQUESTED = 1,
        LOADED = 2,
    };
    Status status = Status::UNLOADED;
    daxa::ImageId texture_id = {};
    AssetName name = {};
};

struct MaterialManifestEntry
{
    u32 diffuse_tex_index = {};
    AssetName name = {};
};

struct MeshManifestEntry
{
    u32 material_index = {};
    daxa::BufferId buffer_id = {};
    // daxa::RaytracingASId raytracing_as_id = {};
    AssetName name = {};
};

struct Entity
{
    daxa_f32mat4x4 transform = {};
    EntityId first_child = {};
    EntityId next_sibling = {};
    EntityId parent = {};
    MeshList mesh_list = {};
    std::array<std::string, 7> mesh_name_list = {};
};

struct Scene
{
    daxa::TaskBuffer t_entity_meta = {};
    daxa::TaskBuffer t_entity_transforms = {};
    daxa::TaskBuffer t_entity_combined_transforms = {};
    daxa::TaskBuffer t_entity_first_children = {};
    daxa::TaskBuffer t_entity_next_siblings = {};
    daxa::TaskBuffer t_entity_parents = {};
    daxa::TaskBuffer t_entity_meshlists = {};
    daxa::TaskBuffer t_materials = {};
    daxa::TaskBuffer t_meshes = {};

    std::vector<TextureManifestEntry> texture_manifest = {};
    std::vector<MaterialManifestEntry> material_manifest = {};
    std::vector<MeshManifestEntry> mesh_manifest = {};
    std::vector<Entity> entities = {};

    // this is mostly used to keep metadata when the scene is still loading
    fastgltf::Scene loading_gltf_scene = {};

    Scene();
    ~Scene();
    auto get_entity_ref(EntityId ent_id) -> EntityRef;
    void load_from_gltf(std::string root_path, std::string glb_name);
};

struct SceneLoader
{
    std::filesystem::path asset_root_folder = ".";

    void load_entities_from_fbx(Scene &scene, AssetManager &asset_manager, std::filesystem::path const &asset_name);
};
