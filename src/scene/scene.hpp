#pragma once

#include <optional>

#include "fastgltf/parser.hpp"

#include "../sandbox.hpp"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/scene.inl"
#include "asset_manager.hpp"

#define MAX_MESHES_PER_MESHGROUP 7
#define MAX_SURFACE_PARAMETERS 3

// Scenes are described by entities and their resources.
// These resources can have complex dependencies between each other.
// We want to be able to load AND UNLOAD the resources asynchronously.
// BUT we want to remember unloaded resources. We never delete metadata.
// We have a manifest of all ever loaded resources that can only grow.
// Resource data may be deleted or loaded later, but their metadata must be present in the manifests.

struct SceneFileManifestEntry
{
    std::filesystem::path path = {};
    std::unique_ptr<fastgltf::glTF> gltf_info = {};
    /// @brief  Offsets of the gltf indices to the loaded manifest indices.
    ///         Subtracting the scene offset from the manifest index gives you the gltf index.
    u32 texture_manifest_offset = {};
    u32 material_manifest_offset = {};
    u32 mesh_group_manifest_offset = {};
    u32 mesh_manifest_offset = {};
};

struct TextureManifestEntry
{
    std::string name = {};
    u32 scene_file_manifest_index = {};
    u32 in_scene_file_index = {};
    struct Runtime
    {
        daxa::ImageId image_id = {};
    };
    std::optional<Runtime> runtime = {};
};

struct MaterialManifestEntry
{
    std::optional<u32> diffuse_tex_index = {};
    std::optional<u32> normal_tex_index = {};
    std::string name = {};
    u32 scene_file_manifest_index = {};
    u32 in_scene_file_index = {};
};

struct MeshManifestEntry
{
    std::optional<u32> material_manifest_index = {};
    u32 scene_file_manifest_index = {};
    struct Runtime
    {
        daxa::BufferId buffer_id = {};
        u32 vertex_count = {};
        u32 meshlet_count = {};
        u32 buffer_offset_meshlets = {};            // Meshlet
        u32 buffer_offset_meshlet_bounds = {};      // BoundingSphere
        u32 buffer_offset_micro_indices = {};       // daxa_u32
        u32 buffer_offset_indirect_vertices = {};   // daxa_u32
        u32 buffer_offset_vertex_positions = {};    // daxa_f32vec3
        u32 buffer_offset_vertex_uvs = {};          // daxa_f32vec2
    };
    std::optional<Runtime> runtime = {};
};

struct MeshGroupManifestEntry
{
    std::array<u32, MAX_MESHES_PER_MESHGROUP> mesh_manifest_indices = {};
    u32 mesh_count = {};
    std::string name = {};
    u32 scene_file_manifest_index = {};
    u32 in_scene_file_index = {};
};

struct Entity
{
    daxa_f32mat4x4 transform = {};
    EntityId first_child = {};
    EntityId next_sibling = {};
    EntityId parent = {};
    u32 model_index = {};
    std::string name = {};
};

struct Scene
{
    daxa::TaskBuffer _t_entity_meta = {};
    daxa::TaskBuffer _t_entity_transforms = {};
    daxa::TaskBuffer _t_entity_combined_transforms = {};
    daxa::TaskBuffer _t_entity_first_children = {};
    daxa::TaskBuffer _t_entity_next_siblings = {};
    daxa::TaskBuffer _t_entity_parents = {};
    daxa::TaskBuffer _t_entity_meshlists = {};
    daxa::TaskBuffer _t_materials = {};
    daxa::TaskBuffer _t_meshes = {};

    std::vector<SceneFileManifestEntry> _scene_file_manifest = {};
    std::vector<TextureManifestEntry> _texture_manifest = {};
    std::vector<MaterialManifestEntry> _material_manifest = {};
    std::vector<MeshManifestEntry> _mesh_manifest = {};
    std::vector<MeshGroupManifestEntry> _mesh_group_manifest = {};

    std::vector<std::optional<Entity>> _entities = {};
    std::vector<u32> _entity_index_free_list = {};

    Scene();
    ~Scene();

    auto get_entity_ref(EntityId ent_id) -> EntityRef;

    enum struct LoadManifestResult
    {
        SUCCESS,
        ERROR_FILE_NOT_FOUND,
        ERROR_COULD_NOT_LOAD_ASSET,
        ERROR_INVALID_GLTF_FILE_TYPE,
        ERROR_PARSING_ASSET_NODES,
    };
    static auto to_string(LoadManifestResult result) -> std::string_view
    {
        switch(result)
        {
            case LoadManifestResult::SUCCESS: return "SUCCESS";
            case LoadManifestResult::ERROR_FILE_NOT_FOUND: return "ERROR_FILE_NOT_FOUND";
            case LoadManifestResult::ERROR_COULD_NOT_LOAD_ASSET: return "ERROR_COULD_NOT_LOAD_ASSET";
            case LoadManifestResult::ERROR_INVALID_GLTF_FILE_TYPE: return "ERROR_INVALID_GLTF_FILE_TYPE";
            case LoadManifestResult::ERROR_PARSING_ASSET_NODES: return "ERROR_PARSING_ASSET_NODES";
            default: return "UNKNOWN";
        }
        return "UNKNOWN";
    }
    auto load_manifest_from_gltf(std::filesystem::path const& root_path, std::filesystem::path const& glb_name) -> LoadManifestResult;
};