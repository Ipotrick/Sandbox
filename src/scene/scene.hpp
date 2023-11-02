#pragma once

#include <optional>

#include "fastgltf/parser.hpp"

#include "../sandbox.hpp"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/scene.inl"
#include "../slot_map.hpp"


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
    std::unique_ptr<fastgltf::Asset> gltf_asset = {};
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
    u32 scene_file_mesh_index = {};
    u32 scene_file_primitive_index = {};
    std::optional<GPUMeshDescriptor> runtime = {};
};

struct MeshGroupManifestEntry
{
    std::array<u32, MAX_MESHES_PER_MESHGROUP> mesh_manifest_indices = {};
    u32 mesh_count = {};
    std::string name = {};
    u32 scene_file_manifest_index = {};
    u32 in_scene_file_index = {};
};

struct RenderEntity;
using RenderEntityId = SlotMap<RenderEntity>::Id;

struct RenderEntity
{
    glm::mat4x3 transform = {};
    std::optional<RenderEntityId> first_child = {};
    std::optional<RenderEntityId> next_sibling = {};
    std::optional<RenderEntityId> parent = {};
    std::optional<u32> mesh_group_manifest_index = {};
    std::string name = {};
};

using RenderEntitySlotMap = SlotMap<RenderEntity>;

struct Scene
{
    /**
     * NOTES:
     * - >on the gpu< the render entities are stores in an structure of arrays fassion
     * - >on the cpu< render entities are stored in an array of structures
     * - arrays and buffers only grow
     * - arrays are NOT nessecarily densly populated with valid entities
     * - growing the entities buffers is done by scene
     * - recording updates to entities is done by scene
     * - WARNING: FOR NOW THE RENDERER ASSUMES TIGHTLY PACKED ENTITIES!
     * - TODO: Upload sparse set to gpu so gpu can tightly iterate!
    */
    daxa::TaskBuffer _gpu_entity_meta = {};
    daxa::TaskBuffer _gpu_entity_transforms = {};
    daxa::TaskBuffer _gpu_entity_combined_transforms = {};
    // UNUSED, but later we wanna do 
    // the compined transform calculation on the gpu!
    daxa::TaskBuffer _gpu_entity_parents = {};                
    daxa::TaskBuffer _gpu_entity_mesh_groups = {};            
    RenderEntitySlotMap _render_entities = {};
    std::vector<RenderEntityId> _dirty_render_entities = {}; 

    /**
     * NOTES:
     * - manifest is mirrored with different types on the gpu (TODO: potential unifications?)
     * - manifest can ONLY GROW, the manifest CAN NOT shrink
     * - baking data for textures and meshes are dynamically loaded and unloaded
     * - unloadable data is marked by a 'runtime' field within the manifest
     * - material-textures and meshes are live load and unloadable
     * - growing the manifest buffers and copying in the constant data is done by scene
     * - recording updates to the runtime manifest data is done by the asset processor
     * */     
    daxa::TaskBuffer _gpu_material_texture_manifest = {};  
    daxa::TaskBuffer _gpu_mesh_manifest = {};
    daxa::TaskBuffer _gpu_material_manifest = {};         
    daxa::TaskBuffer _gpu_mesh_group_manifest = {};
    std::vector<SceneFileManifestEntry> _scene_file_manifest = {};
    std::vector<TextureManifestEntry> _material_texture_manifest = {};
    std::vector<MaterialManifestEntry> _material_manifest = {};
    std::vector<MeshManifestEntry> _mesh_manifest = {};
    std::vector<MeshGroupManifestEntry> _mesh_group_manifest = {};

    Scene(daxa::Device device);
    ~Scene();

    enum struct LoadManifestErrorCode
    {
        FILE_NOT_FOUND,
        COULD_NOT_LOAD_ASSET,
        INVALID_GLTF_FILE_TYPE,
        COULD_NOT_PARSE_ASSET_NODES,
    };
    static auto to_string(LoadManifestErrorCode result) -> std::string_view
    {
        switch(result)
        {
            case LoadManifestErrorCode::FILE_NOT_FOUND: return "FILE_NOT_FOUND";
            case LoadManifestErrorCode::COULD_NOT_LOAD_ASSET: return "COULD_NOT_LOAD_ASSET";
            case LoadManifestErrorCode::INVALID_GLTF_FILE_TYPE: return "INVALID_GLTF_FILE_TYPE";
            case LoadManifestErrorCode::COULD_NOT_PARSE_ASSET_NODES: return "COULD_NOT_PARSE_ASSET_NODES";
            default: return "UNKNOWN";
        }
        return "UNKNOWN";
    }
    auto load_manifest_from_gltf(std::filesystem::path const& root_path, std::filesystem::path const& glb_name) -> std::variant<RenderEntityId, LoadManifestErrorCode>;

    auto record_gpu_manifest_update() -> daxa::ExecutableCommandList;

    daxa::Device _device = {};
};