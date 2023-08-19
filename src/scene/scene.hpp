#pragma once

#include "../sandbox.hpp"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/scene.inl"
#include "asset_manager.hpp"

struct Scene
{
    Scene();
    ~Scene();

    EntityMetaData entity_meta = {};
    daxa_f32mat4x4 entity_transforms[MAX_ENTITY_COUNT] = {};
    daxa_f32mat4x4 entity_combined_transforms[MAX_ENTITY_COUNT] = {};
    EntityId entity_first_children[MAX_ENTITY_COUNT] = {};
    EntityId entity_next_siblings[MAX_ENTITY_COUNT] = {};
    EntityId entity_parents[MAX_ENTITY_COUNT] = {};
    MeshList entity_meshlists[MAX_ENTITY_COUNT] = {};
    EntityId root_entity = {};

    auto create_entity() -> EntityId;
    auto get_entity_ref(EntityId ent_id) -> EntityRef;

    void record_full_entity_update(
        daxa::Device &device, 
        daxa::CommandList &cmd, 
        Scene &scene, 
        daxa::BufferId b_entity_meta,
        daxa::BufferId b_entity_transforms,
        daxa::BufferId b_entity_combined_transforms,
        daxa::BufferId b_entity_first_children,
        daxa::BufferId b_entity_next_siblings,
        daxa::BufferId b_entity_parents,
        daxa::BufferId b_entity_meshlists);
    void process_transforms();
};

struct SceneLoader
{
    std::filesystem::path asset_root_folder = ".";

    void load_entities_from_fbx(Scene &scene, AssetManager &asset_manager, std::filesystem::path const &asset_name);
};
