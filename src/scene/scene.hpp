#pragma once

#include "../sandbox.hpp"
#include "../mesh/mesh.inl"
#include "scene.inl"
#include "asset_manager.hpp"

struct Scene
{
    Scene();
    ~Scene();

    EntityData entities = {};
    EntityId root_entity = {};

    auto create_entity() -> EntityId;
    auto get_entity_ref(EntityId ent_id) -> EntityRef;

    void record_full_entity_update(daxa::Device& device, daxa::CommandList& cmd, Scene& scene, daxa::BufferId static_entities_buffer);
    void set_combined_transforms();
};

struct SceneLoader
{
    std::filesystem::path asset_root_folder = ".";

    void load_entities_from_fbx(Scene &scene, AssetManager &asset_manager, std::filesystem::path const &asset_name);
};
