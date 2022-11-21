#pragma once

#include "../sandbox.hpp"
#include "../mesh/mesh.inl"
#include "../mesh/asset_manager.hpp"

struct EntityId
{
    u32 index;
};

struct EntitySlot
{
    f32vec3 position = {};
    std::shared_ptr<Model> model = {};
    // TODO(pahrens): rotation
    EntityId parent = {};
};

struct SceneLoader
{
    std::filesystem::path asset_root_folder = ".";

    void load_entities_from_fbx(Scene & scene, AssetManager & asset_manager, std::filesystem::path const & asset_name);
};

struct Scene
{
    std::vector<EntitySlot> entities = {};
    std::vector<EntityId> entity_slot_free_list = {};
};