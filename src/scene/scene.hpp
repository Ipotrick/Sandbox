#pragma once

#include "../sandbox.hpp"
#include "../mesh/mesh.inl"
#include "asset_manager.hpp"

struct EntityId
{
    u32 index;
};

struct Transform
{
    f32vec3 position = {};
};

struct EntitySlot
{
    std::shared_ptr<Model> model = {};
    // TODO(pahrens): rotation
    EntityId parent = {};
};

struct EntityReference
{
    Transform *transform;
    std::shared_ptr<Model> *model;
    EntityId *parent_id;
};

struct Scene
{
    std::vector<Transform> entity_transforms = {};
    std::vector<std::shared_ptr<Model>> entity_models = {};
    std::vector<EntityId> entity_parents = {};
    std::vector<EntityId> entity_slot_free_list = {};
    u32 next_entity_index = 0;

    EntityId create_entity();
    EntityReference get_entity_reference(EntityId entity_id);
};

struct SceneLoader
{
    std::filesystem::path asset_root_folder = ".";

    void load_entities_from_fbx(Scene &scene, AssetManager &asset_manager, std::filesystem::path const &asset_name);
};
