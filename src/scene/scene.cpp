#include "scene.hpp"

#include <fstream>

#include <daxa/utils/math_operators.hpp>
using namespace daxa::math_operators;

Scene::Scene()
{
}

Scene::~Scene()
{
}

EntityId Scene::create_entity()
{
    return {this->entities.entity_count++};
}

auto Scene::get_entity_ref(EntityId ent_id) -> EntityRef
{
    return {
        .transform = &this->entities.transform[ent_id.index],
        .first_child = &this->entities.first_child[ent_id.index],
        .next_silbing = &this->entities.next_silbing[ent_id.index],
        .parent = &this->entities.parent[ent_id.index],
        .meshes = &this->entities.meshes[ent_id.index],
    };
}

void Scene::record_full_entity_update(daxa::Device &device, daxa::CommandList &cmd, Scene &scene, daxa::BufferId static_entities_buffer)
{
    auto staging = device.create_buffer({
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .size = sizeof(EntityData),
        .debug_name = "entity update staging",
    });
    cmd.destroy_buffer_deferred(staging);
    *reinterpret_cast<EntityData *>(device.get_host_address(staging)) = scene.entities;
    cmd.copy_buffer_to_buffer({
        .src_buffer = staging,
        .dst_buffer = static_entities_buffer,
        .size = sizeof(EntityData),
    });
}

void Scene::set_combined_transforms()
{
    std::vector<EntityId> frontier(16);
    if (this->root_entity.index == INVALID_ENTITY_INDEX)
    {
        return;
    }
    EntityId scene_ent = this->entities.first_child[this->root_entity.index];
    while (scene_ent.index != INVALID_ENTITY_INDEX)
    {
        frontier.push_back(scene_ent);
        EntityId sibling = this->entities.next_silbing[scene_ent.index];
    }
    while (!frontier.empty())
    {
        EntityId ent = frontier.back();
        frontier.pop_back();
        EntityId parent = this->entities.parent[ent.index];
        this->entities.combined_transform[ent.index] = this->entities.combined_transform[parent.index] * this->entities.transform[ent.index];
        EntityId child = this->entities.first_child[ent.index];
        while (child.index != INVALID_ENTITY_INDEX)
        {
            frontier.push_back(child);
            auto const old_child = child;
            EntityId child = this->entities.next_silbing[old_child.index];
        }
    }
}

void recursive_print_aiNode(aiScene const *aiscene, aiNode *node, u32 depth, std::string &preamble_string)
{
    std::cout << preamble_string << "aiNode::mName: " << node->mName.C_Str() << "\n";
    std::cout << preamble_string << "{\n";
    if (node->mParent)
    {
        std::cout << preamble_string << "  parent node: " << node->mParent->mName.C_Str() << "\n";
    }
    std::cout << preamble_string << "  aiNode::mMeshes:\n";
    std::cout << preamble_string << "  {\n";
    for (u32 *mesh = node->mMeshes; mesh < (node->mMeshes + node->mNumMeshes); ++mesh)
    {
        std::cout << preamble_string << "    aiMesh::mName: " << aiscene->mMeshes[*mesh]->mName.C_Str() << "\n";
    }
    std::cout << preamble_string << "  }\n";
    for (aiNode **child_node = node->mChildren; child_node < (node->mChildren + node->mNumChildren); ++child_node)
    {
        preamble_string += "  ";
        recursive_print_aiNode(aiscene, *child_node, depth + 1, preamble_string);
        preamble_string.pop_back();
        preamble_string.pop_back();
    }
}

void process_meshes(aiScene const *aiscene, AssetManager &asset_manager)
{
}

void process_textures(aiScene const *aiscene, AssetManager &asset_manager)
{
}

void SceneLoader::load_entities_from_fbx(Scene &scene, AssetManager &asset_manager, std::filesystem::path const &asset_name)
{
    std::filesystem::path file_path{asset_root_folder / asset_name};

    Assimp::Importer importer;

    aiScene const *aiscene = importer.ReadFile(file_path.string(), {});

    if (aiscene == nullptr)
    {
        std::cerr << "Error: Assimp failed to load scene with message: \"" << importer.GetErrorString() << "\"" << std::endl;
        return;
    }

    struct FrontierEntry
    {
        EntityId entity_id = {};
        aiNode *node = {};
    };

    for (usize mesh_i = 0; mesh_i < aiscene->mNumMeshes; ++mesh_i)
    {
        auto dummy = asset_manager.get_or_create_mesh(aiscene->mMeshes[mesh_i]);
    }
    std::cout << "total meshlet count: " << asset_manager.total_meshlet_count << std::endl;

    EntityId scene_entity_id = scene.create_entity();
    EntityRef scene_entity = scene.get_entity_ref(scene_entity_id);
    scene.root_entity = scene_entity_id;
    auto ident = glm::identity<glm::mat4x4>();
    *scene_entity.transform = *reinterpret_cast<daxa::types::f32mat4x4 *>(&ident);

    std::vector<FrontierEntry> frontier = {};
    frontier.reserve(128);
    frontier.push_back({
        .entity_id = scene_entity_id,
        .node = aiscene->mRootNode,
    });
    while (!frontier.empty())
    {
        FrontierEntry entry = frontier.back();
        auto [current_entity_id, current_node] = entry;
        frontier.pop_back();
        const auto current_entity = scene.get_entity_ref(current_entity_id);

        usize n = current_node->mNumMeshes;
        ASSERT_M(n <= 7, "max submeshes is 7");

        current_entity.meshes->count = current_node->mNumMeshes;
        for (usize mesh_i = 0; mesh_i < current_node->mNumMeshes; ++mesh_i)
        {
            current_entity.meshes->mesh_indices[mesh_i] = asset_manager.get_or_create_mesh(aiscene->mMeshes[current_node->mMeshes[mesh_i]]).first;
        }
        std::cout << "Node has " << current_node->mNumMeshes << "meshes" << std::endl;

        *current_entity.transform = *reinterpret_cast<daxa::types::f32mat4x4 *>(&current_node->mTransformation);

        for (usize child_i = 0; child_i < current_node->mNumChildren; ++child_i)
        {
            EntityId new_child_id = scene.create_entity();
            auto const new_child = scene.get_entity_ref(new_child_id);
            *new_child.parent = current_entity_id;

            *new_child.next_silbing = *current_entity.first_child;
            *current_entity.first_child = new_child_id;

            frontier.push_back(FrontierEntry{.entity_id = new_child_id, .node = current_node->mChildren[child_i]});
        }
    }

    std::cout << std::flush;
}