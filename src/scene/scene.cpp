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
    return {this->entity_meta.entity_count++};
}

auto Scene::get_entity_ref(EntityId ent_id) -> EntityRef
{
    return {
        .transform = &this->entity_transforms[ent_id.index],
        .first_child = &this->entity_first_children[ent_id.index],
        .next_silbing = &this->entity_next_siblings[ent_id.index],
        .parent = &this->entity_parents[ent_id.index],
        .meshes = &this->entity_meshlists[ent_id.index],
    };
}

void Scene::record_full_entity_update(
    daxa::Device &device, 
    daxa::CommandList &cmd, 
    Scene &scene, 
    daxa::BufferId b_entity_meta,
    daxa::BufferId b_entity_transforms,
    daxa::BufferId b_entity_combined_transforms,
    daxa::BufferId b_entity_first_children,
    daxa::BufferId b_entity_next_siblings,
    daxa::BufferId b_entity_parents,
    daxa::BufferId b_entity_meshlists)
{
    auto upload = [&](auto& src_field, auto& dst_buffer, auto dummy, usize count)
    {
        using DATA_T = decltype(dummy);
        const u32 size = sizeof(DATA_T) * count;
        auto staging = device.create_buffer({
            .size = size,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = device.info_buffer(dst_buffer).name + " staging",
        });
        cmd.destroy_buffer_deferred(staging);
        std::memcpy(reinterpret_cast<DATA_T*>(device.get_host_address(staging)), &src_field, size);
        cmd.copy_buffer_to_buffer({
            .src_buffer = staging,
            .dst_buffer = dst_buffer,
            .size = size,
        });
    };
    upload(this->entity_meta, b_entity_meta, EntityMetaData{}, 1);
    upload(this->entity_transforms, b_entity_transforms, daxa_f32mat4x4{}, MAX_ENTITY_COUNT);
    upload(this->entity_combined_transforms, b_entity_combined_transforms, daxa_f32mat4x4{}, MAX_ENTITY_COUNT);
    upload(this->entity_first_children, b_entity_first_children, EntityId{}, MAX_ENTITY_COUNT);
    upload(this->entity_next_siblings, b_entity_next_siblings, EntityId{}, MAX_ENTITY_COUNT);
    upload(this->entity_parents, b_entity_parents, EntityId{}, MAX_ENTITY_COUNT);
    upload(this->entity_meshlists, b_entity_meshlists, MeshList{}, MAX_ENTITY_COUNT);
}

void Scene::set_combined_transforms()
{
    for (u32 entity_index = 0; entity_index < entity_meta.entity_count; ++entity_index)
    {
        daxa::types::f32mat4x4 combined_transform = entity_transforms[entity_index];
        EntityId parent = entity_parents[entity_index];
        while (parent.index != INVALID_ENTITY_INDEX)
        {
            combined_transform = entity_transforms[parent.index] * combined_transform;
            parent = entity_parents[parent.index];
        }
        entity_combined_transforms[entity_index] = combined_transform;
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

    aiScene const *aiscene = importer.ReadFile(file_path.string(), aiProcess_JoinIdenticalVertices);

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
        auto current_entity_id = entry.entity_id;
        aiNode* current_node = entry.node;
        if (current_entity_id.index == 4) 
        {
            printf("debug point\n");
        }
        frontier.pop_back();
        const auto current_entity = scene.get_entity_ref(current_entity_id);

        usize n = current_node->mNumMeshes;
        ASSERT_M(n <= 7, "max submeshes is 7");

        current_entity.meshes->count = current_node->mNumMeshes;
        for (usize mesh_i = 0; mesh_i < current_node->mNumMeshes; ++mesh_i)
        {
            aiMesh* mesh_ptr = aiscene->mMeshes[current_node->mMeshes[mesh_i]];
            auto fetch = asset_manager.get_or_create_mesh(mesh_ptr);
            current_entity.meshes->mesh_ids[mesh_i] = fetch.first;
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