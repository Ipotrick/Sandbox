#include "scene.hpp"

#include <fstream>

EntityId Scene::create_entity()
{
    return {this->next_entity_index++};
}

EntityReference Scene::get_entity_reference(EntityId entity_id)
{
    return {
        entity_id.index + this->entity_transforms.data(),
        entity_id.index + this->entity_models.data(),
        entity_id.index + this->entity_parents.data(),
    };
}

void recursive_print_aiNode(aiScene const* aiscene, aiNode *node, u32 depth, std::string & preamble_string)
{
    std::cout << preamble_string << "aiNode::mName: " << node->mName.C_Str() << "\n";
    std::cout << preamble_string << "{\n";
    if (node->mParent) { 
        std::cout << preamble_string << "  parent node: " << node->mParent->mName.C_Str() << "\n";
    }
    std::cout << preamble_string << "  aiNode::mMeshes:\n";
    std::cout << preamble_string << "  {\n";
    for (u32* mesh = node->mMeshes; mesh < (node->mMeshes + node->mNumMeshes); ++mesh)
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

void process_meshes(aiScene const * aiscene, AssetManager &asset_manager)
{
    
}

void process_textures(aiScene const * aiscene, AssetManager &asset_manager)
{

}

void SceneLoader::load_entities_from_fbx(Scene &scene, AssetManager &asset_manager, std::filesystem::path const &asset_name)
{
    std::filesystem::path file_path{asset_root_folder / asset_name};

    Assimp::Importer importer;

    aiScene const*aiscene = importer.ReadFile(file_path.string(), {});

    if (aiscene == nullptr)
    {
        std::cerr << "Error: Assimp failed to load scene with message: \"" << importer.GetErrorString() << "\"" << std::endl;
        return;
    }

    struct FrontierEntry
    {
        EntityId parent_id = {};
        aiNode *node = {};
    };

    std::string preamble_string = {""};
    recursive_print_aiNode(aiscene, aiscene->mRootNode, 0, preamble_string);

    for (usize mesh_i = 0; mesh_i < aiscene->mNumMeshes; ++mesh_i)
    {
        auto dummy = asset_manager.get_or_create_mesh(aiscene->mMeshes[mesh_i]);
    }
    std::cout << "total meshlet count: " << asset_manager.total_meshlet_count << std::endl;

    std::vector<FrontierEntry> frontier = {};
    frontier.reserve(128);
    frontier.push_back({
        .parent_id = scene.create_entity(),
        .node = aiscene->mRootNode,
    });
    while (!frontier.empty())
    {
        auto [current_entity, current_node] = frontier.back();
        frontier.pop_back();
        auto ent_ref = scene.get_entity_reference(current_entity);
    }
    std::cout << std::flush;
}