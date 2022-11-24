#include "scene.hpp"

#include <fstream>

#include <assimp/Importer.hpp>
#include <assimp/mesh.h>
#include <assimp/scene.h>
#include <assimp/IOStream.hpp>
#include <assimp/IOSystem.hpp>

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

// My own implementation of IOStream
struct MyIOStream : public Assimp::IOStream
{
    friend class MyIOSystem;
    FILE *stream;
    std::filesystem::path path;

    MyIOStream()
    {

    }

    ~MyIOStream()
    {
        if (this->stream != nullptr)
        {
            fclose(stream);
        }
    }
    size_t Read(void *pvBuffer, size_t pSize, size_t pCount)
    {
        return fread(pvBuffer, pSize, pCount, this->stream);
    }
    size_t Write(const void *pvBuffer, size_t pSize, size_t pCount)
    {
        return fwrite(pvBuffer, pSize, pCount, this->stream);
    }
    aiReturn Seek(size_t pOffset, aiOrigin pOrigin)
    {
        return (fseek(this->stream, pOffset, pOrigin) == 0) ? aiReturn_SUCCESS : aiReturn_FAILURE;
    }
    size_t Tell() const
    {
        return ftell(const_cast<FILE *>(this->stream));
    }
    size_t FileSize() const
    {
        auto prev_pos = ftell(this->stream);
        fseek(this->stream, 0, SEEK_END);
        size_t size = ftell(this->stream);
        fseek(this->stream, prev_pos, SEEK_SET);
        return size;
    }
    void Flush()
    {
        fflush(this->stream);
    }
};

struct MyIOSystem : public Assimp::IOSystem
{
    MyIOSystem()
    {
    }
    ~MyIOSystem()
    {
    }

    // Check whether a specific file exists
    bool Exists(const char* pFile) const override 
    {
        return std::filesystem::exists(pFile);
    }

    // Get the path delimiter character we'd like to see
    char getOsSeparator() const override
    {
        return '/';
    }

    Assimp::IOStream *Open(const char* pFile, const char* pMode = "rb") override
    {
        MyIOStream *stream = new MyIOStream{};
        auto result = fopen_s(&stream->stream, pFile, pMode);
        return (result == 0) ? stream : nullptr;
    }

    void Close(Assimp::IOStream *pFile) override
    {
        delete pFile;
    }
};

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
    importer.SetIOHandler(new MyIOSystem{});

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