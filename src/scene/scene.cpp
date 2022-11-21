#include "scene.hpp"
#include "assimp/Importer.hpp"

#include <fstream>

void SceneLoader::load_entities_from_fbx(Scene & scene, AssetManager & asset_manager, std::filesystem::path const & asset_name)
{
    std::filesystem::path folder_path{ asset_root_folder / asset_name };
    std::ifstream ifs{ folder_path };

    if (!ifs.good())
    {
        std::cerr << "Error: could not open file: \"" << folder_path << "\"" << std::endl;
        return;
    }

    std::string file_content;
    ifs.seekg(0, std::ios::end);
    file_content.reserve(static_cast<usize>(ifs.tellg()));
    ifs.seekg(0, std::ios::beg);
    file_content.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

    Assimp::Importer importer;

    const aiScene* aiscene = importer.ReadFile(file_content, {});

    if (aiscene == nullptr)
    {
        std::cerr << "Error: Assimp failed to load scene with message: \"" << importer.GetErrorString() << "\"" << std::endl;
        return;
    }
}