#pragma once

// #include <assimp/Importer.hpp>
// #include <assimp/mesh.h>
// #include <assimp/scene.h>
// #include <assimp/postprocess.h>
// #include <assimp/IOStream.hpp>
// #include <assimp/IOSystem.hpp>

#include <meshoptimizer.h>

#include "../sandbox.hpp"
#include "../rendering/gpu_context.hpp"
#include "../../shader_shared/asset.inl"
#include "scene.hpp"

using MeshIndex = size_t;
using ImageIndex = size_t;

#define MAX_MESHES 10000

// inline std::string generate_mesh_name(aiMesh* mesh)
// {
//     return
//         std::string(mesh->mName.C_Str()) + std::string(" m:") + std::to_string(mesh->mMaterialIndex);
// }
//
// inline std::string generate_texture_name(aiMaterial * material, aiTextureType type)
// {
//     return std::string(material->GetName().C_Str()) + aiTextureTypeToString(type);
// }

struct AssetProcessor
{
    enum struct AssetLoadResultCode
    {
        SUCCESS,
        ERROR_MISSING_INDEX_BUFFER,
        ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR,
        ERROR_FAULTY_BUFFER_VIEW,
        ERROR_COULD_NOT_OPEN_GLTF,
        ERROR_COULD_NOT_READ_BUFFER_IN_GLTF,
        ERROR_MISSING_VERTEX_POSITIONS,
        ERROR_FAULTY_GLTF_VERTEX_POSITIONS,
    };
    static auto to_string(AssetLoadResultCode code) -> std::string_view
    {
        switch(code)
        {
            case AssetLoadResultCode::SUCCESS: return "SUCCESS";
            case AssetLoadResultCode::ERROR_MISSING_INDEX_BUFFER: return "ERROR_MISSING_INDEX_BUFFER";
            case AssetLoadResultCode::ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR: return "ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR";
            case AssetLoadResultCode::ERROR_FAULTY_BUFFER_VIEW: return "ERROR_FAULTY_BUFFER_VIEW";
            case AssetLoadResultCode::ERROR_COULD_NOT_OPEN_GLTF: return "ERROR_COULD_NOT_OPEN_GLTF";
            case AssetLoadResultCode::ERROR_COULD_NOT_READ_BUFFER_IN_GLTF: return "ERROR_COULD_NOT_READ_BUFFER_IN_GLTF";
            case AssetLoadResultCode::ERROR_MISSING_VERTEX_POSITIONS: return "ERROR_MISSING_VERTEX_POSITIONS";
            case AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_POSITIONS: return "ERROR_FAULTY_GLTF_VERTEX_POSITIONS";
            default: return "UNKNOWN";
        }
    }
    AssetProcessor(daxa::Device device);
    AssetProcessor(AssetProcessor &&) = default;
    ~AssetProcessor();

    /**
     * THREADSAFETY:
     * * internally synchronized, can be called on multiple threads in parallel.
     */
    auto load_texture(Scene &scene, u32 texture_manifest_index) -> AssetLoadResultCode;

    /**
     * THREADSAFETY:
     * * internally synchronized, can be called on multiple threads in parallel.
     */
    auto load_mesh(Scene &scene, u32 mesh_manifest_index) -> AssetLoadResultCode;

    /**
     * THREADSAFETY:
     * * internally synchronized, can be called on multiple threads in parallel.
     */
    auto flush_loading_commands() -> daxa::ExecutableCommandList;

private:
    static inline const std::string VERT_ATTRIB_POSITION_NAME = "POSITION";
    static inline const std::string VERT_ATTRIB_NORMAL_NAME = "NORMAL";
    static inline const std::string VERT_ATTRIB_TEXCOORD0_NAME = "TEXCOORD_0";

    daxa::Device _device = {};
    std::mutex _mtx = {};
};