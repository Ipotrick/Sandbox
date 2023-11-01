#include "asset_processor.hpp"
#include <fastgltf/tools.hpp>
#include <fstream>
#include <FreeImage.h>

#pragma region HELPER

struct RawImageDataFromURIInfo
{
    fastgltf::sources::URI const &uri;
    fastgltf::Asset const &asset;
};

using RawDataRet = std::variant<std::monostate, AssetProcessor::AssetLoadResultCode, std::vector<std::byte>>;
static auto raw_image_data_from_URI(RawImageDataFromURIInfo const &info) -> RawDataRet
{
    return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_READ_BUFFER_IN_GLTF;
}

struct RawImageDataFromBufferViewInfo
{
    fastgltf::sources::BufferView const &buffer_view;
    fastgltf::Asset const &asset;
    // Wihtout the scename.glb part
    std::filesystem::path const scene_dir_path;
};

static auto raw_image_data_from_buffer_view(RawImageDataFromBufferViewInfo const &info) -> RawDataRet
{
    fastgltf::BufferView const &gltf_buffer_view = info.asset.bufferViews.at(info.buffer_view.bufferViewIndex);
    fastgltf::Buffer const &gltf_buffer = info.asset.buffers.at(gltf_buffer_view.bufferIndex);

    if (!std::holds_alternative<fastgltf::sources::URI>(gltf_buffer.data))
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_FAULTY_BUFFER_VIEW;
    }
    fastgltf::sources::URI uri = std::get<fastgltf::sources::URI>(gltf_buffer.data);

    /// NOTE: load the section of the file containing the buffer for the mesh index buffer.
    std::filesystem::path const full_buffer_path = info.scene_dir_path / uri.uri.fspath();
    std::ifstream ifs{full_buffer_path, std::ios::binary};
    if (!ifs)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_OPEN_GLTF;
    }
    /// NOTE: Only load the relevant part of the file containing the view of the buffer we actually need.
    ifs.seekg(gltf_buffer_view.byteOffset + uri.fileByteOffset);
    std::vector<std::byte> raw = {};
    raw.resize(gltf_buffer_view.byteLength);
    /// NOTE: Only load the relevant part of the file containing the view of the buffer we actually need.
    if (!ifs.read(r_cast<char *>(raw.data()), gltf_buffer_view.byteLength))
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_READ_BUFFER_IN_GLTF;
    }
    return raw;
}

#pragma engregion

AssetProcessor::AssetProcessor(daxa::Device device)
// : device{std::move(device)}
{
// call this ONLY when linking with FreeImage as a static library
#ifdef FREEIMAGE_LIB
    FreeImage_Initialise();
#endif
    // this->meshes_buffer = this->device.create_buffer({
    //     .size = sizeof(GPUMesh) * MAX_MESHES,
    //     .name = "meshes buffer",
    // });
    // this->tmeshes = daxa::TaskBuffer{{
    //     .initial_buffers = {
    //         .buffers = std::array{meshes_buffer},
    //     },
    //     .name = "meshes buffer",
    // }};
}

AssetProcessor::~AssetProcessor()
{
// call this ONLY when linking with FreeImage as a static library
#ifdef FREEIMAGE_LIB
    FreeImage_DeInitialise();
#endif
    // device.destroy_buffer(meshes_buffer);
    // for (auto &mesh : meshes)
    // {
    //     device.destroy_buffer(std::bit_cast<daxa::BufferId>(mesh.mesh_buffer));
    // }
}

auto AssetProcessor::load_texture(Scene &scene, u32 texture_manifest_index) -> AssetLoadResultCode
{
    TextureManifestEntry const &texture_entry = scene._texture_manifest.at(texture_manifest_index);
    SceneFileManifestEntry const &scene_entry = scene._scene_file_manifest.at(texture_entry.scene_file_manifest_index);
    fastgltf::Asset const &gltf_asset = *scene_entry.gltf_asset;
    fastgltf::Image const &image = gltf_asset.images.at(texture_entry.in_scene_file_index);
    std::vector<std::byte> raw_data = {};

    RawDataRet ret = {};
    if (auto const *uri = std::get_if<fastgltf::sources::URI>(&image.data))
    {
        auto ret = raw_image_data_from_URI(RawImageDataFromURIInfo{
            .uri = *uri,
            .asset = gltf_asset});
    }
    else if (auto const *buffer_view = std::get_if<fastgltf::sources::BufferView>(&image.data))
    {
        auto ret = raw_image_data_from_buffer_view(RawImageDataFromBufferViewInfo{
            .buffer_view = *buffer_view,
            .asset = gltf_asset,
            .scene_dir_path = std::filesystem::path(scene_entry.path).remove_filename()});
    }

    if (auto const *error = std::get_if<AssetLoadResultCode>(&ret))
    {
        return *error;
    }
    raw_data = std::get<std::vector<std::byte>>(ret);

    // FREE_IMAGE_FORMAT image_format = FIF_UNKNOWN;
    // const char *c_filename = scene._texture_manifest.at(texture_manifest_index).name.c_str();
    // // check file signature and deduce it's format
    // image_format = FreeImage_GetFileType(c_filename);
    // // could not deduce filetype from metadata try to guess the format from the file extension
    // if (image_format == FIF_UNKNOWN)
    // {
    //     image_format = FreeImage_GetFIFFromFilename(c_filename);
    // }
    // // could not deduce filetype at all
    // if (image_format == FIF_UNKNOWN)
    // {
    //     return TextureLoadErrorCode::UNKNOWN_FILETYPE_FORMAT;
    // }

    // FIBITMAP *load_image_pointer(0);
    // BYTE *load_image_data(0);
    // if (FreeImage_FIFSupportsReading(image_format))
    // {
    //     FreeImage_LoadFromHandle();
    //     load_image_pointer = FreeImage_Load(image_format, c_filename);
    // }
}

/// NOTE: Overload ElementTraits for glm vec3 for fastgltf to understand the type.
template <>
struct fastgltf::ElementTraits<glm::vec3> : fastgltf::ElementTraitsBase<float, fastgltf::AccessorType::Vec3>
{
};

template <typename ElemT>
auto load_accessor_data_from_file(
    std::filesystem::path const & root_path,
    fastgltf::Asset const &asset, 
    fastgltf::Accessor const &accesor)
    -> std::variant<std::vector<ElemT>, AssetProcessor::AssetLoadResultCode>
{
    
}

auto AssetProcessor::load_mesh(Scene &scene, u32 mesh_index) -> AssetProcessor::AssetLoadResultCode
{
    MeshManifestEntry &mesh_data = scene._mesh_manifest.at(mesh_index);
    SceneFileManifestEntry &gltf_scene = scene._scene_file_manifest.at(mesh_data.scene_file_manifest_index);
    fastgltf::Asset &gltf_asset = *gltf_scene.gltf_asset;
    fastgltf::Mesh &gltf_mesh = gltf_asset.meshes[mesh_data.scene_file_mesh_index];
    fastgltf::Primitive &gltf_prim = gltf_mesh.primitives[mesh_data.scene_file_primitive_index];

    /// NOTE: Process indices (they are required)
    if (!gltf_prim.indicesAccessor.has_value())
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_MISSING_INDEX_BUFFER;
    }
    fastgltf::Accessor &index_buffer_gltf_accessor = gltf_asset.accessors.at(gltf_prim.indicesAccessor.value());

    bool const index_accessor_valid =
        (index_buffer_gltf_accessor.componentType == fastgltf::ComponentType::UnsignedInt ||
         index_buffer_gltf_accessor.componentType == fastgltf::ComponentType::UnsignedShort) &&
        index_buffer_gltf_accessor.type == fastgltf::AccessorType::Scalar &&
        index_buffer_gltf_accessor.bufferViewIndex.has_value();
    if (!index_accessor_valid)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR;
    }
    fastgltf::BufferView &gltf_buffer_view = gltf_asset.bufferViews.at(index_buffer_gltf_accessor.bufferViewIndex.value());
    fastgltf::Buffer &gltf_buffer = gltf_asset.buffers.at(gltf_buffer_view.bufferIndex);
    if (!std::holds_alternative<fastgltf::sources::URI>(gltf_buffer.data))
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_FAULTY_BUFFER_VIEW;
    }
    fastgltf::sources::URI uri = std::get<fastgltf::sources::URI>(gltf_buffer.data);
    // if (uri.mimeType != fastgltf::MimeType::GltfBuffer)
    // {
    //     return AssetProcessor::AssetLoadResultCode::ERROR_FAULTY_BUFFER_VIEW;
    // }

    /// NOTE: load the section of the file containing the buffer for the mesh index buffer.
    std::filesystem::path const full_buffer_path = gltf_scene.path.remove_filename() / uri.uri.fspath();
    std::ifstream ifs{full_buffer_path, std::ios::binary};
    if (!ifs)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_OPEN_GLTF;
    }
    /// NOTE: Only load the relevant part of the file containing the view of the buffer we actually need.
    ifs.seekg(gltf_buffer_view.byteOffset + index_buffer_gltf_accessor.byteOffset + uri.fileByteOffset);
    std::vector<u16> raw = {};
    raw.resize(gltf_buffer_view.byteLength / 2);
    /// NOTE: Only load the relevant part of the file containing the view of the buffer we actually need.
    auto const elem_byte_size = fastgltf::getElementByteSize(index_buffer_gltf_accessor.type, index_buffer_gltf_accessor.componentType);
    if (!ifs.read(r_cast<char *>(raw.data()), index_buffer_gltf_accessor.count * elem_byte_size))
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_COULD_NOT_READ_BUFFER_IN_GLTF;
    }
    auto fastgltf_index_buffer_adapter = [&](const fastgltf::Buffer &buffer)
    {
        /// NOTE:   We only have a ptr to the loaded data to the accessors section of the buffer.
        ///         Fastgltf expects a ptr to the begin of the buffer, so we just subtract the offsets.
        ///         Fastgltf adds these on in the accessor tool, so in the end it gets the right ptr.
        auto const fastgltf_reverse_byte_offset = (gltf_buffer_view.byteOffset + index_buffer_gltf_accessor.byteOffset);
        return r_cast<std::byte *>(raw.data()) - fastgltf_reverse_byte_offset;
    };

    /// NOTE: Transform the loaded file section into a 32 bit index buffer.
    std::vector<u32> index_buffer(index_buffer_gltf_accessor.count);
    index_buffer.resize(index_buffer_gltf_accessor.count);
    if (index_buffer_gltf_accessor.componentType == fastgltf::ComponentType::UnsignedShort)
    {
        std::vector<u16> u16_index_buffer(index_buffer_gltf_accessor.count);
        fastgltf::copyFromAccessor<u16>(gltf_asset, index_buffer_gltf_accessor, u16_index_buffer.data(), fastgltf_index_buffer_adapter);
        for (size_t i = 0; i < u16_index_buffer.size(); ++i)
        {
            index_buffer[i] = s_cast<u32>(u16_index_buffer[i]);
        }
    }
    else
    {
        fastgltf::copyFromAccessor<u32>(gltf_asset, index_buffer_gltf_accessor, index_buffer.data(), fastgltf_index_buffer_adapter);
    }

    /// NOTE: Load vertex positions
    if (!gltf_prim.attributes.contains(VERT_ATTRIB_POSITION_NAME))
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_MISSING_VERTEX_POSITIONS;
    }
    fastgltf::Accessor &gltf_vertex_pos_accessor = gltf_asset.accessors.at(gltf_prim.attributes[VERT_ATTRIB_POSITION_NAME]);
    bool const gltf_vertex_pos_accessor_valid =
        gltf_vertex_pos_accessor.componentType == fastgltf::ComponentType::Float &&
        gltf_vertex_pos_accessor.type == fastgltf::AccessorType::Vec3;
    if (!gltf_vertex_pos_accessor_valid)
    {
        return AssetProcessor::AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_POSITIONS;
    }
    std::vector<glm::vec3> position_buffer(gltf_vertex_pos_accessor.count);
    auto fastgltf_vert_pos_buffer_adapter = [&](const fastgltf::Buffer &buffer)
    {
        /// NOTE:   We only have a ptr to the loaded data to the accessors section of the buffer.
        ///         Fastgltf expects a ptr to the begin of the buffer, so we just subtract the offsets.
        ///         Fastgltf adds these on in the accessor tool, so in the end it gets the right ptr.
        auto const fastgltf_reverse_byte_offset = (gltf_buffer_view.byteOffset + gltf_vertex_pos_accessor.byteOffset);
        return r_cast<std::byte *>(raw.data()) - fastgltf_reverse_byte_offset;
    };
    fastgltf::copyFromAccessor<glm::vec3>(gltf_asset, gltf_vertex_pos_accessor, r_cast<void *>(position_buffer.data()), fastgltf_vert_pos_buffer_adapter);

    //    ASSERT_M(meshes.size() + 1 < MAX_MESHES, "Exceeded max mesh count!");
    //     // Create entry of mesh in fields.
    //     ASSERT_M(!mesh_lut.contains(unique_name), "All meshes MUST have unique names!");
    //     u32 mesh_index = static_cast<u32>(this->meshes.size());
    //     this->meshes.push_back({});
    //     this->mesh_names.push_back(std::string{unique_name});
    //     this->mesh_lut[this->mesh_names[mesh_index]] = mesh_index;
    //     GPUMesh &mesh = this->meshes.back();
    //     // Create standart index buffer.
    //     std::vector<u32> index_buffer(aimesh->mNumFaces * 3);
    //     for (usize face_i = 0; face_i < aimesh->mNumFaces; ++face_i)
    //     {
    //         for (usize index = 0; index < 3; ++index)
    //         {
    //             index_buffer[face_i * 3 + index] = aimesh->mFaces[face_i].mIndices[index];
    //         }
    //     }
    //     // Texture loading:
    //     {
    //         // Load albedo:
    //         aiMaterial *aimaterial = scene->mMaterials[aimesh->mMaterialIndex];
    //         ASSERT_M(aimaterial != nullptr, "all meshes need a material");
    //         bool const has_albedo = aimaterial->GetTextureCount(aiTextureType_DIFFUSE) > 0;
    //         if (has_albedo)
    //         {
    //             auto [manager_index, tex_id] = get_or_create_texture(scene, aimaterial, aiTextureType_DIFFUSE);
    //             mesh.abledo_tex_id = tex_id.default_view();
    //         }
    //         else
    //         {
    //             mesh.abledo_tex_id = {};
    //         }
    //     }
    //     constexpr usize MAX_VERTICES = MAX_VERTICES_PER_MESHLET;
    //     constexpr usize MAX_TRIANGLES = MAX_TRIANGLES_PER_MESHLET;
    //     // No clue what cone culling is.
    //     constexpr float CONE_WEIGHT = 1.0f;
    //     //std::vector<u32> optimized_indices = {};
    //     //optimized_indices.resize(index_buffer.size());
    //     //meshopt_optimizeVertexCache(optimized_indices.data(), index_buffer.data(), index_buffer.size(), aimesh->mNumVertices);
    //     //index_buffer = optimized_indices;
    //     size_t max_meshlets = meshopt_buildMeshletsBound(index_buffer.size(), MAX_VERTICES, MAX_TRIANGLES);
    //     std::vector<meshopt_Meshlet> meshlets(max_meshlets);
    //     std::vector<u32> meshlet_indirect_vertices(max_meshlets * MAX_VERTICES);
    //     std::vector<u8> meshlet_micro_indices(max_meshlets * MAX_TRIANGLES * 3);
    //     size_t meshlet_count = meshopt_buildMeshlets(
    //         meshlets.data(),
    //         meshlet_indirect_vertices.data(),
    //         meshlet_micro_indices.data(),
    //         index_buffer.data(),
    //         index_buffer.size(),
    //         reinterpret_cast<float *>(aimesh->mVertices),
    //         static_cast<usize>(aimesh->mNumVertices),
    //         sizeof(f32vec3),
    //         MAX_VERTICES,
    //         MAX_TRIANGLES,
    //         CONE_WEIGHT);
    //     std::vector<BoundingSphere> meshlet_bounds(meshlet_count);
    //     for (size_t meshlet_i = 0; meshlet_i < meshlet_count; ++meshlet_i)
    //     {
    //         meshopt_Bounds raw_bounds = meshopt_computeMeshletBounds(
    //             &meshlet_indirect_vertices[meshlets[meshlet_i].vertex_offset],
    //             &meshlet_micro_indices[meshlets[meshlet_i].triangle_offset],
    //             meshlets[meshlet_i].triangle_count,
    //             &aimesh->mVertices[0].x,
    //             static_cast<usize>(aimesh->mNumVertices),
    //             sizeof(f32vec3));
    //         meshlet_bounds[meshlet_i].center[0] = raw_bounds.center[0];
    //         meshlet_bounds[meshlet_i].center[1] = raw_bounds.center[1];
    //         meshlet_bounds[meshlet_i].center[2] = raw_bounds.center[2];
    //         meshlet_bounds[meshlet_i].radius = raw_bounds.radius;
    //     }
    //     // Trimm array sizes.
    //     const meshopt_Meshlet &last = meshlets[meshlet_count - 1];
    //     meshlet_indirect_vertices.resize(last.vertex_offset + last.vertex_count);
    //     meshlet_micro_indices.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
    //     meshlets.resize(meshlet_count);
    //     total_meshlet_count += meshlet_count;
    //     // Determine offsets and size of the buffer containing all mesh data.
    //     u32 allocation_size = 0;
    //     u32 const meshlet_array_offset = allocation_size;
    //     usize const meshlet_array_bytesize = meshlets.size() * sizeof(Meshlet);
    //     allocation_size += meshlet_array_bytesize;
    //     u32 const meshlet_bounds_array_offset = allocation_size;
    //     usize const meshlet_bounds_array_bytesize = meshlet_bounds.size() * sizeof(BoundingSphere);
    //     allocation_size += meshlet_bounds_array_bytesize;
    //     u32 const micro_index_array_offset = allocation_size;
    //     usize const micro_index_array_bytesize = ((meshlet_micro_indices.size() * sizeof(u8)) + sizeof(u32) - 1) / sizeof(u32) * sizeof(u32);
    //     allocation_size += micro_index_array_bytesize;
    //     u32 const indirect_vertex_array_offset = allocation_size;
    //     usize const indirect_vertex_array_bytesize = meshlet_indirect_vertices.size() * sizeof(u32);
    //     allocation_size += indirect_vertex_array_bytesize;
    //     u32 const vertex_positions_array_offset = allocation_size;
    //     usize const vertex_positions_array_bytesize = static_cast<usize>(aimesh->mNumVertices) * sizeof(f32vec3);
    //     allocation_size += vertex_positions_array_bytesize;
    //     u32 const vertex_uvs_array_offset = allocation_size;
    //     usize const vertex_uvs_array_bytesize = static_cast<usize>(aimesh->mNumVertices) * sizeof(f32vec2);
    //     allocation_size += vertex_uvs_array_bytesize;
    //     allocation_size += sizeof(u32); // no clue why i need this... Maybe it is because it tries to do 16 byte loads on vec3s.
    //     // Create mesh.
    //     mesh.mesh_buffer = device.create_buffer({
    //         .size = allocation_size,
    //         .allocate_info = daxa::MemoryFlagBits::DEDICATED_MEMORY,
    //         .name = std::string("GPUMesh Buffer of mesh \"") + std::string(unique_name) + "\"",
    //     });
    //     mesh.meshlet_count = meshlets.size();
    //     mesh.vertex_count = aimesh->mNumVertices;
    //     mesh.meshlets = device.get_device_address(mesh.mesh_buffer) + meshlet_array_offset;
    //     mesh.meshlet_bounds = device.get_device_address(mesh.mesh_buffer) + meshlet_bounds_array_offset;
    //     mesh.micro_indices = device.get_device_address(mesh.mesh_buffer) + micro_index_array_offset;
    //     mesh.indirect_vertices = device.get_device_address(mesh.mesh_buffer) + indirect_vertex_array_offset;
    //     mesh.vertex_positions = device.get_device_address(mesh.mesh_buffer) + vertex_positions_array_offset;
    //     mesh.vertex_uvs = aimesh->GetNumUVChannels() > 0 ? device.get_device_address(mesh.mesh_buffer) + vertex_uvs_array_offset : 0;
    //     mesh.end_ptr = device.get_device_address(mesh.mesh_buffer) + allocation_size;
    //     // Stage buffer upload.
    //     daxa::BufferId staging_buffer = device.create_buffer({
    //         .size = allocation_size,
    //         .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
    //         .name = std::string("Staging buffer for mesh \"") + std::string(unique_name) + "\"",
    //     });
    //     void *staging_buffer_ptr = device.get_host_address(staging_buffer);

    //     void *staging_meshlets = reinterpret_cast<u8 *>(staging_buffer_ptr) + meshlet_array_offset;
    //     std::memcpy(staging_meshlets, meshlets.data(), meshlet_array_bytesize);
    //     void *staging_meshlet_bounds = reinterpret_cast<u8 *>(staging_buffer_ptr) + meshlet_bounds_array_offset;
    //     std::memcpy(staging_meshlet_bounds, meshlet_bounds.data(), meshlet_bounds_array_bytesize);
    //     void *staging_micro_indices = reinterpret_cast<u8 *>(staging_buffer_ptr) + micro_index_array_offset;
    //     std::memcpy(staging_micro_indices, meshlet_micro_indices.data(), micro_index_array_bytesize);
    //     void *staging_indirect_vertices = reinterpret_cast<u8 *>(staging_buffer_ptr) + indirect_vertex_array_offset;
    //     std::memcpy(staging_indirect_vertices, meshlet_indirect_vertices.data(), indirect_vertex_array_bytesize);
    //     void *staging_vertex_positions = reinterpret_cast<u8 *>(staging_buffer_ptr) + vertex_positions_array_offset;
    //     std::memcpy(staging_vertex_positions, aimesh->mVertices, vertex_positions_array_bytesize);
    //     if (aimesh->GetNumUVChannels() > 0)
    //     {
    //         void *staging_vertex_uvs = reinterpret_cast<u8 *>(staging_buffer_ptr) + vertex_uvs_array_offset;
    //         for (u32 i = 0; i < aimesh->mNumVertices; ++i)
    //         {
    //             reinterpret_cast<daxa::f32vec2*>(staging_vertex_uvs)[i] = {aimesh->mTextureCoords[0][i].x, aimesh->mTextureCoords[0][i].y};
    //         }
    //     }
    //     // Record mesh buffer update calls.
    //     if (!this->asset_update_cmd_list.has_value())
    //     {
    //         this->asset_update_cmd_list = this->device.create_command_list({.name = "asset update cmd list"});
    //         this->asset_update_cmd_list.value().pipeline_barrier({
    //             .src_access = daxa::AccessConsts::HOST_WRITE,
    //             .dst_access = daxa::AccessConsts::TRANSFER_READ,
    //         });
    //     }
    //     auto &cmd = this->asset_update_cmd_list.value();
    //     cmd.destroy_buffer_deferred(staging_buffer);
    //     cmd.copy_buffer_to_buffer({
    //         .src_buffer = staging_buffer,
    //         .dst_buffer = mesh.mesh_buffer,
    //         .size = allocation_size,
    //     });
    //     std::cout << "mesh \"" << unique_name << "\" has " << meshlets.size() << " meshlets" << std::endl;
    //     return { mesh_index, &mesh };
    return AssetLoadResultCode::SUCCESS;
}

// auto AssetProcessor::get_texture_if_present(std::string_view unique_name) -> std::optional<std::pair<u32, daxa::ImageId>>
// {
//     if (texture_lut.contains(unique_name))
//     {
//         return {{texture_lut[unique_name], this->textures[texture_lut[unique_name]]}};
//     }
//     return std::nullopt;
// }

// auto AssetProcessor::create_texture(std::string_view unique_name, aiScene const*scene, aiMaterial *aimaterial, aiTextureType type) -> std::pair<u32, daxa::ImageId>
// {
//     // Create entry of mesh in fields.
//     ASSERT_M(!texture_lut.contains(unique_name), "All textures MUST have unique names!");
//     aiString path = {};
//     aiTextureMapping mapping = {};
//     unsigned int uvindex = {};
//     ai_real blend = {};
//     aiTextureOp op = {};
//     aiTextureMapMode mapmode = {};
//     aimaterial->GetTexture(type, 0, &path, &mapping, &uvindex, &blend, &op, &mapmode);

//     ASSERT_M(path.length > 0, "empty path of material texture");
//     if (path.data[0] == '*')
//     {
//         // Assimp loads embeded textures for you, these dont have a path but mark their embeded index in the string instead.

//     }

//     u32 tex_index = static_cast<u32>(this->textures.size());
//     this->textures.push_back({});
//     this->texture_names.push_back(std::string{unique_name});
//     this->texture_lut[this->texture_names[tex_index]] = tex_index;
//     daxa::ImageId &texture_id = this->textures.back();
//     ASSERT_M(strcmp(aitexture->achFormatHint, "rgba888") == 0, "texture format must be rgba888");
//     texture_id = device.create_image({
//         .flags = {},
//         .dimensions = 2,
//         .format = daxa::Format::R8G8B8A8_SRGB,
//         .size = {aitexture->mWidth,aitexture->mHeight,1},
//         .mip_level_count = 1,
//         .array_layer_count = 1,
//         .sample_count = 1,
//         .usage = TEXTURE_USE_FLAGS,
//         .name = std::string(unique_name),
//     });
//     if (!asset_update_cmd_list.has_value())
//     {
//         asset_update_cmd_list = device.create_command_list({});
//     }
//     auto cmd = asset_update_cmd_list.value();
//     auto mem_size = 4 * aitexture->mWidth * aitexture->mHeight;
//     auto staging_buffer = device.create_buffer({
//         .size = mem_size,
//         .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
//         .name = std::string(unique_name) + "_staging buffer",
//     });
//     cmd.destroy_buffer_deferred(staging_buffer);
//     auto host_ptr = device.get_host_address(staging_buffer);
//     std::memcpy(host_ptr, aitexture->pcData, mem_size);
//     texture_upload_list.push_back({staging_buffer, texture_id});
// }

// auto AssetProcessor::get_or_create_texture(aiScene const*scene, aiMaterial *aimaterial, aiTextureType type) -> std::pair<u32, daxa::ImageId>
// {
//     auto const unique_name = generate_texture_name(aimaterial, type);
//     auto ret = get_texture_if_present(unique_name);
//     if (!ret.has_value())
//     {
//         ret = create_texture(unique_name, scene, aimaterial, type);
//     }
//     return ret.value();
// }

// auto AssetProcessor::create_mesh(std::string_view unique_name, aiScene const*scene, aiMesh *aimesh) -> std::pair<u32, GPUMesh const *>
// {
//     ASSERT_M(meshes.size() + 1 < MAX_MESHES, "Exceeded max mesh count!");
//     // Create entry of mesh in fields.
//     ASSERT_M(!mesh_lut.contains(unique_name), "All meshes MUST have unique names!");
//     u32 mesh_index = static_cast<u32>(this->meshes.size());
//     this->meshes.push_back({});
//     this->mesh_names.push_back(std::string{unique_name});
//     this->mesh_lut[this->mesh_names[mesh_index]] = mesh_index;
//     GPUMesh &mesh = this->meshes.back();
//     // Create standart index buffer.
//     std::vector<u32> index_buffer(aimesh->mNumFaces * 3);
//     for (usize face_i = 0; face_i < aimesh->mNumFaces; ++face_i)
//     {
//         for (usize index = 0; index < 3; ++index)
//         {
//             index_buffer[face_i * 3 + index] = aimesh->mFaces[face_i].mIndices[index];
//         }
//     }
//     // Texture loading:
//     {
//         // Load albedo:
//         aiMaterial *aimaterial = scene->mMaterials[aimesh->mMaterialIndex];
//         ASSERT_M(aimaterial != nullptr, "all meshes need a material");
//         bool const has_albedo = aimaterial->GetTextureCount(aiTextureType_DIFFUSE) > 0;
//         if (has_albedo)
//         {
//             auto [manager_index, tex_id] = get_or_create_texture(scene, aimaterial, aiTextureType_DIFFUSE);
//             mesh.abledo_tex_id = tex_id.default_view();
//         }
//         else
//         {
//             mesh.abledo_tex_id = {};
//         }
//     }
//     constexpr usize MAX_VERTICES = MAX_VERTICES_PER_MESHLET;
//     constexpr usize MAX_TRIANGLES = MAX_TRIANGLES_PER_MESHLET;
//     // No clue what cone culling is.
//     constexpr float CONE_WEIGHT = 1.0f;
//     //std::vector<u32> optimized_indices = {};
//     //optimized_indices.resize(index_buffer.size());
//     //meshopt_optimizeVertexCache(optimized_indices.data(), index_buffer.data(), index_buffer.size(), aimesh->mNumVertices);
//     //index_buffer = optimized_indices;
//     size_t max_meshlets = meshopt_buildMeshletsBound(index_buffer.size(), MAX_VERTICES, MAX_TRIANGLES);
//     std::vector<meshopt_Meshlet> meshlets(max_meshlets);
//     std::vector<u32> meshlet_indirect_vertices(max_meshlets * MAX_VERTICES);
//     std::vector<u8> meshlet_micro_indices(max_meshlets * MAX_TRIANGLES * 3);
//     size_t meshlet_count = meshopt_buildMeshlets(
//         meshlets.data(),
//         meshlet_indirect_vertices.data(),
//         meshlet_micro_indices.data(),
//         index_buffer.data(),
//         index_buffer.size(),
//         reinterpret_cast<float *>(aimesh->mVertices),
//         static_cast<usize>(aimesh->mNumVertices),
//         sizeof(f32vec3),
//         MAX_VERTICES,
//         MAX_TRIANGLES,
//         CONE_WEIGHT);
//     std::vector<BoundingSphere> meshlet_bounds(meshlet_count);
//     for (size_t meshlet_i = 0; meshlet_i < meshlet_count; ++meshlet_i)
//     {
//         meshopt_Bounds raw_bounds = meshopt_computeMeshletBounds(
//             &meshlet_indirect_vertices[meshlets[meshlet_i].vertex_offset],
//             &meshlet_micro_indices[meshlets[meshlet_i].triangle_offset],
//             meshlets[meshlet_i].triangle_count,
//             &aimesh->mVertices[0].x,
//             static_cast<usize>(aimesh->mNumVertices),
//             sizeof(f32vec3));
//         meshlet_bounds[meshlet_i].center[0] = raw_bounds.center[0];
//         meshlet_bounds[meshlet_i].center[1] = raw_bounds.center[1];
//         meshlet_bounds[meshlet_i].center[2] = raw_bounds.center[2];
//         meshlet_bounds[meshlet_i].radius = raw_bounds.radius;
//     }
//     // Trimm array sizes.
//     const meshopt_Meshlet &last = meshlets[meshlet_count - 1];
//     meshlet_indirect_vertices.resize(last.vertex_offset + last.vertex_count);
//     meshlet_micro_indices.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
//     meshlets.resize(meshlet_count);
//     total_meshlet_count += meshlet_count;
//     // Determine offsets and size of the buffer containing all mesh data.
//     u32 allocation_size = 0;
//     u32 const meshlet_array_offset = allocation_size;
//     usize const meshlet_array_bytesize = meshlets.size() * sizeof(Meshlet);
//     allocation_size += meshlet_array_bytesize;
//     u32 const meshlet_bounds_array_offset = allocation_size;
//     usize const meshlet_bounds_array_bytesize = meshlet_bounds.size() * sizeof(BoundingSphere);
//     allocation_size += meshlet_bounds_array_bytesize;
//     u32 const micro_index_array_offset = allocation_size;
//     usize const micro_index_array_bytesize = ((meshlet_micro_indices.size() * sizeof(u8)) + sizeof(u32) - 1) / sizeof(u32) * sizeof(u32);
//     allocation_size += micro_index_array_bytesize;
//     u32 const indirect_vertex_array_offset = allocation_size;
//     usize const indirect_vertex_array_bytesize = meshlet_indirect_vertices.size() * sizeof(u32);
//     allocation_size += indirect_vertex_array_bytesize;
//     u32 const vertex_positions_array_offset = allocation_size;
//     usize const vertex_positions_array_bytesize = static_cast<usize>(aimesh->mNumVertices) * sizeof(f32vec3);
//     allocation_size += vertex_positions_array_bytesize;
//     u32 const vertex_uvs_array_offset = allocation_size;
//     usize const vertex_uvs_array_bytesize = static_cast<usize>(aimesh->mNumVertices) * sizeof(f32vec2);
//     allocation_size += vertex_uvs_array_bytesize;
//     allocation_size += sizeof(u32); // no clue why i need this... Maybe it is because it tries to do 16 byte loads on vec3s.
//     // Create mesh.
//     mesh.mesh_buffer = device.create_buffer({
//         .size = allocation_size,
//         .allocate_info = daxa::MemoryFlagBits::DEDICATED_MEMORY,
//         .name = std::string("GPUMesh Buffer of mesh \"") + std::string(unique_name) + "\"",
//     });
//     mesh.meshlet_count = meshlets.size();
//     mesh.vertex_count = aimesh->mNumVertices;
//     mesh.meshlets = device.get_device_address(mesh.mesh_buffer) + meshlet_array_offset;
//     mesh.meshlet_bounds = device.get_device_address(mesh.mesh_buffer) + meshlet_bounds_array_offset;
//     mesh.micro_indices = device.get_device_address(mesh.mesh_buffer) + micro_index_array_offset;
//     mesh.indirect_vertices = device.get_device_address(mesh.mesh_buffer) + indirect_vertex_array_offset;
//     mesh.vertex_positions = device.get_device_address(mesh.mesh_buffer) + vertex_positions_array_offset;
//     mesh.vertex_uvs = aimesh->GetNumUVChannels() > 0 ? device.get_device_address(mesh.mesh_buffer) + vertex_uvs_array_offset : 0;
//     mesh.end_ptr = device.get_device_address(mesh.mesh_buffer) + allocation_size;
//     // Stage buffer upload.
//     daxa::BufferId staging_buffer = device.create_buffer({
//         .size = allocation_size,
//         .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
//         .name = std::string("Staging buffer for mesh \"") + std::string(unique_name) + "\"",
//     });
//     void *staging_buffer_ptr = device.get_host_address(staging_buffer);

//     void *staging_meshlets = reinterpret_cast<u8 *>(staging_buffer_ptr) + meshlet_array_offset;
//     std::memcpy(staging_meshlets, meshlets.data(), meshlet_array_bytesize);
//     void *staging_meshlet_bounds = reinterpret_cast<u8 *>(staging_buffer_ptr) + meshlet_bounds_array_offset;
//     std::memcpy(staging_meshlet_bounds, meshlet_bounds.data(), meshlet_bounds_array_bytesize);
//     void *staging_micro_indices = reinterpret_cast<u8 *>(staging_buffer_ptr) + micro_index_array_offset;
//     std::memcpy(staging_micro_indices, meshlet_micro_indices.data(), micro_index_array_bytesize);
//     void *staging_indirect_vertices = reinterpret_cast<u8 *>(staging_buffer_ptr) + indirect_vertex_array_offset;
//     std::memcpy(staging_indirect_vertices, meshlet_indirect_vertices.data(), indirect_vertex_array_bytesize);
//     void *staging_vertex_positions = reinterpret_cast<u8 *>(staging_buffer_ptr) + vertex_positions_array_offset;
//     std::memcpy(staging_vertex_positions, aimesh->mVertices, vertex_positions_array_bytesize);
//     if (aimesh->GetNumUVChannels() > 0)
//     {
//         void *staging_vertex_uvs = reinterpret_cast<u8 *>(staging_buffer_ptr) + vertex_uvs_array_offset;
//         for (u32 i = 0; i < aimesh->mNumVertices; ++i)
//         {
//             reinterpret_cast<daxa::f32vec2*>(staging_vertex_uvs)[i] = {aimesh->mTextureCoords[0][i].x, aimesh->mTextureCoords[0][i].y};
//         }
//     }
//     // Record mesh buffer update calls.
//     if (!this->asset_update_cmd_list.has_value())
//     {
//         this->asset_update_cmd_list = this->device.create_command_list({.name = "asset update cmd list"});
//         this->asset_update_cmd_list.value().pipeline_barrier({
//             .src_access = daxa::AccessConsts::HOST_WRITE,
//             .dst_access = daxa::AccessConsts::TRANSFER_READ,
//         });
//     }
//     auto &cmd = this->asset_update_cmd_list.value();
//     cmd.destroy_buffer_deferred(staging_buffer);
//     cmd.copy_buffer_to_buffer({
//         .src_buffer = staging_buffer,
//         .dst_buffer = mesh.mesh_buffer,
//         .size = allocation_size,
//     });
//     std::cout << "mesh \"" << unique_name << "\" has " << meshlets.size() << " meshlets" << std::endl;
//     return { mesh_index, &mesh };
// }

// auto AssetProcessor::get_mesh_if_present(std::string_view const &mesh_name) -> std::optional<std::pair<u32, GPUMesh const *>>
// {
//     if (mesh_lut.contains(mesh_name))
//     {
//         return {{mesh_lut[mesh_name], &this->meshes[mesh_lut[mesh_name]]}};
//     }
//     return std::nullopt;
// }
// auto AssetProcessor::get_or_create_mesh(aiScene const *scene, aiMesh * aimesh) -> std::pair<u32, GPUMesh const *>
// {
//     auto const unique_name = generate_mesh_name(aimesh);
//     auto ret = get_mesh_if_present(unique_name);
//     if (!ret.has_value())
//     {
//         ret = create_mesh(unique_name, scene, aimesh);
//     }
//     return ret.value();
// }

// auto AssetProcessor::get_update_commands() -> std::optional<daxa::ExecutableCommandList>
// {
//     if (this->asset_update_cmd_list.has_value())
//     {
//         daxa::CommandEncoder cmd = std::move(this->asset_update_cmd_list.value());
//         this->asset_update_cmd_list.reset();
//         auto staging_buffer = device.create_buffer({ .size = sizeof(GPUMesh) * MAX_MESHES, .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM, .name = "mesh buffer staging upload buffer" });
//         cmd.destroy_buffer_deferred(staging_buffer);
//         auto host_ptr = device.get_host_address_as<GPUMesh>(staging_buffer);
//         for (usize mesh_i = 0; mesh_i < meshes.size(); ++mesh_i)
//         {
//             host_ptr[mesh_i] = meshes[mesh_i];
//         }
//         cmd.pipeline_barrier({
//             .src_access = daxa::AccessConsts::HOST_WRITE,
//             .dst_access = daxa::AccessConsts::TRANSFER_READ,
//         });
//         for (auto [_, texture_id] : texture_upload_list)
//         {
//             cmd.pipeline_barrier_image_transition({
//                 .src_access = {},
//                 .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
//                 .src_layout = daxa::ImageLayout::UNDEFINED,
//                 .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
//                 .image_slice = {},
//                 .image_id = texture_id,
//             });
//         }
//         cmd.copy_buffer_to_buffer({
//             .src_buffer = staging_buffer,
//             .dst_buffer = meshes_buffer,
//             .size = sizeof(GPUMesh) * meshes.size(),
//         });
//         for (auto [staging_buffer, texture_id] : texture_upload_list)
//         {
//             cmd.copy_buffer_to_image({
//                 .buffer = staging_buffer,
//                 .buffer_offset = {},
//                 .image = texture_id,
//                 .image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
//                 .image_slice = {},
//                 .image_offset = {},
//                 .image_extent = device.info_image(texture_id).size,
//             });
//         }
//         cmd.pipeline_barrier({
//             .src_access = daxa::AccessConsts::TRANSFER_WRITE,
//             .dst_access = daxa::AccessConsts::READ,
//         });
//         for (auto [_, texture_id] : texture_upload_list)
//         {
//             cmd.pipeline_barrier_image_transition({
//                 .src_access = daxa::AccessConsts::TRANSFER_WRITE,
//                 .dst_access = daxa::AccessConsts::READ,
//                 .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
//                 .dst_layout = daxa::ImageLayout::READ_ONLY_OPTIMAL,
//                 .image_slice = {},
//                 .image_id = texture_id,
//             });
//         }
//         return cmd.complete_current_commands();
//     }
//     return {};
// }