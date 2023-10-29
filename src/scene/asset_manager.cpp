#include "asset_manager.hpp"

AssetManager::AssetManager(daxa::Device device)
    : device{std::move(device)}
{
    this->meshes_buffer = this->device.create_buffer({
        .size = sizeof(GPUMesh) * MAX_MESHES,
        .name = "meshes buffer",
    });
    this->tmeshes = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{meshes_buffer},
        },
        .name = "meshes buffer",
    }};
}

AssetManager::~AssetManager()
{
    device.destroy_buffer(meshes_buffer);
    for (auto &mesh : meshes)
    {
        device.destroy_buffer(std::bit_cast<daxa::BufferId>(mesh.mesh_buffer));
    }
}

// auto AssetManager::get_texture_if_present(std::string_view unique_name) -> std::optional<std::pair<u32, daxa::ImageId>>
// {
//     if (texture_lut.contains(unique_name))
//     {
//         return {{texture_lut[unique_name], this->textures[texture_lut[unique_name]]}};
//     }
//     return std::nullopt;
// }

// auto AssetManager::create_texture(std::string_view unique_name, aiScene const*scene, aiMaterial *aimaterial, aiTextureType type) -> std::pair<u32, daxa::ImageId>
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

// auto AssetManager::get_or_create_texture(aiScene const*scene, aiMaterial *aimaterial, aiTextureType type) -> std::pair<u32, daxa::ImageId>
// {
//     auto const unique_name = generate_texture_name(aimaterial, type);
//     auto ret = get_texture_if_present(unique_name);
//     if (!ret.has_value())
//     {
//         ret = create_texture(unique_name, scene, aimaterial, type);
//     }
//     return ret.value();
// }

// auto AssetManager::create_mesh(std::string_view unique_name, aiScene const*scene, aiMesh *aimesh) -> std::pair<u32, GPUMesh const *>
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

// auto AssetManager::get_mesh_if_present(std::string_view const &mesh_name) -> std::optional<std::pair<u32, GPUMesh const *>>
// {
//     if (mesh_lut.contains(mesh_name))
//     {
//         return {{mesh_lut[mesh_name], &this->meshes[mesh_lut[mesh_name]]}};
//     }
//     return std::nullopt;
// }
// auto AssetManager::get_or_create_mesh(aiScene const *scene, aiMesh * aimesh) -> std::pair<u32, GPUMesh const *>
// {
//     auto const unique_name = generate_mesh_name(aimesh);
//     auto ret = get_mesh_if_present(unique_name);
//     if (!ret.has_value())
//     {
//         ret = create_mesh(unique_name, scene, aimesh);
//     }
//     return ret.value();
// }

// auto AssetManager::get_update_commands() -> std::optional<daxa::ExecutableCommandList>
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