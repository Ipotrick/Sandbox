#include "scene.hpp"

#include <fstream>

#include <fmt/format.h>
#include <glm/gtx/quaternion.hpp>

Scene::Scene(daxa::Device device)
    : _device{std::move(device)}
{
    /// TODO: THIS IS TEMPORARY! Make manifest and entity buffers growable!
    _gpu_entity_meta.set_buffers(daxa::TrackedBuffers{.buffers = std::array{
                                                          _device.create_buffer({
                                                              .size = sizeof(GPUEntityMetaData),
                                                              .name = "_gpu_entity_meta",
                                                          }),
                                                      }});
    _gpu_entity_transforms.set_buffers(daxa::TrackedBuffers{.buffers = std::array{
                                                                _device.create_buffer({
                                                                    .size = sizeof(daxa_f32mat4x3) * MAX_ENTITY_COUNT,
                                                                    .name = "_gpu_entity_transforms",
                                                                }),
                                                            }});
    _gpu_entity_combined_transforms.set_buffers(daxa::TrackedBuffers{.buffers = std::array{
                                                                         _device.create_buffer({
                                                                             .size = sizeof(daxa_f32mat4x3) * MAX_ENTITY_COUNT,
                                                                             .name = "_gpu_entity_combined_transforms",
                                                                         }),
                                                                     }});
    _gpu_entity_mesh_groups.set_buffers(daxa::TrackedBuffers{.buffers = std::array{
                                                                 _device.create_buffer({
                                                                     .size = sizeof(u32) * MAX_ENTITY_COUNT,
                                                                     .name = "_gpu_entity_mesh_groups",
                                                                 }),
                                                             }});
    _gpu_mesh_manifest.set_buffers(daxa::TrackedBuffers{.buffers = std::array{
                                                            _device.create_buffer({
                                                                .size = sizeof(GPUMesh) * MAX_ENTITY_COUNT,
                                                                .name = "_gpu_mesh_manifest",
                                                            }),
                                                        }});
    _gpu_mesh_group_manifest.set_buffers(daxa::TrackedBuffers{.buffers = std::array{
                                                                  _device.create_buffer({
                                                                      .size = sizeof(GPUMeshGroup) * MAX_ENTITY_COUNT,
                                                                      .name = "_gpu_mesh_group_manifest",
                                                                  }),
                                                              }});
}

Scene::~Scene()
{
    _device.destroy_buffer(_gpu_entity_meta.get_state().buffers[0]);
    _device.destroy_buffer(_gpu_entity_transforms.get_state().buffers[0]);
    _device.destroy_buffer(_gpu_entity_combined_transforms.get_state().buffers[0]);
    _device.destroy_buffer(_gpu_entity_mesh_groups.get_state().buffers[0]);
    _device.destroy_buffer(_gpu_mesh_manifest.get_state().buffers[0]);
    _device.destroy_buffer(_gpu_mesh_group_manifest.get_state().buffers[0]);

    for (auto &mesh : _mesh_manifest)
    {
        if (mesh.runtime.has_value())
        {
            _device.destroy_buffer(std::bit_cast<daxa::BufferId>(mesh.runtime.value().mesh_buffer));
        }
    }
}

// EntityId Scene::create_entity()
// {
//     return {this->entity_meta.entity_count++};
// }
//
// auto Scene::get_entity_ref(EntityId ent_id) -> EntityRef
// {
//     return {
//         .transform = &this->entity_transforms[ent_id.index],
//         .first_child = &this->entity_first_children[ent_id.index],
//         .next_silbing = &this->entity_next_siblings[ent_id.index],
//         .parent = &this->entity_parents[ent_id.index],
//         .meshes = &this->entity_meshlists[ent_id.index],
//     };
// }

// void Scene::record_full_entity_update(
//     daxa::Device &device,
//     daxa::CommandRecorder &cmd,
//     Scene &scene,
//     daxa::BufferId b_entity_meta,
//     daxa::BufferId b_entity_transforms,
//     daxa::BufferId b_entity_combined_transforms,
//     daxa::BufferId b_entity_first_children,
//     daxa::BufferId b_entity_next_siblings,
//     daxa::BufferId b_entity_parents,
//     daxa::BufferId b_entity_meshlists)
// {
//     auto upload = [&](auto& src_field, auto& dst_buffer, auto dummy, usize count)
//     {
//         using DATA_T = decltype(dummy);
//         const u32 size = sizeof(DATA_T) * count;
//         auto staging = device.create_buffer({
//             .size = size,
//             .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
//             .name = std::string(device.info_buffer(dst_buffer).value().name.view()) + " staging",
//         });
//         cmd.destroy_buffer_deferred(staging);
//         std::memcpy(reinterpret_cast<DATA_T*>(device.get_host_address(staging).value()), &src_field, size);
//         cmd.copy_buffer_to_buffer({
//             .src_buffer = staging,
//             .dst_buffer = dst_buffer,
//             .size = size,
//         });
//     };
//     upload(this->entity_meta, b_entity_meta, GPUEntityMetaData{}, 1);
//     upload(this->entity_transforms, b_entity_transforms, daxa_f32mat4x4{}, MAX_ENTITY_COUNT);
//     upload(this->entity_combined_transforms, b_entity_combined_transforms, daxa_f32mat4x4{}, MAX_ENTITY_COUNT);
//     upload(this->entity_first_children, b_entity_first_children, EntityId{}, MAX_ENTITY_COUNT);
//     upload(this->entity_next_siblings, b_entity_next_siblings, EntityId{}, MAX_ENTITY_COUNT);
//     upload(this->entity_parents, b_entity_parents, EntityId{}, MAX_ENTITY_COUNT);
//     upload(this->entity_meshlists, b_entity_meshlists, GPUMeshGroup{}, MAX_ENTITY_COUNT);
// }

// template<typename T, size_t N, size_t M>
// auto y_to_z_up(daxa::detail::GenericMatrix<T,N,M> mat) -> daxa::detail::GenericMatrix<T,N,M>
// {
//     return mat * daxa::types::f32mat4x4{{
//         {1,0,0,0},
//         {0,0,1,0},
//         {0,1,0,0},
//         {0,0,0,1},
//     }};
// }

// void Scene::process_transforms()
// {
//     for (u32 entity_index = 0; entity_index < entity_meta.entity_count; ++entity_index)
//     {
//         daxa::types::f32mat4x4 combined_transform = entity_transforms[entity_index];
//         EntityId parent = entity_parents[entity_index];
//         while (parent.index != INVALID_ENTITY_INDEX)
//         {
//             combined_transform = entity_transforms[parent.index] * combined_transform;
//             parent = entity_parents[parent.index];
//         }
//         entity_combined_transforms[entity_index] = combined_transform;
//     }
//     for (u32 entity_index = 0; entity_index < entity_meta.entity_count; ++entity_index)
//     {
//         // daxa and assimg matrices are stored row major, glsl wants them column major...
//         daxa::types::f32mat4x4 transform = entity_transforms[entity_index];
//         transform = transpose(transform);
//         transform = y_to_z_up(transform);
//         entity_transforms[entity_index] = transform;
//         transform = entity_combined_transforms[entity_index];
//         transform = transpose(transform);
//         transform = y_to_z_up(transform);
//         entity_combined_transforms[entity_index] = transform;
//     }
// }
//
// void recursive_print_aiNode(aiScene const *aiscene, aiNode *node, u32 depth, std::string &preamble_string)
// {
//     std::cout << preamble_string << "aiNode::mName: " << node->mName.C_Str() << "\n";
//     std::cout << preamble_string << "{\n";
//     if (node->mParent)
//     {
//         std::cout << preamble_string << "  parent node: " << node->mParent->mName.C_Str() << "\n";
//     }
//     std::cout << preamble_string << "  aiNode::mMeshes:\n";
//     std::cout << preamble_string << "  {\n";
//     for (u32 *mesh = node->mMeshes; mesh < (node->mMeshes + node->mNumMeshes); ++mesh)
//     {
//         std::cout << preamble_string << "    aiMesh::mName: " << aiscene->mMeshes[*mesh]->mName.C_Str() << "\n";
//     }
//     std::cout << preamble_string << "  }\n";
//     for (aiNode **child_node = node->mChildren; child_node < (node->mChildren + node->mNumChildren); ++child_node)
//     {
//         preamble_string += "  ";
//         recursive_print_aiNode(aiscene, *child_node, depth + 1, preamble_string);
//         preamble_string.pop_back();
//         preamble_string.pop_back();
//     }
// }
//
// void process_meshes(aiScene const *aiscene, AssetProcessor &asset_manager)
// {
// }
//
// void process_textures(aiScene const *aiscene, AssetProcessor &asset_manager)
// {
// }

// void SceneLoader::load_entities_from_fbx(Scene &scene, AssetProcessor &asset_manager, std::filesystem::path const &asset_name)
// {
//     std::filesystem::path file_path{asset_root_folder / asset_name};
//
//     Assimp::Importer importer;
//
//     aiScene const *aiscene = importer.ReadFile(file_path.string(), aiProcess_JoinIdenticalVertices | aiProcess_FindInvalidData);
//
//     if (aiscene == nullptr)
//     {
//         std::cerr << "Error: Assimp failed to load scene with message: \"" << importer.GetErrorString() << "\"" << std::endl;
//         return;
//     }
//
//     struct FrontierEntry
//     {
//         EntityId entity_id = {};
//         aiNode *node = {};
//     };
//
//     for (usize mesh_i = 0; mesh_i < aiscene->mNumMeshes; ++mesh_i)
//     {
//         auto dummy = asset_manager.get_or_create_mesh(aiscene, aiscene->mMeshes[mesh_i]);
//     }
//     std::cout << "total meshlet count: " << asset_manager.total_meshlet_count << std::endl;
//
//     EntityId scene_entity_id = scene.create_entity();
//     EntityRef scene_entity = scene.get_entity_ref(scene_entity_id);
//     scene.root_entity = scene_entity_id;
//     auto ident = glm::identity<glm::mat4x4>();
//     *scene_entity.transform = *reinterpret_cast<daxa::types::f32mat4x4 *>(&ident);
//
//     std::vector<FrontierEntry> frontier = {};
//     frontier.reserve(128);
//     frontier.push_back({
//         .entity_id = scene_entity_id,
//         .node = aiscene->mRootNode,
//     });
//     while (!frontier.empty())
//     {
//         FrontierEntry entry = frontier.back();
//         auto current_entity_id = entry.entity_id;
//         aiNode* current_node = entry.node;
//         if (current_entity_id.index == 4)
//         {
//             printf("debug point\n");
//         }
//         frontier.pop_back();
//         const auto current_entity = scene.get_entity_ref(current_entity_id);
//
//         usize n = current_node->mNumMeshes;
//         ASSERT_M(n <= 7, "max submeshes is 7");
//
//         current_entity.meshes->count = current_node->mNumMeshes;
//         for (usize mesh_i = 0; mesh_i < current_node->mNumMeshes; ++mesh_i)
//         {
//             aiMesh* mesh_ptr = aiscene->mMeshes[current_node->mMeshes[mesh_i]];
//             auto fetch = asset_manager.get_or_create_mesh(aiscene, mesh_ptr);
//             current_entity.meshes->mesh_ids[mesh_i] = fetch.first;
//         }
//         std::cout << "Node has " << current_node->mNumMeshes << "meshes" << std::endl;
//
//         *current_entity.transform = *reinterpret_cast<daxa::types::f32mat4x4 *>(&current_node->mTransformation);
//
//         for (usize child_i = 0; child_i < current_node->mNumChildren; ++child_i)
//         {
//             EntityId new_child_id = scene.create_entity();
//             auto const new_child = scene.get_entity_ref(new_child_id);
//             *new_child.parent = current_entity_id;
//
//             *new_child.next_silbing = *current_entity.first_child;
//             *current_entity.first_child = new_child_id;
//
//             frontier.push_back(FrontierEntry{.entity_id = new_child_id, .node = current_node->mChildren[child_i]});
//         }
//     }
//
//     std::cout << std::flush;
// }

// TODO: Loading god function.
auto Scene::load_manifest_from_gltf(std::filesystem::path const &root_path, std::filesystem::path const &glb_name) -> std::variant<RenderEntityId, LoadManifestErrorCode>
{
#pragma region SETUP
    auto file_path = root_path / glb_name;

    fastgltf::Parser parser{};

    constexpr auto gltf_options =
        fastgltf::Options::DontRequireValidAssetMember |
        fastgltf::Options::AllowDouble;
    // fastgltf::Options::LoadGLBBuffers |
    // fastgltf::Options::LoadExternalBuffers |
    // fastgltf::Options::LoadExternalImages

    fastgltf::GltfDataBuffer data;
    bool const worked = data.loadFromFile(file_path);
    if (!worked)
    {
        return LoadManifestErrorCode::FILE_NOT_FOUND;
    }
    auto type = fastgltf::determineGltfFileType(&data);
    std::unique_ptr<fastgltf::glTF> gltf_file;
    switch (type)
    {
    case fastgltf::GltfType::glTF:
        gltf_file = parser.loadGLTF(&data, file_path.parent_path(), gltf_options);
        break;
    case fastgltf::GltfType::GLB:
        gltf_file = parser.loadBinaryGLTF(&data, file_path.parent_path(), gltf_options);
        break;
    default:
        return LoadManifestErrorCode::INVALID_GLTF_FILE_TYPE;
    }

    if (!gltf_file)
    {
        return LoadManifestErrorCode::COULD_NOT_LOAD_ASSET;
    }

    auto parse_result = gltf_file->parse(fastgltf::Category::All);
    if (!(parse_result == fastgltf::Error::None))
    {
        return LoadManifestErrorCode::COULD_NOT_PARSE_ASSET_NODES;
    }
    auto asset = gltf_file->getParsedAsset();

    u32 const scene_file_manifest_index = s_cast<u32>(_scene_file_manifest.size());
    u32 const texture_manifest_offset = s_cast<u32>(_material_texture_manifest.size());
    u32 const material_manifest_offset = s_cast<u32>(_material_manifest.size());
    u32 const mesh_group_manifest_offset = s_cast<u32>(_mesh_group_manifest.size());
    u32 const mesh_manifest_offset = s_cast<u32>(_mesh_manifest.size());
#pragma endregion

#pragma region POPULATE_TEXTURE_MANIFEST
    /// NOTE: Texture = image + sampler - since we don't care about the samplers we only load the images.
    //        Later when we load in the materials which reference the textures rather than images we just
    //        translate the textures image index and store that in the material
    for (u32 i = 0; i < s_cast<u32>(asset->images.size()); ++i)
    {
        u32 const texture_manifest_index = s_cast<u32>(_material_texture_manifest.size());
        _material_texture_manifest.push_back(TextureManifestEntry{
            .name = asset->images[i].name,
            .scene_file_manifest_index = scene_file_manifest_index,
            .in_scene_file_index = i,
            .runtime = {}, // Loaded later.
        });
        fmt::println("[INFO] Loading texture meta data into manifest:\n  name: {}\n  in scene file index: {}\n  manifest index:  {}",
                     asset->images[i].name,
                     i,
                     texture_manifest_index);
    }
#pragma endregion

#pragma region POPULATE_MATERIAL_MANIFEST
    for (u32 material_index = 0; material_index < s_cast<u32>(asset->materials.size()); material_index++)
    {
        /// NOTE: Because we previously only loaded the images and not textures we now need to translate
        //        the texture indices into image indeces and store that
        auto gltf_texture_to_manifest_texture_index = [&](u32 const texture_index) -> std::optional<u32>
        {
            const bool gltf_texture_has_image_index = asset->textures.at(texture_index).imageIndex.has_value();
            if (!gltf_texture_has_image_index)
            {
                return std::nullopt;
            }
            else
            {
                return s_cast<u32>(asset->textures.at(texture_index).imageIndex.value()) + texture_manifest_offset;
            }
        };

        auto const &material = asset->materials.at(material_index);
        /// NOTE: This will not work once we add multiple threads since some other thread might push to the vector
        //        while we are marking the textures as being used by this material
        const u32 material_manifest_index = _material_manifest.size();
        const bool has_pbr_info = material.pbrData.has_value();
        const bool has_normal_texture = material.normalTexture.has_value();
        const bool has_diffuse_texture = has_pbr_info ? material.pbrData.value().baseColorTexture.has_value() : false;
        std::optional<u32> diffuse_texture_index = {};
        std::optional<u32> normal_texture_index = {};
        if (has_diffuse_texture)
        {
            const u32 texture_index = s_cast<u32>(material.pbrData.value().baseColorTexture.value().textureIndex);
            diffuse_texture_index = gltf_texture_to_manifest_texture_index(texture_index);
            _material_texture_manifest.at(diffuse_texture_index.value()).material_manifest_indices.push_back(material_manifest_index);
        }
        if (has_normal_texture)
        {
            const u32 texture_index = s_cast<u32>(material.pbrData.value().baseColorTexture.value().textureIndex);
            normal_texture_index = gltf_texture_to_manifest_texture_index(texture_index);
            _material_texture_manifest.at(normal_texture_index.value()).material_manifest_indices.push_back(material_manifest_index);
        }
        _material_manifest.push_back({.diffuse_tex_index = diffuse_texture_index,
                                      .normal_tex_index = normal_texture_index,
                                      .name = material.name,
                                      .scene_file_manifest_index = scene_file_manifest_index,
                                      .in_scene_file_index = material_index});
    }
#pragma endregion

#pragma region POPULATE_MESHGROUP_AND_MESH_MANIFEST
    /// NOTE: fastgltf::Mesh is a MeshGroup
    std::array<u32, MAX_MESHES_PER_MESHGROUP> mesh_manifest_indices;
    for (u32 mesh_group_index = 0; mesh_group_index < s_cast<u32>(asset->meshes.size()); mesh_group_index++)
    {
        auto const &mesh_group = asset->meshes.at(mesh_group_index);
        u32 const mesh_group_manifest_index = s_cast<u32>(_mesh_group_manifest.size());
        /// NOTE: fastgltf::Primitive is Mesh
        for (u32 mesh_index = 0; mesh_index < s_cast<u32>(mesh_group.primitives.size()); mesh_index++)
        {
            u32 const mesh_manifest_entry = _mesh_manifest.size();
            auto const &mesh = mesh_group.primitives.at(mesh_index);
            mesh_manifest_indices.at(mesh_index) = mesh_manifest_entry;
            std::optional<u32> material_manifest_index = mesh.materialIndex.has_value() ? std::optional{s_cast<u32>(mesh.materialIndex.value()) + material_manifest_offset} : std::nullopt;
            _mesh_manifest.push_back(MeshManifestEntry{
                .material_manifest_index = std::move(material_manifest_index),
                .scene_file_manifest_index = scene_file_manifest_index,
                .scene_file_mesh_index = mesh_group_index,
                .scene_file_primitive_index = mesh_index,
            });
            _new_mesh_manifest_entries += 1;
        }

        _mesh_group_manifest.push_back(MeshGroupManifestEntry{
            .mesh_manifest_indices = std::move(mesh_manifest_indices),
            .mesh_count = s_cast<u32>(mesh_group.primitives.size()),
            .name = mesh_group.name,
            .scene_file_manifest_index = scene_file_manifest_index,
            .in_scene_file_index = mesh_group_index});
        _new_mesh_group_manifest_entries += 1;
        mesh_manifest_indices.fill(0u);
    }
#pragma endregion

#pragma region POPULATE_RENDER_ENTITIES
    /// NOTE: fastgltf::Node is Entity
    ASSERT_M(asset->nodes.size() != 0, "[ERROR][load_manifest_from_gltf()] Empty node array - what to do now?");
    std::vector<RenderEntityId> node_index_to_entity_id = {};
    /// NOTE: Here we allocate space for each entity and create a translation table between node index and entity id
    for (u32 node_index = 0; node_index < s_cast<u32>(asset->nodes.size()); node_index++)
    {
        node_index_to_entity_id.push_back(_render_entities.create_slot());
        _dirty_render_entities.push_back(node_index_to_entity_id.back());
    }
    for (u32 node_index = 0; node_index < s_cast<u32>(asset->nodes.size()); node_index++)
    {
        // TODO: For now store transform as a matrix - later should be changed to something else (TRS: translation, rotor, scale).
        auto fastgltf_to_glm_mat4x3_transform = [](std::variant<fastgltf::Node::TRS, fastgltf::Node::TransformMatrix> const &trans) -> glm::mat4x3
        {
            glm::mat4x3 ret_trans;
            if (auto const *trs = std::get_if<fastgltf::Node::TRS>(&trans))
            {
                if (trs->translation[0] != 0.0f)
                {
                //     printf("penis\n");
                }
                auto const scaled = glm::scale(glm::identity<glm::mat4x4>(), glm::vec3(trs->scale[0], trs->scale[1], trs->scale[2]));
                auto const quat_rotation_mat = glm::toMat4(glm::quat(trs->rotation[3], trs->rotation[0], trs->rotation[1], trs->rotation[2])) * scaled;
                auto const rotated_scaled = quat_rotation_mat * scaled;
                auto const translated_rotated_scaled = glm::translate(
                    glm::identity<glm::mat4x4>(),
                    glm::vec3(trs->translation[0], trs->translation[1], trs->translation[2])) * rotated_scaled;
                /// NOTE: As the last row is always (0,0,0,1) we dont store it.
                ret_trans = glm::mat4x3(translated_rotated_scaled);
            }
            else if (auto const *trs = std::get_if<fastgltf::Node::TransformMatrix>(&trans))
            {
                // Gltf and glm matrices are column major.
                ret_trans = glm::mat4x3(std::bit_cast<glm::mat4x4>(*trs));
            }
            return ret_trans;
        };

        fastgltf::Node const &node = asset->nodes[node_index];
        RenderEntityId const parent_r_ent_id = node_index_to_entity_id[node_index];
        RenderEntity &r_ent = *_render_entities.slot(parent_r_ent_id);
        r_ent.mesh_group_manifest_index = node.meshIndex.has_value() ? std::optional<u32>(s_cast<u32>(node.meshIndex.value()) + mesh_group_manifest_offset) : std::optional<u32>(std::nullopt);
        r_ent.transform = fastgltf_to_glm_mat4x3_transform(node.transform);
        if (!node.children.empty())
        {
            r_ent.first_child = node_index_to_entity_id[node.children[0]];
        }
        for (u32 curr_child_vec_idx = 0; curr_child_vec_idx < node.children.size(); curr_child_vec_idx++)
        {
            u32 const curr_child_node_idx = node.children[curr_child_vec_idx];
            RenderEntityId const curr_child_r_ent_id = node_index_to_entity_id[curr_child_node_idx];
            RenderEntity &curr_child_r_ent = *_render_entities.slot(curr_child_r_ent_id);
            curr_child_r_ent.parent = parent_r_ent_id;
            bool const has_next_sibling = curr_child_vec_idx < (node.children.size() - 1ull);
            if (has_next_sibling)
            {
                RenderEntityId const next_r_ent_child_id = node_index_to_entity_id[node.children[curr_child_vec_idx + 1]];
                curr_child_r_ent.next_sibling = next_r_ent_child_id;
            }
        }
    }

    /// NOTE: Find all root render entities (aka render entities that have no parent) and store them as
    //        Child root entites under scene root node
    RenderEntityId root_r_ent_id = _render_entities.create_slot({.transform = glm::mat4x3(glm::identity<glm::mat4x3>()),
                                                                 .first_child = std::nullopt,
                                                                 .next_sibling = std::nullopt,
                                                                 .parent = std::nullopt,
                                                                 .mesh_group_manifest_index = std::nullopt,
                                                                 .name = glb_name.string() + " scene root node"});
    _dirty_render_entities.push_back(root_r_ent_id);
    RenderEntity &root_r_ent = *_render_entities.slot(root_r_ent_id);
    std::optional<RenderEntityId> root_r_ent_prev_child = {};
    for (u32 node_index = 0; node_index < s_cast<u32>(asset->nodes.size()); node_index++)
    {
        RenderEntityId const r_ent_id = node_index_to_entity_id[node_index];
        RenderEntity &r_ent = *_render_entities.slot(r_ent_id);
        if (!r_ent.parent.has_value())
        {
            r_ent.parent = root_r_ent_id;
            if (!root_r_ent_prev_child.has_value()) // First child
            {
                root_r_ent.first_child = r_ent_id;
            }
            else // We have other root children already
            {
                _render_entities.slot(root_r_ent_prev_child.value())->next_sibling = r_ent_id;
            }
            root_r_ent_prev_child = r_ent_id;
        }
    }

#pragma endregion

    _scene_file_manifest.push_back(SceneFileManifestEntry{
        .path = file_path,
        .gltf_info = std::move(gltf_file),
        .gltf_asset = std::move(asset),
        .texture_manifest_offset = texture_manifest_offset,
        .material_manifest_offset = material_manifest_offset,
        .mesh_group_manifest_offset = mesh_group_manifest_offset,
        .mesh_manifest_offset = mesh_manifest_offset,
    });
    return root_r_ent_id;
}

auto Scene::record_gpu_manifest_update() -> daxa::ExecutableCommandList
{
    auto recorder = _device.create_command_recorder({});
    /// TODO: Make buffers resize.

    // Calculate required staging buffer size:
    usize required_staging_size = 0;
    required_staging_size += sizeof(GPUEntityMetaData);                              // _gpu_entity_meta
    required_staging_size += sizeof(daxa_f32mat4x3) * _dirty_render_entities.size(); // _gpu_entity_transforms
    required_staging_size += sizeof(daxa_f32mat4x3) * _dirty_render_entities.size(); // _gpu_entity_combined_transforms
    required_staging_size += sizeof(GPUMeshGroup) * _dirty_render_entities.size();   // _gpu_entity_mesh_groups
    daxa::BufferId staging_buffer = _device.create_buffer({
        .size = required_staging_size,
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = "entities update staging",
    });
    recorder.destroy_buffer_deferred(staging_buffer);
    usize staging_offset = 0;
    std::byte *host_ptr = _device.get_host_address(staging_buffer).value();
    *r_cast<GPUEntityMetaData *>(host_ptr) = {
        .entity_count = s_cast<u32>(_render_entities.size()),
    };
    recorder.copy_buffer_to_buffer({
        .src_buffer = staging_buffer,
        .dst_buffer = _gpu_entity_meta.get_state().buffers[0],
        .src_offset = staging_offset,
        .size = sizeof(GPUEntityMetaData),
    });
    staging_offset += sizeof(GPUEntityMetaData);

    /**
     * TODO:
     * - replace with compute shader
     * - write two arrays, one containing entity ids other containing update data
     * - write compute shader that reads both arrays, they then write the updates from staging to entity arrays
     */
    /// NOTE: Update dirty entities.
    for (u32 i = 0; i < _dirty_render_entities.size(); ++i)
    {
        usize offset = (staging_offset + (sizeof(glm::mat4x3) * 2 + sizeof(u32)) * i);
        u32 entity_index = _dirty_render_entities[i].index;
        glm::mat4 transform4 = glm::mat4(
                glm::vec4(_render_entities.slot(_dirty_render_entities[i])->transform[0], 0.0f),
                glm::vec4(_render_entities.slot(_dirty_render_entities[i])->transform[1], 0.0f),
                glm::vec4(_render_entities.slot(_dirty_render_entities[i])->transform[2], 0.0f),
                glm::vec4(_render_entities.slot(_dirty_render_entities[i])->transform[3], 1.0f));
        glm::mat4 combined_transform4 = transform4;
        std::optional<RenderEntityId> parent = _render_entities.slot(_dirty_render_entities[i])->parent;
        while (parent.has_value())
        {
            glm::mat4x3 parent_transform4 = glm::mat4(
                glm::vec4(_render_entities.slot(parent.value())->transform[0], 0.0f),
                glm::vec4(_render_entities.slot(parent.value())->transform[1], 0.0f),
                glm::vec4(_render_entities.slot(parent.value())->transform[2], 0.0f),
                glm::vec4(_render_entities.slot(parent.value())->transform[3], 1.0f));
            combined_transform4 = parent_transform4 * combined_transform4;
            parent = _render_entities.slot(parent.value())->parent;
        }
        u32 mesh_group_manifest_index = _render_entities.slot(_dirty_render_entities[i])->mesh_group_manifest_index.value_or(INVALID_MANIFEST_INDEX);
        struct RenderEntityUpdateStagingMemoryView
        {
            glm::mat4x3 transform;
            glm::mat4x3 combined_transform;
            u32 mesh_group_manifest_index;
        };
        *r_cast<RenderEntityUpdateStagingMemoryView *>(host_ptr + offset) = {
            .transform = transform4,
            .combined_transform = combined_transform4,
            .mesh_group_manifest_index = mesh_group_manifest_index,
        };
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _gpu_entity_transforms.get_state().buffers[0],
            .src_offset = offset + offsetof(RenderEntityUpdateStagingMemoryView, transform),
            .dst_offset = sizeof(glm::mat4x3) * entity_index,
            .size = sizeof(glm::mat4x3),
        });
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _gpu_entity_combined_transforms.get_state().buffers[0],
            .src_offset = offset + offsetof(RenderEntityUpdateStagingMemoryView, combined_transform),
            .dst_offset = sizeof(glm::mat4x3) * entity_index,
            .size = sizeof(glm::mat4x3),
        });
        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_buffer,
            .dst_buffer = _gpu_entity_mesh_groups.get_state().buffers[0],
            .src_offset = offset + offsetof(RenderEntityUpdateStagingMemoryView, mesh_group_manifest_index),
            .dst_offset = sizeof(u32) * entity_index,
            .size = sizeof(u32),
        });
    }

    if (_new_mesh_group_manifest_entries > 0)
    {
        u32 const mesh_group_staging_buffer_size = sizeof(GPUMeshGroup) * _new_mesh_group_manifest_entries;
        daxa::BufferId mesh_group_staging_buffer = _device.create_buffer({.size = mesh_group_staging_buffer_size,
                                                                          .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
                                                                          .name = "mesh group update staging buffer"});
        recorder.destroy_buffer_deferred(mesh_group_staging_buffer);
        GPUMeshGroup *staging_ptr = _device.get_host_address_as<GPUMeshGroup>(mesh_group_staging_buffer).value();
        u32 const mesh_group_manifest_offset = _mesh_group_manifest.size() - _new_mesh_group_manifest_entries;
        for (u32 new_mesh_group_idx = 0; new_mesh_group_idx < _new_mesh_group_manifest_entries; new_mesh_group_idx++)
        {
            u32 const mesh_group_manifest_idx = mesh_group_manifest_offset + new_mesh_group_idx;
            staging_ptr[new_mesh_group_idx].count = _mesh_group_manifest.at(mesh_group_manifest_idx).mesh_count;
            std::memcpy(
                &staging_ptr[new_mesh_group_idx].mesh_manifest_indices,
                _mesh_group_manifest.at(mesh_group_manifest_idx).mesh_manifest_indices.data(),
                sizeof(_mesh_group_manifest.at(mesh_group_manifest_idx).mesh_manifest_indices));
        }
        recorder.copy_buffer_to_buffer({
            .src_buffer = mesh_group_staging_buffer,
            .dst_buffer = _gpu_mesh_group_manifest.get_state().buffers[0],
            .src_offset = 0,
            .dst_offset = mesh_group_manifest_offset * sizeof(GPUMeshGroup),
            .size = sizeof(GPUMeshGroup) * _new_mesh_group_manifest_entries,
        });
    }
    if (_new_mesh_manifest_entries > 0)
    {
        u32 const mesh_update_staging_buffer_size = sizeof(GPUMesh) * _new_mesh_manifest_entries;
        u32 const mesh_manifest_offset = _mesh_manifest.size() - _new_mesh_manifest_entries;
        daxa::BufferId mesh_staging_buffer = _device.create_buffer({.size = mesh_update_staging_buffer_size,
                                                                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
                                                                    .name = "mesh update staging buffer"});
        recorder.destroy_buffer_deferred(mesh_staging_buffer);
        GPUMesh *staging_ptr = _device.get_host_address_as<GPUMesh>(mesh_staging_buffer).value();
        std::vector<GPUMesh> tmp_meshes(_new_mesh_manifest_entries);
        std::memcpy(staging_ptr, tmp_meshes.data(), _new_mesh_manifest_entries);

        recorder.copy_buffer_to_buffer({
            .src_buffer = mesh_staging_buffer,
            .dst_buffer = _gpu_mesh_manifest.get_state().buffers[0],
            .src_offset = 0,
            .dst_offset = mesh_manifest_offset * sizeof(GPUMesh),
            .size = sizeof(GPUMesh) * _new_mesh_manifest_entries,
        });
    }
    /// TODO: Taskgraph this shit.
    recorder.pipeline_barrier({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::READ_WRITE,
    });

    _new_mesh_manifest_entries = 0;
    _new_mesh_group_manifest_entries = 0;
    return recorder.complete_current_commands();
}