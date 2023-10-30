#include "scene.hpp"

#include <fstream>

#include <fmt/format.h>
#include <glm/gtx/quaternion.hpp>

Scene::Scene()
{
}

Scene::~Scene()
{
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
//     upload(this->entity_meta, b_entity_meta, EntityMetaData{}, 1);
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
// void process_meshes(aiScene const *aiscene, AssetManager &asset_manager)
// {
// }
//
// void process_textures(aiScene const *aiscene, AssetManager &asset_manager)
// {
// }

// void SceneLoader::load_entities_from_fbx(Scene &scene, AssetManager &asset_manager, std::filesystem::path const &asset_name)
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
auto Scene::load_manifest_from_gltf(std::filesystem::path const &root_path, std::filesystem::path const &glb_name) -> LoadManifestResult
{
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
        return LoadManifestResult::ERROR_FILE_NOT_FOUND;
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
        return LoadManifestResult::ERROR_INVALID_GLTF_FILE_TYPE;
    }

    if (!gltf_file)
    {
        return LoadManifestResult::ERROR_COULD_NOT_LOAD_ASSET;
    }

    auto parse_result = gltf_file->parse(fastgltf::Category::All);
    if (!(parse_result == fastgltf::Error::None))
    {
        return LoadManifestResult::ERROR_PARSING_ASSET_NODES;
    }
    auto asset = gltf_file->getParsedAsset();

    u32 const scene_file_manifest_index = s_cast<u32>(_scene_file_manifest.size());
    u32 const texture_manifest_offset = s_cast<u32>(_texture_manifest.size());
    u32 const material_manifest_offset = s_cast<u32>(_material_manifest.size());
    u32 const mesh_group_manifest_offset = s_cast<u32>(_mesh_group_manifest.size());
    u32 const mesh_manifest_offset = s_cast<u32>(_mesh_manifest.size());

    /// NOTE: Texture = image + sampler since we don't care about the samplers we only load the images
    //        Later when we load in the materials which reference the textures rather than images we just
    //        translate the textures image index and store that in the material
    for (u32 i = 0; i < s_cast<u32>(asset->images.size()); ++i)
    {
        u32 const texture_manifest_index = s_cast<u32>(_texture_manifest.size());
        _texture_manifest.push_back(TextureManifestEntry{
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

    for (u32 material_index = 0; material_index < s_cast<u32>(asset->materials.size()); material_index++)
    {
        auto const &material = asset->materials.at(material_index);
        const bool has_pbr_info = material.pbrData.has_value();
        const bool has_normal_texture = material.normalTexture.has_value();
        const bool has_diffuse_texture = has_pbr_info ? material.pbrData.value().baseColorTexture.has_value() : false;
        _material_manifest.push_back({.diffuse_tex_index = has_diffuse_texture ? material.pbrData.value().baseColorTexture.value().textureIndex : std::optional<u32>{},
                                      .normal_tex_index = has_normal_texture ? material.normalTexture.value().textureIndex : std::optional<u32>{},
                                      .name = material.name,
                                      .scene_file_manifest_index = scene_file_manifest_index,
                                      .in_scene_file_index = material_index});
    }

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
            _mesh_manifest.push_back({.material_manifest_index = std::move(material_manifest_index),
                                      .scene_file_manifest_index = scene_file_manifest_index});
        }

        _mesh_group_manifest.push_back(MeshGroupManifestEntry{
            .mesh_manifest_indices = std::move(mesh_manifest_indices),
            .mesh_count = s_cast<u32>(mesh_group.primitives.size()),
            .name = mesh_group.name,
            .scene_file_manifest_index = scene_file_manifest_index,
            .in_scene_file_index = mesh_group_index});
        mesh_manifest_indices.fill(0u);
    }

    /// NOTE: fastgltf::Node is Entity
    ASSERT_M(asset->nodes.size() != 0, "[ERROR][load_manifest_from_gltf()] Empty node array - what to do now?");
    std::vector<RenderEntityId> node_index_to_entity_id = {};
    for (u32 node_index = 0; node_index < s_cast<u32>(asset->nodes.size()); node_index++)
    {
        node_index_to_entity_id.push_back(_render_entities.create_slot());
    }
    // In the first pass find all root nodes (aka nodes that have no parent)
    for (u32 node_index = 0; node_index < s_cast<u32>(asset->nodes.size()); node_index++)
    {
        // TODO: For now store transform as a matrix - later should be changed to something else (TRS: translation, rotor, scale).
        auto fastgltf_to_glm_mat4x3_transform = [](std::variant<fastgltf::Node::TRS, fastgltf::Node::TransformMatrix> const &trans) -> glm::mat4x3
        {
            glm::mat4x3 ret_trans;
            if (auto const *trs = std::get_if<fastgltf::Node::TRS>(&trans))
            {
                auto const scaled = glm::scale(glm::identity<glm::mat4x4>(), glm::vec3(trs->scale.at[0], trs->scale.at[1], trs->scale[2]));
                auto const rotated_scaled = glm::toMat4(glm::qua<float>(trs->rotation.at[0], trs->rotation.at[1], trs->rotation.at[2], trs->rotation.at[3])) * scaled;
                auto const translated_rotated_scaled = glm::translate(
                    rotated_scaled,
                    glm::vec3(trs->scale[0], trs->scale[1], trs->scale[2]));
                ret_trans[0] = translated_rotated_scaled[0];
                ret_trans[1] = translated_rotated_scaled[1];
                ret_trans[2] = translated_rotated_scaled[2];
            }
            else if (auto const *trs = std::get_if<fastgltf::Node::TransformMatrix>(&trans))
            {
                for (u32 col = 0; col < 4; col++)
                {
                    for (u32 row = 0; row < 3; row++)
                    {
                        ret_trans[col][row] = trs[col * 4 + row];
                    }
                }
            }
            return ret_trans;
        };
        fastgltf::Node const &node = asset->nodes[node_index];
        auto const parent_r_ent_idx = node_index_to_entity_id[node_index];
        RenderEntity &r_ent = *_render_entities.slot(parent_r_ent_idx);

        r_ent.mesh_group_manifest_index = node.meshIndex.has_value() ?
            std::optional<u32>(s_cast<u32>(node.meshIndex.value()) + mesh_group_manifest_offset) :
            std::optional<u32>(std::nullopt);

        r_ent.transform = fastgltf_to_glm_mat4x3_transform(node.transform);
        if (node.children.size() > 0)
        {
            auto const first_child_r_ent_index = node_index_to_entity_id[node.children[0]];
            auto * prev_r_ent_child = *_render_entities.slot(first_child_r_ent_index);
            for (u32 curr_child_vec_idx = 0; curr_child_vec_idx < node.children.size(); curr_child_vec_idx++)
            {
                u32 const curr_child_node_idx = node.children[curr_child_vec_idx];
                auto const curr_child_r_ent_idx = node_index_to_entity_id[curr_child_node_idx];
                auto &curr_child_r_ent = *_render_entities.slot(node_index_to_entity_id[curr_child_r_ent_idx]);
                curr_child_r_ent.first_child = first_child_r_ent_index;
                curr_child_r_ent.parent = parent_r_ent_idx;
                prev_r_ent_child->next_sibling = curr_child_r_ent_idx;
                prev_r_ent_child = &curr_child_r_ent;
            }
        }
    }

    _scene_file_manifest.push_back(SceneFileManifestEntry{
        .path = file_path,
        .gltf_info = std::move(gltf_file),
        .texture_manifest_offset = texture_manifest_offset,
        .material_manifest_offset = material_manifest_offset,
        .mesh_group_manifest_offset = mesh_group_manifest_offset,
        .mesh_manifest_offset = mesh_manifest_offset,
    });
    return LoadManifestResult::SUCCESS;
}

auto record_gpu_manifest_update() -> daxa::ExecutableCommandList
{
}