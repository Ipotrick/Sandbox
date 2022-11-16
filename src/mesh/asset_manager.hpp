#pragma once

#include "../sandbox.hpp"
#include "../gpu_context.hpp"
#include "mesh.inl"

struct AssetManager
{
    daxa::Device device = {};

    daxa::BufferId meshlet_index_buffer = {};
    daxa::BufferId meshlet_vertex_positions_buffer = {};
    daxa::BufferId meshlet_vertex_uvs_buffer = {};
    daxa::BufferId meshlet_vertex_normals_buffer = {};
    daxa::BufferId meshlet_vertex_tangents_buffer = {};

    daxa::TaskBufferId t_meshlet_index_buffer = {};
    daxa::TaskBufferId t_meshlet_vertex_positions_buffer = {};
    daxa::TaskBufferId t_meshlet_vertex_uvs_buffer = {};
    daxa::TaskBufferId t_meshlet_vertex_normals_buffer = {};
    daxa::TaskBufferId t_meshlet_vertex_tangents_buffer = {};

    std::unordered_map<std::string, daxa::ImageId> albedo_textures = {};
    std::unordered_map<std::string, daxa::ImageId> normal_textures = {};
    std::unordered_map<std::string, Mesh> meshes = {};
};