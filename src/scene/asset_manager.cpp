#include "asset_manager.hpp"

AssetManager::AssetManager(daxa::Device device)
    : device{std::move(device)}
{
    this->meshes_buffer = this->device.create_buffer({
        .size = sizeof(Mesh) * MAX_MESHES,
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
        device.destroy_buffer(mesh.mesh_buffer);
    }
}