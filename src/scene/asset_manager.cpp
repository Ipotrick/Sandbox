#include "asset_manager.hpp"

static constexpr usize INDEX_BUFFER_SIZE = 50'000'000;
static constexpr usize VERTEX_BUFFER_SIZE = 200'000'000;

AssetManager::AssetManager(daxa::Device device)
    : device{ std::move(device) }
{
}