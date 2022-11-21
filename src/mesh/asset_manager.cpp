#include "asset_manager.hpp"

static constexpr usize INDEX_BUFFER_SIZE = 50'000'000;
static constexpr usize VERTEX_BUFFER_SIZE = 200'000'000;

AssetManager::AssetManager(daxa::Device device)
{
    this->meshlet_index_buffer = device.create_buffer({
        .memory_flags = daxa::MemoryFlagBits::DEDICATED_MEMORY,
        .size = INDEX_BUFFER_SIZE,
        .debug_name = "Index Buffer",
    });

    this->meshlet_vertex_positions_buffer = device.create_buffer({
        .memory_flags = daxa::MemoryFlagBits::DEDICATED_MEMORY,
        .size = VERTEX_BUFFER_SIZE,
        .debug_name = "Index Buffer",
    });
}