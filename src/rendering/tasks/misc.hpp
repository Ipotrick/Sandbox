#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_list.hpp>
#include "../gpu_context.hpp"

struct ClearInstantiatedMeshletsHeaderTask
{
    struct Uses
    {
        daxa::BufferTransferWrite instantiated_meshlets{};
    } uses = {};
    static constexpr inline std::string_view NAME = "ClearInstantiatedMeshletsHeaderTask";
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::BufferId src_buf = {};
        daxa::CommandList cmd = ti.get_command_list();
        auto alloc = ti.get_allocator().allocate(sizeof(INDIRECT_COMMAND_BYTE_SIZE)).value();
        if (context->settings.indexed_id_rendering != 0)
        {
            *reinterpret_cast<DispatchIndirectStruct *>(alloc.host_address) = DispatchIndirectStruct{
                .x = 0,
                .y = 1,
                .z = 1,
            };
        }
        else
        {
            *reinterpret_cast<DrawIndirectStruct *>(alloc.host_address) = DrawIndirectStruct{
                .vertex_count = 0,
                .instance_count = 1,
                .first_vertex = 0,
                .first_instance = 0,
            };
        }
        auto const src = ti.get_allocator().get_buffer();
        auto const dst = uses.instantiated_meshlets.buffer();
        cmd.copy_buffer_to_buffer({
            .src_buffer = src,
            .src_offset = alloc.buffer_offset,
            .dst_buffer = dst,
            .dst_offset = 0,
            .size = sizeof(INDIRECT_COMMAND_BYTE_SIZE),
        });
        cmd.copy_buffer_to_buffer({
            .src_buffer = src,
            .src_offset = alloc.buffer_offset,
            .dst_buffer = dst,
            .dst_offset = sizeof(INDIRECT_COMMAND_BYTE_SIZE),
            .size = sizeof(INDIRECT_COMMAND_BYTE_SIZE),
        });
    }
};