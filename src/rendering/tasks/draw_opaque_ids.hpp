#pragma once

#include "../gpu_context.hpp"
#include "draw_opaque_ids.inl"

inline static constexpr std::string_view DRAW_OPAQUE_IDS_PIPELINE_NAME = "draw opaque ids";

inline static const daxa::RasterPipelineCompileInfo DRAW_OPAQUE_IDS_PIPELINE_INFO{
    .vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"draw_opaque_ids.inl"},
    },
    .fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"draw_opaque_ids.inl"},
    },
    .color_attachments = {
        daxa::RenderAttachment{
            .format = daxa::Format::R32_UINT,
        },
    },
    .depth_test = {
        .depth_attachment_format = daxa::Format::D32_SFLOAT,
        .enable_depth_test = true,
        .enable_depth_write = true,
        .depth_test_compare_op = daxa::CompareOp::GREATER_OR_EQUAL,
        .min_depth_bounds = 1.0f,
        .max_depth_bounds = 0.0f,
    },
    .push_constant_size = sizeof(DrawOpaqueIdsPush),
    .name = std::string{DRAW_OPAQUE_IDS_PIPELINE_NAME},
};

inline void t_draw_opaque_ids(
    daxa::TaskList &task_list,
    GPUContext &context,
    daxa::TaskBufferHandle vertex_id_buffer,
    daxa::TaskBufferHandle instanced_meshlets,
    daxa::TaskBufferHandle meshes,
    daxa::TaskImageHandle opaque_ids,
    daxa::TaskImageHandle p_depth)
{
    //auto depth = p_depth.subslice({.image_aspect = daxa::ImageAspectFlagBits::DEPTH});
    //task_list.add_task({
    //    .uses = {
    //        daxa::BufferIndexRead{ vertex_id_buffer },
    //        daxa::BufferVertexShaderRead{ instanced_meshlets },
    //        daxa::BufferShaderRead{ meshes },
    //        daxa::BufferDrawIndirectInfoRead{ context.draw_opaque_id_info_buffer.t_id },
    //        daxa::ImageColorAttachment{opaque_ids},
    //        daxa::ImageDepthAttachment{depth},
    //    },
    //    .task = [=](daxa::TaskRuntimeInterface const &runtime)
    //    {
    //        daxa::CommandList cmd = runtime.get_command_list();
    //        cmd.begin_renderpass({
    //            .color_attachments = {
    //                daxa::RenderAttachmentInfo{
    //                    .image_view = runtime.get_images(opaque_ids)[0].default_view(),
    //                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
    //                    .load_op = daxa::AttachmentLoadOp::CLEAR,
    //                    .store_op = daxa::AttachmentStoreOp::STORE,
    //                    .clear_value = daxa::ClearValue{std::array<f32, 4>{1.f, 1.f, 1.f, 1.f}},
    //                },
    //            },
    //            .depth_attachment = daxa::RenderAttachmentInfo{
    //                .image_view = runtime.get_images(depth)[0].default_view(),
    //                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
    //                .load_op = daxa::AttachmentLoadOp::CLEAR,
    //                .store_op = daxa::AttachmentStoreOp::STORE,
    //                .clear_value = daxa::ClearValue{daxa::DepthValue{0.0f, 0}},
    //            },
    //            .render_area = daxa::Rect2D{
    //                .width = (runtime.get_device().info_image(runtime.get_images(opaque_ids)[0]).size.x),
    //                .height = (runtime.get_device().info_image(runtime.get_images(opaque_ids)[0]).size.y),
    //            },
    //        });
    //        cmd.set_pipeline(*context.raster_pipelines.at(DRAW_OPAQUE_IDS_PIPELINE_NAME));
    //        cmd.push_constant(DrawOpaqueIdsPush{
    //            .dummy = 64,
    //        });
    //        // cmd.draw_indirect({
    //        //     .draw_command_buffer = context.draw_opaque_id_info_buffer.id,
    //        //     .draw_count = 1,
    //        //     .is_indexed = true,
    //        // });
    //        cmd.end_renderpass();
    //    },
    //    .name = std::string{DRAW_OPAQUE_IDS_PIPELINE_NAME},
    //});
}