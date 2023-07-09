#pragma once
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../../shaders/visbuffer.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

DAXA_DECL_TASK_USES_BEGIN(DrawOpaqueId, 1)
DAXA_TASK_USE_IMAGE(u_visbuffer, daxa_RWImage2Df32, COLOR_ATTACHMENT)
DAXA_TASK_USE_IMAGE(u_debug_image, daxa_RWImage2Df32, COLOR_ATTACHMENT)
DAXA_TASK_USE_IMAGE(u_depth_image, daxa_RWImage2Df32, DEPTH_ATTACHMENT)
DAXA_TASK_USE_BUFFER(u_draw_info_index_buffer, daxa_BufferPtr(daxa_u32), INDEX_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(InstantiatedMeshlet), VERTEX_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_meshlists, daxa_BufferPtr(MeshList), VERTEX_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_debug, daxa_RWBufferPtr(daxa_u32vec4), VERTEX_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), VERTEX_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_combined_transforms, daxa_BufferPtr(daxa_f32mat4x4), VERTEX_SHADER_READ)
DAXA_DECL_TASK_USES_END()

#if __cplusplus

#include "../gpu_context.hpp"

inline static constexpr std::string_view DRAW_OPAQUE_IDS_PIPELINE_NAME = {DrawOpaqueId::NAME};

inline static const daxa::RasterPipelineCompileInfo DRAW_OPAQUE_IDS_PIPELINE_INFO{
    .vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"./src/rendering/tasks/draw_opaque_ids.glsl"},
    },
    .fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"./src/rendering/tasks/draw_opaque_ids.glsl"},
    },
    .color_attachments = {
        daxa::RenderAttachment{
            .format = daxa::Format::R32_UINT,
        },
        daxa::RenderAttachment{
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .blend = daxa::BlendInfo{
                .blend_enable = true,
                .src_color_blend_factor = daxa::BlendFactor::SRC_ALPHA,
                .dst_color_blend_factor = daxa::BlendFactor::ONE_MINUS_SRC_ALPHA,
                .color_blend_op = daxa::BlendOp::ADD,
                .src_alpha_blend_factor = daxa::BlendFactor::ONE,
                .dst_alpha_blend_factor = daxa::BlendFactor::ONE,
                .alpha_blend_op = daxa::BlendOp::ADD,
            },
        },
    },
    .depth_test = {
        .depth_attachment_format = daxa::Format::D32_SFLOAT,
        .enable_depth_test = true,
        .enable_depth_write = true,
        .depth_test_compare_op = daxa::CompareOp::GREATER_OR_EQUAL,
        .min_depth_bounds = 0.0f,
        .max_depth_bounds = 1.0f,
    },
    .name = std::string{DrawOpaqueId{}.name},
};

struct DrawOpaqueIdTask : DrawOpaqueId
{
    std::shared_ptr<daxa::RasterPipeline> pipeline = {};
    GPUContext *context = {};
    u32 pass = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::CommandList cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        daxa::ImageId visbuffer = uses.u_visbuffer.image();
        daxa::ImageId debug_image = uses.u_debug_image.image();
        daxa::ImageId depth_image = uses.u_depth_image.image();
        bool const clear_attachments = pass == 0;
        cmd.begin_renderpass({
            .color_attachments = {
                daxa::RenderAttachmentInfo{
                    .image_view = visbuffer.default_view(),
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = clear_attachments ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD,
                    .store_op = daxa::AttachmentStoreOp::STORE,
                    .clear_value = daxa::ClearValue{std::array<u32, 4>{INVALID_PIXEL_ID, 0, 0, 0}},
                },
                daxa::RenderAttachmentInfo{
                    .image_view = debug_image.default_view(),
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = clear_attachments ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD,
                    .store_op = daxa::AttachmentStoreOp::STORE,
                    .clear_value = daxa::ClearValue{std::array<f32, 4>{1.f, 1.f, 1.f, 1.f}},
                },
            },
            .depth_attachment = daxa::RenderAttachmentInfo{
                .image_view = depth_image.default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = clear_attachments ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{daxa::DepthValue{0.0f, 0}},
            },
            .render_area = daxa::Rect2D{
                .width = (ti.get_device().info_image(visbuffer).size.x),
                .height = (ti.get_device().info_image(visbuffer).size.y),
            },
        });
        cmd.set_pipeline(*pipeline);

        cmd.set_index_buffer(uses.u_draw_info_index_buffer.buffer(), INDIRECT_COMMAND_BYTE_SIZE * 2);
        auto const indirect_offset = static_cast<usize>(pass == 0 ? 0 : INDIRECT_COMMAND_BYTE_SIZE);
        cmd.draw_indirect({
            .draw_command_buffer = uses.u_draw_info_index_buffer.buffer(),
            .draw_command_buffer_read_offset = indirect_offset,
            .draw_count = 1,
            .draw_command_stride = INDIRECT_COMMAND_BYTE_SIZE,
            .is_indexed = context->settings.indexed_id_rendering != 0,
        });
        cmd.end_renderpass();
    }
};

#endif