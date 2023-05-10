#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_list.inl>

#include "../../../shaders/shared.inl"
#include "../../mesh/mesh.inl"
#include "../../mesh/visbuffer_meshlet_util.inl"

DAXA_INL_TASK_USE_BEGIN(DrawVisbufferBase, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_BUFFER(u_triangle_list, daxa_BufferPtr(TriangleDrawList), DRAW_INDIRECT_INFO_READ)
DAXA_INL_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(InstantiatedMeshlets), SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), SHADER_READ)
DAXA_INL_TASK_USE_IMAGE(u_vis_image, daxa_RWImage2Du32, COLOR_ATTACHMENT)
DAXA_INL_TASK_USE_IMAGE(u_debug_image, daxa_RWImage2Df32, COLOR_ATTACHMENT)
DAXA_INL_TASK_USE_IMAGE(u_depth_image, daxa_RWImage2Df32, DEPTH_ATTACHMENT)
DAXA_INL_TASK_USE_END()

#if __cplusplus
#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const DRAW_VISBUFFER_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/filter_visible_meshlets.glsl";

struct DrawVisbuffer : DrawVisbufferBase
{
    inline static const daxa::RasterPipelineCompileInfo COMPILE_INFO {
        .vertex_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        },
        .fragment_shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
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
            .depth_test_compare_op = daxa::CompareOp::GREATER,
            .min_depth_bounds = 0.0f,
            .max_depth_bounds = 1.0f,
        },
        .name = std::string{DrawVisbufferBase::NAME},
    };
    bool clear_attachments = {};
    GPUContext *context = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::ImageId vis_image = uses.u_vis_image.image();
        daxa::ImageId debug_image = uses.u_debug_image.image();
        daxa::ImageId depth_image = uses.u_depth_image.image();
        auto cmd = ti.get_command_list();
        cmd.begin_renderpass({
            .color_attachments = {
                daxa::RenderAttachmentInfo{
                    .image_view = vis_image.default_view(),
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
                .width = (ti.get_device().info_image(vis_image).size.x),
                .height = (ti.get_device().info_image(vis_image).size.y),
            },
        });
        cmd.set_pipeline(*context->raster_pipelines.at(DrawVisbufferBase::NAME));
        cmd.draw_indirect({
            .draw_command_buffer = uses.u_triangle_list.buffer(),
            .draw_count = 1,
            .draw_command_stride = sizeof(DrawIndirectStruct),
        });
        cmd.end_renderpass();
    }
};

#endif