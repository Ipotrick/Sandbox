#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shaders/shared.inl"
#include "../../mesh/mesh.inl"
#include "../../mesh/visbuffer_meshlet_util.inl"

#if __cplusplus || defined(DrawVisbufferWriteCommand_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(DrawVisbufferWriteCommand, 1)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(InstantiatedMeshlets), SHADER_READ)
DAXA_TASK_USE_BUFFER(u_command, daxa_RWBufferPtr(DrawIndirectStruct), COMPUTE_SHADER_WRITE)
DAXA_DECL_TASK_USES_END()
#endif
#if __cplusplus || !defined(DrawVisbufferWriteCommand_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(DrawVisbuffer, 1)
// When drawing triangles, this draw command has triangle ids appended to the end of the command.
DAXA_TASK_USE_BUFFER(u_draw_command, daxa_BufferPtr(DrawIndirectStruct), DRAW_INDIRECT_INFO_READ)
DAXA_TASK_USE_BUFFER(u_triangle_list, daxa_BufferPtr(TriangleList), DRAW_INDIRECT_INFO_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(InstantiatedMeshlets), SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), SHADER_READ)
DAXA_TASK_USE_IMAGE(u_vis_image, REGULAR_2D, COLOR_ATTACHMENT)
DAXA_TASK_USE_IMAGE(u_debug_image, REGULAR_2D, COLOR_ATTACHMENT)
DAXA_TASK_USE_IMAGE(u_depth_image, REGULAR_2D, DEPTH_ATTACHMENT)
DAXA_DECL_TASK_USES_END()
#endif

#define DRAW_VISBUFFER_TRIANGLES 1
#define DRAW_VISBUFFER_MESHLETS 0

#define DRAW_VISBUFFER_CLEAR_ATTACHMENTS 1
#define DRAW_VISBUFFER_DONT_CLEAR_ATTACHMENTS 0

struct DrawVisbufferPush
{
    daxa_u32 tris_or_meshlets;
};

#if __cplusplus
#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const DRAW_VISBUFFER_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/draw_visbuffer.glsl";

using DrawVisbufferWriteCommandTask = WriteIndirectDispatchArgsBaseTask<
    DrawVisbufferWriteCommand,
    DRAW_VISBUFFER_SHADER_PATH>;

struct DrawVisbufferTask : DrawVisbuffer
{
    inline static const daxa::RasterPipelineCompileInfo PIPELINE_COMPILE_INFO {
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
        .push_constant_size = sizeof(DrawVisbufferPush),
        .name = std::string{DrawVisbuffer::NAME},
    };
    GPUContext *context = {};
    bool clear_attachments = {};
    bool tris_or_meshlets = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::ImageId vis_image = uses.u_vis_image.image();
        daxa::ImageId debug_image = uses.u_debug_image.image();
        daxa::ImageId depth_image = uses.u_depth_image.image();
        auto cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
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
        cmd.set_pipeline(*context->raster_pipelines.at(DrawVisbuffer::NAME));
        cmd.push_constant(DrawVisbufferPush{
            .tris_or_meshlets = (tris_or_meshlets ? 1u : 0u),
        });
        cmd.draw_indirect({
            .draw_command_buffer = uses.u_draw_command.buffer(),
            .draw_count = 1,
            .draw_command_stride = sizeof(DrawIndirectStruct),
        });
        cmd.end_renderpass();
    }
};

inline void task_draw_visbuffer(GPUContext * context, daxa::TaskGraph & task_graph, DrawVisbuffer::Uses uses, const bool clear_attachments, const bool draw_triangles)
{
    if (!draw_triangles)
    {
        auto command_buffer = task_graph.create_transient_buffer({
            .size = static_cast<u32>(sizeof(DrawIndirectStruct)),
            .name = "draw visbuffer command buffer",
        });
        uses.u_draw_command.handle = command_buffer;
        task_graph.add_task(DrawVisbufferWriteCommandTask{
            {.uses = {
                .u_instantiated_meshlets = uses.u_instantiated_meshlets,
                .u_command = uses.u_draw_command.handle,
            }},
            context,
        });
        uses.u_triangle_list.handle = task_graph.create_transient_buffer({
            .size = 4,
            .name = "dummy",
        });
    }
    task_graph.add_task(DrawVisbufferTask{
        {.uses = uses},
        context,
        clear_attachments,
        draw_triangles,
    });
}

#endif