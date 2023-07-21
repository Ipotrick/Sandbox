#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shaders/shared.inl"
#include "../../mesh/mesh.inl"
#include "../../mesh/visbuffer_meshlet_util.inl"

#if __cplusplus || defined(DrawVisbufferWriteCommand_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(DrawVisbufferWriteCommand, 1)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(InstantiatedMeshlets), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_triangle_list, daxa_BufferPtr(TriangleList), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshlet_list, daxa_BufferPtr(VisibleMeshletList), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_command, daxa_RWBufferPtr(DrawIndirectStruct), COMPUTE_SHADER_WRITE)
DAXA_DECL_TASK_USES_END()
#endif
#if __cplusplus || !defined(DrawVisbufferWriteCommand_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(DrawVisbuffer, 1)
// When drawing triangles, this draw command has triangle ids appended to the end of the command.
DAXA_TASK_USE_BUFFER(u_draw_command, daxa_BufferPtr(DrawIndirectStruct), DRAW_INDIRECT_INFO_READ)
DAXA_TASK_USE_BUFFER(u_triangle_list, daxa_BufferPtr(TriangleList), VERTEX_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshlet_list, daxa_BufferPtr(VisibleMeshletList), VERTEX_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(InstantiatedMeshlets), VERTEX_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), VERTEX_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_combined_transforms, daxa_BufferPtr(daxa_f32mat4x4), VERTEX_SHADER_READ)
DAXA_TASK_USE_IMAGE(u_vis_image, REGULAR_2D, COLOR_ATTACHMENT)
DAXA_TASK_USE_IMAGE(u_debug_image, REGULAR_2D, COLOR_ATTACHMENT)
DAXA_TASK_USE_IMAGE(u_depth_image, REGULAR_2D, DEPTH_ATTACHMENT)
DAXA_DECL_TASK_USES_END()
#endif

#define DRAW_VISBUFFER_TRIANGLES 1
#define DRAW_VISBUFFER_MESHLETS_DIRECTLY 0
#define DRAW_VISBUFFER_MESHLETS_INDIRECT 2

#define DRAW_FIRST_PASS 0
#define DRAW_SECOND_PASS 1

#define DRAW_VISBUFFER_DEPTH_ONLY 1
#define DRAW_VISBUFFER_NO_DEPTH_ONLY 0

struct DrawVisbufferPush
{
    daxa_u32 pass;
    daxa_u32 mode;
};

#if __cplusplus
#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const DRAW_VISBUFFER_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/draw_visbuffer.glsl";

using DrawVisbufferWriteCommandTask = WriteIndirectDispatchArgsPushBaseTask<
    DrawVisbufferWriteCommand,
    DRAW_VISBUFFER_SHADER_PATH,
    DrawVisbufferPush>;

inline static const daxa::RasterPipelineCompileInfo PIPELINE_COMPILE_INFO_DrawVisbufferTask_DEPTH_ONLY{
    .vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"DEPTH_ONLY", "1"}}},
    },
    .fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"DEPTH_ONLY", "1"}}},
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
    .name = std::string{"DrawVisbufferDepthOnly"},
};

inline static const daxa::RasterPipelineCompileInfo PIPELINE_COMPILE_INFO_DrawVisbufferTask = []()
{
    auto ret = PIPELINE_COMPILE_INFO_DrawVisbufferTask_DEPTH_ONLY;
    ret.color_attachments = {
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
    };
    ret.fragment_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH}};
    ret.vertex_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH}};
    ret.name = "DrawVisbuffer";
    return ret;
}();

struct DrawVisbufferTask
{
    DAXA_USE_TASK_HEADER(DrawVisbuffer)
    inline static const daxa::RasterPipelineCompileInfo PIPELINE_COMPILE_INFO[2] = {
        PIPELINE_COMPILE_INFO_DrawVisbufferTask,
        PIPELINE_COMPILE_INFO_DrawVisbufferTask_DEPTH_ONLY};
    GPUContext *context = {};
    u32 pass = {};
    u32 mode = {};
    bool depth_only = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::ImageId vis_image = uses.u_vis_image.image();
        daxa::ImageId debug_image = uses.u_debug_image.image();
        daxa::ImageId depth_image = uses.u_depth_image.image();
        auto cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        daxa::RenderPassBeginInfo render_pass_begin_info{
            .depth_attachment = daxa::RenderAttachmentInfo{
                .image_view = depth_image.default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = pass == DRAW_FIRST_PASS ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{daxa::DepthValue{0.0f, 0}},
            },
            .render_area = daxa::Rect2D{
                .width = (ti.get_device().info_image(depth_image).size.x),
                .height = (ti.get_device().info_image(depth_image).size.y),
            },
        };
        if (!depth_only)
        {
            render_pass_begin_info.color_attachments = {
                daxa::RenderAttachmentInfo{
                    .image_view = vis_image.default_view(),
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = pass == DRAW_FIRST_PASS ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD,
                    .store_op = daxa::AttachmentStoreOp::STORE,
                    .clear_value = daxa::ClearValue{std::array<u32, 4>{INVALID_PIXEL_ID, 0, 0, 0}},
                },
                daxa::RenderAttachmentInfo{
                    .image_view = debug_image.default_view(),
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = pass == DRAW_FIRST_PASS ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD,
                    .store_op = daxa::AttachmentStoreOp::STORE,
                    .clear_value = daxa::ClearValue{std::array<f32, 4>{1.f, 1.f, 1.f, 1.f}},
                },
            };
        }
        cmd.begin_renderpass(render_pass_begin_info);
        cmd.set_pipeline(*context->raster_pipelines.at(depth_only ? PIPELINE_COMPILE_INFO[1].name : PIPELINE_COMPILE_INFO[0].name));
        cmd.push_constant(DrawVisbufferPush{
            .pass = pass,
            .mode = mode,
        });
        cmd.draw_indirect({
            .draw_command_buffer = uses.u_draw_command.buffer(),
            .draw_count = 1,
            .draw_command_stride = sizeof(DrawIndirectStruct),
        });
        cmd.end_renderpass();
    }
};

inline void task_draw_visbuffer(GPUContext *context, daxa::TaskGraph &task_graph, DrawVisbuffer::Uses uses, const u32 pass, const u32 mode, const bool depth_only)
{
    auto command_buffer = task_graph.create_transient_buffer({
        .size = static_cast<u32>(sizeof(DrawIndirectStruct)),
        .name = std::string("draw visbuffer command buffer") + context->dummy_string(),
    });
    uses.u_draw_command.handle = command_buffer;
    switch (mode)
    {
        case DRAW_VISBUFFER_MESHLETS_DIRECTLY:
        {
            uses.u_triangle_list.handle = task_graph.create_transient_buffer({.size = 4, .name = context->dummy_string()});
            uses.u_meshlet_list.handle = task_graph.create_transient_buffer({.size = 4, .name = context->dummy_string()});
            break;
        }
        case DRAW_VISBUFFER_MESHLETS_INDIRECT:
        {
            uses.u_triangle_list.handle = task_graph.create_transient_buffer({.size = 4, .name = context->dummy_string()});
            break;
        }
        case DRAW_VISBUFFER_TRIANGLES:
        {
            uses.u_meshlet_list.handle = task_graph.create_transient_buffer({.size = 4, .name = context->dummy_string()});
            break;
        }
        default: break;
    }
    task_graph.add_task(DrawVisbufferWriteCommandTask{
        .uses = {
            .u_instantiated_meshlets = uses.u_instantiated_meshlets.handle,
            .u_triangle_list = uses.u_triangle_list.handle,
            .u_meshlet_list = uses.u_meshlet_list.handle,
            .u_command = uses.u_draw_command.handle,
        },
        .context = context,
        .push = DrawVisbufferPush{.pass = pass, .mode = mode },
    });
    if (depth_only)
    {
        uses.u_debug_image.handle = task_graph.create_transient_image({.size = {1, 1, 1}, .name = context->dummy_string()});
        uses.u_vis_image.handle = task_graph.create_transient_image({.size = {1, 1, 1}, .name = context->dummy_string()});
    }
    task_graph.add_task(DrawVisbufferTask{
        .uses = uses,
        .context = context,
        .pass = pass,
        .mode = mode,
        .depth_only = depth_only,
    });
}

#endif