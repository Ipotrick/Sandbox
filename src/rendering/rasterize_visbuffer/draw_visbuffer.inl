#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/mesh.inl"
#include "../../../shader_shared/visbuffer.inl"

#if __cplusplus || defined(DrawVisbufferWriteCommand_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(DrawVisbufferWriteCommand, 1)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(InstantiatedMeshlets), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_command, daxa_u64, COMPUTE_SHADER_WRITE)
DAXA_DECL_TASK_USES_END()
#endif
#if __cplusplus || defined(NO_MESH_SHADER) || defined(MESH_SHADER)
DAXA_DECL_TASK_USES_BEGIN(DrawVisbuffer, 1)
// When drawing triangles, this draw command has triangle ids appended to the end of the command.
DAXA_TASK_USE_BUFFER(u_command, daxa_u64, DRAW_INDIRECT_INFO_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(InstantiatedMeshlets), GRAPHICS_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), GRAPHICS_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_combined_transforms, daxa_BufferPtr(daxa_f32mat4x4), GRAPHICS_SHADER_READ)
DAXA_TASK_USE_IMAGE(u_vis_image, REGULAR_2D, COLOR_ATTACHMENT)
//DAXA_TASK_USE_IMAGE(u_debug_image, REGULAR_2D, COLOR_ATTACHMENT)
DAXA_TASK_USE_IMAGE(u_depth_image, REGULAR_2D, DEPTH_ATTACHMENT)
DAXA_DECL_TASK_USES_END()
#endif
#if __cplusplus || defined(MESH_SHADER_CULL_AND_DRAW)
DAXA_DECL_TASK_USES_BEGIN(DrawVisbufferMeshShaderCullAndDraw, 1)
// When drawing triangles, this draw command has triangle ids appended to the end of the command.
DAXA_TASK_USE_BUFFER(u_command, daxa_u64, DRAW_INDIRECT_INFO_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(InstantiatedMeshlets), GRAPHICS_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(Mesh), GRAPHICS_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_combined_transforms, daxa_BufferPtr(daxa_f32mat4x4), GRAPHICS_SHADER_READ)
DAXA_TASK_USE_IMAGE(u_vis_image, REGULAR_2D, COLOR_ATTACHMENT)
//DAXA_TASK_USE_IMAGE(u_debug_image, REGULAR_2D, COLOR_ATTACHMENT)
DAXA_TASK_USE_IMAGE(u_depth_image, REGULAR_2D, DEPTH_ATTACHMENT)
DAXA_DECL_TASK_USES_END()
#endif

#define DRAW_VISBUFFER_PASS_ONE 0
#define DRAW_VISBUFFER_PASS_TWO 1
#define DRAW_VISBUFFER_PASS_OBSERVER 2

struct DrawVisbufferWriteCommandPush
{
    daxa_u32 pass;
    daxa_u32 mesh_shader;
};

struct DrawVisbufferPush
{
    daxa_u32 pass;
};

#if __cplusplus
#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"
#include "cull_meshlets.inl"

static constexpr inline char const DRAW_VISBUFFER_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/draw_visbuffer.glsl";

static inline daxa::DepthTestInfo DRAW_VISBUFFER_DEPTH_TEST_INFO = {
    .depth_attachment_format = daxa::Format::D32_SFLOAT,
    .enable_depth_test = true,
    .enable_depth_write = true,
    .depth_test_compare_op = daxa::CompareOp::GREATER,
    .min_depth_bounds = 0.0f,
    .max_depth_bounds = 1.0f,
};

static inline std::vector<daxa::RenderAttachment> DRAW_VISBUFFER_RENDER_ATTACHMENT_INFOS = {
    daxa::RenderAttachment{
        .format = daxa::Format::R32_UINT,
    },
    //daxa::RenderAttachment{
    //    .format = daxa::Format::R16G16B16A16_SFLOAT,
    //    .blend = daxa::BlendInfo{
    //        .blend_enable = true,
    //        .src_color_blend_factor = daxa::BlendFactor::SRC_ALPHA,
    //        .dst_color_blend_factor = daxa::BlendFactor::ONE_MINUS_SRC_ALPHA,
    //        .color_blend_op = daxa::BlendOp::ADD,
    //        .src_alpha_blend_factor = daxa::BlendFactor::ONE,
    //        .dst_alpha_blend_factor = daxa::BlendFactor::ONE,
    //        .alpha_blend_op = daxa::BlendOp::ADD,
    //    },
    //},
};

using DrawVisbufferWriteCommandTask = WriteIndirectDispatchArgsPushBaseTask<
    DrawVisbufferWriteCommand,
    DRAW_VISBUFFER_SHADER_PATH,
    DrawVisbufferWriteCommandPush>;

inline static const daxa::RasterPipelineCompileInfo DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_NO_MESH_SHADER = []()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.depth_test = DRAW_VISBUFFER_DEPTH_TEST_INFO;
    ret.color_attachments = DRAW_VISBUFFER_RENDER_ATTACHMENT_INFOS;
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"NO_MESH_SHADER", "1"}}},
    };
    ret.vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"NO_MESH_SHADER", "1"}}},
    };
    ret.name = "DrawVisbuffer";
    ret.push_constant_size = sizeof(DrawVisbufferPush);
    return ret;
}();

inline static const daxa::RasterPipelineCompileInfo DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER = []()
{
    auto ret = daxa::RasterPipelineCompileInfo{};
    ret.depth_test = DRAW_VISBUFFER_DEPTH_TEST_INFO;
    ret.color_attachments = DRAW_VISBUFFER_RENDER_ATTACHMENT_INFOS;
    ret.fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"MESH_SHADER", "1"}}},
    };
    ret.mesh_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{DRAW_VISBUFFER_SHADER_PATH},
        .compile_options = {.defines = {{"MESH_SHADER", "1"}}},
    };
    ret.name = "DrawVisbufferMeshShader";
    ret.push_constant_size = sizeof(DrawVisbufferPush);
    return ret;
}();

struct DrawVisbufferTask
{
    DAXA_USE_TASK_HEADER(DrawVisbuffer)
    inline static const daxa::RasterPipelineCompileInfo PIPELINE_COMPILE_INFO = DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_NO_MESH_SHADER;
    GPUContext *context = {};
    u32 pass = {};
    bool mesh_shader = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::ImageId vis_image = uses.u_vis_image.image();
        //daxa::ImageId debug_image = uses.u_debug_image.image();
        daxa::ImageId depth_image = uses.u_depth_image.image();
        auto cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        const bool clear_images = pass == DRAW_VISBUFFER_PASS_ONE || pass == DRAW_VISBUFFER_PASS_OBSERVER;
        auto load_op = clear_images ? daxa::AttachmentLoadOp::CLEAR : daxa::AttachmentLoadOp::LOAD;
        daxa::RenderPassBeginInfo render_pass_begin_info{
            .depth_attachment = daxa::RenderAttachmentInfo{
                .image_view = depth_image.default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{daxa::DepthValue{0.0f, 0}},
            },
            .render_area = daxa::Rect2D{
                .width = (ti.get_device().info_image(depth_image).size.x),
                .height = (ti.get_device().info_image(depth_image).size.y),
            },
        };
        render_pass_begin_info.color_attachments = {
            daxa::RenderAttachmentInfo{
                .image_view = vis_image.default_view(),
                .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                .load_op = load_op,
                .store_op = daxa::AttachmentStoreOp::STORE,
                .clear_value = daxa::ClearValue{std::array<u32, 4>{INVALID_TRIANGLE_ID, 0, 0, 0}},
            },
            //daxa::RenderAttachmentInfo{
            //    .image_view = debug_image.default_view(),
            //    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
            //    .load_op = load_op,
            //    .store_op = daxa::AttachmentStoreOp::STORE,
            //    .clear_value = daxa::ClearValue{std::array<f32, 4>{1.f, 1.f, 1.f, 1.f}},
            //},
        };
        cmd.begin_renderpass(render_pass_begin_info);
        if (mesh_shader)
        {
            cmd.set_pipeline(*context->raster_pipelines.at(DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER.name));
        }
        else
        {
            cmd.set_pipeline(*context->raster_pipelines.at(PIPELINE_COMPILE_INFO.name));
        }
        cmd.push_constant(DrawVisbufferPush{
            .pass = pass,
        });
        if (mesh_shader)
        {
            cmd.draw_mesh_tasks_indirect({
                .indirect_buffer = uses.u_command.buffer(),
                .offset = 0,
                .draw_count = 1,
                .stride = sizeof(DispatchIndirectStruct),
            });
        }
        else
        {
            cmd.draw_indirect({
                .draw_command_buffer = uses.u_command.buffer(),
                .draw_count = 1,
                .draw_command_stride = sizeof(DrawIndirectStruct),
            });
        }
        cmd.end_renderpass();
    }
};

struct TaskCullAndDrawVisbufferInfo
{
    GPUContext *context = {};
    daxa::TaskGraph & tg;
    const bool enable_mesh_shader = {};
    daxa::TaskBufferView cull_meshlets_commands = {};
    daxa::TaskBufferView meshlet_cull_indirect_args = {};
    daxa::TaskBufferView entity_meta_data = {};
    daxa::TaskBufferView entity_meshlists = {};
    daxa::TaskBufferView entity_combined_transforms = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView entity_meshlet_visibility_bitfield_offsets = {};
    daxa::TaskBufferView entity_meshlet_visibility_bitfield_arena = {};
    daxa::TaskImageView hiz = {};
    daxa::TaskBufferView instantiated_meshlets = {};
    daxa::TaskImageView vis_image = {};
    daxa::TaskImageView debug_image = {};
    daxa::TaskImageView depth_image = {};
};
inline void task_cull_and_draw_visbuffer(TaskCullAndDrawVisbufferInfo const & info)
{
    if (info.enable_mesh_shader)
    {
        //tg.add_task(CullAndDrawVisBufferTask{
        //    .uses = {
        //        .u_command = cull_meshlets_command,
        //        .instantiated_meshlets = instantiated_meshlets,
        //    },
        //    .context = context,
        //});
    }
    else
    {
        auto draw_command = info.tg.create_transient_buffer({
            .size = static_cast<u32>(std::max(sizeof(DrawIndirectStruct), sizeof(DispatchIndirectStruct))),
            .name = std::string("draw visbuffer command buffer") + info.context->dummy_string(),
        });
        // clear to zero, rest of values will be initialized by CullMeshletsTask.
        task_clear_buffer(info.tg, draw_command, 0);
        info.tg.add_task(CullMeshletsTask{
            .uses = {
                .u_commands = info.cull_meshlets_commands,
                .u_meshlet_cull_indirect_args = info.meshlet_cull_indirect_args,
                .u_entity_meta_data = info.entity_meta_data,
                .u_entity_meshlists = info.entity_meshlists,
                .u_entity_combined_transforms = info.entity_combined_transforms,
                .u_meshes = info.meshes,
                .u_entity_meshlet_visibility_bitfield_offsets = info.entity_meshlet_visibility_bitfield_offsets,
                .u_entity_meshlet_visibility_bitfield_arena = info.entity_meshlet_visibility_bitfield_arena,
                .u_hiz = info.hiz,
                .u_instantiated_meshlets = info.instantiated_meshlets,
                .u_draw_command = draw_command,
            },
            .context = info.context,
        });
        info.tg.add_task(DrawVisbufferTask{
            .uses = {
                .u_command = draw_command,
                .u_instantiated_meshlets = info.instantiated_meshlets,
                .u_meshes = info.meshes,
                .u_combined_transforms = info.entity_combined_transforms,
                .u_vis_image = info.vis_image,
                //.u_debug_image = info.debug_image,
                .u_depth_image = info.depth_image,
            },
            .context = info.context,
            .pass = DRAW_VISBUFFER_PASS_TWO,
            .mesh_shader = false,
        });
    }
}

struct TaskDrawVisbufferInfo
{
    GPUContext *context = {};
    daxa::TaskGraph & tg;
    DrawVisbuffer::Uses uses = {};
    const bool enable_mesh_shader = {};
    const u32 pass = {};
    daxa::TaskBufferView instantiated_meshlets = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView combined_transforms = {};
    daxa::TaskImageView vis_image = {};
    daxa::TaskImageView debug_image = {};
    daxa::TaskImageView depth_image = {};
};
inline void task_draw_visbuffer(TaskDrawVisbufferInfo const & info)
{
    auto draw_command = info.tg.create_transient_buffer({
        .size = static_cast<u32>(std::max(sizeof(DrawIndirectStruct), sizeof(DispatchIndirectStruct))),
        .name = std::string("draw visbuffer command buffer") + info.context->dummy_string(),
    });
    info.tg.add_task(DrawVisbufferWriteCommandTask{
        .uses = {
            .u_instantiated_meshlets = info.instantiated_meshlets,
            .u_command = draw_command,
        },
        .context = info.context,
        .push = DrawVisbufferWriteCommandPush{.pass = info.pass, .mesh_shader = info.enable_mesh_shader ? 1u : 0u},
    });
    info.tg.add_task(DrawVisbufferTask{
        .uses = {
            .u_command = draw_command,
            .u_instantiated_meshlets = info.instantiated_meshlets,
            .u_meshes = info.meshes,
            .u_combined_transforms = info.combined_transforms,
            .u_vis_image = info.vis_image,
            //.u_debug_image = info.debug_image,
            .u_depth_image = info.depth_image,
        },
        .context = info.context,
        .pass = info.pass,
        .mesh_shader = info.enable_mesh_shader,
    });
}
#endif