#pragma once

#include "../gpu_context.hpp"
#include "../../../shaders/triangle_shared.inl"

inline static constexpr std::string_view TRIANGLE_PIPELINE_NAME = "triangle pipeline";

inline static const daxa::RasterPipelineCompileInfo TRIANGLE_PIPELINE_INFO{
    .vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"triangle.glsl"},
        .compile_options = {
            .defines = {{"ENTRY_VERTEX", ""}},
        },
    },
    .fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"triangle.glsl"},
        .compile_options = {
            .defines = {{"ENTRY_FRAGMENT", ""}},
        },
    },
    .color_attachments = {
        daxa::RenderAttachment{
            .format = daxa::Format::B8G8R8A8_SRGB,
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
    .push_constant_size = sizeof(TriangleTaskPushConstant),
    .name = std::string{TRIANGLE_PIPELINE_NAME},
};

struct TriangleTaskInfo
{
    daxa::TaskList &task_list;
    GPUContext &context;
    daxa::TaskImageId t_swapchain_image;
    daxa::TaskImageId t_depth_image;
    daxa::TaskBufferId t_shader_globals;
};

inline void t_draw_triangle(TriangleTaskInfo const &info)
{
    info.task_list.add_task({
        .used_buffers = {
            daxa::TaskBufferUse{info.t_shader_globals, daxa::TaskBufferAccess::SHADER_READ_ONLY}},
        .used_images = {
            daxa::TaskImageUse{info.t_swapchain_image, daxa::TaskImageAccess::COLOR_ATTACHMENT, {}},
            daxa::TaskImageUse{info.t_depth_image, daxa::TaskImageAccess::DEPTH_ATTACHMENT, {.image_aspect = daxa::ImageAspectFlagBits::DEPTH}},
        },
        .task = [=](daxa::TaskRuntimeInterface const &runtime)
        {
            daxa::CommandList cmd = runtime.get_command_list();
            daxa::ImageId swapchain_image = runtime.get_images(info.t_swapchain_image)[0];
            daxa::ImageId depth_image = runtime.get_images(info.t_depth_image)[0];
            cmd.begin_renderpass({
                .color_attachments = {
                    daxa::RenderAttachmentInfo{
                        .image_view = swapchain_image.default_view(),
                        .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                        .load_op = daxa::AttachmentLoadOp::CLEAR,
                        .store_op = daxa::AttachmentStoreOp::STORE,
                        .clear_value = daxa::ClearValue{std::array<f32, 4>{1.f, 1.f, 1.f, 1.f}},
                    },
                },
                .depth_attachment = daxa::RenderAttachmentInfo{
                    .image_view = depth_image.default_view(),
                    .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .load_op = daxa::AttachmentLoadOp::CLEAR,
                    .store_op = daxa::AttachmentStoreOp::STORE,
                    .clear_value = daxa::ClearValue{daxa::DepthValue{0.0f, 0}},
                },
                .render_area = daxa::Rect2D{
                    .width = (runtime.get_device().info_image(swapchain_image).size.x),
                    .height = (runtime.get_device().info_image(swapchain_image).size.y),
                },
            });
            cmd.set_pipeline(*info.context.raster_pipelines.at(TRIANGLE_PIPELINE_NAME));
            cmd.push_constant(TriangleTaskPushConstant{
                .globals = runtime.get_device().get_device_address(runtime.get_buffers(info.t_shader_globals)[0]),
            });
            cmd.draw({
                .vertex_count = 3,
                .instance_count = 1,
                .first_vertex = 0,
                .first_instance = 0,
            });
            cmd.end_renderpass();
        },
        .name = std::string{TRIANGLE_PIPELINE_NAME},
    });
}