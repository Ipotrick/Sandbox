#pragma once

#include "../render_context.hpp"
#include "../../../shaders/triangle_shared.inl"

inline static const daxa::RasterPipelineInfo TRIANGLE_TASK_RASTER_PIPE_INFO{
    .vertex_shader_info = daxa::ShaderInfo{
        .source = daxa::ShaderFile{"triangle.glsl"},
        .compile_options = {
            .defines = {{"_VERTEX", ""}},
        },
    },
    .fragment_shader_info = daxa::ShaderInfo{
        .source = daxa::ShaderFile{"triangle.glsl"},
        .compile_options = {
            .defines = {{"_FRAGMENT", ""}},
        },
    },
    .color_attachments = {
        daxa::RenderAttachment{
            .format = daxa::Format::B8G8R8A8_SRGB,
        },
    },
    .push_constant_size = sizeof(TriangleTaskPushConstant),
};

struct TriangleTaskInfo
{
    daxa::TaskList &task_list;
    RenderContext &context;
    daxa::TaskImageId t_swapchain_image;
};
inline void t_draw_triangle(TriangleTaskInfo const &info)
{
    info.task_list.add_task({
        .used_images = {
            daxa::TaskImageUse{info.t_swapchain_image, daxa::TaskImageAccess::COLOR_ATTACHMENT, {}},
        },
        .task = [=](daxa::TaskRuntime const &runtime)
        {
            daxa::CommandList cmd = runtime.get_command_list();
            daxa::ImageId swapchain_image = runtime.get_images(info.t_swapchain_image)[0];
            std::cout
                << "renderarea size: ("
                << (runtime.get_device().info_image(swapchain_image).size.x)
                << ", "
                << (runtime.get_device().info_image(swapchain_image).size.y)
                << ")"
                << std::endl;
            cmd.begin_renderpass({
                .color_attachments = {
                    daxa::RenderAttachmentInfo{
                        .image_view = swapchain_image.default_view(),
                        .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                        .load_op = daxa::AttachmentLoadOp::CLEAR,
                        .store_op = daxa::AttachmentStoreOp::STORE,
                        .clear_value = daxa::ClearValue{std::array<f32, 4>{0.f, 0.f, 0.f, 0.f}},
                    },
                },
                .render_area = daxa::Rect2D{
                    .width = (runtime.get_device().info_image(swapchain_image).size.x),
                    .height = (runtime.get_device().info_image(swapchain_image).size.y),
                },
            });
            cmd.set_pipeline(info.context.triangle_pipe);
            cmd.draw({
                .vertex_count = 3,
                .instance_count = 1,
                .first_vertex = 0,
                .first_instance = 0,
            });
            cmd.end_renderpass();
        },
        .debug_name = "Sandbox Triangle Task",
    });
}