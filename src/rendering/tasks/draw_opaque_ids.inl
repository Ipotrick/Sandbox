#pragma once
#include <daxa/daxa.inl>
#include <daxa/utils/task_list.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

DAXA_INL_TASK_USE_BEGIN(DrawOpaqueId, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_IMAGE(id_image, daxa_RWImage2Df32, COLOR_ATTACHMENT)
DAXA_INL_TASK_USE_IMAGE(depth_image, daxa_RWImage2Df32, DEPTH_ATTACHMENT)
DAXA_INL_TASK_USE_BUFFER(globals, daxa_BufferPtr(ShaderGlobals), VERTEX_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(draw_info_index_buffer, daxa_BufferPtr(daxa_u32), INDEX_READ)
DAXA_INL_TASK_USE_BUFFER(instanciated_meshlets, daxa_BufferPtr(InstanciatedMeshlet), VERTEX_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(entity_meshlists, daxa_BufferPtr(MeshList), VERTEX_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(meshes, daxa_BufferPtr(Mesh), VERTEX_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(combined_transforms, daxa_BufferPtr(daxa_f32mat4x4), VERTEX_SHADER_READ)
DAXA_INL_TASK_USE_END()

#if __cplusplus

#include "../gpu_context.hpp"

inline static constexpr std::string_view DRAW_OPAQUE_IDS_PIPELINE_NAME = { DrawOpaqueId::NAME };

inline static const daxa::RasterPipelineCompileInfo DRAW_OPAQUE_IDS_PIPELINE_INFO{
    .vertex_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"./src/rendering/tasks/draw_opaque_ids.inl"},
    },
    .fragment_shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"./src/rendering/tasks/draw_opaque_ids.inl"},
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
    .name = std::string{DrawOpaqueId{}.name},
};

struct DrawOpaqueIdTask : DrawOpaqueId
{
    std::shared_ptr<daxa::RasterPipeline> pipeline = {}; 
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::CommandList cmd = ti.get_command_list();
        cmd.set_constant_buffer(ti.uses.constant_buffer_set_info());
        daxa::ImageId id_image = uses.id_image.image();
        daxa::ImageId depth_image = uses.depth_image.image();
        cmd.begin_renderpass({
            .color_attachments = {
                daxa::RenderAttachmentInfo{
                    .image_view = id_image.default_view(),
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
                .width = (ti.get_device().info_image(id_image).size.x),
                .height = (ti.get_device().info_image(id_image).size.y),
            },
        });
        cmd.set_pipeline(*pipeline);
        cmd.set_index_buffer(uses.draw_info_index_buffer.buffer(), 32 /*draw info*/);
        cmd.draw_indirect({
            .draw_command_buffer = uses.draw_info_index_buffer.buffer(),
            .draw_command_buffer_read_offset = 0,
            .draw_count = 1,
            .draw_command_stride = 32,
            .is_indexed = true,
        });
        cmd.end_renderpass();
    }
};

#elif DAXA_SHADER
#extension GL_EXT_debug_printf : enable
#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_VERTEX
void main()
{
    uint vertex_id = gl_VertexIndex;
    uint instanciated_meshlet_index = 0;
    uint micro_index = 0;
    decode_vertex_id(vertex_id, instanciated_meshlet_index, micro_index);

    // daxa_u32 entity_index;
    // daxa_u32 mesh_id;
    // daxa_u32 mesh_index;
    // daxa_u32 meshlet_index;
    InstanciatedMeshlet instanciated_meshlet = instanciated_meshlets[instanciated_meshlet_index].value;

    // daxa_BufferId mesh_buffer;
    // daxa_u32 meshlet_count;
    // daxa_BufferPtr(Meshlet) meshlets;
    // daxa_BufferPtr(BoundingSphere) meshlet_bounds;
    // daxa_BufferPtr(daxa_u32) micro_indices;
    // daxa_BufferPtr(daxa_u32) indirect_vertices;
    // daxa_BufferPtr(daxa_f32vec3) vertex_positions;
    daxa_BufferPtr(Mesh) mesh = meshes + instanciated_meshlet.mesh_id;
    
    // daxa_u32 indirect_vertex_offset;
    // daxa_u32 micro_indices_offset;
    // daxa_u32 vertex_count;
    // daxa_u32 triangle_count;
    Meshlet meshlet = mesh.value.meshlets[instanciated_meshlet.meshlet_index].value;
    const uint vertex_index = mesh.value.indirect_vertices[meshlet.indirect_vertex_offset + micro_index].value;
    const vec4 vertex_position = vec4(mesh.value.vertex_positions[vertex_index].value, 1);
    mat4 model_matrix = combined_transforms[instanciated_meshlet.entity_index].value;

    gl_Position = globals.value.camera_view_projection * vertex_position;
}
#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_FRAGMENT
layout(location = 0) out vec4 fout_color;
void main()
{
    fout_color = vec4(0,0,0,1);
}
#endif

#endif