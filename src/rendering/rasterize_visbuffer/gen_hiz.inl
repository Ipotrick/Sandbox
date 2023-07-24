#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/mesh.inl"

#define GEN_HIZ_X 32
#define GEN_HIZ_Y 32
#define GEN_HIZ_LEVELS_PER_DISPATCH 6

struct GenHizPush
{
    daxa_ImageViewId src;
    daxa_ImageViewId dst[GEN_HIZ_LEVELS_PER_DISPATCH];
    daxa_u32 sample_depth;
    daxa_u32 width;
    daxa_u32 height;
    daxa_u32 mips;
};

#if __cplusplus

#include <format>
#include "../gpu_context.hpp"

inline static const daxa::ComputePipelineCompileInfo GEN_HIZ_PIPELINE_COMPILE_INFO{
    .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{"./src/rendering/rasterize_visbuffer/gen_hiz.glsl"}},
    .push_constant_size = sizeof(GenHizPush),
    .name = std::string{"GenHiz"},
};

daxa::TaskImageView task_gen_hiz(GPUContext * context, daxa::TaskGraph & task_graph, daxa::TaskImageView src)
{
    const u32vec2 hiz_size = u32vec2(context->settings.render_target_size.x / 2, context->settings.render_target_size.y / 2);
    const u32 mips = static_cast<u32>(std::ceil(std::log2(std::max(hiz_size.x, hiz_size.y))));
    daxa::TaskImageView hiz = task_graph.create_transient_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = { hiz_size.x, hiz_size.y, 1 },
        .mip_level_count = mips,
        .array_layer_count = 1,
        .sample_count = 1,
        .name = "hiz",
    });
    for (u32 dispatch_i = 0; dispatch_i < (mips + GEN_HIZ_LEVELS_PER_DISPATCH - 1) / GEN_HIZ_LEVELS_PER_DISPATCH; ++dispatch_i)
    {
        using namespace daxa::task_resource_uses;
        const u32 first_mip = dispatch_i * GEN_HIZ_LEVELS_PER_DISPATCH;
        const u32 last_mip = std::min(mips, (dispatch_i + 1) * GEN_HIZ_LEVELS_PER_DISPATCH) - 1;
        const u32 mips_this_dispatch = last_mip - first_mip + 1;
        std::vector<daxa::GenericTaskResourceUse> uses = {};
        daxa::TaskImageView src_view = (dispatch_i == 0 ? src.view({.base_mip_level = 0}) : hiz.view({.base_mip_level = first_mip-1}));
        uses.push_back(ImageComputeShaderSampled<>{ src_view });
        daxa::TaskImageView dst_views[GEN_HIZ_LEVELS_PER_DISPATCH] = { {}, {}, {}, {}, {} };
        for (u32 i = 0; i < mips_this_dispatch; ++i)
        {
            dst_views[i] = hiz.view({.base_mip_level = first_mip + i});
            uses.push_back(ImageComputeShaderStorageWriteOnly<>{ dst_views[i] });                                 
        }
        task_graph.add_task({
            .uses = uses,
            .task = [=](daxa::TaskInterface ti)
            {
                auto cmd = ti.get_command_list();
                auto & device = ti.get_device();
                cmd.set_uniform_buffer(context->shader_globals_set_info);
                cmd.set_pipeline(*context->compute_pipelines.at(GEN_HIZ_PIPELINE_COMPILE_INFO.name));
                auto const src_x = device.info_image(ti.uses[src_view].image()).size.x >> (ti.uses[src_view].handle.slice.base_mip_level);
                auto const src_y = device.info_image(ti.uses[src_view].image()).size.y >> (ti.uses[src_view].handle.slice.base_mip_level);
                auto const x = std::max(1u,device.info_image(ti.uses[dst_views[0]].image()).size.x >> (ti.uses[dst_views[0]].handle.slice.base_mip_level));
                auto const y = std::max(1u,device.info_image(ti.uses[dst_views[0]].image()).size.y >> (ti.uses[dst_views[0]].handle.slice.base_mip_level));
                auto const dispatch_x = round_up_div(x, GEN_HIZ_X);
                auto const dispatch_y = round_up_div(y, GEN_HIZ_Y);
                GenHizPush push{ 
                    .src = ti.uses[src_view].view(),
                    .dst = {{},{},{},{},{}},
                    .sample_depth = dispatch_i == 0 ? 1u : 0u,
                    .width = src_x, 
                    .height = src_y, 
                    .mips = mips_this_dispatch,
                };
                for (u32 i = 0; i < mips_this_dispatch; ++i)
                {
                    push.dst[i] = ti.uses[dst_views[i]].view();
                }
                cmd.push_constant(push);
                cmd.dispatch(dispatch_x, dispatch_y, 1);
            },
            .name = std::format("gen hiz level {} to (inclusive) {}", first_mip, last_mip),
        });
    }
    return hiz;
}

#endif