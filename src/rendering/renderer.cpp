#include "renderer.hpp"

#include "../scene/scene.inl"

#include "tasks/generate_index_buffer.inl"
#include "tasks/prefix_sum.inl"

#include "tasks/triangle.hpp"
#include "tasks/find_visible_meshlets.hpp"
#include "tasks/draw_opaque_ids.hpp"

Renderer::Renderer(Window *window, GPUContext *context, Scene *scene, AssetManager *asset_manager)
    : window{window},
      context{context},
      scene{scene},
      asset_manager{asset_manager}
{
    entity_meta = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(EntityMetaData) * MAX_ENTITY_COUNT,
                    .name = "entity_meta",
                }),
            },
        },
        .name = "entity_meta",
    }};
    entity_transforms = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(daxa_f32mat4x4) * MAX_ENTITY_COUNT,
                    .name = "entity_transforms",
                }),
            },
        },
        .name = "entity_transforms",
    }};
    entity_combined_transforms = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(daxa_f32mat4x4) * MAX_ENTITY_COUNT,
                    .name = "entity_combined_transforms",
                }),
            },
        },
        .name = "entity_combined_transforms",
    }};
    entity_first_children = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(EntityId) * MAX_ENTITY_COUNT,
                    .name = "entity_first_children",
                }),
            },
        },
        .name = "entity_first_children",
    }};
    entity_next_silbings = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(EntityId) * MAX_ENTITY_COUNT,
                    .name = "entity_next_silbings",
                }),
            },
        },
        .name = "entity_next_silbings",
    }};
    entity_parents = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(EntityId) * MAX_ENTITY_COUNT,
                    .name = "entity_parents",
                }),
            },
        },
        .name = "entity_parents",
    }};
    entity_meshlists = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(MeshList) * MAX_ENTITY_COUNT,
                    .name = "entity_meshlists",
                }),
            },
        },
        .name = "entity_meshlists",
    }};

    globals = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(ShaderGlobals),
                    .name = "globals",
                }),
            },
        },
        .name = "globals",
    }};
    instanciated_meshlets = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(MeshletDrawInfo) * MAX_DRAWN_MESHLETS + /*reserved space for a counter*/ 16,
                    .name = "instanciated_meshlets",
                }),
            },
        },
        .name = "instanciated_meshlets",
    }};
    index_buffer = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = TRIANGLE_SIZE * MAX_DRAWN_TRIANGLES + /*reserved space for a counter*/ 16,
                    .name = "index_buffer",
                }),
            },
        },
        .name = "index_buffer",
    }};
    ent_meshlet_count_prefix_sum_buffer = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = static_cast<u32>(sizeof(u32)) * round_up_to_multiple(MAX_ENTITY_COUNT, PREFIX_SUM_WORKGROUP_SIZE),
                    .name = "ent_meshlet_count_prefix_sum_buffer",
                }),
            },
        },
        .name = "ent_meshlet_count_prefix_sum_buffer",
    }};
    ent_meshlet_count_partial_sum_buffer = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = round_up_to_multiple(round_up_div((sizeof(u32) * MAX_ENTITY_COUNT), PREFIX_SUM_WORKGROUP_SIZE), PREFIX_SUM_WORKGROUP_SIZE),
                    .name = "ent_meshlet_count_partial_sum_buffer",
                }),
            },
        },
        .name = "ent_meshlet_count_partial_sum_buffer",
    }};
    // First 16 bytes are reserved for a counter variable.
    draw_opaque_id_info_buffer = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(DrawOpaqueDrawInfo),
                    .name = "draw_opaque_id_info_buffer",
                }),
            },
        },
        .name = "draw_opaque_id_info_buffer",
    }};

    buffers = {
        entity_meta,
        entity_transforms,
        entity_combined_transforms,
        entity_first_children,
        entity_next_silbings,
        entity_parents,
        entity_meshlists,
        globals,
        instanciated_meshlets,
        index_buffer,
        ent_meshlet_count_prefix_sum_buffer,
        ent_meshlet_count_partial_sum_buffer,
        draw_opaque_id_info_buffer};

    depth = daxa::TaskImage{{
        .name = "depth",
    }};
    swapchain_image = daxa::TaskImage{{
        .swapchain_image = true,
        .name = "swapchain_image",
    }};

    images = {
        depth};

    recreate_resizable_images();

    compile_pipelines();

    main_task_list = create_main_task_list();
}

Renderer::~Renderer()
{
    for (auto &tbuffer : buffers)
    {
        for (auto buffer : tbuffer.get_state().buffers)
        {
            this->context->device.destroy_buffer(buffer);
        }
    }
    for (auto &timage : images)
    {
        for (auto image : timage.get_state().images)
        {
            this->context->device.destroy_image(image);
        }
    }
    this->context->device.wait_idle();
    this->context->device.collect_garbage();
}

void Renderer::compile_pipelines()
{
    std::vector<std::tuple<std::string_view, daxa::RasterPipelineCompileInfo>> rasters = {
        {TRIANGLE_PIPELINE_NAME, TRIANGLE_PIPELINE_INFO},
    };
    for (auto [name, info] : rasters)
    {
        auto compilation_result = this->context->pipeline_manager.add_raster_pipeline(info);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->raster_pipelines[name] = compilation_result.value();
    }
    std::vector<std::tuple<std::string_view, daxa::ComputePipelineCompileInfo>> computes = {
        {PREFIX_SUM_NAME, PREFIX_SUM_PIPELINE_INFO},
        {PrefixSumMeshletTask{}.name, PREFIX_SUM_MESHLETS_PIPELINE_INFO},
        {PREFIX_SUM_TWO_PASS_FINALIZE_NAME, PREFIX_SUM_TWO_PASS_FINALIZE_PIPELINE_INFO},
        {FIND_VISIBLE_MESHLETS_PIPELINE_NAME, FIND_VISIBLE_MESHLETS_PIPELINE_INFO},
        {GENERATE_INDEX_BUFFER_NAME, GENERATE_INDEX_BUFFER_PIPELINE_INFO},
    };
    for (auto [name, info] : computes)
    {
        auto compilation_result = this->context->pipeline_manager.add_compute_pipeline(info);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->compute_pipelines[name] = compilation_result.value();
    }
}

void Renderer::recreate_resizable_images()
{
    if (!depth.get_state().images.empty() && !depth.get_state().images[0].is_empty())
    {
        context->device.destroy_image(depth.get_state().images[0]);
    }
    depth.set_images({
        .images = std::array{
            this->context->device.create_image({
                .format = daxa::Format::D32_SFLOAT,
                .aspect = daxa::ImageAspectFlagBits::DEPTH,
                .size = {this->window->get_width(), this->window->get_height(), 1},
                .usage = daxa::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
                .name = depth.info().name,
            })},
    });
}

void Renderer::window_resized()
{
    if (this->window->size.x == 0 || this->window->size.y == 0)
    {
        return;
    }
    this->context->swapchain.resize();
    recreate_resizable_images();
}

auto Renderer::create_main_task_list() -> daxa::TaskList
{
    using namespace daxa;
    TaskList task_list{{
        .device = this->context->device,
        .swapchain = this->context->swapchain,
        .name = "Sandbox main TaskList",
    }};
    for (auto const &tbuffer : buffers)
    {
        task_list.use_persistent_buffer(tbuffer);
    }
    task_list.use_persistent_buffer(asset_manager->tmeshes);
    for (auto const &timage : images)
    {
        task_list.use_persistent_image(timage);
    }
    task_list.use_persistent_image(swapchain_image);

    task_list.add_task({
        .uses = {
            BufferHostTransferWrite{globals},
        },
        .task = [=](daxa::TaskInterface ti)
        {
            auto cmd = ti.get_command_list();
            auto staging_buffer = ti.get_device().create_buffer({
                .size = sizeof(ShaderGlobals),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "ShaderGlobals staging buffer",
            });
            cmd.destroy_buffer_deferred(staging_buffer);
            *ti.get_device().get_host_address_as<ShaderGlobals>(staging_buffer) = context->shader_globals;
            cmd.copy_buffer_to_buffer({
                .src_buffer = staging_buffer,
                .dst_buffer = ti.uses[globals].buffer(),
                .size = sizeof(ShaderGlobals),
            });
        },
        .name = "buffer uploads",
    });

    task_list.add_task(PrefixSumMeshletTask{
        .uses = {
            .meshes = asset_manager->tmeshes.handle(),
            .entity_meta = entity_meta.handle(),
            .entity_meshlists = entity_meshlists.handle(),
            .ent_meshlet_count_prefix_sum_buffer = ent_meshlet_count_prefix_sum_buffer.handle(),
        },
        .pipeline = context->compute_pipelines[PrefixSumMeshletTask{}.name],
        .config = {
            .entity_count = &scene->entity_meta.entity_count,
        },
    });

    static constexpr u32 STRIDE = PREFIX_SUM_WORKGROUP_SIZE;
    static constexpr u32 OFFSET = PREFIX_SUM_WORKGROUP_SIZE - 1;
    task_list.add_task(PrefixSumTask{
        .uses = {
            .src = ent_meshlet_count_prefix_sum_buffer.handle(),
            .dst = ent_meshlet_count_partial_sum_buffer.handle(),
        },
        .pipeline = context->compute_pipelines[PrefixSumTask{}.name],
        .config = {
            .src_stride = &STRIDE,
            .src_offset = &OFFSET,
            .value_count = &this->context->meshlet_sums_step2_dispatch_size,
        },
    });

    task_list.add_task(PrefixSumFinalizeTask{
        .uses = {
            .partial_sums = ent_meshlet_count_partial_sum_buffer.handle(),
            .values = ent_meshlet_count_prefix_sum_buffer.handle(),
        },
        .pipeline = context->compute_pipelines[PrefixSumFinalizeTask{}.name],
        .config = {
            .value_count = &scene->entity_meta.entity_count,
        },
    });

    t_find_visible_meshlets(
        this->context,
        task_list,
        ent_meshlet_count_prefix_sum_buffer,
        entity_meta,
        entity_meshlists,
        asset_manager->tmeshes,
        instanciated_meshlets,
        [=]()
        {
            return this->asset_manager->total_meshlet_count;
        });

    task_list.add_task({
        .uses = {
            daxa::BufferHostTransferWrite{index_buffer},
        },
        .task = [=](daxa::TaskInterface ti)
        {
            daxa::CommandList cmd = ti.get_command_list();
            auto alloc = this->context->transient_mem.allocate(sizeof(u32)).value();
            *reinterpret_cast<u32 *>(alloc.host_address) = 0;
            cmd.copy_buffer_to_buffer({
                .src_buffer = this->context->transient_mem.get_buffer(),
                .src_offset = alloc.buffer_offset,
                .dst_buffer = ti.uses[index_buffer].buffer(),
                .dst_offset = 0,
                .size = sizeof(u32),
            });
        },
        .name = "clear triangle count of index buffer",
    });

    task_list.add_task(GenIndexBufferTask{
        {
            .uses = {
                .meshes = asset_manager->tmeshes.handle(),
                .instanciated_meshlets = instanciated_meshlets.handle(),
                .index_buffer_and_count = index_buffer.handle(),
            },
        },
        context});
    // t_generate_index_buffer(
    //     this->context,
    //     task_list,
    //     meshes_buffer_tid,
    //     this->context->instanciated_meshlets.t_id,
    //     this->context->index_buffer.t_id,
    //     [=]()
    //     { return this->asset_manager->total_meshlet_count; });

    t_draw_triangle({
        .task_list = task_list,
        .context = *(this->context),
        .t_swapchain_image = swapchain_image,
        .t_depth_image = depth,
        .t_shader_globals = globals,
    });

    task_list.submit({.additional_signal_timeline_semaphores = &submit_info.signal_timeline_semaphores});
    task_list.present({});
    task_list.complete({});
    return task_list;
}

void Renderer::render_frame(CameraInfo const &camera_info)
{
    if (this->window->size.x == 0 || this->window->size.y == 0)
    {
        return;
    }
    this->context->shader_globals.camera_view = *reinterpret_cast<f32mat4x4 const *>(&camera_info.view);
    this->context->shader_globals.camera_projection = *reinterpret_cast<f32mat4x4 const *>(&camera_info.proj);
    this->context->shader_globals.camera_view_projection = *reinterpret_cast<f32mat4x4 const *>(&camera_info.vp);

    this->context->meshlet_sums_step2_dispatch_size = (scene->entity_meta.entity_count + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE;
    this->context->total_meshlet_count = this->asset_manager->total_meshlet_count;

    auto swapchain_image = context->swapchain.acquire_next_image();
    if (swapchain_image.is_empty())
    {
        return;
    }
    this->swapchain_image.set_images({.images = std::array{swapchain_image}});

    this->submit_info = {};
    this->submit_info.signal_timeline_semaphores = {
        {this->context->transient_mem.get_timeline_semaphore(), this->context->transient_mem.timeline_value()},
    };
    main_task_list.execute({});
}