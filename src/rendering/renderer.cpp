#include "renderer.hpp"

#include "../scene/scene.inl"

#include "tasks/fill_meshlet_buffer.inl"
#include "tasks/fill_index_buffer.inl"
#include "tasks/analyze_visbuffer.inl"
#include "tasks/prefix_sum.inl"
#include "tasks/draw_opaque_ids.inl"
#include "tasks/write_swapchain.inl"

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
    entity_debug = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(daxa_u32vec4) * MAX_ENTITY_COUNT,
                    .name = "entity_debug",
                }),
            },
        },
        .name = "entity_debug",
    }};

    instantiated_meshlets = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(InstantiatedMeshletInfo) * MAX_DRAWN_MESHLETS + /*reserved space for dispatch indirect info*/ 32,
                    .name = "instantiated_meshlets",
                }),
            },
        },
        .name = "instantiated_meshlets",
    }};
    instantiated_meshlet_visibility_counters = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(daxa_u32) * MAX_DRAWN_MESHLETS,
                    .name = "instantiated_meshlet_visibility_counters",
                }),
            },
        },
        .name = "instantiated_meshlet_visibility_counters",
    }};
    visible_meshlets = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(InstantiatedMeshletInfo) * MAX_DRAWN_MESHLETS + /*reserved space for dispatch indirect info*/ 32,
                    .name = "visible_meshlets",
                }),
            },
        },
        .name = "visible_meshlets",
    }};
    index_buffer = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = TRIANGLE_SIZE * MAX_DRAWN_TRIANGLES + /*reserved space for draw inidrect info*/ 32,
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

    buffers = {
        entity_meta,
        entity_transforms,
        entity_combined_transforms,
        entity_first_children,
        entity_next_silbings,
        entity_parents,
        entity_meshlists,
        entity_debug,
        instantiated_meshlets,
        instantiated_meshlet_visibility_counters,
        index_buffer,
        ent_meshlet_count_prefix_sum_buffer,
        ent_meshlet_count_partial_sum_buffer};

    swapchain_image = daxa::TaskImage{{
        .swapchain_image = true,
        .name = "swapchain_image",
    }};
    depth = daxa::TaskImage{{
        .name = "depth",
    }};
    visbuffer = daxa::TaskImage{{
        .name = "visbuffer",
    }};
    debug_image = daxa::TaskImage{{
        .name = "debug_image",
    }};

    images = {
        debug_image,
        visbuffer,
        depth,
    };

    frame_buffer_images = {
        {
            {
                .format = daxa::Format::D32_SFLOAT,
                .aspect = daxa::ImageAspectFlagBits::DEPTH,
                .usage = daxa::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
                .name = depth.info().name,
            },
            depth,
        },
        {
            {
                .format = daxa::Format::R32_UINT,
                .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT |
                         daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
                .name = visbuffer.info().name,
            },
            visbuffer,
        },
        {
            {
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT |
                         daxa::ImageUsageFlagBits::TRANSFER_DST |
                         daxa::ImageUsageFlagBits::TRANSFER_SRC |
                         daxa::ImageUsageFlagBits::SHADER_READ_WRITE |
                         daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
                .name = debug_image.info().name,
            },
            debug_image,
        },
    };

    recreate_framebuffer();

    compile_pipelines();

    context->settings.indexed_id_rendering = 1;
    context->settings.update_culling_matrix = 1;

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
        {DrawOpaqueIdTask::NAME, DRAW_OPAQUE_IDS_PIPELINE_INFO},
    };
    for (auto [name, info] : rasters)
    {
        auto compilation_result = this->context->pipeline_manager.add_raster_pipeline(info);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->raster_pipelines[name] = compilation_result.value();
    }
    std::vector<std::tuple<std::string_view, daxa::ComputePipelineCompileInfo>> computes = {
        {PrefixSumTask{}.name, PREFIX_SUM_PIPELINE_INFO},
        {PrefixSumMeshletTask{}.name, PREFIX_SUM_MESHLETS_PIPELINE_INFO},
        {PrefixSumFinalizeTask{}.name, PREFIX_SUM_TWO_PASS_FINALIZE_PIPELINE_INFO},
        {FillMeshletBufferTask::NAME, FILL_MESHLET_BUFFER_PIPELINE_INFO},
        {FillIndexBufferTask::NAME, FILL_INDEX_BUFFER_PIPELINE_INFO},
        {WriteSwapchainTask::NAME, WRITE_SWAPCHAIN_PIPELINE_INFO},
        {AnalyzeVisbufferTask::NAME, ANALYZE_VISBUFFER_PIPELINE_INFO},
    };
    for (auto [name, info] : computes)
    {
        auto compilation_result = this->context->pipeline_manager.add_compute_pipeline(info);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->compute_pipelines[name] = compilation_result.value();
    }
}

void Renderer::recreate_framebuffer()
{
    for (auto &[info, timg] : frame_buffer_images)
    {
        if (!timg.get_state().images.empty() && !timg.get_state().images[0].is_empty())
        {
            context->device.destroy_image(timg.get_state().images[0]);
        }
        auto new_info = info;
        new_info.size = {this->window->get_width(), this->window->get_height(), 1};
        timg.set_images({.images = std::array{this->context->device.create_image(new_info)}});
    }
}

void Renderer::window_resized()
{
    if (this->window->size.x == 0 || this->window->size.y == 0)
    {
        return;
    }
    this->context->swapchain.resize();
    recreate_framebuffer();
}

auto Renderer::create_main_task_list() -> daxa::TaskList
{
    // Rendering process:
    //  - update metadata
    //  - clear buffer containing list of drawn meshlets
    //      - IMPORTANT: This list of drawn meshlets has a counter of visible pixels for each drawn meshlet
    //                   These counters are written to when anaylizing the id buffer and used to create the list of visible meshlets in the end
    //  - insert list of visible meshlets into list of drawn meshlets
    //  - draw list of visible meshlets from last frame depth only
    //  - build hiz from depth initial depth
    //  - three possible paths:
    //      - draw indirect indexed count:
    //          - cull instances, expand list of to be drawn instances
    //          - cull meshlets, concatinate visible meshlets to list of drawn meshlets
    //          - cull triangles, expand index buffer
    //          - draw indexed indirect count
    //      - draw indirect count:
    //          - cull instances, expand list of to be drawn instances
    //          - cull meshlets, concatinate visible meshlets to list of drawn meshlets
    //          - draw indirect count
    //      - dispatch tasks indirect count
    //          - cull instances, expand list of to be drawn instances
    //          - dispatch task shaders indirect count:
    //              - cull meshlets, dispatch mesh shaders for non culled meshlets
    //              - cull triangles, create triangles and give the index buffer to the rasterizer
    //  - build hiz depth
    //  - analyze visbuffer tiles
    //      - count visible pixels for each drawn meshlet
    //      - add bits to materiak mask for tile
    //      - write material id as depth value to material depth
    //  - scan list of drawn meshlets:
    //      - if a meshlet has a visible triangle count of over 0, append it to visible meshlet list for next frames use.
    //  - draw opaque:
    //      - generate g buffer for post processing
    //      - write final color result
    //      - method:
    //          - draw tiles only for tiles that have material present
    //          - set depth test to equal material depth
    //  - blurr pass for bloom
    //      - take last frames bloom into account for temporal effect
    //      - also anaylze brightness for tonemapping
    //  - write swapchain
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
    auto depth_handle = depth.handle().subslice({.image_aspect = daxa::ImageAspectFlagBits::DEPTH});

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

    task_list.add_task({
        .uses = {
            daxa::BufferTransferWrite{instantiated_meshlets},
        },
        .task = [=](daxa::TaskInterface ti)
        {
            daxa::CommandList cmd = ti.get_command_list();
            auto alloc = this->context->transient_mem.allocate(sizeof(DispatchIndirectStruct)).value();
            *reinterpret_cast<DispatchIndirectStruct *>(alloc.host_address) = DispatchIndirectStruct{
                .x = 0,
                .y = 1,
                .z = 1,
            };
            cmd.copy_buffer_to_buffer({
                .src_buffer = this->context->transient_mem.get_buffer(),
                .src_offset = alloc.buffer_offset,
                .dst_buffer = ti.uses[instantiated_meshlets].buffer(),
                .dst_offset = 0,
                .size = sizeof(DispatchIndirectStruct),
            });
        },
        .name = "clear instantiated meshlet counter",
    });

    task_list.add_task(FillMeshletBufferTask{
        {
            .uses = {
                .u_prefix_sum_mehslet_counts = ent_meshlet_count_prefix_sum_buffer.handle(),
                .u_entity_meta_data = entity_meta.handle(),
                .u_entity_meshlists = entity_meshlists.handle(),
                .u_meshes = asset_manager->tmeshes.handle(),
                .u_instantiated_meshlets = instantiated_meshlets.handle(),
            },
        },
        context->compute_pipelines[FillMeshletBufferTask::NAME],
        context,
        &this->asset_manager->total_meshlet_count,
    });

    task_list.add_task({
        .uses = {
            daxa::BufferTransferWrite{index_buffer},
        },
        .task = [=](daxa::TaskInterface ti)
        {
            daxa::CommandList cmd = ti.get_command_list();
            if (context->settings.indexed_id_rendering)
            {
                auto alloc = this->context->transient_mem.allocate(sizeof(DrawIndexedIndirectStruct)).value();
                *reinterpret_cast<DrawIndexedIndirectStruct *>(alloc.host_address) = DrawIndexedIndirectStruct{
                    .index_count = {},
                    .instance_count = 1,
                    .first_index = {},
                    .vertex_offset = {},
                    .first_instance = {},
                };
                cmd.copy_buffer_to_buffer({
                    .src_buffer = this->context->transient_mem.get_buffer(),
                    .src_offset = alloc.buffer_offset,
                    .dst_buffer = ti.uses[index_buffer].buffer(),
                    .dst_offset = 0,
                    .size = sizeof(DrawIndexedIndirectStruct),
                });
            }
            else
            {
                auto alloc = this->context->transient_mem.allocate(sizeof(DrawIndirectStruct)).value();
                *reinterpret_cast<DrawIndirectStruct *>(alloc.host_address) = DrawIndirectStruct{
                    .vertex_count = {},
                    .instance_count = 1,
                    .first_vertex = {},
                    .first_instance = {},
                };
                cmd.copy_buffer_to_buffer({
                    .src_buffer = this->context->transient_mem.get_buffer(),
                    .src_offset = alloc.buffer_offset,
                    .dst_buffer = ti.uses[index_buffer].buffer(),
                    .dst_offset = 0,
                    .size = sizeof(DrawIndirectStruct),
                });
            }
        },
        .name = "clear triangle count of index buffer",
    });

    task_list.add_task(FillIndexBufferTask{
        {
            .uses = {
                .u_meshes = asset_manager->tmeshes.handle(),
                .u_instantiated_meshlets = instantiated_meshlets.handle(),
                .u_index_buffer_and_count = index_buffer.handle(),
            },
        },
        context,
    });

    task_list.add_task(DrawOpaqueIdTask{
        {
            .uses = {
                .u_visbuffer = visbuffer.handle(),
                .u_debug_image = debug_image.handle(),
                .u_depth_image = depth_handle,
                .u_draw_info_index_buffer = index_buffer.handle(),
                .u_instantiated_meshlets = instantiated_meshlets.handle(),
                .u_entity_meshlists = entity_meshlists.handle(),
                .u_entity_debug = entity_debug.handle(),
                .u_meshes = asset_manager->tmeshes.handle(),
                .u_combined_transforms = entity_combined_transforms.handle(),
            },
        },
        context->raster_pipelines[DrawOpaqueIdTask{}.name],
        context,
    });

    task_list.add_task({
        .uses = { daxa::BufferTransferWrite{instantiated_meshlet_visibility_counters} },
        .task = [=](daxa::TaskInterface ti)
        {
            ti.get_command_list().clear_buffer({
                .buffer = ti.uses[instantiated_meshlet_visibility_counters].buffer(),
                .clear_value = 0,
                .size = ti.get_device().info_buffer(ti.uses[instantiated_meshlet_visibility_counters].buffer()).size,
            });
        },
        .name = "clear instantiated meshlet counters buffer",
    });

    task_list.add_task(AnalyzeVisbufferTask{
        {
            .uses = {
                .u_visbuffer = visbuffer.handle(),
                .u_instantiated_meshlet_counters = instantiated_meshlet_visibility_counters.handle(),
            },
        },
        context->compute_pipelines[AnalyzeVisbufferTask::NAME],
        context,
    });

    task_list.add_task(WriteSwapchainTask{
        {.uses = {swapchain_image.handle(), debug_image.handle()}},
        context->compute_pipelines[WriteSwapchainTask::NAME],
    });

    task_list.submit({.additional_signal_timeline_semaphores = &submit_info.signal_timeline_semaphores});
    task_list.present({});
    task_list.complete({});
    return task_list;
}

void Renderer::render_frame(CameraInfo const &camera_info, f32 const delta_time)
{
    if (this->window->size.x == 0 || this->window->size.y == 0)
    {
        return;
    }
    auto opt = context->pipeline_manager.reload_all();
    if (opt.has_value())
    {
        std::cout << opt.value().to_string() << std::endl;
    }
    u32 const flight_frame_index = context->swapchain.get_cpu_timeline_value() % (context->swapchain.info().max_allowed_frames_in_flight + 1);

    bool const settings_changed = context->settings != context->prev_settings;
    if (settings_changed)
    {
        this->main_task_list = create_main_task_list();
    }
    
    this->context->shader_globals.globals.camera_view = *reinterpret_cast<f32mat4x4 const *>(&camera_info.view);
    this->context->shader_globals.globals.camera_projection = *reinterpret_cast<f32mat4x4 const *>(&camera_info.proj);
    this->context->shader_globals.globals.camera_view_projection = *reinterpret_cast<f32mat4x4 const *>(&camera_info.vp);
    if (context->settings.update_culling_matrix)
    {
        this->context->shader_globals.globals.cull_camera_view_projection = *reinterpret_cast<f32mat4x4 const *>(&camera_info.vp);
    }
    this->context->shader_globals.globals.frame_index = static_cast<u32>(context->swapchain.get_cpu_timeline_value());
    this->context->shader_globals.globals.delta_time = delta_time;
    this->context->shader_globals.globals.settings = this->context->settings;

    context->device.get_host_address_as<ShaderGlobalsBlock>(context->shader_globals_buffer)[flight_frame_index] = context->shader_globals;
    context->shader_globals_ptr = context->device.get_device_address(context->shader_globals_buffer) + sizeof(ShaderGlobalsBlock) * flight_frame_index;
    context->shader_globals_set_info = {
        .slot = SHADER_GLOBALS_SLOT,
        .buffer = context->shader_globals_buffer,
        .size = sizeof(ShaderGlobalsBlock),
        .offset = sizeof(ShaderGlobalsBlock) * flight_frame_index,
    };  

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
    context->prev_settings = context->settings; 
}