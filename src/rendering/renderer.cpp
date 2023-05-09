#include "renderer.hpp"

#include "../scene/scene.inl"

#include "rasterize_visbuffer/filter_visible_meshlets.inl"

#include "tasks/misc.hpp"
#include "tasks/allocate_meshlet_visibility_bitfields.inl"
#include "tasks/fill_meshlet_buffer.inl"
#include "tasks/write_draw_opaque_index_buffer.inl"
#include "tasks/patch_draw_opaque_indirect.inl"
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
    entity_visibility_bitfield_offsets = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(EntityVisibilityBitfieldOffsets) * MAX_ENTITY_COUNT,
                    .name = "entity_visibility_bitfield_offsets",
                }),
            },
        },
        .name = "entity_visibility_bitfield_offsets",
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
                    .size = sizeof(InstantiatedMeshlets),
                    .name = "instantiated_meshlets",
                }),
            },
        },
        .name = "instantiated_meshlets",
    }};
    initial_pass_triangles = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(TriangleDrawList),
                    .name = "initial_pass_triangles",
                }),
            },
        },
        .name = "initial_pass_triangles",
    }};
    instantiated_meshlet_visibility_counters = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(daxa_u32) * MAX_INSTANTIATED_MESHLETS,
                    .name = "instantiated_meshlet_visibility_counters",
                }),
            },
        },
        .name = "instantiated_meshlet_visibility_counters",
    }};
    instantiated_meshlets_last_frame = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(InstantiatedMeshlet) * MAX_INSTANTIATED_MESHLETS +  + 2 * INDIRECT_COMMAND_BYTE_SIZE,
                    .name = "instantiated_meshlets_last_frame",
                }),
            },
        },
        .name = "instantiated_meshlets_last_frame",
    }};
    meshlet_visibility_bitfield = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = VISIBLE_ENTITY_MESHLETS_BITFIELD_SCRATCH + /*reserved space for atomic back counter*/ 32,
                    .name = "meshlet_visibility_bitfield",
                }),
            },
        },
        .name = "meshlet_visibility_bitfield",
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
        instantiated_meshlets_last_frame,
        entity_visibility_bitfield_offsets,
        meshlet_visibility_bitfield,
        entity_debug,
        instantiated_meshlets,
        initial_pass_triangles,
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
                         daxa::ImageUsageFlagBits::TRANSFER_SRC |
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
    context->settings.update_culling_information = 1;

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
        {WriteDrawOpaqueIndexBufferTask::NAME, WRITE_DRAW_OPAQUE_INDEX_BUFFER_PIPELINE_INFO},
        {PatchDrawOpaqueIndirectTask::NAME, PATCH_DRAW_OPAQUE_INDIRECT_PIPELINE_INFO},
        {WriteSwapchainTask::NAME, WRITE_SWAPCHAIN_PIPELINE_INFO},
        {AnalyzeVisbufferTask::NAME, ANALYZE_VISBUFFER_PIPELINE_INFO},
        {AllocateMeshletVisibilityTask::NAME, ALLOCATE_MESHLET_VISIBILITY_PIPELINE_INFO},
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
    //  - first draw pass:
    //      - meshshader path:
    //          - drawTasksIndirect on the previous frames meshlet list
    //              - cull meshlets that have 0 visible triangles
    //              - write out not culled meshlets to current frame meshlet list.
    //              - cull triangles that are not set as visible
    //      - fallback path:
    //          - dispatch compute shader on old frames meshlet list
    //              - cull meshlets that have 0 visible triangles
    //              - write out not culled meshlets to current frame meshlet list.
    //              - (potentially) write out triangle buffer for culled triangle compacted draw
    //          - draw indirect on new meshlist (or poentially triangle list)
    //  - build HIZ depth
    //  - second draw pass:
    //      - meshshader path:
    //          - drawTasksIndirectCount with one draw per instance
    //              - cull instances on depth and frustum, dispatch n meshshaders each meshlet in surviving instances
    //              - cull meshlets on depth and frustum.
    //              - write out surviving meshlets to meshlist.
    //              - cull triangles that are not set as visible
    //      - fallback path:
    //          - dispatch compute shader on on all instances, build compact buffer of surviving instances
    //          - dispatch on surviving instances, build prefix sum on meshlet count.
    //          - dispatch for each meshlet, binary search meshlet identiy, cull meshlet on depth and frustum, write survivers to meshlist
    //          - dispatch for each surviving meshlet, cull triangles on depth and frustum, write out trangle id buffer
    //          - draw indirect on triangle id buffer
    //  - analyze visbuffer
    //      - set meshlet triangle visibility bitmasks 
    //  - blit debug image to swapchain
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

    // Using the last frames visbuffer and meshlet visibility bitmasks, filter the visible meshlets into a list.
    // This list of meshlets will be written to the list of instantiated meshlets of the current frame.
    task_list.add_task(FilterVisibleMeshlets{
        {.uses={
            .u_instantiated_meshlets = instantiated_meshlets_last_frame.handle(),
            .u_meshlet_visibility_bitmasks = meshlet_visibility_bitfield.handle(),
            .u_filtered_meshlets = instantiated_meshlets.handle(),
            .u_filtered_triangles = initial_pass_triangles.handle(),
        }},
        .context = context,
    });

    if (!context->settings.enable_mesh_shader)
    {
        task_list.add_task(BuildInitialMeshletList{

        })
    }

    auto draw_opaque_indirect_command_buffer = task_list.create_transient_buffer({
        .size = static_cast<u32>(std::max(sizeof(DrawIndexedIndirectStruct), sizeof(DrawIndirectStruct))),
        .name = "draw_opaque_indirect_command_buffer",
    });
    
    task_list.add_task(PatchDrawOpaqueIndirectTask{
        {.uses = { instantiated_meshlets.handle() }},
        .context = context,
    });

    if (context->settings.indexed_id_rendering)
    {
        task_list.add_task({
            .uses = {
                daxa::BufferTransferWrite{index_buffer},
            },
            .task = [=](daxa::TaskInterface ti)
            {
                daxa::CommandList cmd = ti.get_command_list();
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
            },
            .name = "clear triangle count of index buffer",
        });
        // First pass id rendering:
        task_list.add_task(WriteDrawOpaqueIndexBufferTask{
            {
                .uses = {
                    .u_meshes = asset_manager->tmeshes.handle(),
                    .u_meshlet_list = instantiated_meshlets.handle(),
                    .u_index_buffer_and_count = index_buffer.handle(),
                },
            },
            context,
        });
    }

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
        /*pass:*/ 0 ,
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

    task_list.add_task(FillMeshletBufferTask{
        {
            .uses = {
                .u_prefix_sum_mehslet_counts = ent_meshlet_count_prefix_sum_buffer.handle(),
                .u_entity_meta_data = entity_meta.handle(),
                .u_entity_meshlists = entity_meshlists.handle(),
                .u_entity_visibility_bitfield_offsets = entity_visibility_bitfield_offsets.handle(),
                .u_meshlet_visibility_bitfield = meshlet_visibility_bitfield.handle(),
                .u_meshes = asset_manager->tmeshes.handle(),
                .u_instantiated_meshlets = instantiated_meshlets.handle(),
            },
        },
        context->compute_pipelines[FillMeshletBufferTask::NAME],
        context,
        &this->asset_manager->total_meshlet_count,
        .cull_alredy_visible_meshlets = true,
    });

    if (context->settings.indexed_id_rendering)
    {
        task_list.add_task({
            .uses = {
                daxa::BufferTransferWrite{index_buffer},
            },
            .task = [=](daxa::TaskInterface ti)
            {
                daxa::CommandList cmd = ti.get_command_list();
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
            },
            .name = "clear triangle count of index buffer",
        });

        task_list.add_task(WriteDrawOpaqueIndexBufferTask{
            {
                .uses = {
                    .u_meshes = asset_manager->tmeshes.handle(),
                    .u_meshlet_list = instantiated_meshlets.handle(),
                    .u_index_buffer_and_count = index_buffer.handle(),
                },
            },
            context,
        });
    }

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
        /*pass:*/ 1,
    });

    if (context->settings.update_culling_information)
    {
        // TODO: replace with compute indirect clear, that only clears the dirty part of the buffers.
        task_list.add_task({
            .uses = { daxa::BufferTransferWrite{instantiated_meshlet_visibility_counters} },
            .task = [=](daxa::TaskInterface ti)
            {
                ti.get_command_list().clear_buffer({
                    .buffer = ti.uses[instantiated_meshlet_visibility_counters].buffer(),
                    .size = ti.get_device().info_buffer(ti.uses[instantiated_meshlet_visibility_counters].buffer()).size,
                    .clear_value = 0,
                });
            },
            .name = "clear instantiated meshlet counters buffer",
        });
        // TODO: replace with compute indirect clear, that only clears the dirty part of the buffers.
        task_list.add_task({
            .uses = { 
                daxa::BufferTransferWrite{entity_visibility_bitfield_offsets},
                daxa::BufferTransferWrite{meshlet_visibility_bitfield},
            },
            .task = [=](daxa::TaskInterface ti)
            {
                auto cmd = ti.get_command_list();
                cmd.clear_buffer({
                    .buffer = ti.uses[entity_visibility_bitfield_offsets].buffer(),
                    .size = ti.get_device().info_buffer(ti.uses[entity_visibility_bitfield_offsets].buffer()).size,
                    .clear_value = 0,
                });
                cmd.clear_buffer({
                    .buffer = ti.uses[meshlet_visibility_bitfield].buffer(),
                    .size = ti.get_device().info_buffer(ti.uses[meshlet_visibility_bitfield].buffer()).size,
                    .clear_value = 0,
                });
            },
            .name = "clear entity_visibility_bitfield_offsets and meshlet_visibility_bitfield",
        });

        task_list.add_task(ClearInstantiatedMeshletsHeaderTask{
            .uses = {instantiated_meshlets_last_frame.handle()},
            .context = context,
        });

        task_list.add_task(AllocateMeshletVisibilityTask{
            {
                .uses = {
                    .u_meshlists = entity_meshlists.handle(),
                    .u_meshes = asset_manager->tmeshes.handle(),
                    .u_entity_meta = entity_meta.handle(),
                    .u_visibility_bitfield_sratch = meshlet_visibility_bitfield.handle(),
                    .u_meshlet_visibilities = entity_visibility_bitfield_offsets.handle(),
                },
            },
            context,
            scene,
            context->compute_pipelines[AllocateMeshletVisibilityTask::NAME],
        });
    }

    task_list.add_task(AnalyzeVisbufferTask{
        {
            .uses = {
                .u_visbuffer = visbuffer.handle(),
                .u_instantiated_meshlets = instantiated_meshlets.handle(),
                .u_entity_visibility_bitfield_offsets = entity_visibility_bitfield_offsets.handle(),
                .u_instantiated_meshlet_counters = instantiated_meshlet_visibility_counters.handle(),
                .u_meshlet_visibility_bitfield = meshlet_visibility_bitfield.handle(),
                .u_instantiated_meshlets_last_frame = instantiated_meshlets_last_frame.handle(),
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
    if (context->settings.update_culling_information)
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

    if(this->context->settings.update_culling_information != 0)
    {
        // visible meshlets from last frame become the first instantiated meshlets of the current frame.
        this->instantiated_meshlets_last_frame.swap_buffers(this->instantiated_meshlets);
    }

    this->submit_info = {};
    this->submit_info.signal_timeline_semaphores = {
        {this->context->transient_mem.get_timeline_semaphore(), this->context->transient_mem.timeline_value()},
    };
    main_task_list.execute({});
    context->prev_settings = context->settings; 
}