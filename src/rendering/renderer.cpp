#include "renderer.hpp"

#include "../scene/scene.inl"

#include "rasterize_visbuffer/filter_visible_meshlets.inl"
#include "rasterize_visbuffer/draw_visbuffer.inl"
#include "rasterize_visbuffer/cull_meshes.inl"
#include "rasterize_visbuffer/cull_meshlets.inl"

#include "tasks/prefix_sum.inl"

// #include "tasks/misc.hpp"
// #include "tasks/allocate_meshlet_visibility_bitfields.inl"
// #include "tasks/fill_meshlet_buffer.inl"
// #include "tasks/write_draw_opaque_index_buffer.inl"
// #include "tasks/patch_draw_opaque_indirect.inl"
// #include "tasks/analyze_visbuffer.inl"
// #include "tasks/prefix_sum.inl"
// #include "tasks/draw_opaque_ids.inl"
// #include "tasks/write_swapchain.inl"

Renderer::Renderer(Window *window, GPUContext *context, Scene *scene, AssetManager *asset_manager)
    : window{window},
      context{context},
      scene{scene},
      asset_manager{asset_manager}
{
    zero_buffer = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(daxa_u32),
                    .name = "zero_buffer",
                }),
            },
        },
        .name = "zero_buffer",
    }};
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
    entity_visibility_bitfield = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(daxa_u32vec4) * VISIBLE_ENTITY_MESHLETS_BITFIELD_SCRATCH,
                    .name = "entity_visibility_bitfield",
                }),
            },
        },
        .name = "entity_visibility_bitfield",
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

    mesh_draw_list = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(MeshDrawList),
                    .name = "mesh_draw_list",
                }),
            },
        },
        .name = "mesh_draw_list",
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
    instantiated_meshlets_last_frame = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(InstantiatedMeshlets),
                    .name = "instantiated_meshlets_last_frame",
                }),
            },
        },
        .name = "instantiated_meshlets_last_frame",
    }};
    triangle_draw_list = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(TriangleDrawList),
                    .name = "triangle_draw_list",
                }),
            },
        },
        .name = "triangle_draw_list",
    }};

    buffers = {
        zero_buffer,
        entity_meta,
        entity_transforms,
        entity_combined_transforms,
        entity_first_children,
        entity_next_silbings,
        entity_parents,
        entity_meshlists,
        entity_visibility_bitfield_offsets,
        entity_visibility_bitfield,
        entity_debug,
        instantiated_meshlets_last_frame,
        initial_pass_triangles,
        mesh_draw_list,
        instantiated_meshlets,
        triangle_draw_list};

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

    context->settings.enable_mesh_shader = 0;
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
        {DrawVisbuffer::NAME, DrawVisbuffer::PIPELINE_COMPILE_INFO},
    };
    for (auto [name, info] : rasters)
    {
        auto compilation_result = this->context->pipeline_manager.add_raster_pipeline(info);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->raster_pipelines[name] = compilation_result.value();
    }
    std::vector<std::tuple<std::string_view, daxa::ComputePipelineCompileInfo>> computes = {
        {FilterVisibleMeshletsCommandWrite::NAME, FilterVisibleMeshletsCommandWrite::PIPELINE_COMPILE_INFO},
        {FilterVisibleMeshlets::NAME, FilterVisibleMeshlets::PIPELINE_COMPILE_INFO},
        {CullMeshesCommandWrite::NAME, CullMeshesCommandWrite::PIPELINE_COMPILE_INFO},
        {CullMeshes::NAME, CullMeshes::PIPELINE_COMPILE_INFO},
        {PrefixSumCommandWrite::NAME, PrefixSumCommandWrite::PIPELINE_COMPILE_INFO},
        {PrefixSumUpsweep::NAME, PrefixSumUpsweep::PIPELINE_COMPILE_INFO},
        {PrefixSumDownsweep::NAME, PrefixSumDownsweep::PIPELINE_COMPILE_INFO},
        {CullMeshletsCommandWrite::NAME, CullMeshletsCommandWrite::PIPELINE_COMPILE_INFO},
        {CullMeshlets::NAME, CullMeshlets::PIPELINE_COMPILE_INFO},
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

void Renderer::clear_select_buffers()
{
    using namespace daxa;
    TaskGraph list{{
        .device = this->context->device,
        .swapchain = this->context->swapchain,
        .name = "clear task list",
    }};
    list.use_persistent_buffer(instantiated_meshlets);
    list.use_persistent_buffer(instantiated_meshlets_last_frame);
    list.add_task({
        .uses = {
            BufferTransferWrite{instantiated_meshlets},
            BufferTransferWrite{instantiated_meshlets_last_frame},
        },
        .task = [=](TaskInterface ti)
        {
            auto cmd = ti.get_command_list();
            cmd.clear_buffer({ti.uses[instantiated_meshlets].buffer(), 0, sizeof(u32)*2, 0});
            cmd.clear_buffer({ti.uses[instantiated_meshlets_last_frame].buffer(), 0, sizeof(u32)*2, 0});
        }
    });
    list.submit({});
    list.complete({});
    list.execute({});
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

auto Renderer::create_main_task_list() -> daxa::TaskGraph
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
    TaskGraph task_list{{
        .device = this->context->device,
        .swapchain = this->context->swapchain,
        .name = "Sandbox main TaskGraph",
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

    // Using the last frames visbuffer and meshlet visibility bitmasks, filter the visible meshlets into a list.
    // This list of meshlets will be written to the list of instantiated meshlets of the current frame.
    task_filter_visible_meshlets(
        context,
        task_list,
        {
            .u_entity_visibility_bitfield_offsets_prev = entity_visibility_bitfield_offsets,
            .u_entity_visibility_bitfield_prev = entity_visibility_bitfield,
            .u_instantiated_meshlets_prev = instantiated_meshlets_last_frame,
            .u_instantiated_meshlets = instantiated_meshlets,
            .u_triangle_draw_list = triangle_draw_list,
        });
    // Draw initial triangles to the visbuffer using the previously generated meshlets and triangle lists.
    task_list.add_task(DrawVisbuffer{
        {.uses = {
             .u_draw_command = initial_pass_triangles,
             .u_instantiated_meshlets = instantiated_meshlets,
             .u_meshes = asset_manager->tmeshes,
             .u_vis_image = visbuffer,
             .u_debug_image = debug_image,
             .u_depth_image = depth,
         }},
        .context = context,
        .clear_attachments = true,
        .tris_or_meshlets = DRAW_VISBUFFER_TRIANGLES,
    });
    // After the visible triangles of the last frame are drawn, we must test if something else became visible between frames.
    // For that we need a hiz depth map to cull meshes, meshlets and triangles efficiently.
    // TODO: build hiz
    // Cull meshes
    tasks_cull_meshes(
        context,
        task_list,
        {
            .u_meshes = asset_manager->tmeshes,
            .u_entity_meta = entity_meta,
            .u_entity_meshlists = entity_meshlists,
            .u_entity_transforms = entity_transforms,
            .u_entity_combined_transforms = entity_combined_transforms,
            .u_mesh_draw_list = mesh_draw_list,
        });
    // For the non mesh shader path we now need to build a prefix sum over the count of meshlets of surviving meshes.
    task_prefix_sum(PrefixSumTaskGroupInfo{
        .context = context,
        .task_list = task_list,
        .value_count = entity_meta,
        .value_count_uint_offset = offsetof(EntityMetaData, entity_count) / sizeof(u32),
        .values = mesh_draw_list,
        .src_uint_offset = offsetof(MeshDrawList, mesh_dispatch_indirects) / sizeof(u32),
        .src_uint_stride = sizeof(DispatchIndirectStruct) / sizeof(u32),
    });
    task_cull_meshlets(
        context,
        task_list,
        {
            .u_mesh_draw_list = mesh_draw_list,
            .u_entity_meta_data = entity_meta,
            .u_entity_meshlists = entity_meshlists,
            .u_entity_visibility_bitfield_offsets = entity_visibility_bitfield_offsets,
            .u_meshlet_visibility_bitfield = entity_visibility_bitfield,
            .u_meshes = asset_manager->tmeshes,
            .u_instantiated_meshlets = instantiated_meshlets,
        });
    auto second_visbuffer_draw_command = task_list.create_transient_buffer({
        .size = sizeof(DrawIndirectStruct),
        .name = "second_visbuffer_draw_command",
    });
    task_list.add_task({
        .uses = {
            BufferTransferWrite{second_visbuffer_draw_command},
            BufferTransferRead{instantiated_meshlets},
        },
        .task = [=](daxa::TaskInterface ti)
        {
            auto ab = ti.get_allocator().get_buffer();
            auto cmd = ti.get_command_list();
            auto alloc0 = ti.get_allocator().allocate(sizeof(u32)).value();
            *reinterpret_cast<u32 *>(alloc0.host_address) = MAX_TRIANGLES_PER_MESHLET * 3;
            auto alloc1 = ti.get_allocator().allocate(sizeof(daxa_u32vec2)).value();
            *reinterpret_cast<daxa_u32vec2 *>(alloc1.host_address) = daxa_u32vec2(0, 0);
            cmd.copy_buffer_to_buffer({ab, alloc0.buffer_offset, ti.uses[second_visbuffer_draw_command].buffer(), 0, sizeof(u32)});
            cmd.copy_buffer_to_buffer({ab, alloc1.buffer_offset, ti.uses[second_visbuffer_draw_command].buffer(), offsetof(DrawIndirectStruct, first_vertex), sizeof(u32)});
            cmd.copy_buffer_to_buffer({
                ti.uses[instantiated_meshlets].buffer(),
                offsetof(InstantiatedMeshlets, second_pass_count),
                ti.uses[second_visbuffer_draw_command].buffer(),
                offsetof(DrawIndirectStruct, instance_count),
                .size = static_cast<u32>(sizeof(u32)),
            });
        },
        .name = "write second_visbuffer_draw_command",
    });
    task_list.add_task(DrawVisbuffer{
        {.uses = {
             .u_draw_command = second_visbuffer_draw_command,
             .u_instantiated_meshlets = instantiated_meshlets,
             .u_meshes = asset_manager->tmeshes,
             .u_vis_image = visbuffer,
             .u_debug_image = debug_image,
             .u_depth_image = depth,
         }},
        .context = context,
        .clear_attachments = false,
        .tris_or_meshlets = DRAW_VISBUFFER_MESHLETS,
    });
    // TODO: analyze visbuffer

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
    auto reloaded_result = context->pipeline_manager.reload_all();
    if (auto reload_err = std::get_if<daxa::PipelineReloadError>(&reloaded_result))
    {
        std::cout << "Failed to reload " << reload_err->message << '\n';
    }
    if (auto _ = std::get_if<daxa::PipelineReloadSuccess>(&reloaded_result))
    {
        std::cout << "Successfully reloaded!\n";
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

    if (this->context->settings.update_culling_information != 0)
    {
        // visible meshlets from last frame become the first instantiated meshlets of the current frame.
        this->instantiated_meshlets_last_frame.swap_buffers(this->instantiated_meshlets);
    }

    if (this->context->shader_globals.globals.frame_index == 0)
    {
        clear_select_buffers();
    }

    this->submit_info = {};
    this->submit_info.signal_timeline_semaphores = {
        {this->context->transient_mem.get_timeline_semaphore(), this->context->transient_mem.timeline_value()},
    };
    main_task_list.execute({});
    context->prev_settings = context->settings;
}