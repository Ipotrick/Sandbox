#include "renderer.hpp"

#include "../scene/scene.inl"

#include "rasterize_visbuffer/draw_visbuffer.inl"
#include "rasterize_visbuffer/cull_meshes.inl"
#include "rasterize_visbuffer/cull_meshlets.inl"
#include "rasterize_visbuffer/analyze_visbuffer.inl"
#include "rasterize_visbuffer/gen_hiz.inl"
#include "rasterize_visbuffer/prepopulate_inst_meshlets.inl"

#include "tasks/prefix_sum.inl"
#include "tasks/write_swapchain.inl"

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
    instantiated_meshlets_prev = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(InstantiatedMeshlets),
                    .name = "instantiated_meshlets_prev",
                }),
            },
        },
        .name = "instantiated_meshlets_prev",
    }};
    visible_meshlets = daxa::TaskBuffer{{
        .initial_buffers = {
            .buffers = std::array{
                context->device.create_buffer({
                    .size = sizeof(VisibleMeshletList),
                    .name = "visible_meshlets",
                }),
            },
        },
        .name = "visible_meshlets",
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
        instantiated_meshlets,
        instantiated_meshlets_prev,
        visible_meshlets};

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
                .usage = daxa::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = depth.info().name,
            },
            depth,
        },
        {
            {
                .format = daxa::Format::R32_UINT,
                .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT |
                         daxa::ImageUsageFlagBits::TRANSFER_SRC |
                         daxa::ImageUsageFlagBits::SHADER_SAMPLED,
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
                         daxa::ImageUsageFlagBits::SHADER_STORAGE |
                         daxa::ImageUsageFlagBits::SHADER_SAMPLED,
                .name = debug_image.info().name,
            },
            debug_image,
        },
    };

    recreate_framebuffer();

    compile_pipelines();

    context->settings.enable_mesh_shader = 0;
    context->settings.update_culling_information = 1;
    context->settings.render_target_x = window->size.x;
    context->settings.render_target_y = window->size.y;

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
        {DrawVisbufferTask::PIPELINE_COMPILE_INFO[0].name, DrawVisbufferTask::PIPELINE_COMPILE_INFO[0]},
        {DrawVisbufferTask::PIPELINE_COMPILE_INFO[1].name, DrawVisbufferTask::PIPELINE_COMPILE_INFO[1]},
    };
    for (auto [name, info] : rasters)
    {
        auto compilation_result = this->context->pipeline_manager.add_raster_pipeline(info);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->raster_pipelines[name] = compilation_result.value();
    }
    std::vector<std::tuple<std::string_view, daxa::ComputePipelineCompileInfo>> computes = {
        {SetEntityMeshletVisibilityBitMasksTask::NAME, SetEntityMeshletVisibilityBitMasksTask::PIPELINE_COMPILE_INFO},
        {PrepopulateInstantiatedMeshletsTask::NAME, PrepopulateInstantiatedMeshletsTask::PIPELINE_COMPILE_INFO},
        {PrepopulateInstantiatedMeshletsCommandWriteTask::NAME, PrepopulateInstantiatedMeshletsCommandWriteTask::PIPELINE_COMPILE_INFO},
        {AnalyzeVisBufferTask2::NAME, AnalyzeVisBufferTask2::PIPELINE_COMPILE_INFO},
        {GEN_HIZ_PIPELINE_COMPILE_INFO.name, GEN_HIZ_PIPELINE_COMPILE_INFO},
        {WriteSwapchainTask::NAME, WriteSwapchainTask::PIPELINE_COMPILE_INFO},
        {DrawVisbufferWriteCommandTask::NAME, DrawVisbufferWriteCommandTask::PIPELINE_COMPILE_INFO},
        {CullMeshesCommandWriteTask::NAME, CullMeshesCommandWriteTask::PIPELINE_COMPILE_INFO},
        {CullMeshesTask::NAME, CullMeshesTask::PIPELINE_COMPILE_INFO},
        {PrefixSumCommandWriteTask::NAME, PrefixSumCommandWriteTask::PIPELINE_COMPILE_INFO},
        {PrefixSumUpsweepTask::NAME, PrefixSumUpsweepTask::PIPELINE_COMPILE_INFO},
        {PrefixSumDownsweepTask::NAME, PrefixSumDownsweepTask::PIPELINE_COMPILE_INFO},
        {CullMeshletsCommandWriteTask::NAME, CullMeshletsCommandWriteTask::PIPELINE_COMPILE_INFO},
        {CullMeshletsTask::NAME, CullMeshletsTask::PIPELINE_COMPILE_INFO},
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
    task_clear_buffer(list, instantiated_meshlets, 0, sizeof(u32));
    list.use_persistent_buffer(visible_meshlets);
    task_clear_buffer(list, visible_meshlets, 0, sizeof(u32));
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
    //      - dispatch mesh cull over all active entities in scene
    //          - append 5 indirect argument buffers, each corresponding to one indirect dispatch later
    //          - each of the 5 indirect dispatches blong to a granularity of meshlet culling.
    //          - 1st buffer is at the granularity of one arg per meshlet culled
    //          - 2nd buffer is at the granularity of one arg per 8 meshlets culled
    //          - 3rd buffer is at the granularity of one arg per 64 meshlets culled
    //          - 4th buffer is at the granularity of one arg per 512 meshlets culled
    //          - 5th buffer is at the granularity of one arg per 4096 meshlets culled
    //          - each mesh cull thread can put out up to 8 indirect args in up to two indirect arg buffers.
    //          - for example: mesh of 673 meshlets would generate 1 arg in the 512 buffer and 3 args in the 64 meshlet arg buffer, leaving the last arg partially empty.
    //          - this way no mesh writes extreme amounds of data out while still only wasting up to 1/16 of threads in meshlet culling.
    //      - meshshader path:
    //          - drawTasksIndirect over the 5 indirect commands generated by the mesh culling
    //              - cull meshlets on depth and frustum.
    //              - write out surviving meshlets to meshlist.
    //              - cull triangles that are not set as visible
    //              - draw indirect on triangle id buffer
    //      - fallback path:
    //          - dispatch over the 5 indirect commands from mesh culling
    //              - do serialization loop over same mesh index, to reduce meshshader dispatches for the same mesh.
    //              - cull meshlet on depth and frustum, write survivers to meshlist
    //                  - write out surviing meshlets into draw indirect arg buffer
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
    auto entity_meshlet_visibility_bitfield_offsets = task_list.create_transient_buffer({sizeof(EntityMeshletVisibilityBitfieldOffsets) * MAX_ENTITY_COUNT + sizeof(u32), "entity_meshlet_visibility_bitfield_offsets"});
    auto entity_meshlet_visibility_bitfield_arena = task_list.create_transient_buffer({ENTITY_MESHLET_VISIBILITY_ARENA_SIZE, "entity_meshlet_visibility_bitfield_arena"});
    task_prepopulate_instantiated_meshlets(
        context,
        task_list,
        PrepopInfo{
            .meshes = asset_manager->tmeshes,
            .visible_meshlets_prev = visible_meshlets,
            .instantiated_meshlets_prev = instantiated_meshlets_prev,
            .instantiated_meshlets = instantiated_meshlets,
            .entity_meshlet_visibility_bitfield_offsets = entity_meshlet_visibility_bitfield_offsets,
            .entity_meshlet_visibility_bitfield_arena = entity_meshlet_visibility_bitfield_arena,
        }
    );
    // Draw initial triangles to the visbuffer using the previously generated meshlets and triangle lists.
    task_draw_visbuffer(
        context,
        task_list,
        DrawVisbuffer::Uses{
            .u_draw_command = {},   // Set inside
            .u_triangle_list = {},  // Unused
            .u_meshlet_list = {},   // Unused
            .u_instantiated_meshlets = instantiated_meshlets,
            .u_meshes = asset_manager->tmeshes,
            .u_combined_transforms = entity_combined_transforms,
            .u_vis_image = visbuffer,
            .u_debug_image = debug_image,
            .u_depth_image = depth,
        },
        DRAW_FIRST_PASS,
        DRAW_VISBUFFER_MESHLETS_DIRECTLY,
        DRAW_VISBUFFER_NO_DEPTH_ONLY);
    auto hiz = task_gen_hiz(context, task_list, depth);
    auto meshlet_cull_indirect_args = task_list.create_transient_buffer({
        .size = sizeof(MeshletCullIndirectArgTable) + sizeof(MeshletCullIndirectArg) * MAX_INSTANTIATED_MESHLETS * 2,
        .name = "meshlet_cull_indirect_args",
    });
    tasks_cull_meshes(
        context,
        task_list,
        {
            .u_meshes = asset_manager->tmeshes,
            .u_entity_meta = entity_meta,
            .u_entity_meshlists = entity_meshlists,
            .u_entity_transforms = entity_transforms,
            .u_entity_combined_transforms = entity_combined_transforms,
            .u_meshlet_cull_indirect_args = meshlet_cull_indirect_args,
        });
    // When culling Meshlets, we consider 3 reasons to cull:
    // - out of frustum
    // - covered by hiz depth
    // - was drawn in first pass (all visible meshlets from last frame are drawn in first pass)
    task_cull_meshlets(
        context,
        task_list,
        {
            .u_meshlet_cull_indirect_args = meshlet_cull_indirect_args,
            .u_entity_meta_data = entity_meta,
            .u_entity_meshlists = entity_meshlists,
            .u_meshes = asset_manager->tmeshes,
            .u_entity_meshlet_visibility_bitfield_offsets = entity_meshlet_visibility_bitfield_offsets,
            .u_entity_meshlet_visibility_bitfield_arena = entity_meshlet_visibility_bitfield_arena,
            .u_instantiated_meshlets = instantiated_meshlets,
        });
    task_draw_visbuffer(
        context,
        task_list,
        {
            .u_draw_command = {},  // Set inside the function in case of meshlet draw
            .u_triangle_list = {}, // Only used for triangle draw
            .u_meshlet_list = {},
            .u_instantiated_meshlets = instantiated_meshlets,
            .u_meshes = asset_manager->tmeshes,
            .u_combined_transforms = entity_combined_transforms,
            .u_vis_image = visbuffer,
            .u_debug_image = debug_image,
            .u_depth_image = depth,
        },
        DRAW_SECOND_PASS,
        DRAW_VISBUFFER_MESHLETS_DIRECTLY,
        DRAW_VISBUFFER_NO_DEPTH_ONLY);
    auto meshlet_visibility_bitfields = task_list.create_transient_buffer({
        .size = static_cast<u32>(sizeof(daxa_u32vec4) * MAX_INSTANTIATED_MESHLETS),
        .name = "meshlet_visibility_counters",
    });
    task_clear_buffer(task_list, meshlet_visibility_bitfields, 0);
    auto visible_meshlets_bitfield = task_list.create_transient_buffer({sizeof(daxa_u32) * MAX_INSTANTIATED_MESHLETS, "visible meshlets bitfield"});
    task_clear_buffer(task_list, visible_meshlets, 0, 4);
    task_clear_buffer(task_list, visible_meshlets_bitfield, 0);
    // While analyzing the visbuffer, the bitfields are written.
    // Before that can be done it needs to be cleared, as it still contains last frames contents.
    task_clear_buffer(task_list, entity_meshlet_visibility_bitfield_arena, 0);
    task_list.add_task(AnalyzeVisBufferTask2{
        .uses = {
            .u_visbuffer = visbuffer,
            .u_instantiated_meshlets = instantiated_meshlets,
            .u_meshlet_visibility_bitfield = visible_meshlets_bitfield,
            .u_visible_meshlets = visible_meshlets,
        },
        .context = context,
    });
    task_list.add_task(WriteSwapchainTask{
        .uses = {
            .swapchain = swapchain_image,
            .debug_image = debug_image,
        },
        .context = context,
    });

    task_list.submit({});
    task_list.present({});
    task_list.complete({});
    // std::cout << task_list.get_debug_string() << std::endl;
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
    context->settings.render_target_x = this->window->size.x;
    context->settings.render_target_y = this->window->size.y;

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

    instantiated_meshlets.swap_buffers(instantiated_meshlets_prev);

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