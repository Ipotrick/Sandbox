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
                         daxa::ImageUsageFlagBits::SHADER_STORAGE |
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
    context->settings.enable_observer = 0;
    update_settings();
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
        {DrawVisbufferTask::PIPELINE_COMPILE_INFO.name, DrawVisbufferTask::PIPELINE_COMPILE_INFO},
        #if COMPILE_IN_MESH_SHADER
        {DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER_CULL_AND_DRAW.name, DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER_CULL_AND_DRAW},
        {DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER.name, DRAW_VISBUFFER_PIPELINE_COMPILE_INFO_MESH_SHADER},
        #endif
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
    //  - clear buffers
    //  - analyze last frames visible meshlet list:
    //    - create a bitfield and lookuptable into bitfield, describing what meshlet of each mesh of each entity is visible or not.
    //    - prepopulate instantiated meshlets buffer with meshlets visible from last frame
    //  - first draw pass:
    //    - depth, visbuffer
    //    - draws instantiated meshlets, written by prepopulate pass
    //    - meshshader path:
    //      - drawTasksIndirect on the previous frames meshlet list
    //        - cull meshlets that have 0 visible triangles
    //        - write out not culled meshlets to current frame meshlet list.
    //        - cull triangles that are not set as visible
    //      - fallback path:
    //        - go over all instantiated meshlets to that point and draw them instanced non indexed
    //  - build HIZ depth
    //  - cull meshes against: hiz, frustum
    //  - second draw pass:
    //    - meshshader path:
    //      - dispatch task shaders:
    //        - cull meshlets against: hiz, frustum, whether or not they were drawn in first pass
    //        - cull triangles against: hiz, frustum
    //    - fallback path:
    //      - compute shader cull meshlets against: hiz, frustum, whether or not they were drawn in first pass
    //      - draw instanced indirect meshlets
    //  - analyze visbuffer, create list of visible meshlets from this frame
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
    task_draw_visbuffer({
        .context = context,
        .tg = task_list,
        .enable_mesh_shader = context->settings.enable_mesh_shader != 0,
        .pass = DRAW_VISBUFFER_PASS_ONE,
        .instantiated_meshlets = instantiated_meshlets,
        .meshes = asset_manager->tmeshes,
        .combined_transforms = entity_combined_transforms,
        .vis_image = visbuffer,
        .debug_image = debug_image,
        .depth_image = depth,
    });
    auto hiz = task_gen_hiz(context, task_list, depth);
    const u32vec2 hiz_size = u32vec2(context->settings.render_target_size.x / 2, context->settings.render_target_size.y / 2);
    const u32 hiz_mips = static_cast<u32>(std::ceil(std::log2(std::max(hiz_size.x, hiz_size.y))));
    auto meshlet_cull_indirect_args = task_list.create_transient_buffer({
        .size = sizeof(MeshletCullIndirectArgTable) + sizeof(MeshletCullIndirectArg) * MAX_INSTANTIATED_MESHLETS * 2,
        .name = "meshlet_cull_indirect_args",
    });
    auto cull_meshlets_commands = task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct) * 32,
        .name = "CullMeshletsCommands",
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
            .u_hiz = hiz.view({.level_count = hiz_mips}),
            .u_meshlet_cull_indirect_args = meshlet_cull_indirect_args,
            .u_cull_meshlets_commands = cull_meshlets_commands,
        });
    // When culling Meshlets, we consider 3 reasons to cull:
    // - out of frustum
    // - covered by hiz depth
    // - was drawn in first pass (all visible meshlets from last frame are drawn in first pass)
    task_cull_and_draw_visbuffer({
        .context = context,
        .tg = task_list,
        .enable_mesh_shader = context->settings.enable_mesh_shader == 1,
        .cull_meshlets_commands = cull_meshlets_commands,
        .meshlet_cull_indirect_args = meshlet_cull_indirect_args,
        .entity_meta_data = entity_meta,
        .entity_meshlists = entity_meshlists,
        .entity_combined_transforms = entity_combined_transforms,
        .meshes = asset_manager->tmeshes,
        .entity_meshlet_visibility_bitfield_offsets = entity_meshlet_visibility_bitfield_offsets,
        .entity_meshlet_visibility_bitfield_arena = entity_meshlet_visibility_bitfield_arena,
        .hiz = hiz.view({.level_count = hiz_mips}),
        .instantiated_meshlets = instantiated_meshlets,
        .vis_image = visbuffer,
        .debug_image = debug_image,
        .depth_image = depth,
    });
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
    if (context->settings.enable_observer)
    {
        task_draw_visbuffer({
            .context = context,
            .tg = task_list,
            .enable_mesh_shader = context->settings.enable_mesh_shader != 0,
            .pass = DRAW_VISBUFFER_PASS_OBSERVER,
            .instantiated_meshlets = instantiated_meshlets,
            .meshes = asset_manager->tmeshes,
            .combined_transforms = entity_combined_transforms,
            .vis_image = visbuffer,
            .debug_image = debug_image,
            .depth_image = depth,
        });
    }
    task_list.add_task(WriteSwapchainTask{
        .uses = {
            .swapchain = swapchain_image,
            .vis_image = visbuffer,
            .u_instantiated_meshlets = instantiated_meshlets,
        },
        .context = context,
    });

    task_list.submit({});
    task_list.present({});
    task_list.complete({});
    // std::cout << task_list.get_debug_string() << std::endl;
    return task_list;
}

void Renderer::update_settings()
{
    context->settings.render_target_size.x = window->size.x;
    context->settings.render_target_size.y = window->size.y;
    context->settings.render_target_size_inv = {1.0f / context->settings.render_target_size.x, 1.0f / context->settings.render_target_size.y};
}

void Renderer::render_frame(CameraInfo const &camera_info, CameraInfo const &observer_camera_info, f32 const delta_time)
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
    daxa::u32vec2 render_target_size = {static_cast<u32>(this->window->size.x), static_cast<u32>(this->window->size.y)};
    this->update_settings();
    this->context->shader_globals.globals.settings = context->settings;
    bool const settings_changed = context->settings != context->prev_settings;
    if (settings_changed)
    {
        this->main_task_list = create_main_task_list();
    }
    this->context->prev_settings = this->context->settings;

    // Set Shader Globals.
    this->context->shader_globals.globals.settings = this->context->settings;
    this->context->shader_globals.globals.frame_index = static_cast<u32>(context->swapchain.get_cpu_timeline_value());
    this->context->shader_globals.globals.delta_time = delta_time;
    this->context->shader_globals.globals.observer_camera_up = *reinterpret_cast<f32vec3 const *>(&observer_camera_info.up);
    this->context->shader_globals.globals.observer_camera_pos = *reinterpret_cast<f32vec3 const *>(&observer_camera_info.pos);
    this->context->shader_globals.globals.observer_camera_view = *reinterpret_cast<f32mat4x4 const *>(&observer_camera_info.view);
    this->context->shader_globals.globals.observer_camera_projection = *reinterpret_cast<f32mat4x4 const *>(&observer_camera_info.proj);
    this->context->shader_globals.globals.observer_camera_view_projection = *reinterpret_cast<f32mat4x4 const *>(&observer_camera_info.vp);
    {
        this->context->shader_globals.globals.camera_up = *reinterpret_cast<f32vec3 const *>(&camera_info.up);
        this->context->shader_globals.globals.camera_pos = *reinterpret_cast<f32vec3 const *>(&camera_info.pos);
        this->context->shader_globals.globals.camera_view = *reinterpret_cast<f32mat4x4 const *>(&camera_info.view);
        this->context->shader_globals.globals.camera_projection = *reinterpret_cast<f32mat4x4 const *>(&camera_info.proj);
        this->context->shader_globals.globals.camera_view_projection = *reinterpret_cast<f32mat4x4 const *>(&camera_info.vp);
        this->context->shader_globals.globals.camera_near_plane_normal = *reinterpret_cast<f32vec3 const *>(&camera_info.camera_near_plane_normal);
        this->context->shader_globals.globals.camera_right_plane_normal = *reinterpret_cast<f32vec3 const *>(&camera_info.camera_right_plane_normal);
        this->context->shader_globals.globals.camera_left_plane_normal = *reinterpret_cast<f32vec3 const *>(&camera_info.camera_left_plane_normal);
        this->context->shader_globals.globals.camera_top_plane_normal = *reinterpret_cast<f32vec3 const *>(&camera_info.camera_top_plane_normal);
        this->context->shader_globals.globals.camera_bottom_plane_normal = *reinterpret_cast<f32vec3 const *>(&camera_info.camera_bottom_plane_normal);
    }
    // Upload Shader Globals.
    context->device.get_host_address_as<ShaderGlobalsBlock>(context->shader_globals_buffer)[flight_frame_index] = context->shader_globals;
    context->shader_globals_ptr = context->device.get_device_address(context->shader_globals_buffer) + sizeof(ShaderGlobalsBlock) * flight_frame_index;
    context->shader_globals_set_info = {
        .slot = SHADER_GLOBALS_SLOT,
        .buffer = context->shader_globals_buffer,
        .size = sizeof(ShaderGlobalsBlock),
        .offset = sizeof(ShaderGlobalsBlock) * flight_frame_index,
    };

    auto swapchain_image = context->swapchain.acquire_next_image();
    if (swapchain_image.is_empty())
    {
        return;
    }
    this->swapchain_image.set_images({.images = std::array{swapchain_image}});
    instantiated_meshlets.swap_buffers(instantiated_meshlets_prev);

    if (static_cast<u32>(context->swapchain.get_cpu_timeline_value()) == 0)
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