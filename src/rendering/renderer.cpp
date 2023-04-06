#include "renderer.hpp"

#include "../scene/scene.inl"

#include "tasks/triangle.hpp"
#include "tasks/prefix_sum.hpp"
#include "tasks/find_visible_meshlets.hpp"
#include "tasks/generate_index_buffer.hpp"
#include "tasks/draw_opaque_ids.hpp"

Renderer::Renderer(Window *window, GPUContext *context, Scene *scene, AssetManager *asset_manager)
    : window{window},
      context{context},
      scene{scene},
      asset_manager{asset_manager},
      main_task_list{this->create_main_task_list()}
{
    recreate_resizable_images();
}

Renderer::~Renderer()
{
    this->context->device.wait_idle();
    this->context->device.collect_garbage();
}

void Renderer::compile_pipelines()
{
    {
        auto compilation_result = this->context->pipeline_manager.add_raster_pipeline(TRIANGLE_PIPELINE_INFO);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->raster_pipelines[TRIANGLE_PIPELINE_NAME] = compilation_result.value();
    }
    {
        auto compilation_result = this->context->pipeline_manager.add_compute_pipeline(PREFIX_SUM_PIPELINE_INFO);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->compute_pipelines[PREFIX_SUM_PIPELINE_NAME] = compilation_result.value();
    }
    {
        auto compilation_result = this->context->pipeline_manager.add_compute_pipeline(PREFIX_SUM_MESHLETS_PIPELINE_INFO);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->compute_pipelines[PREFIX_SUM_MESHLETS_PIPELINE_NAME] = compilation_result.value();
    }
    {
        auto compilation_result = this->context->pipeline_manager.add_compute_pipeline(PREFIX_SUM_TWO_PASS_FINALIZE_PIPELINE_INFO);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->compute_pipelines[PREFIX_SUM_TWO_PASS_FINALIZE_PIPELINE_NAME] = compilation_result.value();
    }
    {
        auto compilation_result = this->context->pipeline_manager.add_compute_pipeline(FIND_VISIBLE_MESHLETS_PIPELINE_INFO);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->compute_pipelines[FIND_VISIBLE_MESHLETS_PIPELINE_NAME] = compilation_result.value();
    }
    {
        auto compilation_result = this->context->pipeline_manager.add_compute_pipeline(GENERATE_INDEX_BUFFER_PIPELINE_INFO);
        std::cout << compilation_result.to_string() << std::endl;
        this->context->compute_pipelines[GENERATE_INDEX_BUFFER_PIPELINE_NAME] = compilation_result.value();
    }
}

void Renderer::recreate_resizable_images()
{
    if (!this->context->depth_image.id.is_empty())
    {
        this->main_task_list.remove_runtime_image(this->context->depth_image.t_id, this->context->depth_image.id);
        this->context->device.destroy_image(this->context->depth_image.id);
    }
    this->context->depth_image.id = this->context->device.create_image({
        .format = daxa::Format::D32_SFLOAT,
        .aspect = daxa::ImageAspectFlagBits::DEPTH,
        .size = {this->window->get_width(), this->window->get_height(), 1},
        .usage = daxa::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT | daxa::ImageUsageFlagBits::SHADER_READ_ONLY,
        .name = "depth image",
    });
    this->main_task_list.add_runtime_image(this->context->depth_image.t_id, this->context->depth_image.id);
}

void Renderer::window_resized()
{
    if (this->window->size.x == 0 || this->window->size.y == 0)
    {
        return;
    }
    recreate_resizable_images();
    this->context->swapchain.resize();
}

auto Renderer::create_main_task_list() -> daxa::TaskList
{
    using namespace daxa;
    TaskList task_list{{
        .device = this->context->device,
        .swapchain = this->context->swapchain,
        .name = "Sandbox main TaskList",
    }};
    this->context->swapchain_image.t_id = task_list.create_task_image({
        .swapchain_image = true,
        .name = "swapchain",
    });
    task_list.add_runtime_image(context->swapchain_image.t_id, context->swapchain_image.id);
    this->context->depth_image.t_id = task_list.create_task_image({
        .name = "depth",
    });
    this->context->globals_buffer.t_id = task_list.create_task_buffer({
        .name = "globals",
    });
    task_list.add_runtime_buffer(this->context->globals_buffer.t_id, this->context->globals_buffer.id);

    this->context->entity_meta_data.t_id = task_list.create_task_buffer({
        .name = "entity_meta_data",
    });
    task_list.add_runtime_buffer(this->context->entity_meta_data.t_id, this->context->entity_meta_data.id);
    this->context->entity_transforms.t_id = task_list.create_task_buffer({
        .name = "entity_transforms",
    });
    task_list.add_runtime_buffer(this->context->entity_transforms.t_id, this->context->entity_transforms.id);
    this->context->entity_combined_transforms.t_id = task_list.create_task_buffer({
        .name = "entity_combined_transforms",
    });
    task_list.add_runtime_buffer(this->context->entity_combined_transforms.t_id, this->context->entity_combined_transforms.id);
    this->context->entity_first_children.t_id = task_list.create_task_buffer({
        .name = "entity_first_children",
    });
    task_list.add_runtime_buffer(this->context->entity_first_children.t_id, this->context->entity_first_children.id);
    this->context->entity_next_silbings.t_id = task_list.create_task_buffer({
        .name = "entity_next_silbings",
    });
    task_list.add_runtime_buffer(this->context->entity_next_silbings.t_id, this->context->entity_next_silbings.id);
    this->context->entity_parents.t_id = task_list.create_task_buffer({
        .name = "entity_parents",
    });
    task_list.add_runtime_buffer(this->context->entity_parents.t_id, this->context->entity_parents.id);
    this->context->entity_meshlists.t_id = task_list.create_task_buffer({
        .name = "entity_meshes",
    });
    task_list.add_runtime_buffer(this->context->entity_meshlists.t_id, this->context->entity_meshlists.id);

    this->context->ent_meshlet_count_prefix_sum_buffer.t_id = task_list.create_task_buffer({
        .name = "ent_meshlet_count_prefix_sum_buffer",
    });
    task_list.add_runtime_buffer(this->context->ent_meshlet_count_prefix_sum_buffer.t_id, this->context->ent_meshlet_count_prefix_sum_buffer.id);
    this->context->ent_meshlet_count_partial_sum_buffer.t_id = task_list.create_task_buffer({
        .name = "ent_meshlet_count_partial_sum_buffer",
    });
    task_list.add_runtime_buffer(this->context->ent_meshlet_count_partial_sum_buffer.t_id, this->context->ent_meshlet_count_partial_sum_buffer.id);
    auto meshes_buffer_tid = task_list.create_task_buffer({
        .name = "meshes_buffer_tid",
    });
    task_list.add_runtime_buffer(meshes_buffer_tid, this->asset_manager->meshes_buffer);
    this->context->instanciated_meshlets.t_id = task_list.create_task_buffer({
        .name = "instanciated_meshlets",
    });
    task_list.add_runtime_buffer(this->context->instanciated_meshlets.t_id, this->context->instanciated_meshlets.id);
    this->context->index_buffer.t_id = task_list.create_task_buffer({
        .name = "index buffer",
    });
    task_list.add_runtime_buffer(this->context->index_buffer.t_id, this->context->index_buffer.id);

    task_list.add_task({
        .used_buffers = {
            {this->context->globals_buffer.t_id, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
        },
        .task = [&](daxa::TaskRuntimeInterface const &runtime)
        {
            auto cmd = runtime.get_command_list();
            auto staging_buffer = runtime.get_device().create_buffer({
                .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .size = sizeof(ShaderGlobals),
                .name = "ShaderGlobals staging buffer",
            });
            cmd.destroy_buffer_deferred(staging_buffer);
            *runtime.get_device().get_host_address_as<ShaderGlobals>(staging_buffer) = context->shader_globals;
            cmd.copy_buffer_to_buffer({
                .src_buffer = staging_buffer,
                .dst_buffer = this->context->globals_buffer.id,
                .size = sizeof(ShaderGlobals),
            });
        },
        .name = "buffer uploads",
    });

    task_list.add_task({
        .used_buffers = {
            daxa::TaskBufferUse{this->context->entity_meta_data.t_id, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            daxa::TaskBufferUse{this->context->entity_meshlists.t_id, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            daxa::TaskBufferUse{this->context->ent_meshlet_count_prefix_sum_buffer.t_id, daxa::TaskBufferAccess::COMPUTE_SHADER_WRITE_ONLY},
        },
        .task = [=](daxa::TaskRuntimeInterface const &runtime)
        {
            daxa::CommandList cmd = runtime.get_command_list();
            cmd.set_pipeline(*(this->context->compute_pipelines[PREFIX_SUM_MESHLETS_PIPELINE_NAME]));
            cmd.push_constant(PrefixSumMeshletCountPush{
                .entity_meta_data = context->device.get_device_address(context->entity_meta_data.id),
                .entity_meshlists = context->device.get_device_address(context->entity_meshlists.id),
                .meshes = context->device.get_device_address(asset_manager->meshes_buffer),
                .dst = context->device.get_device_address(context->ent_meshlet_count_prefix_sum_buffer.id),
            });
            cmd.dispatch((scene->entity_meta.entity_count + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE, 1, 1);
        },
        .name = std::string{PREFIX_SUM_PIPELINE_NAME},
    });

    t_prefix_sum(
        this->context,
        task_list,
        this->context->ent_meshlet_count_prefix_sum_buffer.t_id,
        this->context->ent_meshlet_count_partial_sum_buffer.t_id,
        [=]() -> std::tuple<u32, u32, u32>
        {
            return std::make_tuple<u32, u32, u32>(
                PREFIX_SUM_WORKGROUP_SIZE,
                PREFIX_SUM_WORKGROUP_SIZE - 1,
                (scene->entity_meta.entity_count + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE);
        });

    t_prefix_sum_two_pass_finalize(
        this->context,
        task_list,
        this->context->ent_meshlet_count_partial_sum_buffer.t_id,
        this->context->ent_meshlet_count_prefix_sum_buffer.t_id,
        [=]() -> u32
        {
            return scene->entity_meta.entity_count;
        });

    t_find_visible_meshlets(
        this->context,
        task_list,
        this->context->ent_meshlet_count_prefix_sum_buffer.t_id,
        this->context->entity_meta_data.t_id,
        this->context->entity_meshlists.t_id,
        meshes_buffer_tid,
        this->context->instanciated_meshlets.t_id,
        [=]()
        {
            return this->asset_manager->total_meshlet_count;
        });

    task_list.add_task({
        .used_buffers = {
            daxa::TaskBufferUse{this->context->index_buffer.t_id, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
        },
        .task = [=](daxa::TaskRuntimeInterface const &runtime)
        {
            daxa::CommandList cmd = runtime.get_command_list();
            auto alloc = this->context->transient_mem.allocate(sizeof(u32)).value();
            *reinterpret_cast<u32 *>(alloc.host_address) = 0;
            cmd.copy_buffer_to_buffer({
                .src_buffer = this->context->transient_mem.get_buffer(),
                .src_offset = alloc.buffer_offset,
                .dst_buffer = runtime.get_buffers(this->context->index_buffer.t_id)[0],
                .dst_offset = 0,
                .size = sizeof(u32),
            });
        },
        .name = "clear triangle count of index buffer",
    });

    t_generate_index_buffer(
        this->context,
        task_list,
        meshes_buffer_tid,
        this->context->instanciated_meshlets.t_id,
        this->context->index_buffer.t_id,
        [=]()
        { return this->asset_manager->total_meshlet_count; });

    t_draw_triangle({
        .task_list = task_list,
        .context = *(this->context),
        .t_swapchain_image = this->context->swapchain_image.t_id,
        .t_depth_image = this->context->depth_image.t_id,
        .t_shader_globals = this->context->globals_buffer.t_id,
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

    main_task_list.remove_runtime_image(context->swapchain_image.t_id, context->swapchain_image.id);
    context->swapchain_image.id = context->swapchain.acquire_next_image();
    if (context->swapchain_image.id.is_empty())
    {
        return;
    }
    main_task_list.add_runtime_image(context->swapchain_image.t_id, context->swapchain_image.id);

    this->submit_info = {};
    this->submit_info.signal_timeline_semaphores = {
        {this->context->transient_mem.get_timeline_semaphore(), this->context->transient_mem.timeline_value()},
    };
    main_task_list.execute({});
}