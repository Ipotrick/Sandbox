#include "renderer.hpp"

#include "../scene/scene.inl"

#include "tasks/triangle.hpp"
#include "tasks/prefix_sum.hpp"

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
        .debug_name = "depth image",
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
        .debug_name = "Sandbox main TaskList",
    }};
    this->context->swapchain_image.t_id = task_list.create_task_image({
        .swapchain_image = true,
        .debug_name = "Sandbox main Tasklist Swapchain Task Image",
    });
    task_list.add_runtime_image(context->swapchain_image.t_id, context->swapchain_image.id);
    this->context->depth_image.t_id = task_list.create_task_image({
        .debug_name = "depth image",
    });
    this->context->globals_buffer.t_id = task_list.create_task_buffer({
        .debug_name = "Shader Globals TaskBuffer",
    });
    task_list.add_runtime_buffer(this->context->globals_buffer.t_id, this->context->globals_buffer.id);
    this->context->entity_data_buffer.t_id = task_list.create_task_buffer({
        .debug_name = "entity_data_buffer",
    });
    task_list.add_runtime_buffer(this->context->entity_data_buffer.t_id, this->context->entity_data_buffer.id);
    this->context->ent_meshlet_count_prefix_sum_buffer.t_id = task_list.create_task_buffer({
        .debug_name = "ent_meshlet_count_prefix_sum_buffer",
    });
    task_list.add_runtime_buffer(this->context->ent_meshlet_count_prefix_sum_buffer.t_id, this->context->ent_meshlet_count_prefix_sum_buffer.id);
    this->context->ent_meshlet_count_partial_sum_buffer.t_id = task_list.create_task_buffer({
        .debug_name = "ent_meshlet_count_partial_sum_buffer",
    });
    task_list.add_runtime_buffer(this->context->ent_meshlet_count_partial_sum_buffer.t_id, this->context->ent_meshlet_count_partial_sum_buffer.id);

    task_list.add_task({
        .used_buffers = {
            {this->context->globals_buffer.t_id, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
        },
        .task = [&](daxa::TaskRuntime const &runtime)
        {
            auto cmd = runtime.get_command_list();
            auto staging_buffer = runtime.get_device().create_buffer({
                .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .size = sizeof(ShaderGlobals),
                .debug_name = "ShaderGlobals staging buffer",
            });
            cmd.destroy_buffer_deferred(staging_buffer);
            *runtime.get_device().get_host_address_as<ShaderGlobals>(staging_buffer) = context->shader_globals;
            cmd.copy_buffer_to_buffer({
                .src_buffer = staging_buffer,
                .dst_buffer = this->context->globals_buffer.id,
                .size = sizeof(ShaderGlobals),
            });
        },
        .debug_name = "buffer uploads",
    });

    task_list.add_task({
        .used_buffers = {
            daxa::TaskBufferUse{this->context->entity_data_buffer.t_id, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            daxa::TaskBufferUse{this->context->ent_meshlet_count_prefix_sum_buffer.t_id, daxa::TaskBufferAccess::COMPUTE_SHADER_WRITE_ONLY},
        },
        .task = [=](daxa::TaskRuntime const &runtime)
        {
            daxa::CommandList cmd = runtime.get_command_list();
            cmd.set_pipeline(*(this->context->compute_pipelines[PREFIX_SUM_MESHLETS_PIPELINE_NAME]));
            cmd.push_constant(PrefixSumMeshletCountPush{
                .entities = context->device.get_device_address(context->entity_data_buffer.id),
                .meshes = context->device.get_device_address(asset_manager->meshes_buffer),
                .dst = context->device.get_device_address(context->ent_meshlet_count_prefix_sum_buffer.id),
            });
            cmd.dispatch((scene->entities.entity_count + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE, 1, 1);
        },
        .debug_name = std::string{PREFIX_SUM_PIPELINE_NAME},
    });

    t_prefix_sum(
        this->context,
        task_list,
        this->context->ent_meshlet_count_prefix_sum_buffer.t_id,
        this->context->ent_meshlet_count_partial_sum_buffer.t_id,
        [=]()
        {
            return std::make_tuple<u32, u32, u32>(
                PREFIX_SUM_WORKGROUP_SIZE,
                PREFIX_SUM_WORKGROUP_SIZE - 1,
                (scene->entities.entity_count + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE);
        });

    t_prefix_sum_two_pass_finalize(
        this->context,
        task_list,
        this->context->ent_meshlet_count_partial_sum_buffer.t_id,
        this->context->ent_meshlet_count_prefix_sum_buffer.t_id,
        [=]()
        {
            return scene->entities.entity_count;
        });

    t_draw_triangle({
        .task_list = task_list,
        .context = *(this->context),
        .t_swapchain_image = this->context->swapchain_image.t_id,
        .t_depth_image = this->context->depth_image.t_id,
        .t_shader_globals = this->context->globals_buffer.t_id,
    });

    task_list.submit({});
    task_list.present({});
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

    main_task_list.execute();
}