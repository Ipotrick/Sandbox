#include "application.hpp"

Application::Application()
    : window{ 400, 300, "sandbox" }
    , renderer{ window }
    , asset_manager{ renderer.context.device }
{
    std::cout << "Application::Application" << std::endl;
    this->renderer.compile_pipelines();
    this->scene_loader = SceneLoader{ "./assets/" };
    scene_loader.load_entities_from_fbx(this->scene, this->asset_manager, "Bistro_v5_2/BistroExterior.fbx");
}

auto Application::run() -> i32
{
    std::cout << "Application::run" << std::endl;
    while(keep_running)
    {
        std::cout << "Application::run::loop" << std::endl;
        glfwPollEvents();
        i32vec2 new_window_size;
        glfwGetWindowSize(this->window.glfw_handle, &new_window_size.x, &new_window_size.y);
        if (this->window.size.x != new_window_size.x || this->window.size.y != new_window_size.y)
        {
            this->window.size = new_window_size;
            renderer.window_resized(this->window);
        }
        keep_running &= !static_cast<bool>(glfwWindowShouldClose(this->window.glfw_handle));

        this->update();
        this->renderer.render_frame(this->window);
    }
    return 0;
}

void Application::update()
{

}

Application::~Application()
{
    std::cout << "Application::~Application" << std::endl;
}