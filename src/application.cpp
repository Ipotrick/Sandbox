#include "application.hpp"

Application::Application()
    : window{ 400, 300, "sandbox" }
    , renderer{ window }
{
    std::cout << "Application::Application" << std::endl;
}

auto Application::run() -> i32
{
    std::cout << "Application::run" << std::endl;
    while(keep_running)
    {
        glfwPollEvents();
        glfwGetWindowSize(this->window.glfw_handle, &window.size.x, &window.size.y);
        keep_running &= !static_cast<bool>(glfwWindowShouldClose(this->window.glfw_handle));
    }
    return 0;
}

Application::~Application()
{
    std::cout << "Application::~Application" << std::endl;
}