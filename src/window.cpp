#include "window.hpp"

Window::Window(i32 width, i32 height, std::string_view name)
    : size{width, height},
      name{name},
      glfw_handle{
          [=]()
          {
              glfwInit();
              return glfwCreateWindow(width, height, name.data(), nullptr, nullptr);
          }()}
{
}

Window::~Window()
{
    glfwDestroyWindow(this->glfw_handle);
    glfwTerminate();
}