#pragma once

// Standart headers:
// Library headers:
// Project headers:
#include "sandbox.hpp"

struct Window
{
    Window(i32 width, i32 height, std::string_view name);
    ~Window();
    
    i32vec2 size;
    std::string_view name;
    GLFWwindow* glfw_handle;
};