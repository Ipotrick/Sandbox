#pragma once

// Standart headers:
// Library headers:
// Project headers:
#include "sandbox.hpp"
#include "window.hpp"
#include "rendering/renderer.hpp"

struct Application
{
    Application();
    ~Application();
    auto run() -> i32;

    Window window;
    Renderer renderer;
    bool keep_running{ true };
};