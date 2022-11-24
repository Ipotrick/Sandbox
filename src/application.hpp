#pragma once

// Standart headers:
// Library headers:
// Project headers:
#include "sandbox.hpp"
#include "window.hpp"
#include "rendering/renderer.hpp"
#include "scene/scene.hpp"

struct Application
{
    Application();
    ~Application();
    auto run() -> i32;
    void update();

    Window window;
    Renderer renderer;
    AssetManager asset_manager;
    Scene scene;
    SceneLoader scene_loader;
    bool keep_running{ true };
};