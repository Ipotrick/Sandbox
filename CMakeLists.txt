cmake_minimum_required (VERSION 3.21)
project(Sandbox)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "binaries")
add_executable(${PROJECT_NAME} 
    "src/main.cpp"
    "src/sandbox.cpp"
    "src/application.cpp"
    "src/window.cpp"
    "src/rendering/renderer.cpp"
    "src/rendering/gpu_context.cpp"
    "src/scene/asset_manager.cpp"
    "src/scene/scene.cpp"
    "src/ui.cpp"
)
find_package(fmt CONFIG REQUIRED)
find_package(daxa CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)     
find_package(fastgltf CONFIG REQUIRED)
find_package(meshoptimizer CONFIG REQUIRED)
find_package(EnTT CONFIG REQUIRED)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE
    fmt::fmt
    daxa::daxa
    glfw
    meshoptimizer::meshoptimizer
    fastgltf::fastgltf
    EnTT::EnTT
)