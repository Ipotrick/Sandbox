cmake_minimum_required (VERSION 3.21)
project(Sandbox)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "binaries")
add_executable(${PROJECT_NAME} 
    "src/main.cpp"
    "src/sandbox.cpp"
    "src/application.cpp"
    "src/window.cpp"
    "src/rendering/renderer.cpp"
)
find_package(daxa CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE
    daxa::daxa
    glfw
)
# This creates a marko define that can be used to find the daxa include folder for shader compilation.
set(DAXA_INCLUDE_DIR "$<TARGET_FILE_DIR:Sandbox>/../../vcpkg_installed/x64-$<LOWER_CASE:$<PLATFORM_ID>>/include")
target_compile_definitions(${PROJECT_NAME} PRIVATE DAXA_SHADER_INCLUDE_DIR="${DAXA_INCLUDE_DIR}")