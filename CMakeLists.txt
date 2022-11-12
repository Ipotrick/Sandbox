cmake_minimum_required (VERSION 3.21)
project(Sandbox)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "binaries")
add_executable(${PROJECT_NAME} 
    "src/main.cpp"
)
find_package(daxa CONFIG REQUIRED)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE
    daxa::daxa
)