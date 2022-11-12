cmake_minimum_required (VERSION 3.21)
project(Sandbox)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "binaries")
add_executable(${PROJECT_NAME} 
    "src/main.cpp"
)