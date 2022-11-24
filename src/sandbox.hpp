#pragma once

// Standart headers:
#include <thread>
#include <chrono>
#include <string_view>
#include <filesystem>
#include <unordered_map>
#include <span>
#include <cstdlib>
// Library headers:
#include <GLFW/glfw3.h>
#include <daxa/daxa.hpp>
#include <daxa/utils/task_list.hpp>
// Project headers:
#include "shared.inl"

using namespace daxa::types;
using namespace std::chrono_literals;

#if defined(_DEBUG)
#include <iostream>
#define ASSERT_M(x, m)                                            \
    [&]() {                                                       \
        if (!(x))                                                 \
        {                                                         \
            std::cerr << "ASSERTION FAILURE: " << m << std::endl; \
            std::abort();                                            \
        }                                                         \
    }()
#else
#define ASSERT_M(x, m)
#endif