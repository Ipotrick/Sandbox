#pragma once

// Standart headers:
#include <thread>
#include <iostream>
#include <chrono>
#include <string_view>
#include <filesystem>
#include <unordered_map>
#include <span>
#include <cstdlib>
// Library headers:
#include <GLFW/glfw3.h>
#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/mem.hpp>
#define GLM_DEPTH_ZERO_TO_ONEW
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
// Project headers:
#include "../shader_shared/shared.inl"

using namespace daxa::types;
using namespace std::chrono_literals;

#if defined(_DEBUG)
#include <iostream>
#define ASSERT_M(x, m)                                            \
    [&]() {                                                       \
        if (!(x))                                                 \
        {                                                         \
            std::cerr << "ASSERTION FAILURE: " << m << std::endl; \
            std::abort();                                         \
        }                                                         \
    }()
#else
#define ASSERT_M(x, m)
#endif

#define s_cast static_cast
#define d_cast dynamic_cast
#define r_cast reinterpret_cast

#ifndef defer
struct defer_dummy {};
template <class F> struct deferrer { F f; ~deferrer() { f(); } };
template <class F> deferrer<F> operator*(defer_dummy, F f) { return {f}; }
#define DEFER_(LINE) zz_defer##LINE
#define DEFER(LINE) DEFER_(LINE)
#define defer auto DEFER(__LINE__) = defer_dummy{} *[&]()
#endif // defer