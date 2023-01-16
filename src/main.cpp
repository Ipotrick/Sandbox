#include "application.hpp"

#include <variant>

auto main() -> i32
{
    auto app = std::make_unique<Application>();
    return app->run();
}
