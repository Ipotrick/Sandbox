#include "application.hpp"

auto main() -> i32
{
    auto app = std::make_unique<Application>();
    return app->run();
}
