#include "window.hpp"

void close_callback(GLFWwindow* window) {
    WindowState* self = reinterpret_cast<WindowState*>(glfwGetWindowUserPointer(window));
    self->b_close_requested = true;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
		WindowState* self = reinterpret_cast<WindowState*>(glfwGetWindowUserPointer(window));
		if (action == GLFW_PRESS) {
			self->key_down[key] = true;
		} else if (action == GLFW_RELEASE) {
			self->key_down[key] = false;
		}
	}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    WindowState* self = reinterpret_cast<WindowState*>(glfwGetWindowUserPointer(window));
    if (action == GLFW_PRESS) {
        self->mouse_button_down[button] = true;
    } else if (action == GLFW_RELEASE) {
        self->mouse_button_down[button] = false;
    }
}

void cursor_move_callback(GLFWwindow* window, double xpos, double ypos) {
    WindowState* self = reinterpret_cast<WindowState*>(glfwGetWindowUserPointer(window));
    self->cursor_change_x = static_cast<i32>(std::floor(xpos)) - self->old_cursor_pos_x;
    self->cursor_change_y = static_cast<i32>(std::floor(ypos)) - self->old_cursor_pos_y;
}

Window::Window(i32 width, i32 height, std::string_view name)
    : size{width, height},
      name{name},
      glfw_handle{
          [=]()
          {
              glfwInit();
              glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
              return glfwCreateWindow(width, height, name.data(), nullptr, nullptr);
          }()},
          window_state{ std::make_unique<WindowState>() }
{
    glfwSetWindowUserPointer(glfw_handle, window_state.get());

    glfwSetWindowCloseCallback(glfw_handle, close_callback);
    glfwSetKeyCallback(glfw_handle, key_callback);
    glfwSetMouseButtonCallback(glfw_handle, mouse_button_callback);
    glfwSetCursorPosCallback(glfw_handle, cursor_move_callback);
}

Window::~Window()
{
    glfwDestroyWindow(this->glfw_handle);
    glfwTerminate();
}

const std::string& Window::get_name() {
    return name;
}

bool Window::is_focused() const {
    return focused;
}

// keys

bool Window::key_pressed(Key key) const {
    return window_state->key_down[key];
}

bool Window::key_just_pressed(Key key) const {
    return !window_state->key_down_old[key] && window_state->key_down[key];
}

bool Window::key_just_released(Key key) const {
    return window_state->key_down_old[key] && !window_state->key_down[key];
}

// buttons

bool Window::button_pressed(Button button) const {
    return window_state->mouse_button_down[button];
}

bool Window::button_just_pressed(Button button) const {
    return !window_state->mouse_button_down_old[button] && window_state->mouse_button_down[button];
}

bool Window::button_just_released(Button button) const {
    return window_state->mouse_button_down_old[button] && !window_state->mouse_button_down[button];
}

// cursor

i32 Window::get_cursor_x() const{
    double x,y;
    glfwGetCursorPos(glfw_handle, &x,&y);
    return static_cast<i32>(std::floor(x));
}

i32 Window::get_cursor_y() const {
    double x,y;
    glfwGetCursorPos(glfw_handle, &x,&y);
    return static_cast<i32>(std::floor(y));
}

i32 Window::get_cursor_change_x() const {
    return window_state->cursor_change_x;
}

i32 Window::get_cursor_change_y() const {
    return window_state->cursor_change_y;
}

bool Window::is_cursor_over_window() const {
    double x,y;
    glfwGetCursorPos(glfw_handle, &x,&y);
    i32 width, height;
    glfwGetWindowSize(glfw_handle, &width, &height);
    return x >= 0 && x <= width && y >= 0 && y <= height;
}

void Window::capture_cursor() {
    glfwSetInputMode(glfw_handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void Window::release_cursor() {
    glfwSetInputMode(glfw_handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

bool Window::is_cursor_captured() const {
    return glfwGetInputMode(glfw_handle, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
}

bool Window::update(f32 deltaTime) 	{
    window_state->key_down_old = window_state->key_down;
    window_state->mouse_button_down_old = window_state->mouse_button_down;
    window_state->old_cursor_pos_x = get_cursor_x();
    window_state->old_cursor_pos_y = get_cursor_y();
    window_state->cursor_change_x = {};
    window_state->cursor_change_y = {};
    glfwPollEvents();
    if (is_cursor_captured()) {
        glfwSetCursorPos(glfw_handle, -10000, -10000);
    }
    return window_state->b_close_requested;
}

void Window::set_width(u32 width) {
    i32 oldW, oldH;
    glfwGetWindowSize(glfw_handle, &oldW, &oldH);
    glfwSetWindowSize(glfw_handle, width, oldH);
}

void Window::set_height(u32 height) {
    i32 oldW, oldH;
    glfwGetWindowSize(glfw_handle, &oldW, &oldH);
    glfwSetWindowSize(glfw_handle, oldW, height);
}

u32 Window::get_width() const {
    i32 w, h;
    glfwGetWindowSize(glfw_handle, &w, &h);
    return w;
}

u32 Window::get_height() const {
    i32 w, h;
    glfwGetWindowSize(glfw_handle, &w, &h);
    return h;
}