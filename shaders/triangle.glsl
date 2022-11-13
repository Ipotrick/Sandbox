#define DAXA_SHADER_NO_NAMESPACE_PRIMITIVES
#include <daxa/daxa.glsl>
#include "triangle_shared.inl"

DAXA_USE_PUSHCONSTANT(TriangleTaskPushConstant)

const vec4 positions[] = {
    vec4(-0.5, 0.5, 0.0, 1.0),
    vec4( 0.5, 0.5, 0.0, 1.0),
    vec4( 0.0,-0.5, 0.0, 1.0)
};

const vec4 colors[] = {
    vec4(1.0f, 0.0f, 0.0f, 1.0f),
    vec4(0.0f, 1.0f, 0.0f, 1.0f),
    vec4(0.0f, 0.0f, 1.0f, 1.0f)
};

#ifdef _VERTEX
layout(location = 0) out f32vec4 v_col;
void main()
{
    v_col = colors[gl_VertexIndex];
    gl_Position = positions[gl_VertexIndex];
}
#endif

#ifdef _FRAGMENT
layout(location = 0) in f32vec4 v_col;
layout(location = 0) out f32vec4 color;
void main()
{
    color = v_col;
}
#endif