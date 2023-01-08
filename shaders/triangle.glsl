#define DAXA_ENABLE_SHADER_NO_NAMESPACE 1
#include <triangle_shared.inl>
#include <../src/mesh/mesh.inl>

layout(push_constant, scalar) uniform Push
{
    TriangleTaskPushConstant push;
};

const vec4 positions[] = {
    vec4(-0.5, 0.0, -0.5, 1.0),
    vec4( 0.5, 0.0, -0.5, 1.0),
    vec4( 0.0, 0.0,  0.5, 1.0)
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
    gl_Position = deref(push.globals).camera_view_projection * positions[gl_VertexIndex];
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