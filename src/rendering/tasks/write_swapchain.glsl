#extension GL_EXT_debug_printf : enable

#define DAXA_ENABLE_IMAGE_OVERLOADS_BASIC 1
#include <daxa/daxa.inl>

#include "write_swapchain.inl"

DAXA_USE_PUSH_CONSTANT(WriteSwapchainPush, push)
layout(local_size_x = WRITE_SWAPCHAIN_WG_X, local_size_y = WRITE_SWAPCHAIN_WG_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    vec4 result = vec4(0,0,0,1);


    const vec4 debug_value = texelFetch(debug_image, index, 0);
    result.xyz = result.xyz * (1.0f - debug_value.a) + debug_value.xyz * debug_value.a;
    result.xyz = debug_value.xyz;

    imageStore(swapchain, index, result);
}