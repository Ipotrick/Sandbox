#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "patch_draw_opaque_indirect.inl"
#include "../../mesh/visbuffer_meshlet_util.glsl"

layout(local_size_x = 1) in;
void main()
{
    const bool indexed_id_rendering = globals.settings.indexed_id_rendering != 0;
    if (indexed_id_rendering)
    {
        InstantiatedMeshletsViewDispatchIndirect view = InstantiatedMeshletsViewDispatchIndirect(u_meshlets);
        view.first_pass.x = 1;
        view.first_pass.z = 1;
        view.second_pass.x = 1;
        view.second_pass.z = 1;
    }
    else
    {
        InstantiatedMeshletsViewDrawIndirect view = InstantiatedMeshletsViewDrawIndirect(u_meshlets);
        view.first_pass.vertex_count = MAX_TRIANGLES_PER_MESHLET * 3;
        view.first_pass.first_vertex = 0;
        view.first_pass.first_instance = 0;
        view.second_pass.vertex_count = MAX_TRIANGLES_PER_MESHLET * 3;
        view.second_pass.first_vertex = 0;
        view.second_pass.first_instance = 0;
    }
}