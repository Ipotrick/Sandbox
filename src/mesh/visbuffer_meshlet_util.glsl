#pragma once

#include <daxa/daxa.inl>
#include "mesh.inl"

#include "../../shaders/visbuffer.glsl"

DAXA_BUFFER_REFERENCE_LAYOUT buffer InstantiatedMeshletsView
{
    daxa_u32 padd0;
    daxa_u32 first_pass_meshlet_count;
    daxa_u32 padd1[3];
    daxa_u32 second_pass_meshlet_count;
    daxa_u32 padd2[2];
    InstantiatedMeshlet meshlets[];
};

DAXA_BUFFER_REFERENCE_LAYOUT readonly buffer ROInstantiatedMeshletsView
{
    daxa_u32 padd0;
    daxa_u32 first_pass_meshlet_count;
    daxa_u32 padd1[3];
    daxa_u32 second_pass_meshlet_count;
    daxa_u32 padd2[2];
    InstantiatedMeshlet meshlets[];
};

DAXA_BUFFER_REFERENCE_LAYOUT buffer InstantiatedMeshletsViewDrawIndirect
{
    DrawIndirectStruct first_pass;
    DrawIndirectStruct second_pass;
    InstantiatedMeshlet meshlets[];
};

DAXA_BUFFER_REFERENCE_LAYOUT buffer InstantiatedMeshletsViewDispatchIndirect
{
    DispatchIndirectStruct first_pass;
    DispatchIndirectStruct second_pass;
    InstantiatedMeshlet meshlets[];
};