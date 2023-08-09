#pragma once

#include "visbuffer.inl"

void encode_triangle_id(uint instantiated_meshlet_index, uint triangle_index, out uint id)
{
    id = (instantiated_meshlet_index << 7) | (triangle_index);
}

uint meshlet_instance_index_from_triangle_id(uint id)
{
    return id >> 7;
}

uint triangle_index_from_triangle_id(uint id)
{
    return id & 0x7F;
}

void decode_triangle_id(uint id, out uint instantiated_meshlet_index, out uint triangle_index)
{
    instantiated_meshlet_index = meshlet_instance_index_from_triangle_id(id);
    triangle_index = triangle_index_from_triangle_id(id);
}