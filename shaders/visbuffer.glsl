#pragma once

#include <daxa/daxa.inl>

void encode_pixel_id(uint instantiated_meshlet_index, uint triangle_index, out uint id)
{
    id = (instantiated_meshlet_index << 7) | (triangle_index);
}

void decode_pixel_id(uint id, out uint instantiated_meshlet_index, out uint triangle_index)
{
    instantiated_meshlet_index = id >> 7;
    triangle_index = id & 0x7F;
}