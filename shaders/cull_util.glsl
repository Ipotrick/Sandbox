#pragma once

#include <daxa/daxa.inl>

struct NdcBounds
{
    vec3 ndc_min;
    vec3 ndc_max;
    uint valid_vertices;
};

void init_ndc_bounds(inout NdcBounds ndc_bounds)
{
    ndc_bounds.ndc_min = vec3(0);
    ndc_bounds.ndc_max = vec3(0);
    ndc_bounds.valid_vertices = 0;
}

// All vertex positions MUST be in front of the near plane!
void add_vertex_to_ndc_bounds(inout NdcBounds ndc_bounds, vec3 ndc_pos)
{
    {
        if (ndc_bounds.valid_vertices == 0)
        {
            ndc_bounds.ndc_min = ndc_pos;
            ndc_bounds.ndc_max = ndc_pos;
        }
        else
        {
            ndc_bounds.ndc_min = vec3(
                min(ndc_pos.x, ndc_bounds.ndc_min.x),
                min(ndc_pos.y, ndc_bounds.ndc_min.y),
                min(ndc_pos.z, ndc_bounds.ndc_min.z)
            );
            ndc_bounds.ndc_max = vec3(
                max(ndc_pos.x, ndc_bounds.ndc_max.x),
                max(ndc_pos.y, ndc_bounds.ndc_max.y),
                max(ndc_pos.z, ndc_bounds.ndc_max.z)
            );
        }
        ndc_bounds.valid_vertices += 1;
    }
}



bool is_in_frustum(NdcBounds bounds)
{
    return !(bounds.ndc_max.z < 0.0f ||
            bounds.ndc_min.x > 1.0f ||
            bounds.ndc_min.y > 1.0f ||
            bounds.ndc_max.x < -1.0f ||
            bounds.ndc_max.y < -1.0f);
}