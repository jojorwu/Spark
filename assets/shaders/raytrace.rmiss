#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "raytrace_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main()
{
    payload.color = vec3(0.01, 0.01, 0.05);
    payload.dist = -1.0;
    payload.hit = 0;
    payload.normal = vec3(0, 0, 0);
}
