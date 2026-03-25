#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload {
    vec3 color;
    float dist;
    uint hit;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main()
{
    payload.color = vec3(0.01, 0.01, 0.05); // Very dark blue background
    payload.dist = -1.0;
    payload.hit = 0;
}
