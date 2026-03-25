#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload {
    vec3 color;
    float dist;
    uint hit;
    vec3 normal;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main()
{
    payload.color = vec3(0.01, 0.01, 0.05);
    payload.dist = -1.0;
    payload.hit = 0;
    payload.normal = vec3(0, 0, 0);
}
