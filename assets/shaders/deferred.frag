#version 450

#ifndef MSAA_SAMPLES
#define MSAA_SAMPLES 1
#endif

#if MSAA_SAMPLES > 1
layout (input_attachment_index = 0, binding = 0) uniform subpassInputMS inputAlbedo;
layout (input_attachment_index = 1, binding = 1) uniform subpassInputMS inputNormal;
layout (input_attachment_index = 2, binding = 2) uniform subpassInputMS inputPosition;
layout (input_attachment_index = 3, binding = 3) uniform subpassInputMS inputDepth;
#else
layout (input_attachment_index = 0, binding = 0) uniform subpassInput inputAlbedo;
layout (input_attachment_index = 1, binding = 1) uniform subpassInput inputNormal;
layout (input_attachment_index = 2, binding = 2) uniform subpassInput inputPosition;
layout (input_attachment_index = 3, binding = 3) uniform subpassInput inputDepth;
#endif

struct Light {
    vec4 pos;
    vec4 color;
};

layout (std430, binding = 4) buffer LightBuffer {
    Light lights[];
};

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    mat4 lightViewProj;
    uint lightCount;
} push;

layout (location = 0) out vec4 outColor;

vec3 calculateLighting(vec3 albedo, vec3 normal, vec3 position) {
    vec3 lighting = albedo * 0.1; // Ambient
    for (int i = 0; i < push.lightCount; i++) {
        vec3 L = normalize(lights[i].pos.xyz - position);
        float dist = length(lights[i].pos.xyz - position);
        float attenuation = lights[i].color.a / (dist * dist);
        float diffuse = max(dot(normal, L), 0.0);
        lighting += albedo * lights[i].color.rgb * diffuse * attenuation;
    }
    return lighting;
}

void main()
{
#if MSAA_SAMPLES > 1
    vec3 color = vec3(0.0);
    for (int i = 0; i < MSAA_SAMPLES; i++) {
        vec3 albedo = subpassLoad(inputAlbedo, i).rgb;
        vec3 normal = subpassLoad(inputNormal, i).rgb;
        vec3 position = subpassLoad(inputPosition, i).rgb;
        color += calculateLighting(albedo, normal, position);
    }
    outColor = vec4(color / float(MSAA_SAMPLES), 1.0);
#else
    vec3 albedo = subpassLoad(inputAlbedo).rgb;
    vec3 normal = subpassLoad(inputNormal).rgb;
    vec3 position = subpassLoad(inputPosition).rgb;
    outColor = vec4(calculateLighting(albedo, normal, position), 1.0);
#endif
}
