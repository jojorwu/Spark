#version 450
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D hdrSampler;
layout(binding = 1) uniform sampler2D bloomSampler;
layout(binding = 2) uniform sampler2D fogSampler;
layout(binding = 3) uniform sampler2D spriteSampler;

layout(push_constant) uniform PostProcessParams {
    float exposure;
    float gamma;
    float enable_bloom;
    float padding;
} params;

// ACES Tone Mapping
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdrColor = texture(hdrSampler, inUV).rgb;
    vec3 bloomColor = texture(bloomSampler, inUV).rgb;
    vec3 fogColor = texture(fogSampler, inUV).rgb;
    vec4 spriteColor = texture(spriteSampler, inUV);

    // Combine with Fog (additive/volumetric)
    // Note: In some setups fog is already in HDR color if applied during lighting pass
    // but here we have a separate VolumetricColor pass.
    vec3 color = hdrColor + fogColor;

    // Add Bloom
    if (params.enable_bloom > 0.5) {
        color += bloomColor;
    }

    // Exposure
    color *= params.exposure;

    // Blend Sprites (Alpha blend over HDR)
    color = mix(color, spriteColor.rgb, spriteColor.a);

    // Tone Mapping
    vec3 mapped = aces(color);

    // Gamma Correction
    mapped = pow(mapped, vec3(1.0 / params.gamma));

    outColor = vec4(mapped, 1.0);
}
