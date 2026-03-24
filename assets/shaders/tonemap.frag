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
    float bloom_enabled;
    float vignette_intensity;
    float vignette_smoothness;
    float chromatic_aberration;
    float film_grain;
    float time;
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

float noise(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // Chromatic Aberration
    vec2 dist = inUV - 0.5;
    vec2 r_offset = dist * params.chromatic_aberration;

    vec3 hdrColor;
    hdrColor.r = texture(hdrSampler, inUV - r_offset).r;
    hdrColor.g = texture(hdrSampler, inUV).g;
    hdrColor.b = texture(hdrSampler, inUV + r_offset).b;

    vec3 bloomColor = texture(bloomSampler, inUV).rgb;
    vec3 fogColor = texture(fogSampler, inUV).rgb;
    vec4 spriteColor = texture(spriteSampler, inUV);

    // Combine with Fog
    vec3 color = hdrColor + fogColor;

    // Add Bloom
    if (params.bloom_enabled > 0.5) {
        color += bloomColor;
    }

    // Exposure
    color *= params.exposure;

    // Blend Sprites
    color = mix(color, spriteColor.rgb, spriteColor.a);

    // Tone Mapping
    vec3 mapped = aces(color);

    // Gamma Correction
    mapped = pow(mapped, vec3(1.0 / params.gamma));

    // Vignette
    float d = length(inUV - 0.5);
    float vignette = smoothstep(params.vignette_intensity, params.vignette_intensity - params.vignette_smoothness, d);
    mapped *= vignette;

    // Film Grain
    float grain = noise(inUV + params.time) * params.film_grain;
    mapped += grain;

    outColor = vec4(mapped, 1.0);
}
