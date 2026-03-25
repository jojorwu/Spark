#version 450
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0) uniform sampler2D hdrSampler;
layout(set = 1, binding = 1) uniform sampler2D bloomSampler;
layout(set = 1, binding = 2) uniform sampler2D fogSampler;
layout(set = 1, binding = 3) uniform sampler2D spriteSampler;
layout(set = 1, binding = 4) uniform sampler2D velocitySampler;
layout(set = 1, binding = 5) buffer LuminanceBuffer {
    float avgLuminance;
    float targetExposure;
} lum;
layout(set = 1, binding = 6) uniform sampler2D dofSampler;

layout(set = 0, binding = 0) uniform sampler3D luts[16];

layout(push_constant) uniform PostProcessParams {
    float exposure;
    float gamma;
    float bloom_enabled;
    float vignette_intensity;
    float vignette_smoothness;
    float chromatic_aberration;
    float film_grain;
    float motion_blur_strength;
    float auto_exposure_enabled;
    float dof_enabled;
    float lut_index;
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
    vec2 velocity = texture(velocitySampler, inUV).rg;

    // Motion Blur (Simple multi-tap)
    vec3 hdrColor = vec3(0.0);
    if (params.motion_blur_strength > 0.0) {
        const int samples = 8;
        for (int i = 0; i < samples; i++) {
            vec2 offset = velocity * (float(i) / float(samples - 1) - 0.5) * params.motion_blur_strength;

            // Chromatic Aberration integrated into blur
            vec2 dist = (inUV + offset) - 0.5;
            vec2 r_offset = dist * params.chromatic_aberration;

            hdrColor.r += texture(hdrSampler, inUV + offset - r_offset).r;
            hdrColor.g += texture(hdrSampler, inUV + offset).g;
            hdrColor.b += texture(hdrSampler, inUV + offset + r_offset).b;
        }
        hdrColor /= float(samples);
    } else {
        // Just Chromatic Aberration
        vec2 dist = inUV - 0.5;
        vec2 r_offset = dist * params.chromatic_aberration;
        hdrColor.r = texture(hdrSampler, inUV - r_offset).r;
        hdrColor.g = texture(hdrSampler, inUV).g;
        hdrColor.b = texture(hdrSampler, inUV + r_offset).b;
    }

    vec3 bloomColor = texture(bloomSampler, inUV).rgb;
    vec3 fogColor = texture(fogSampler, inUV).rgb;
    vec4 spriteColor = texture(spriteSampler, inUV);

    // Combine with Fog
    vec3 color = hdrColor + fogColor;

    // Apply DoF
    if (params.dof_enabled > 0.5) {
        vec3 dofColor = texture(dofSampler, inUV).rgb;
        color = dofColor + fogColor;
    }

    // Add Bloom
    if (params.bloom_enabled > 0.5) {
        color += bloomColor;
    }

    // Exposure
    float exposure = params.exposure;
    if (params.auto_exposure_enabled > 0.5) {
        exposure *= lum.targetExposure;
    }
    color *= exposure;

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

    // Color Grading (LUT)
    if (params.lut_index >= 0.0) {
        int idx = int(params.lut_index);
        mapped = texture(luts[idx], mapped).rgb;
    }

    // Film Grain
    float grain = noise(inUV + params.time) * params.film_grain;
    mapped += grain;

    outColor = vec4(mapped, 1.0);
}
