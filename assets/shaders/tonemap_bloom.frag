#version 450
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D hdrSampler;
layout(binding = 1) uniform sampler2D bloomSampler;
layout(binding = 2) uniform sampler2D fogSampler;

layout(push_constant) uniform PushConstants {
    float exposure;
    float gamma;
    float enable_bloom;
    float padding;
} push;

// ACES Tone Mapping (Narkowicz 2015)
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
    vec3 hdrColor = texture(hdrSampler, inUV).rgb;
    vec3 bloomColor = texture(bloomSampler, inUV).rgb;
    vec3 fogColor = texture(fogSampler, inUV).rgb;

    // Mix HDR with bloom (if enabled) and fog
    vec3 result = hdrColor;
    if (push.enable_bloom > 0.5) {
        result += bloomColor;
    }
    result += fogColor;
    result *= push.exposure;

    // ACES Tone Mapping
    vec3 mapped = ACESFilm(result);

    // Gamma correction
    mapped = pow(mapped, vec3(1.0 / push.gamma));

    outColor = vec4(mapped, 1.0);
}
