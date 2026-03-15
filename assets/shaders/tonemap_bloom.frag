#version 450
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D hdrSampler;
layout(binding = 1) uniform sampler2D bloomSampler;

void main() {
    vec3 hdrColor = texture(hdrSampler, inUV).rgb;
    vec3 bloomColor = texture(bloomSampler, inUV).rgb;

    vec3 result = hdrColor + bloomColor; // Additive blend

    // Reinhard tone mapping
    vec3 mapped = result / (result + vec3(1.0));
    // Gamma correction
    mapped = pow(mapped, vec3(1.0 / 2.2));

    outColor = vec4(mapped, 1.0);
}
