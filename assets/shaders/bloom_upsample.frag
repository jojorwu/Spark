#version 450

layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outColor;

layout (binding = 0) uniform sampler2D srcTexture;

layout (push_constant) uniform PushConstants {
    float filterRadius;
} push;

void main() {
    float x = push.filterRadius;
    float y = push.filterRadius;

    // Take 9 samples, with a 3x3 kernel
    // a - b - c
    // d - e - f
    // g - h - i
    // Weights:
    // 1/16 - 2/16 - 1/16
    // 2/16 - 4/16 - 2/16
    // 1/16 - 2/16 - 1/16

    vec3 a = texture(srcTexture, vec2(inUV.x - x, inUV.y + y)).rgb;
    vec3 b = texture(srcTexture, vec2(inUV.x,     inUV.y + y)).rgb;
    vec3 c = texture(srcTexture, vec2(inUV.x + x, inUV.y + y)).rgb;

    vec3 d = texture(srcTexture, vec2(inUV.x - x, inUV.y)).rgb;
    vec3 e = texture(srcTexture, vec2(inUV.x,     inUV.y)).rgb;
    vec3 f = texture(srcTexture, vec2(inUV.x + x, inUV.y)).rgb;

    vec3 g = texture(srcTexture, vec2(inUV.x - x, inUV.y - y)).rgb;
    vec3 h = texture(srcTexture, vec2(inUV.x,     inUV.y - y)).rgb;
    vec3 i = texture(srcTexture, vec2(inUV.x + x, inUV.y - y)).rgb;

    vec3 upsample = e*4.0;
    upsample += (b+d+f+h)*2.0;
    upsample += (a+c+g+i);
    upsample *= 1.0 / 16.0;

    outColor = vec4(upsample, 1.0);
}
