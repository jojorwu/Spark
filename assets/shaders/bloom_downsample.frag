#version 450

layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outColor;

layout (binding = 0) uniform sampler2D srcTexture;

layout (push_constant) uniform PushConstants {
    vec2 srcResolution;
} push;

void main() {
    vec2 srcTexelSize = 1.0 / push.srcResolution;
    float x = srcTexelSize.x;
    float y = srcTexelSize.y;

    // Take 13 samples as specified in Karis 2013
    // a - b - c
    // - d - e -
    // f - g - h
    // - i - j -
    // k - l - m

    vec3 a = texture(srcTexture, vec2(inUV.x - 2*x, inUV.y + 2*y)).rgb;
    vec3 b = texture(srcTexture, vec2(inUV.x,       inUV.y + 2*y)).rgb;
    vec3 c = texture(srcTexture, vec2(inUV.x + 2*x, inUV.y + 2*y)).rgb;

    vec3 d = texture(srcTexture, vec2(inUV.x - x,   inUV.y + y)).rgb;
    vec3 e = texture(srcTexture, vec2(inUV.x + x,   inUV.y + y)).rgb;

    vec3 f = texture(srcTexture, vec2(inUV.x - 2*x, inUV.y)).rgb;
    vec3 g = texture(srcTexture, vec2(inUV.x,       inUV.y)).rgb;
    vec3 h = texture(srcTexture, vec2(inUV.x + 2*x, inUV.y)).rgb;

    vec3 i = texture(srcTexture, vec2(inUV.x - x,   inUV.y - y)).rgb;
    vec3 j = texture(srcTexture, vec2(inUV.x + x,   inUV.y - y)).rgb;

    vec3 k = texture(srcTexture, vec2(inUV.x - 2*x, inUV.y - 2*y)).rgb;
    vec3 l = texture(srcTexture, vec2(inUV.x,       inUV.y - 2*y)).rgb;
    vec3 m = texture(srcTexture, vec2(inUV.x + 2*x, inUV.y - 2*y)).rgb;

    vec3 downsample = e*0.125;
    downsample += (a+c+k+m)*0.03125;
    downsample += (b+d+f+h+i+j+l)*0.0625;
    downsample += g*0.125;

    outColor = vec4(max(downsample, 0.0001), 1.0);
}
