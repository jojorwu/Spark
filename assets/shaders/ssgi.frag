#version 450

layout (set = 0, binding = 0) uniform GlobalUBO {
    mat4 viewProj;
    mat4 lightViewProj[4];
    mat4 invViewProj;
    vec4 cameraPos;
} global;

layout (set = 1, binding = 0) uniform sampler2D hdrTex;
layout (set = 1, binding = 1) uniform sampler2D normalTex;
layout (set = 1, binding = 2) uniform sampler2D depthTex;

layout (location = 0) out vec4 outColor;

void main() {
    vec2 uv = gl_FragCoord.xy / textureSize(hdrTex, 0).xy;
    vec3 N = texture(normalTex, uv).rgb * 2.0 - 1.0;
    float depth = texture(depthTex, uv).r;

    // Very basic SSGI approximation: sample nearby pixels and accumulate reflected light
    vec3 indirect = vec3(0.0);
    for(int i = -2; i <= 2; i++) {
        for(int j = -2; j <= 2; j++) {
            if(i == 0 && j == 0) continue;
            vec2 offset = vec2(i, j) * 0.005;
            vec3 sampleColor = texture(hdrTex, uv + offset).rgb;
            vec3 sampleNormal = texture(normalTex, uv + offset).rgb * 2.0 - 1.0;
            float weight = max(0.0, dot(N, sampleNormal));
            indirect += sampleColor * weight;
        }
    }

    outColor = vec4(indirect * 0.1, 1.0);
}
