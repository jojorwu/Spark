#version 450

layout (set = 0, binding = 0) uniform sampler2D ssaoInput;
layout (set = 0, binding = 1) uniform sampler2D gDepth;

layout (location = 0) out float outColor;

const float BLUR_SHARPNESS = 40.0;

void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(ssaoInput, 0));
    vec2 texCoord = gl_FragCoord.xy / textureSize(ssaoInput, 0);

    float centerSSAO = texture(ssaoInput, texCoord).r;
    float centerDepth = texture(gDepth, texCoord).r;

    float result = 0.0;
    float totalWeight = 0.0;

    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            float sampleSSAO = texture(ssaoInput, texCoord + offset).r;
            float sampleDepth = texture(gDepth, texCoord + offset).r;

            // Bilateral filter weight: Gaussian space weight * Depth-based edge weight
            float weight = 1.0 / (1.0 + abs(float(x)) + abs(float(y)));
            float depthWeight = exp(-abs(sampleDepth - centerDepth) * BLUR_SHARPNESS);
            weight *= depthWeight;

            result += sampleSSAO * weight;
            totalWeight += weight;
        }
    }

    outColor = result / max(totalWeight, 0.0001);
}
