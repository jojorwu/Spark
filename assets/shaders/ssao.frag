#version 450

layout (set = 0, binding = 0) uniform sampler2D gNormal;
layout (set = 0, binding = 1) uniform sampler2D gDepth;
layout (set = 0, binding = 2) uniform sampler2D texNoise;

layout (constant_id = 0) const int SSAO_KERNEL_SIZE = 64;
layout (constant_id = 1) const float SSAO_RADIUS = 0.5;
layout (constant_id = 2) const float SSAO_BIAS = 0.025;

layout (set = 0, binding = 3) uniform SSAOParams {
    vec4 samples[SSAO_KERNEL_SIZE];
    mat4 projection;
    mat4 view;
    vec2 screenSize;
} params;

layout (location = 0) out float outOcclusion;

void main() {
    vec2 texCoord = gl_FragCoord.xy / params.screenSize;

    float depth = texture(gDepth, texCoord).r;
    if (depth == 1.0) {
        outOcclusion = 1.0;
        return;
    }

    // Reconstruct view-space position
    vec4 ndc = vec4(texCoord * 2.0 - 1.0, depth, 1.0);
    vec4 viewPos = inverse(params.projection) * ndc;
    viewPos /= viewPos.w;

    vec3 normal = normalize(texture(gNormal, texCoord).rgb * 2.0 - 1.0);
    normal = mat3(params.view) * normal; // Normal to view space

    vec2 noiseScale = params.screenSize / 4.0;
    vec3 randomVec = normalize(texture(texNoise, texCoord * noiseScale).xyz);

    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < SSAO_KERNEL_SIZE; i++) {
        vec3 samplePos = TBN * params.samples[i].xyz;
        samplePos = viewPos.xyz + samplePos * SSAO_RADIUS;

        vec4 offset = vec4(samplePos, 1.0);
        offset = params.projection * offset;
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        float sampleDepth = texture(gDepth, offset.xy).r;
        vec4 sampleNdc = vec4(offset.xy * 2.0 - 1.0, sampleDepth, 1.0);
        vec4 sampleViewPos = inverse(params.projection) * sampleNdc;
        sampleViewPos /= sampleViewPos.w;

        float rangeCheck = smoothstep(0.0, 1.0, SSAO_RADIUS / abs(viewPos.z - sampleViewPos.z));
        occlusion += (sampleViewPos.z >= samplePos.z + SSAO_BIAS ? 1.0 : 0.0) * rangeCheck;
    }

    outOcclusion = 1.0 - (occlusion / float(SSAO_KERNEL_SIZE));
}
