#version 450

layout (set = 0, binding = 0) uniform sampler2D gNormal;
layout (set = 0, binding = 1) uniform sampler2D gDepth;
layout (set = 0, binding = 2) uniform sampler2D texNoise;

layout (constant_id = 0) const int SSAO_KERNEL_SIZE = 64;

layout (set = 0, binding = 3) uniform SSAOParams {
    vec4 samples[SSAO_KERNEL_SIZE];
    mat4 projection;
    mat4 view;
    vec2 screenSize;
    float radius;
    float bias;
    float strength;
} params;

layout (location = 0) out float outOcclusion;

// Reconstruct view-space position from depth more efficiently
vec3 getViewPos(vec2 uv) {
    float depth = texture(gDepth, uv).r;
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewPos = inverse(params.projection) * ndc;
    return viewPos.xyz / viewPos.w;
}

void main() {
    vec2 uv = gl_FragCoord.xy / params.screenSize;

    float depth = texture(gDepth, uv).r;
    if (depth == 1.0) {
        outOcclusion = 1.0;
        return;
    }

    vec3 viewPos = getViewPos(uv);
    vec3 normal = normalize(texture(gNormal, uv).rgb * 2.0 - 1.0);
    normal = mat3(params.view) * normal;

    vec2 noiseScale = params.screenSize / 4.0;
    vec3 randomVec = normalize(texture(texNoise, uv * noiseScale).xyz);

    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < SSAO_KERNEL_SIZE; i++) {
        vec3 samplePos = TBN * params.samples[i].xyz;
        samplePos = viewPos + samplePos * params.radius;

        vec4 offset = vec4(samplePos, 1.0);
        offset = params.projection * offset;
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        float sampleDepth = texture(gDepth, offset.xy).r;
        vec3 sampleViewPos = getViewPos(offset.xy);

        float rangeCheck = smoothstep(0.0, 1.0, params.radius / abs(viewPos.z - sampleViewPos.z));
        occlusion += (sampleViewPos.z >= samplePos.z + params.bias ? 1.0 : 0.0) * rangeCheck;
    }

    outOcclusion = 1.0 - (occlusion / float(SSAO_KERNEL_SIZE)) * params.strength;
}
