#version 450

#ifndef MSAA_SAMPLES
#define MSAA_SAMPLES 1
#endif

layout (set = 0, binding = 0) uniform GlobalUBO {
    mat4 viewProj;
    mat4 lightViewProj;
    mat4 invViewProj;
    vec4 cameraPos;
} global;

#if MSAA_SAMPLES > 1
layout (set = 1, input_attachment_index = 0, binding = 0) uniform subpassInputMS inputAlbedo;
layout (set = 1, input_attachment_index = 1, binding = 1) uniform subpassInputMS inputNormal;
layout (set = 1, input_attachment_index = 2, binding = 2) uniform subpassInputMS inputPBR;
layout (set = 1, input_attachment_index = 3, binding = 3) uniform subpassInputMS inputDepth;
#else
layout (set = 1, input_attachment_index = 0, binding = 0) uniform subpassInput inputAlbedo;
layout (set = 1, input_attachment_index = 1, binding = 1) uniform subpassInput inputNormal;
layout (set = 1, input_attachment_index = 2, binding = 2) uniform subpassInput inputPBR;
layout (set = 1, input_attachment_index = 3, binding = 3) uniform subpassInput inputDepth;
#endif

layout (set = 1, binding = 4) uniform sampler2D shadowMap;

struct Light {
    vec4 pos;
    vec4 color;
};

layout (set = 1, std430, binding = 5) buffer LightBuffer {
    Light lights[];
};

layout(push_constant) uniform PushConstants {
    uint lightCount;
    float metallic;
    float roughness;
    float width;
    float height;
} push;

layout (location = 0) out vec4 outColor;

const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

/**
 * Percentage-Closer Filtering (PCF) for soft shadows.
 */
float calculateShadow(vec3 worldPos) {
    vec4 shadowCoord = global.lightViewProj * vec4(worldPos, 1.0);
    shadowCoord.xyz /= shadowCoord.w;

    // Convert to [0, 1] range
    shadowCoord.xy = shadowCoord.xy * 0.5 + 0.5;

    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    float bias = 0.005;

    // 3x3 PCF Kernel
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, shadowCoord.xy + vec2(x, y) * texelSize).r;
            shadow += shadowCoord.z - bias > pcfDepth ? 0.5 : 1.0;
        }
    }
    return shadow / 9.0;
}

/**
 * Physically Based Rendering (PBR) lighting using Cook-Torrance BRDF.
 */
vec3 calculatePBRLighting(vec3 albedo, vec3 normal, vec3 worldPos, float metallic, float roughness, vec3 viewPos) {
    vec3 N = normalize(normal);
    vec3 V = normalize(viewPos - worldPos);
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    vec3 Lo = vec3(0.0);
    float shadow = calculateShadow(worldPos);

    for (int i = 0; i < push.lightCount; i++) {
        vec3 L = normalize(lights[i].pos.xyz - worldPos);
        vec3 H = normalize(V + L);
        float distance = length(lights[i].pos.xyz - worldPos);
        float attenuation = lights[i].color.a / (distance * distance);
        vec3 radiance = lights[i].color.rgb * attenuation;

        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL * shadow;
    }

    vec3 ambient = vec3(0.03) * albedo;
    vec3 color = ambient + Lo;
    return color;
}

vec3 worldPosFromDepth(float depth, vec2 texCoord) {
    vec4 clipSpacePos = vec4(texCoord * 2.0 - 1.0, depth, 1.0);
    vec4 worldSpacePos = global.invViewProj * clipSpacePos;
    return worldSpacePos.xyz / worldSpacePos.w;
}

void main()
{
    vec2 texCoord = gl_FragCoord.xy / vec2(push.width, push.height);

    vec3 viewPos = global.cameraPos.xyz;

#if MSAA_SAMPLES > 1
    vec3 color = vec3(0.0);
    for (int i = 0; i < MSAA_SAMPLES; i++) {
        vec3 albedo = subpassLoad(inputAlbedo, i).rgb;
        vec3 normal = subpassLoad(inputNormal, i).rgb * 2.0 - 1.0;
        float depth = subpassLoad(inputDepth, i).r;
        vec3 position = worldPosFromDepth(depth, texCoord);
        vec2 pbr = subpassLoad(inputPBR, i).rg;
        color += calculatePBRLighting(albedo, normal, position, pbr.x, pbr.y, viewPos);
    }
    outColor = vec4(color / float(MSAA_SAMPLES), 1.0);
#else
    vec3 albedo = subpassLoad(inputAlbedo).rgb;
    vec3 normal = subpassLoad(inputNormal).rgb * 2.0 - 1.0;
    float depth = subpassLoad(inputDepth).r;
    vec3 position = worldPosFromDepth(depth, texCoord);
    vec2 pbr = subpassLoad(inputPBR).rg;
    outColor = vec4(calculatePBRLighting(albedo, normal, position, pbr.x, pbr.y, viewPos), 1.0);
#endif
}
