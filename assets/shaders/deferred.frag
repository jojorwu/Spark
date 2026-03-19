#version 450

#ifndef MSAA_SAMPLES
#define MSAA_SAMPLES 1
#endif

layout (set = 0, binding = 0) uniform GlobalUBO {
    mat4 viewProj;
    mat4 lightViewProj[4];
    mat4 invViewProj;
    vec4 cameraPos;
    vec4 frustum[6];
    vec4 cascadeSplits;
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

layout (set = 1, binding = 4) uniform sampler2DArray shadowMap;
layout (set = 1, binding = 7) uniform sampler2D ssaoTex;
layout (set = 1, binding = 8) uniform samplerCube irradianceMap;
layout (set = 1, binding = 9) uniform samplerCube specularMap;
layout (set = 1, binding = 10) uniform sampler2D brdfLUT;

struct Light {
    vec4 pos;
    vec4 color;
};

layout (set = 1, std430, binding = 5) readonly buffer LightBuffer {
    Light lights[];
};

struct LightGrid {
    uint offset;
    uint count;
};

layout (set = 0, std430, binding = 6) readonly buffer LightGridBuffer {
    LightGrid lightGrids[];
};

layout (set = 0, std430, binding = 7) readonly buffer GlobalIndexList {
    uint globalIndexList[];
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
 * Percentage-Closer Filtering (PCF) for soft shadows with Cascaded Shadow Maps.
 */
float calculateShadow(vec3 worldPos, float linearDepth) {
    uint cascadeIdx = 0;
    for (uint i = 0; i < 3; ++i) {
        if (linearDepth > global.cascadeSplits[i]) {
            cascadeIdx = i + 1;
        }
    }

    vec4 shadowCoord = global.lightViewProj[cascadeIdx] * vec4(worldPos, 1.0);
    shadowCoord.xyz /= shadowCoord.w;

    shadowCoord.xy = shadowCoord.xy * 0.5 + 0.5;

    float shadow = 0.0;
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0).xy);
    float bias = 0.005;

    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, vec3(shadowCoord.xy + vec2(x, y) * texelSize, cascadeIdx)).r;
            shadow += shadowCoord.z - bias > pcfDepth ? 0.5 : 1.0;
        }
    }
    return shadow / 9.0;
}

/**
 * Physically Based Rendering (PBR) lighting using Cook-Torrance BRDF.
 */
vec3 calculatePBRLighting(vec3 albedo, vec3 normal, vec3 worldPos, float metallic, float roughness, vec3 viewPos, float ssao, vec2 uv, float depth) {
    vec3 N = normalize(normal);
    vec3 V = normalize(viewPos - worldPos);
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    vec3 Lo = vec3(0.0);
    float shadow = calculateShadow(worldPos, depth);

    // Clustered Light Lookup
    // 16x9x24 grid
    uint zSlices = 24;
    float zNear = 0.1;
    float zFar = 100.0;

    uint clusterZ = uint(max(0.0, log(depth / zNear) * float(zSlices) / log(zFar / zNear)));
    uvec2 clusterXY = uvec2(uv * vec2(16, 9));
    uint clusterIndex = clusterXY.x + clusterXY.y * 16 + clusterZ * 16 * 9;

    LightGrid grid = lightGrids[clusterIndex];

    for (uint i = 0; i < grid.count; i++) {
        uint lightIdx = globalIndexList[grid.offset + i];
        vec3 L = normalize(lights[lightIdx].pos.xyz - worldPos);
        vec3 H = normalize(V + L);
        float distance = length(lights[lightIdx].pos.xyz - worldPos);
        float attenuation = lights[lightIdx].color.w / (distance * distance); // Range in w
        vec3 radiance = lights[lightIdx].color.rgb * attenuation;

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

    vec3 R = reflect(-V, N);
    vec3 F = fresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse = irradiance * albedo;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(specularMap, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 envSpecular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = (kD * diffuse + envSpecular) * ssao;
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
    float ssao = texture(ssaoTex, texCoord).r;

        vec3 albedo = subpassLoad(inputAlbedo, i).rgb;
        vec3 normal = subpassLoad(inputNormal, i).rgb * 2.0 - 1.0;
        float depth = subpassLoad(inputDepth, i).r;
        vec3 position = worldPosFromDepth(depth, texCoord);
        vec2 pbr = subpassLoad(inputPBR, i).rg;

        // linear depth for clustering
        float linearDepth = (2.0 * 0.1) / (100.0 + 0.1 - depth * (100.0 - 0.1));

        color += calculatePBRLighting(albedo, normal, position, pbr.x, pbr.y, viewPos, ssao, texCoord, linearDepth);
    }
    outColor = vec4(color / float(MSAA_SAMPLES), 1.0);
#else
    float ssao = texture(ssaoTex, texCoord).r;
    vec3 albedo = subpassLoad(inputAlbedo).rgb;
    vec3 normal = subpassLoad(inputNormal).rgb * 2.0 - 1.0;
    float depth = subpassLoad(inputDepth).r;
    vec3 position = worldPosFromDepth(depth, texCoord);
    vec2 pbr = subpassLoad(inputPBR).rg;

    float linearDepth = (2.0 * 0.1) / (100.0 + 0.1 - depth * (100.0 - 0.1));

    outColor = vec4(calculatePBRLighting(albedo, normal, position, pbr.x, pbr.y, viewPos, ssao, texCoord, linearDepth), 1.0);
#endif
}
