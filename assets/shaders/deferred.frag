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

const vec2 poissonDisk[16] = vec2[](
   vec2( -0.94201624, -0.39906216 ), vec2( 0.94558609, -0.76890725 ),
   vec2( -0.09418410, -0.92938870 ), vec2( 0.34495938, 0.29387760 ),
   vec2( -0.91588581, 0.45771432 ), vec2( -0.81544232, -0.87912464 ),
   vec2( -0.38204462, 0.62251975 ), vec2( 0.12984571, -0.44454745 ),
   vec2( 0.24271246, -0.99031590 ), vec2( 0.76482570, -0.54023540 ),
   vec2( 0.58900311, 0.88790979 ), vec2( 0.16543990, 0.12354469 ),
   vec2( -0.18837480, -0.06806012 ), vec2( -0.31343943, -0.45406813 ),
   vec2( 0.53742981, -0.47373420 ), vec2( -0.03576915, 0.74851071 )
);

float interleavedGradientNoise(vec2 n) {
    return fract(sin(dot(n, vec2(12.9898, 78.233))) * 43758.5453);
}

float calculateShadow(vec3 worldPos, float linearDepth, vec3 N) {
    uint cascadeIdx = 0;
    for (uint i = 0; i < 3; ++i) {
        if (linearDepth > global.cascadeSplits[i]) {
            cascadeIdx = i + 1;
        }
    }

    vec4 shadowCoord = global.lightViewProj[cascadeIdx] * vec4(worldPos, 1.0);
    shadowCoord.xyz /= shadowCoord.w;
    shadowCoord.xy = shadowCoord.xy * 0.5 + 0.5;

    if (shadowCoord.z > 1.0) return 1.0;

    vec3 L = normalize(vec3(global.lightViewProj[0][0][2], global.lightViewProj[0][1][2], global.lightViewProj[0][2][2]));
    float bias = max(0.002 * (1.0 - dot(N, L)), 0.0005);
    if (cascadeIdx == 3) bias *= 2.0;

    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0).xy);
    float noise = interleavedGradientNoise(gl_FragCoord.xy);
    float cosT = cos(noise * 2.0 * 3.14159);
    float sinT = sin(noise * 2.0 * 3.14159);
    mat2 rot = mat2(cosT, sinT, -sinT, cosT);

    float avgBlockerDepth = 0.0;
    float blockers = 0.0;
    float searchRadius = 5.0 * texelSize.x;
    for (int i = 0; i < 8; i++) {
        vec2 offset = (rot * poissonDisk[i]) * searchRadius;
        float depth = texture(shadowMap, vec3(shadowCoord.xy + offset, cascadeIdx)).r;
        if (depth < shadowCoord.z - bias) {
            avgBlockerDepth += depth;
            blockers += 1.0;
        }
    }

    if (blockers < 1.0) return 1.0;
    avgBlockerDepth /= blockers;

    float penumbraSize = (shadowCoord.z - avgBlockerDepth) * 10.0 / avgBlockerDepth;
    float filterRadius = clamp(penumbraSize * texelSize.x * 20.0, texelSize.x, 10.0 * texelSize.x);

    float shadow = 0.0;
    for (int i = 0; i < 16; i++) {
        vec2 offset = (rot * poissonDisk[i]) * filterRadius;
        float pcfDepth = texture(shadowMap, vec3(shadowCoord.xy + offset, cascadeIdx)).r;
        shadow += (shadowCoord.z - bias > pcfDepth) ? 0.0 : 1.0;
    }

    return shadow / 16.0;
}

float calculateContactShadow(vec3 worldPos, vec3 L) {
    vec4 startPos = global.viewProj * vec4(worldPos, 1.0);
    startPos.xyz /= startPos.w;
    startPos.xy = startPos.xy * 0.5 + 0.5;

    vec3 rayDir = (global.viewProj * vec4(L, 0.0)).xyz;
    rayDir = normalize(rayDir);

    float shadow = 1.0;
    vec3 currentPos = startPos.xyz;
    for(int i=0; i<16; i++) {
        currentPos += rayDir * 0.01;
        if(currentPos.x < 0.0 || currentPos.x > 1.0 || currentPos.y < 0.0 || currentPos.y > 1.0) break;
        float depth = texture(inputDepth, currentPos.xy).r; // Note: simplified access
        if(currentPos.z > depth + 0.0001) {
            shadow = 0.0;
            break;
        }
    }
    return shadow;
}

vec3 calculatePBRLighting(vec3 albedo, vec3 normal, vec3 worldPos, float metallic, float roughness, vec3 viewPos, float ssao, vec2 uv, float depth) {
    vec3 N = normalize(normal);
    vec3 V = normalize(viewPos - worldPos);
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    vec3 Lo = vec3(0.0);
    float shadow = calculateShadow(worldPos, depth, N);

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
        float attenuation = lights[lightIdx].color.w / (distance * distance);
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
