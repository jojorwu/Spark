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
layout (set = 1, binding = 11) uniform sampler2D ssgiTex;

struct Light {
    vec4 pos_range; // pos.xyz, range
    vec4 dir_type;  // dir.xyz, type
    vec4 color_intensity;
    vec4 spot_angles; // inner, outer, 0, 0
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
    float ssgiIntensity;
    uint shadowPCF;
    float zNear;
    float zFar;
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
 * Вычисляет коэффициент затенения для заданной позиции.
 * Calculates shadow factor for a given world position using Cascaded Shadow Maps.
 */
float calculateShadow(vec3 worldPos, float linearDepth, vec3 N) {
    vec3 L = normalize(vec3(0.5, 1.0, 0.5)); // Directional light dir
    float bias = max(0.005 * (1.0 - dot(N, L)), 0.001);

    // Выбор подходящего каскада на основе линейной глубины.
    // Shadow cascade selection based on linear depth splits.
    uint cascadeIdx = 0;
    if (linearDepth > global.cascadeSplits[0]) cascadeIdx = 1;
    if (linearDepth > global.cascadeSplits[1]) cascadeIdx = 2;
    if (linearDepth > global.cascadeSplits[2]) cascadeIdx = 3;

    vec4 shadowCoord = global.lightViewProj[cascadeIdx] * vec4(worldPos, 1.0);
    shadowCoord.xyz /= shadowCoord.w;
    shadowCoord.xy = shadowCoord.xy * 0.5 + 0.5;

    if (shadowCoord.z > 1.0) return 1.0;

    float shadow = 0.0;
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0).xy);

    if (push.shadowPCF == 0) {
        float pcfDepth = texture(shadowMap, vec3(shadowCoord.xy, cascadeIdx)).r;
        shadow = shadowCoord.z - bias > pcfDepth ? 0.0 : 1.0;
        return shadow;
    }

    // 3x3 PCF
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, vec3(shadowCoord.xy + vec2(x, y) * texelSize, cascadeIdx)).r;
            shadow += shadowCoord.z - bias > pcfDepth ? 0.0 : 1.0;
        }
    }
    return shadow / 9.0;
}

/**
 * Основная функция расчета PBR освещения (Cook-Torrance BRDF).
 * Main PBR lighting calculation using Cook-Torrance BRDF.
 */
vec3 calculatePBRLighting(vec3 albedo, vec3 normal, vec3 worldPos, float metallic, float roughness, vec3 viewPos, float ssao, vec3 ssgi, float depth, LightGrid grid, float shadow) {
    vec3 N = normalize(normal);
    vec3 V = normalize(viewPos - worldPos);
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    vec3 Lo = vec3(0.0);

    // Итерация по источникам света в текущем кластере.
    // Iterate over local lights assigned to this cluster.
    for (uint i = 0; i < grid.count; i++) {
        uint lightIdx = globalIndexList[grid.offset + i];
        Light light = lights[lightIdx];

        vec3 L;
        float attenuation = 1.0;

        if (light.dir_type.w == 0.0) { // Directional
            L = normalize(-light.dir_type.xyz);
        } else { // Point or Spot
            L = normalize(light.pos_range.xyz - worldPos);
            float dist = length(light.pos_range.xyz - worldPos);
            attenuation = max(0.0, 1.0 - (dist / light.pos_range.w));
            attenuation *= attenuation;

            if (light.dir_type.w == 2.0) { // Spot
                float theta = dot(L, normalize(-light.dir_type.xyz));
                float epsilon = light.spot_angles.x - light.spot_angles.y;
                float intensity = clamp((theta - light.spot_angles.y) / epsilon, 0.0, 1.0);
                attenuation *= intensity;
            }
        }

        vec3 H = normalize(V + L);
        vec3 radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation;

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
        float s = (light.dir_type.w == 0.0) ? shadow : 1.0; // Apply shadow only to directional for now
        Lo += (kD * albedo / PI + specular) * radiance * NdotL * s;
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

    vec3 ambient = (kD * (diffuse + ssgi) + envSpecular) * ssao;
    vec3 color = ambient + Lo;
    return color;
}

vec3 worldPosFromDepth(float depth, vec2 clipXY) {
    vec4 clipSpacePos = vec4(clipXY, depth, 1.0);
    vec4 worldSpacePos = global.invViewProj * clipSpacePos;
    return worldSpacePos.xyz / worldSpacePos.w;
}

void main()
{
    vec2 texCoord = gl_FragCoord.xy / vec2(push.width, push.height);
    vec2 clipXY = texCoord * 2.0 - 1.0;
    vec3 viewPos = global.cameraPos.xyz;

    // Расчет индекса кластера один раз для пикселя (оптимизация для MSAA).
    // Calculate cluster index once per pixel (optimization for MSAA).
#if MSAA_SAMPLES > 1
    float mainDepth = subpassLoad(inputDepth, 0).r;
    vec3 mainNormal = subpassLoad(inputNormal, 0).rgb * 2.0 - 1.0;
#else
    float mainDepth = subpassLoad(inputDepth).r;
    vec3 mainNormal = subpassLoad(inputNormal).rgb * 2.0 - 1.0;
#endif

    float linearMainDepth = (2.0 * push.zNear) / (push.zFar + push.zNear - mainDepth * (push.zFar - push.zNear));

    uint zSlices = 24;
    uint clusterZ = uint(max(0.0, log(linearMainDepth / push.zNear) * float(zSlices) / log(push.zFar / push.zNear)));
    uvec2 clusterXY = uvec2(texCoord * vec2(16, 9));
    uint clusterIndex = clusterXY.x + clusterXY.y * 16 + clusterZ * 16 * 9;

    // Предварительный расчет теней (используем основную глубину для эффективности).
    // Pre-calculate shadows using the primary depth for efficiency.
    vec3 mainPos = worldPosFromDepth(mainDepth, clipXY);
    float shadow = calculateShadow(mainPos, linearMainDepth, mainNormal);

    float ssao = texture(ssaoTex, texCoord).r;
    vec3 ssgi = texture(ssgiTex, texCoord).rgb * push.ssgiIntensity;
    LightGrid grid = lightGrids[clusterIndex];

#if MSAA_SAMPLES > 1
    vec3 color = vec3(0.0);
    for (int i = 0; i < MSAA_SAMPLES; i++) {
        vec3 albedo = subpassLoad(inputAlbedo, i).rgb;
        vec3 normal = subpassLoad(inputNormal, i).rgb * 2.0 - 1.0;
        float depth = subpassLoad(inputDepth, i).r;
        vec3 position = worldPosFromDepth(depth, clipXY);
        vec2 pbr = subpassLoad(inputPBR, i).rg;
        float linearDepth = (2.0 * push.zNear) / (push.zFar + push.zNear - depth * (push.zFar - push.zNear));

        color += calculatePBRLighting(albedo, normal, position, pbr.x, pbr.y, viewPos, ssao, ssgi, linearDepth, grid, shadow);
    }
    outColor = vec4(color / float(MSAA_SAMPLES), 1.0);
#else
    vec3 albedo = subpassLoad(inputAlbedo).rgb;
    vec3 normal = subpassLoad(inputNormal).rgb * 2.0 - 1.0;
    float depth = subpassLoad(inputDepth).r;
    vec3 position = worldPosFromDepth(depth, clipXY);
    vec2 pbr = subpassLoad(inputPBR).rg;
    float linearDepth = (2.0 * push.zNear) / (push.zFar + push.zNear - depth * (push.zFar - push.zNear));

    outColor = vec4(calculatePBRLighting(albedo, normal, position, pbr.x, pbr.y, viewPos, ssao, ssgi, linearDepth, grid, shadow), 1.0);
#endif
}
