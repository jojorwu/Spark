#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inWorldPos;
layout(location = 3) in vec3 inColor;
layout(location = 4) in flat uint inMaterialIndex;
layout(location = 5) in vec4 inCurrPos;
layout(location = 6) in vec4 inPrevPos;

struct MaterialData {
    vec4 albedoFactor;
    vec4 emissiveFactor;
    float metallicFactor;
    float roughnessFactor;
    float alphaCutoff;
    uint flags;
    int albedoTexture;
    int normalTexture;
    int metallicRoughnessTexture;
    int emissiveTexture;
    int occlusionTexture;
    int padding[3];
};

layout(std430, set = 0, binding = 5) readonly buffer MaterialBuffer {
    MaterialData materials[];
};

layout(set = 1, binding = 0) uniform sampler2D textures[];

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outPBR;
layout(location = 3) out vec2 outVelocity;

layout(push_constant) uniform PushConstants {
    uint lightCount;
    float metallic;
    float roughness;
    float width;
    float height;
    uint  padding;
    uint64_t objectBufferAddress;
    mat4 prevViewProj;
} push;

void main() {
    MaterialData mat = materials[inMaterialIndex];

    vec4 albedo = mat.albedoFactor * vec4(inColor, 1.0);
    if (mat.albedoTexture >= 0) {
        albedo *= texture(textures[nonuniformEXT(mat.albedoTexture)], inTexCoord);
    }

    if (albedo.a < mat.alphaCutoff) {
        discard;
    }

    vec3 normal = normalize(inNormal);
    if (mat.normalTexture >= 0) {
        vec3 tangentNormal = texture(textures[nonuniformEXT(mat.normalTexture)], inTexCoord).xyz * 2.0 - 1.0;
        // Basic TBN calculation if needed, or just use world space normals for now
        // For simplicity in this step, we'll stick to vertex normals if no TBN is passed.
    }

    float metallic = mat.metallicFactor;
    float roughness = mat.roughnessFactor;
    if (mat.metallicRoughnessTexture >= 0) {
        vec4 mrSample = texture(textures[nonuniformEXT(mat.metallicRoughnessTexture)], inTexCoord);
        metallic *= mrSample.b;
        roughness *= mrSample.g;
    }

    outAlbedo = albedo;
    outNormal = vec4(normal * 0.5 + 0.5, 1.0);
    outPBR = vec4(metallic, roughness, 0.0, 1.0);

    vec2 currPos = (inCurrPos.xy / inCurrPos.w) * 0.5 + 0.5;
    vec2 prevPos = (inPrevPos.xy / inPrevPos.w) * 0.5 + 0.5;
    outVelocity = currPos - prevPos;
}
