#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inWorldPos;
layout(location = 3) in vec3 inColor;
layout(location = 4) in flat uint inMaterialIndex;

layout(location = 0) out vec4 outColor;

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

void main() {
    MaterialData mat = materials[inMaterialIndex];

    vec4 albedo = mat.albedoFactor * vec4(inColor, 1.0);
    if (mat.albedoTexture >= 0) {
        albedo *= texture(textures[nonuniformEXT(mat.albedoTexture)], inTexCoord);
    }

    // Basic lighting for forward pass (transparent objects)
    // In a full engine, we'd use the same PBR logic here or a simplified version.
    vec3 N = normalize(inNormal);
    vec3 L = normalize(vec3(10.0, 10.0, 10.0)); // Hardcoded light for now
    float diff = max(dot(N, L), 0.2);

    outColor = vec4(albedo.rgb * diff, albedo.a);
}
