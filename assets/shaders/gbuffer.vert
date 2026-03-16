#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in mat4 inModel; // Locations 4, 5, 6, 7

layout (set = 0, binding = 0) uniform GlobalUBO {
    mat4 viewProj;
    mat4 lightViewProj;
    mat4 invViewProj;
} global;

layout(push_constant) uniform PushConstants {
    uint lightCount;
    float metallic;
    float roughness;
    float width;
    float height;
} push;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outTexCoord;
layout(location = 2) out vec3 outWorldPos;
layout(location = 3) out vec3 outColor;

void main() {
    vec4 worldPos = inModel * vec4(inPos, 1.0);
    outWorldPos = worldPos.xyz;
    outNormal = mat3(inModel) * inNormal;
    outTexCoord = inTexCoord;
    outColor = inColor;
    gl_Position = global.viewProj * worldPos;
}
