#version 450

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inWorldPos;
layout(location = 3) in vec3 inColor;

layout(set = 1, binding = 0) uniform sampler2D textures[];

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outPBR;

layout(push_constant) uniform PushConstants {
    uint lightCount;
    float metallic;
    float roughness;
    float width;
    float height;
    uint textureIndex;
} push;

void main() {
    outAlbedo = texture(textures[nonuniformEXT(push.textureIndex)], inTexCoord) * vec4(inColor, 1.0);
    // Transform normal to [0, 1] range for UNORM storage
    outNormal = vec4(normalize(inNormal) * 0.5 + 0.5, 1.0);
    outPBR = vec4(push.metallic, push.roughness, 0.0, 1.0);
}
