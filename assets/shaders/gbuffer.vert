#version 450
#extension GL_ARB_shader_draw_parameters : enable
#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

struct ObjectData {
    mat4 model;
    vec4 sphere;
    uint indexCount;
    uint firstIndex;
    int  vertexOffset;
    uint materialIndex;
};

layout(buffer_reference, std430) readonly buffer ObjectDataRef {
    ObjectData objects[];
};

layout (set = 0, binding = 0) uniform GlobalUBO {
    mat4 viewProj;
    mat4 lightViewProj;
    mat4 invViewProj;
    vec4 cameraPos;
    vec4 frustum[6];
} global;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outTexCoord;
layout(location = 2) out vec3 outWorldPos;
layout(location = 3) out vec3 outColor;
layout(location = 4) out flat uint outMaterialIndex;

layout(push_constant) uniform PushConstants {
    uint lightCount;
    float metallic;
    float roughness;
    float width;
    float height;
    uint  padding;
    uint64_t objectBufferAddress;
} push;

void main() {
    ObjectDataRef objectBuffer = ObjectDataRef(push.objectBufferAddress);
    uint objIdx = gl_InstanceIndex;
    mat4 model = objectBuffer.objects[objIdx].model;

    vec4 worldPos = model * vec4(inPos, 1.0);
    outWorldPos = worldPos.xyz;
    outNormal = mat3(model) * inNormal;
    outTexCoord = inTexCoord;
    outColor = inColor;
    outMaterialIndex = objectBuffer.objects[objIdx].materialIndex;

    gl_Position = global.viewProj * worldPos;
}
