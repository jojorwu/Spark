#version 450
#extension GL_ARB_shader_draw_parameters : enable
#extension GL_EXT_buffer_reference : require

struct Vertex {
    float pos[3];
    uint normal;
    uint texCoord;
    uint color;
};

layout(buffer_reference, std430) readonly buffer VertexBufferRef {
    Vertex vertices[];
};

struct ObjectData {
    vec4 modelRow0;
    vec4 modelRow1;
    vec4 modelRow2;
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
layout(location = 5) out vec4 outCurrPos;
layout(location = 6) out vec4 outPrevPos;

layout(push_constant) uniform PushConstants {
    uint lightCount;
    float metallic;
    float roughness;
    float width;
    float height;
    uint  padding;
    uint64_t objectBufferAddress;
    mat4 prevViewProj;
    uint64_t vertexBufferAddress;
} push;

vec3 unpackNormal(uint p) {
    vec3 n = vec3(float(p & 0x3FF), float((p >> 10) & 0x3FF), float((p >> 20) & 0x3FF));
    return n / 1023.0 * 2.0 - 1.0;
}

vec2 unpackTexCoord(uint p) {
    return vec2(float(p & 0xFFFF), float(p >> 16)) / 65535.0;
}

vec3 unpackColor(uint p) {
    return vec3(float(p & 0xFF), float((p >> 8) & 0xFF), float((p >> 16) & 0xFF)) / 255.0;
}

void main() {
    ObjectDataRef objectBuffer = ObjectDataRef(push.objectBufferAddress);
    VertexBufferRef vertexBuffer = VertexBufferRef(push.vertexBufferAddress);

    uint objIdx = gl_InstanceIndex;
    ObjectData obj = objectBuffer.objects[objIdx];

    Vertex v = vertexBuffer.vertices[gl_VertexIndex];
    vec3 pos = vec3(v.pos[0], v.pos[1], v.pos[2]);

    vec4 worldPos;
    worldPos.x = dot(obj.modelRow0, vec4(pos, 1.0));
    worldPos.y = dot(obj.modelRow1, vec4(pos, 1.0));
    worldPos.z = dot(obj.modelRow2, vec4(pos, 1.0));
    worldPos.w = 1.0;
    outWorldPos = worldPos.xyz;

    vec3 localNormal = unpackNormal(v.normal);
    outNormal.x = dot(obj.modelRow0.xyz, localNormal);
    outNormal.y = dot(obj.modelRow1.xyz, localNormal);
    outNormal.z = dot(obj.modelRow2.xyz, localNormal);
    outNormal = normalize(outNormal);

    outTexCoord = unpackTexCoord(v.texCoord);
    outColor = unpackColor(v.color);
    outMaterialIndex = objectBuffer.objects[objIdx].materialIndex;

    outCurrPos = global.viewProj * worldPos;
    outPrevPos = push.prevViewProj * worldPos;

    gl_Position = outCurrPos;
}
