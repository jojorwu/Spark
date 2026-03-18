#version 450
#extension GL_ARB_shader_draw_parameters : enable
#extension GL_EXT_buffer_reference : require

layout(location = 0) in vec3 inPos;

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

layout(push_constant) uniform Push {
    mat4 light_view_proj;
    uint64_t objectBufferAddress;
} push;

void main() {
    ObjectDataRef objectBuffer = ObjectDataRef(push.objectBufferAddress);
    uint objIdx = gl_InstanceIndex;
    mat4 model = objectBuffer.objects[objIdx].model;

    gl_Position = push.light_view_proj * model * vec4(inPos, 1.0);
}
