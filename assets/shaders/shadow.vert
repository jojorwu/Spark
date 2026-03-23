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

layout(push_constant) uniform Push {
    mat4 light_view_proj;
    uint64_t objectBufferAddress;
    uint64_t vertexBufferAddress;
} push;

void main() {
    ObjectDataRef objectBuffer = ObjectDataRef(push.objectBufferAddress);
    VertexBufferRef vertexBuffer = VertexBufferRef(push.vertexBufferAddress);

    uint objIdx = gl_InstanceIndex;
    mat4 model = mat4(
        vec4(objectBuffer.objects[objIdx].modelRow0.x, objectBuffer.objects[objIdx].modelRow1.x, objectBuffer.objects[objIdx].modelRow2.x, 0.0),
        vec4(objectBuffer.objects[objIdx].modelRow0.y, objectBuffer.objects[objIdx].modelRow1.y, objectBuffer.objects[objIdx].modelRow2.y, 0.0),
        vec4(objectBuffer.objects[objIdx].modelRow0.z, objectBuffer.objects[objIdx].modelRow1.z, objectBuffer.objects[objIdx].modelRow2.z, 0.0),
        vec4(objectBuffer.objects[objIdx].modelRow0.w, objectBuffer.objects[objIdx].modelRow1.w, objectBuffer.objects[objIdx].modelRow2.w, 1.0)
    );

    Vertex v = vertexBuffer.vertices[gl_VertexIndex];
    vec3 pos = vec3(v.pos[0], v.pos[1], v.pos[2]);

    gl_Position = push.light_view_proj * model * vec4(pos, 1.0);
}
