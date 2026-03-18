#version 450

layout (location = 0) out vec3 nearPoint;
layout (location = 1) out vec3 farPoint;

layout (set = 0, binding = 0) uniform GlobalUBO {
    mat4 viewProj;
    mat4 lightViewProj;
    mat4 invViewProj;
    vec4 cameraPos;
    vec4 frustum[6];
} global;

vec3 UnprojectPoint(float x, float y, float z, mat4 invVP) {
    vec4 unprojectedPoint =  invVP * vec4(x, y, z, 1.0);
    return unprojectedPoint.xyz / unprojectedPoint.w;
}

void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec4 pos = vec4(p * 2.0f - 1.0f, 1.0f, 1.0f); // Fullscreen quad at far plane

    nearPoint = UnprojectPoint(pos.x, pos.y, 0.0, global.invViewProj).xyz;
    farPoint = UnprojectPoint(pos.x, pos.y, 1.0, global.invViewProj).xyz;

    gl_Position = pos;
}
