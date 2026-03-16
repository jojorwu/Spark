#version 450
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in mat4 instanceModel;

layout(push_constant) uniform Push {
    mat4 light_view_proj;
} push;

void main() {
    gl_Position = push.light_view_proj * instanceModel * vec4(inPos, 1.0);
}
