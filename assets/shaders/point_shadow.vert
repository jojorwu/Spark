#version 450

layout(location = 0) in vec4 in_pos;

layout(push_constant) uniform Push {
    mat4 light_vp;
    vec4 light_pos;
} pc;

layout(location = 0) out vec3 v_world_pos;

void main() {
    gl_Position = pc.light_vp * in_pos;
    v_world_pos = in_pos.xyz;
}
