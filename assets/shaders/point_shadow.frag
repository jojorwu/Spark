#version 450

layout(location = 0) in vec3 v_world_pos;

layout(push_constant) uniform Push {
    mat4 light_vp;
    vec4 light_pos;
} pc;

void main() {
    float dist = length(v_world_pos - pc.light_pos.xyz);
    dist = dist / pc.light_pos.w; // Far plane in w
    gl_FragDepth = dist;
}
