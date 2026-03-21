#version 450

struct Particle {
    vec4 pos;
    vec4 vel;
    vec4 color;
    float life;
    float size;
    vec2 padding;
};

layout(std430, binding = 0) buffer Particles {
    Particle particles[];
};

layout(set = 1, binding = 0) uniform GlobalUBO {
    mat4 vp;
    mat4 lvp[4];
    mat4 inv_vp;
    vec4 camera_pos;
    vec4 frustum[6];
    vec4 cascade_splits;
} global;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_uv;

void main() {
    uint particle_idx = gl_VertexIndex / 4;
    uint vertex_in_particle = gl_VertexIndex % 4;

    Particle p = particles[particle_idx];
    if (p.life <= 0.0) {
        gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    vec3 pos = p.pos.xyz;
    vec3 cam_right = normalize(vec3(global.vp[0][0], global.vp[1][0], global.vp[2][0]));
    vec3 cam_up = normalize(vec3(global.vp[0][1], global.vp[1][1], global.vp[2][1]));

    vec2 uv = vec2((vertex_in_particle << 1) & 2, vertex_in_particle & 2);
    vec3 billboard_pos = pos + (cam_right * (uv.x - 0.5) * p.size) + (cam_up * (uv.y - 0.5) * p.size);

    gl_Position = global.vp * vec4(billboard_pos, 1.0);
    out_color = p.color;
    out_uv = uv;
}
