#version 450

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;
layout(location = 2) out flat int v_tex_idx;

struct SpriteData {
    mat4 model;
    vec4 color;
    vec2 size;
    int texture_idx;
    int padding;
};

layout(std430, set = 0, binding = 1) readonly buffer SpriteBuffer {
    SpriteData sprites[];
};

layout(push_constant) uniform Push {
    mat4 view_proj;
} pc;

void main() {
    SpriteData sprite = sprites[gl_InstanceIndex];
    vec2 pos = in_pos * sprite.size;
    gl_Position = pc.view_proj * sprite.model * vec4(pos, 0.0, 1.0);
    v_uv = in_uv;
    v_color = sprite.color;
    v_tex_idx = sprite.texture_idx;
}
