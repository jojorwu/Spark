#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) in flat int v_tex_idx;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D textures[];

void main() {
    vec4 tex_color = vec4(1.0);
    if (v_tex_idx >= 0) {
        tex_color = texture(textures[nonuniformEXT(v_tex_idx)], v_uv);
    }
    f_color = v_color * tex_color;
    if (f_color.a < 0.1) discard;
}
