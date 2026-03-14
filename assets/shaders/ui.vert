#version 450
layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_tc;
layout(location = 2) in vec4 in_srgba;

layout(location = 0) out vec4 v_rgba;
layout(location = 1) out vec2 v_tc;

layout(push_constant) uniform Push {
    vec2 screen_size;
} push;

void main() {
    gl_Position = vec4(
        2.0 * in_pos.x / push.screen_size.x - 1.0,
        2.0 * in_pos.y / push.screen_size.y - 1.0,
        0.0,
        1.0
    );
    v_rgba = in_srgba;
    v_tc = in_tc;
}
