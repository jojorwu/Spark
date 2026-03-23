#version 450

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec4 out_hdr;

void main() {
    float dist = length(in_uv - 0.5);
    if (dist > 0.5) discard;

    float alpha = in_color.a * (1.0 - dist * 2.0);
    out_hdr = vec4(in_color.rgb * 5.0, alpha); // Emissive particles
}
