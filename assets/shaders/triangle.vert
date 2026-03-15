#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in mat4 instanceModel;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec4 fragShadowCoord;

layout(push_constant) uniform Push {
    mat4 view_proj;
    mat4 light_view_proj;
} push;

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

const mat4 biasMat = mat4(
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );

void main() {
    vec4 worldPos = instanceModel * vec4(inPos, 1.0);
    gl_Position = push.view_proj * worldPos;
    fragColor = inColor;
    fragTexCoord = inTexCoord;
    fragShadowCoord = (biasMat * push.light_view_proj) * worldPos;
}
