#version 450

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inWorldPos;
layout(location = 3) in vec3 inColor;

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outPosition;

void main() {
    outAlbedo = texture(texSampler, inTexCoord) * vec4(inColor, 1.0);
    outNormal = vec4(normalize(inNormal), 1.0);
    outPosition = vec4(inWorldPos, 1.0);
}
