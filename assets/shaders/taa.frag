#version 450

layout (set = 0, binding = 0) uniform sampler2D currentFrame;
layout (set = 0, binding = 1) uniform sampler2D historyFrame;
layout (set = 0, binding = 2) uniform sampler2D velocityBuffer;
layout (set = 0, binding = 3) uniform sampler2D depthBuffer;

layout (location = 0) out vec4 outColor;

void main() {
    vec2 texCoord = gl_FragCoord.xy / textureSize(currentFrame, 0);
    vec2 velocity = texture(velocityBuffer, texCoord).rg;
    vec2 historyCoord = texCoord - velocity;

    vec4 current = texture(currentFrame, texCoord);

    if (historyCoord.x < 0.0 || historyCoord.x > 1.0 || historyCoord.y < 0.0 || historyCoord.y > 1.0) {
        outColor = current;
        return;
    }

    vec4 history = texture(historyFrame, historyCoord);

    // Neighborhood clamping
    vec4 minColor = vec4(1000.0), maxColor = vec4(-1000.0);
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec4 c = textureOffset(currentFrame, texCoord, ivec2(x, y));
            minColor = min(minColor, c);
            maxColor = max(maxColor, c);
        }
    }
    history = clamp(history, minColor, maxColor);

    float feedback = 0.9;
    outColor = mix(current, history, feedback);
}
