#version 450
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D hdrSampler;

layout(push_constant) uniform PushConstants {
    float exposure;
    float gamma;
    float bloom_enabled;
    float vignette_intensity;
    float vignette_smoothness;
    float chromatic_aberration;
    float film_grain;
    float motion_blur_strength;
    float auto_exposure_enabled;
    float dof_enabled;
    float lut_index;
    float time;
    float rt_reflections_enabled;
    float rt_shadows_enabled;
    float rt_ao_enabled;
    float rt_gi_enabled;
    float bloom_intensity;
    float bloom_threshold;
} params;

void main() {
    vec3 color = texture(hdrSampler, inUV).rgb;

    // Smooth thresholding (Karis 2013)
    float brightness = max(color.r, max(color.g, color.b));
    float threshold = params.bloom_threshold;
    float knee = 0.1;
    float soft = brightness - threshold + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 0.00001);
    float contribution = max(soft, brightness - threshold);
    contribution /= max(brightness, 0.00001);

    outColor = vec4(color * contribution, 1.0);
}
