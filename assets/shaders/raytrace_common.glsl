struct RayPayload {
    vec3 color;
    float dist;
    uint hit;
    vec3 normal;
    uint material_index;
    float roughness;
    float metallic;
    vec3 emissive;
};

struct Vertex {
    vec3 pos;
    float padding;
    uint normal;
    uint tex_coord;
    uint color;
    uint tangent;
};

struct MeshData {
    vec4 model_row0;
    vec4 model_row1;
    vec4 model_row2;
    vec4 sphere;
    uint index_count;
    uint first_index;
    int vertex_offset;
    uint material_index;
};

struct MaterialData {
    vec4 albedo_factor;
    vec4 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float alpha_cutoff;
    uint flags;
    int albedo_texture;
    int normal_texture;
    int metallic_roughness_texture;
    int emissive_texture;
    int occlusion_texture;
    int padding[3];
};

struct Light {
    vec4 pos_range; // pos.xyz, range
    vec4 dir_type;  // dir.xyz, type (0: Dir, 1: Point, 2: Spot)
    vec4 col_intensity; // col.rgb, intensity
    vec4 spot_angles; // inner, outer, _, _
};

vec3 unpackNormal(uint p) {
    vec3 n;
    n.x = float(p & 0x3FF) / 1023.0 * 2.0 - 1.0;
    n.y = float((p >> 10) & 0x3FF) / 1023.0 * 2.0 - 1.0;
    n.z = float((p >> 20) & 0x3FF) / 1023.0 * 2.0 - 1.0;
    return normalize(n);
}

#include "pbr_lighting.glsl"

vec3 brdf(vec3 L, vec3 V, vec3 N, vec3 albedo, float metallic, float roughness) {
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    vec3 specular = numerator / denominator;

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    return (kD * albedo / PI + specular) * NdotL;
}
