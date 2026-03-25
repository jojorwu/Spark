struct RayPayload {
    vec3 color;
    float dist;
    uint hit;
    vec3 normal;
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

vec3 unpackNormal(uint p) {
    vec3 n;
    n.x = float(p & 0x3FF) / 1023.0 * 2.0 - 1.0;
    n.y = float((p >> 10) & 0x3FF) / 1023.0 * 2.0 - 1.0;
    n.z = float((p >> 20) & 0x3FF) / 1023.0 * 2.0 - 1.0;
    return normalize(n);
}
