#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable

struct RayPayload {
    vec3 color;
    float dist;
    uint hit;
    vec3 normal;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;
hitAttributeEXT vec2 attribs;

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

layout(binding = 2, set = 1, scalar) buffer Vertices { Vertex v[]; } vertices;
layout(binding = 3, set = 1) buffer Indices { uint i[]; } indices;
layout(binding = 4, set = 1, scalar) buffer Meshes { MeshData m[]; } meshes;

vec3 unpackNormal(uint p) {
    vec3 n;
    n.x = float(p & 0x3FF) / 1023.0 * 2.0 - 1.0;
    n.y = float((p >> 10) & 0x3FF) / 1023.0 * 2.0 - 1.0;
    n.z = float((p >> 20) & 0x3FF) / 1023.0 * 2.0 - 1.0;
    return normalize(n);
}

void main()
{
  uint instanceID = gl_InstanceCustomIndexEXT;
  MeshData mesh = meshes.m[instanceID];

  uint primitiveID = gl_PrimitiveID;
  uint i0 = indices.i[mesh.first_index + 3 * primitiveID + 0];
  uint i1 = indices.i[mesh.first_index + 3 * primitiveID + 1];
  uint i2 = indices.i[mesh.first_index + 3 * primitiveID + 2];

  Vertex v0 = vertices.v[mesh.vertex_offset + int(i0)];
  Vertex v1 = vertices.v[mesh.vertex_offset + int(i1)];
  Vertex v2 = vertices.v[mesh.vertex_offset + int(i2)];

  vec3 n0 = unpackNormal(v0.normal);
  vec3 n1 = unpackNormal(v1.normal);
  vec3 n2 = unpackNormal(v2.normal);

  const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
  vec3 normal = normalize(n0 * barycentricCoords.x + n1 * barycentricCoords.y + n2 * barycentricCoords.z);

  mat3 normalMatrix = mat3(gl_ObjectToWorldEXT);
  normal = normalize(normalMatrix * normal);

  payload.color = vec3(0.7);
  payload.dist = gl_HitTEXT;
  payload.hit = 1;
  payload.normal = normal;
}
