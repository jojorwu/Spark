#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable

struct RayPayload {
    vec3 color;
    float dist;
    uint hit;
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

layout(binding = 2, set = 1, scalar) buffer Vertices { Vertex v[]; } vertices;
layout(binding = 3, set = 1) buffer Indices { uint i[]; } indices;

vec3 unpackNormal(uint p) {
    vec3 n;
    n.x = float(p & 0x3FF) / 1023.0 * 2.0 - 1.0;
    n.y = float((p >> 10) & 0x3FF) / 1023.0 * 2.0 - 1.0;
    n.z = float((p >> 20) & 0x3FF) / 1023.0 * 2.0 - 1.0;
    return normalize(n);
}

void main()
{
  uint primitiveID = gl_PrimitiveID;
  uint i0 = indices.i[3 * primitiveID + 0];
  uint i1 = indices.i[3 * primitiveID + 1];
  uint i2 = indices.i[3 * primitiveID + 2];

  Vertex v0 = vertices.v[i0];
  Vertex v1 = vertices.v[i1];
  Vertex v2 = vertices.v[i2];

  vec3 n0 = unpackNormal(v0.normal);
  vec3 n1 = unpackNormal(v1.normal);
  vec3 n2 = unpackNormal(v2.normal);

  const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
  vec3 normal = normalize(n0 * barycentricCoords.x + n1 * barycentricCoords.y + n2 * barycentricCoords.z);

  payload.color = normal * 0.5 + 0.5;
  payload.dist = gl_HitTEXT;
  payload.hit = 1;
}
