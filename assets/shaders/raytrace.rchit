#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#include "raytrace_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;
hitAttributeEXT vec2 attribs;

layout(binding = 2, set = 1, scalar) buffer Vertices { Vertex v[]; } vertices;
layout(binding = 3, set = 1) buffer Indices { uint i[]; } indices;
layout(binding = 4, set = 1, scalar) buffer Meshes { MeshData m[]; } meshes;
layout(binding = 5, set = 1, scalar) buffer Materials { MaterialData m[]; } materials;

void main()
{
  uint instanceID = gl_InstanceCustomIndexEXT;
  MeshData mesh = meshes.m[instanceID];
  MaterialData mat = materials.m[mesh.material_index];

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

  payload.color = mat.albedo_factor.rgb;
  payload.dist = gl_HitTEXT;
  payload.hit = 1;
  payload.normal = normal;
  payload.material_index = mesh.material_index;
  payload.roughness = mat.roughness_factor;
  payload.metallic = mat.metallic_factor;
  payload.emissive = mat.emissive_factor.rgb;
}
