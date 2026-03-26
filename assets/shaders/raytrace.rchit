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

layout(set = 2, binding = 0) uniform sampler2D bindless_textures[];

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

  vec3 albedo = mat.albedo_factor.rgb;
  if (mat.albedo_texture >= 0) {
      uint u0 = indices.i[mesh.first_index + 3 * primitiveID + 0];
      uint u1 = indices.i[mesh.first_index + 3 * primitiveID + 1];
      uint u2 = indices.i[mesh.first_index + 3 * primitiveID + 2];
      Vertex vt0 = vertices.v[mesh.vertex_offset + int(u0)];
      Vertex vt1 = vertices.v[mesh.vertex_offset + int(u1)];
      Vertex vt2 = vertices.v[mesh.vertex_offset + int(u2)];
      vec2 uv0 = unpackHalf2x16(vt0.tex_coord);
      vec2 uv1 = unpackHalf2x16(vt1.tex_coord);
      vec2 uv2 = unpackHalf2x16(vt2.tex_coord);
      vec2 uv = uv0 * barycentricCoords.x + uv1 * barycentricCoords.y + uv2 * barycentricCoords.z;
      albedo *= texture(bindless_textures[nonuniformEXT(mat.albedo_texture)], uv).rgb;
  }

  payload.color = albedo;
  payload.dist = gl_HitTEXT;
  payload.hit = 1;
  payload.normal = normal;
  payload.material_index = mesh.material_index;
  payload.roughness = mat.roughness_factor;
  payload.metallic = mat.metallic_factor;
  payload.emissive = mat.emissive_factor.rgb;
}
