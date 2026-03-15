#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec4 fragShadowCoord;
layout(location = 3) in vec3 fragNormal;
layout(location = 4) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D texSampler;
layout(binding = 1) uniform sampler2DShadow shadowSampler;

layout(push_constant) uniform Push {
    mat4 view_proj;
    mat4 light_view_proj;
    vec3 light_pos;
    float _pad1;
    vec3 light_color;
    float _pad2;
} push;

float textureProj(vec4 shadowCoord, vec2 off)
{
	float shadow = 1.0;
	if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 )
	{
		float dist = texture(shadowSampler, vec3(shadowCoord.st + off, shadowCoord.z));
		if (shadowCoord.w > 0.0 && dist < shadowCoord.z)
		{
			shadow = 0.1;
		}
	}
	return shadow;
}

float filterPCF(vec4 sc)
{
	ivec2 texDim = textureSize(shadowSampler, 0);
	float scale = 1.5;
	float dx = scale * 1.0 / float(texDim.x);
	float dy = scale * 1.0 / float(texDim.y);

	float shadowFactor = 0.0;
	int count = 0;
	int range = 1;

	for (int x = -range; x <= range; x++)
	{
		for (int y = -range; y <= range; y++)
		{
			shadowFactor += textureProj(sc, vec2(dx*x, dy*y));
			count++;
		}

	}
	return shadowFactor / count;
}

void main() {
    float shadow = filterPCF(fragShadowCoord / fragShadowCoord.w);

    vec3 N = normalize(fragNormal);
    vec3 L = normalize(push.light_pos - fragPos);

    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * push.light_color;

    vec3 ambient = vec3(0.05);

    vec3 lighting = (ambient + diffuse * shadow) * fragColor;

    outColor = vec4(lighting, 1.0);
}
