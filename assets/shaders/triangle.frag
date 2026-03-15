#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec4 fragShadowCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D texSampler;
layout(binding = 1) uniform sampler2DShadow shadowSampler;

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
    outColor = vec4(fragColor * shadow, 1.0);
}
