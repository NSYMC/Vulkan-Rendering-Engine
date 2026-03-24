#version 450
layout(location = 0) in vec2 inNDC;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform SkyPush {
	mat4 invViewProj;
	vec3 cameraPos;
	float _pad0;
	vec3 skyTop;
	float _pad1;
	vec3 skyBottom;
	float _pad2;
} sky;

void main() {
	// Ray at far plane (NDC z = 1)
	vec4 ndcFar = vec4(inNDC, 1.0, 1.0);
	vec4 worldFar = sky.invViewProj * ndcFar;
	vec3 worldPos = worldFar.xyz / worldFar.w;
	vec3 viewDir = normalize(worldPos - sky.cameraPos);

	// Gradient: up (viewDir.y = 1) -> skyTop, down (viewDir.y = -1) -> skyBottom
	float t = viewDir.y * 0.5 + 0.5;
	t = smoothstep(0.0, 1.0, t);  // softer transition
	vec3 color = mix(sky.skyBottom, sky.skyTop, t);

	outColor = vec4(color, 1.0);

	// Sky is at far plane so scene draws on top
	gl_FragDepth = 1.0;
}
