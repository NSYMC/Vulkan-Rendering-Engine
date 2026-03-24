#version 450
// Shadow map pass: only output depth from the light's view.
// We only need gl_Position in light clip space; no color output.

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 tex;
layout(location = 3) in vec3 inputNormal;

layout(set = 0, binding = 0) uniform ShadowUBO {
	mat4 lightViewProj;
} shadowUbo;

layout(push_constant) uniform PushModel {
	mat4 model;
} pushModel;

void main() {
	// Transform vertex to light clip space; the rasterizer will write depth.
	gl_Position = shadowUbo.lightViewProj * pushModel.model * vec4(pos, 1.0);
}
