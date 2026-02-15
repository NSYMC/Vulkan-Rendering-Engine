#version 450 // use gls 4.5

layout(location = 0) in vec3 pos;  // output colour for vertez (location is required)
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 tex;
layout(location = 3) in vec3 inputNormal;

layout(set = 0, binding = 0) uniform UboViewProjection{
	mat4 projection;
	mat4 view;
} uboViewProjection;


layout(set = 0, binding = 1) uniform UboModel{
	mat4 model;
} uboModel;


layout(push_constant) uniform PushModel {
	mat4 model;
} pushModel;

layout(location = 0) out vec3 fragCol;
layout(location = 1) out vec2 fragTex;
layout(location = 4) out vec3 worldPos;
layout(location = 5) out vec3 worldNormal;


void main() {
	gl_Position = uboViewProjection.projection * uboViewProjection.view * pushModel.model *vec4(pos,1.0);
	
	fragCol = color;
	fragTex = tex;
	
	mat4 modelMatrix = pushModel.model;
	
	worldPos = (modelMatrix * vec4(pos,1.0)).xyz;
	
	worldNormal = normalize(mat3(transpose(inverse(modelMatrix))) * inputNormal);
}