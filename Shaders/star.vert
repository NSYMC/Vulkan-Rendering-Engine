#version 450
// Yildiz kuresi: pozisyon + normal; opsiyonel kenar nabzi (vertex displacement)

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(set = 0, binding = 0) uniform UboViewProjection {
	mat4 projection;
	mat4 view;
} uboViewProjection;

layout(push_constant) uniform StarPush {
	mat4 model;       // 0
	float time;       // 64
	float temperature;// 68
	float _pad0;      // 72
	float _pad1;      // 76
} starPush;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 worldNormal;

// Basit 3D hash (kenar nabzi icin vertex displacement)
float hash31(vec3 p) {
	vec3 q = fract(p * vec3(127.1, 269.5, 419.2));
	return fract(q.x + q.y + q.z);
}
float noise3(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	float n = mix(
		mix(mix(hash31(i), hash31(i + vec3(1,0,0)), f.x),
		    mix(hash31(i + vec3(0,1,0)), hash31(i + vec3(1,1,0)), f.x), f.y),
		mix(mix(hash31(i + vec3(0,0,1)), hash31(i + vec3(1,0,1)), f.x),
		    mix(hash31(i + vec3(0,1,1)), hash31(i + vec3(1,1,1)), f.x), f.y), f.z);
	return n;
}

void main() {
	vec3 pos = inPos;
	vec3 N = normalize(inNormal);

	// Kenar nabzi: dusuk frekansli gurultu ile vertex'i normal boyunca hafifce yerinden oynat
	float pulse = noise3(pos * 2.0 + starPush.time * 0.3) * 2.0 - 1.0;
	float amp = 0.02;
	pos += N * pulse * amp;

	vec4 worldPos4 = starPush.model * vec4(pos, 1.0);
	worldPos = worldPos4.xyz;
	worldNormal = normalize(mat3(transpose(inverse(starPush.model))) * N);

	gl_Position = uboViewProjection.projection * uboViewProjection.view * worldPos4;
}
