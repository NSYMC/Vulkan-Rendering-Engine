#version 450
// Yildiz: sicaklik -> blackbody renk, FBM ile granulation (kaynayan plazma), zamanla animasyon

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 worldNormal;

layout(push_constant) uniform StarPush {
	mat4 model;
	float time;
	float temperature;
	float _pad0;
	float _pad1;
} starPush;

layout(location = 0) out vec4 outColor;

// --- 3D value noise (FBM icin) ---
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise3(vec3 v) {
	const vec2 C = vec2(1.0/6.0, 1.0/3.0);
	const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
	vec3 i  = floor(v + dot(v, C.yyy));
	vec3 x0 = v - i + dot(i, C.xxx);
	vec3 g = step(x0.yzx, x0.xyz);
	vec3 l = 1.0 - g;
	vec3 i1 = min(g.xyz, l.zxy);
	vec3 i2 = max(g.xyz, l.zxy);
	vec3 x1 = x0 - i1 + C.xxx;
	vec3 x2 = x0 - i2 + C.yyy;
	vec3 x3 = x0 - D.yyy;
	i = mod289(i);
	vec4 p = permute(permute(permute(
		i.z + vec4(0.0, i1.z, i2.z, 1.0))
		+ i.y + vec4(0.0, i1.y, i2.y, 1.0))
		+ i.x + vec4(0.0, i1.x, i2.x, 1.0));
	float n_ = 0.142857142857;
	vec3 ns = n_ * D.wyz - D.xzx;
	vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
	vec4 x_ = floor(j * ns.z);
	vec4 y_ = floor(j - 7.0 * x_);
	vec4 x = x_ *ns.x + ns.yyyy;
	vec4 y = y_ *ns.x + ns.yyyy;
	vec4 h = 1.0 - abs(x) - abs(y);
	vec4 b0 = vec4(x.xy, y.xy);
	vec4 b1 = vec4(x.zw, y.zw);
	vec4 s0 = floor(b0)*2.0 + 1.0;
	vec4 s1 = floor(b1)*2.0 + 1.0;
	vec4 sh = -step(h, vec4(0.0));
	vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
	vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
	vec3 p0 = vec3(a0.xy, h.x);
	vec3 p1 = vec3(a0.zw, h.y);
	vec3 p2 = vec3(a1.xy, h.z);
	vec3 p3 = vec3(a1.zw, h.w);
	vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
	p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
	vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), vec4(0.0));
	m = m * m;
	return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

// FBM: cok katmanli gurultu (granulation)
float fbm3(vec3 p) {
	float f = 0.0;
	float a = 0.5;
	float freq = 4.0;
	for (int i = 0; i < 5; i++) {
		f += a * snoise3(p * freq);
		freq *= 2.0;
		a *= 0.5;
	}
	return f;
}

// Sicaklik (Kelvin) -> Blackbody RGB (3000K kirmizi, 5800K sari/beyaz, 15000K+ mavi/beyaz)
vec3 blackbody(float T) {
	T = clamp(T, 1000.0, 40000.0);
	float t = T / 100.0;
	float r, g, b;
	if (t <= 66.0) {
		r = 1.0;
		g = clamp(0.390081578 * log(max(t, 0.01)) - 0.631841443, 0.0, 1.0);
		b = t <= 19.0 ? 0.0 : clamp(0.543206789 * log(max(t - 10.0, 0.01)) - 1.196254089, 0.0, 1.0);
	} else {
		r = clamp(1.292936186 * pow(t - 60.0, -0.1332047592), 0.0, 1.0);
		g = clamp(1.129890861 * pow(t - 60.0, -0.0755148492), 0.0, 1.0);
		b = 1.0;
	}
	return vec3(r, g, b);
}

void main() {
	float T = starPush.temperature;
	// Sicak yildizlar biraz daha hizli kaynasin
	float timeScale = 1.0 + (T - 3000.0) / 30000.0;
	vec3 samplePos = worldPos * 3.0 + vec3(starPush.time * timeScale * 0.5, 0.0, 0.0);
	float n = fbm3(samplePos);
	// Granulation: koyu (soguk) ve parlak (sicak) noktalar; 0..1 araligina cek
	float gran = n * 0.5 + 0.5;
	gran = 0.7 + 0.35 * gran;

	vec3 baseColor = blackbody(T);
	vec3 finalColor = baseColor * gran;

	// Hafif parlaklik (yildiz parlar)
	float intensity = 1.0 + 0.15 * (1.0 + n);
	finalColor *= intensity;
	finalColor = clamp(finalColor, vec3(0.0), vec3(1.5));

	outColor = vec4(finalColor, 1.0);
}
