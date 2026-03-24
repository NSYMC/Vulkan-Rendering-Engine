#version 450
// Set to 1 to visualize shadow map depth (grayscale); 0 = normal
#define DEBUG_SHADOW_MAP 0
// Set to 1 to visualize shadow factor (0=black shadow, 1=white lit); 0 = normal
#define DEBUG_SHADOW_FACTOR 0

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTex;
layout(location = 4) in vec3 worldPos;
layout(location = 5) in vec3 worldNormal;
layout(location = 6) in vec3 objectCenter;  // vertex'ten: obje merkezi (yildiz isik yonu)

layout(set = 1, binding=0) uniform sampler2D textureSampler;
layout(set = 0, binding = 2) uniform sampler2D shadowMap;

// Point light (must match PointLightUbo: 3 x vec4)
struct PointLightData {
	vec4 position;   // xyz = world pos, w = enabled (0/1)
	vec4 color;      // rgb, w = intensity
	vec4 params;     // x = range
};

// Spot light (must match SpotLightUbo: 4 x vec4)
struct SpotLightData {
	vec4 position;   // xyz = world pos, w = enabled (0/1)
	vec4 direction;  // xyz = normalized direction
	vec4 color;      // rgb, w = intensity
	vec4 params;     // x = range, y = cos(innerAngle), z = cos(outerAngle)
};

layout(set = 0, binding = 1) uniform LightUBO
{
	vec4 direction;
	vec4 color;
	vec4 starPosition;  // xyz = star world pos; w=1 => L = normalize(xyz - objectCenter)
	mat4 lightViewProj;
	vec4 sunParams;  // x=intensity, y=ambientStrength, z=shadowBiasScale, w=shadowDarken
	vec4 shadowParams;  // x=shadowSoftness
	int numPointLights;
	vec3 _padPoint;
	PointLightData pointLights[8];
	int numSpotLights;
	vec3 _padSpot;
	SpotLightData spotLights[4];
} light;

layout(location = 0) out vec4 outColour;

const float INV_SHADOW_MAP_SIZE = 1.0 / 8192.0;
const float SHADOW_BIAS = 0.001;
const float SHADOW_BIAS_SLOPE = 0.003;  // egim acisinda ek bias (dumduz kesen cizgiyi azaltir)

float sampleShadowPCF(vec2 uv, float fragDepth, float bias) {
    float r = clamp(light.shadowParams.x, 1.0, 2.0);
    int ir = int(ceil(r));
    ir = min(ir, 2);
    float step = INV_SHADOW_MAP_SIZE;
    float sum = 0.0;
    float n = 0.0;
    for (int x = -ir; x <= ir; x++) {
        for (int y = -ir; y <= ir; y++) {
            vec2 s = clamp(uv + vec2(float(x), float(y)) * step, 0.0, 1.0);
            float d = texture(shadowMap, s).r;
            if (d >= 0.9999)
                sum += 1.0;
            else
                sum += (fragDepth <= d + bias) ? 1.0 : 0.0;
            n += 1.0;
        }
    }
    return n > 0.0 ? sum / n : 1.0;
}

void main() {

    vec3 N = normalize(worldNormal);
    // Yildiz isigi: yon = objenin merkezinden yildiza (vertex'ten gelen objectCenter; obje hareket edince guncellenir)
    vec3 toStar = light.starPosition.xyz - objectCenter;
    float distToStar = length(toStar);
    vec3 L = (light.starPosition.w > 0.5 && distToStar > 1e-5)
        ? (toStar / distToStar)
        : normalize(light.direction.xyz);
    float NdotL = max(dot(N, L), 0.0);

    vec3 lightColor = max(light.color.rgb, vec3(0.5));
    float sunIntensity = max(0.0, light.sunParams.x);
    float ambStrength = clamp(light.sunParams.y, 0.0, 1.0);
    vec3 ambient = ambStrength * lightColor;
    vec3 diffuse = lightColor * NdotL * sunIntensity;

    vec4 lightClip = light.lightViewProj * vec4(worldPos, 1.0);
    float w = lightClip.w;
    vec3 ndc = (abs(w) > 1e-6) ? (lightClip.xyz / w) : vec3(0.0);
    vec2 shadowUV = ndc.xy * 0.5 + 0.5;
    float fragDepthInLight = clamp(ndc.z, 0.0, 1.0);

    // Isik kaynaginin arkasindaki yuzeylere golge dusurme (F1 gunesin ote tarafindayken dunyaya golge vermesin)
    // direction = sahne -> yildiz; fragment isik onundeyse (fragment - yildiz) ile direction ayni yonde olmali
    vec3 fragToLight = light.starPosition.xyz - worldPos;
    float behindLight = dot(fragToLight, light.direction.xyz);
    bool fragmentBehindLight = (light.starPosition.w > 0.5) && (behindLight <= 0.0);

    // Slope-scaled bias: yuzevin isiga gore egimi arttikca bias artar (dumduz kesen golge cizgisini azaltir)
    float bias = (SHADOW_BIAS + SHADOW_BIAS_SLOPE * (1.0 - NdotL)) * light.sunParams.z;

    bool outsideFrustum = any(lessThan(ndc.xy, vec2(-1.02))) || any(greaterThan(ndc.xy, vec2(1.02))) || ndc.z < -0.01 || ndc.z > 1.01;
    float pcfShadow = (fragmentBehindLight || outsideFrustum) ? 1.0 : sampleShadowPCF(clamp(shadowUV, 0.0, 1.0), fragDepthInLight, bias);

    // Frustum kenarinda yumusak gecis: bazı acilarda dumduz kesen cizgiyi kaldir
    float edgeX = max(0.0, max(-ndc.x - 1.0, ndc.x - 1.0));
    float edgeY = max(0.0, max(-ndc.y - 1.0, ndc.y - 1.0));
    float edgeDist = max(edgeX, edgeY);
    float edgeBlend = smoothstep(0.0, 0.08, edgeDist);
    float shadowFactor = mix(pcfShadow, 1.0, edgeBlend);

    // shadowParams.y: 0=normal | 1=UV (sunum) | 2=sert golge | 3=derinlik kontrast — F5-F8
    int dbg = int(light.shadowParams.y + 0.5);
    if (dbg == 1) {
        // Gölge haritasinda bu piksel hangi (u,v) texel'e düsüyor — zeminde gradient, F1'de farkli renkler
        vec2 uv = clamp(shadowUV, 0.0, 1.0);
        outColour = vec4(uv.x, uv.y, 0.15, 1.0);
        return;
    }
    if (dbg == 2) {
        // PCF yok: tek ornek — siyah=golge beyaz=aydinlik (sunumda net)
        if (fragmentBehindLight || outsideFrustum) {
            outColour = vec4(1.0, 1.0, 1.0, 1.0);
            return;
        }
        float d = texture(shadowMap, clamp(shadowUV, 0.0, 1.0)).r;
        float hard = (fragDepthInLight <= d + bias) ? 1.0 : 0.0;
        outColour = vec4(hard, hard, hard, 1.0);
        return;
    }
    if (dbg == 3) {
        // Haritadaki ham derinligi kontrastla ac (düz gri yerine fark gorunur)
        float sd = texture(shadowMap, clamp(shadowUV, 0.0, 1.0)).r;
        float vis = pow(clamp(sd, 0.001, 1.0), 0.35);
        outColour = vec4(vec3(vis), 1.0);
        return;
    }

    float shadowDarken = clamp(light.sunParams.w, 0.0, 1.0);
    float shadow = mix(shadowDarken, 1.0, shadowFactor);

    vec3 lightContribution = ambient + diffuse * shadow;

    // Point lights (user-controlled count, position, color, intensity, range) – no shadows
    int n = clamp(light.numPointLights, 0, 8);
    for (int i = 0; i < n; i++) {
        if (light.pointLights[i].position.w < 0.5) continue;  // disabled
        vec3 lightPos = light.pointLights[i].position.xyz;
        vec3 L = normalize(lightPos - worldPos);
        float NdotL = max(dot(N, L), 0.0);
        float d = length(lightPos - worldPos);
        float range = max(light.pointLights[i].params.x, 0.001);
        float att = 1.0 / (1.0 + (d / range) * (d / range));  // smooth falloff
        float intensity = max(0.0, light.pointLights[i].color.w);
        vec3 pColor = max(light.pointLights[i].color.rgb, vec3(0.0));
        lightContribution += pColor * NdotL * intensity * att;
    }

    // Spot lights (cone: direction = isigin baktigi yon, inner/outer angle)
    int nSpot = clamp(light.numSpotLights, 0, 4);
    for (int i = 0; i < nSpot; i++) {
        if (light.spotLights[i].position.w < 0.5) continue;  // disabled
        vec3 lightPos = light.spotLights[i].position.xyz;
        vec3 toFrag = worldPos - lightPos;   // isiktan fragmente vektor
        float d = length(toFrag);
        vec3 lightToFrag = (d > 1e-6) ? (toFrag / d) : vec3(0.0, 1.0, 0.0);  // birim: isik -> fragment
        vec3 fragToLight = -lightToFrag;    // diffuse icin: fragment -> isik
        vec3 spotDir = normalize(light.spotLights[i].direction.xyz);  // isigin baktigi yon (isiktan disari)
        // Koni: fragment koni icinde mi? lightToFrag ile spotDir ayni yonde olmali (cosTheta yukari)
        float cosTheta = dot(lightToFrag, spotDir);
        float cosOuter = light.spotLights[i].params.z;  // dis acinin cos'u (kucuk)
        float cosInner = light.spotLights[i].params.y;  // ic acinin cos'u (buyuk)
        float spotAtt = clamp((cosTheta - cosOuter) / max(cosInner - cosOuter, 0.001), 0.0, 1.0);
        float range = max(light.spotLights[i].params.x, 0.001);
        float distAtt = 1.0 / (1.0 + (d / range) * (d / range));
        float NdotL = max(dot(N, fragToLight), 0.0);   // Lambert: normal * (fragment -> isik)
        float intensity = max(0.0, light.spotLights[i].color.w);
        vec3 sColor = max(light.spotLights[i].color.rgb, vec3(0.0));
        lightContribution += sColor * NdotL * intensity * distAtt * spotAtt;
    }

    lightContribution = max(lightContribution, vec3(0.03));

    const float exposure = 0.85;
    lightContribution = vec3(1.0) - exp(-lightContribution * exposure);
    lightContribution = clamp(lightContribution, vec3(0.02), vec3(1.0));

#if DEBUG_SHADOW_MAP
    float shadowDepth = texture(shadowMap, clamp(shadowUV, 0.0, 1.0)).r;
    outColour = vec4(vec3(shadowDepth), 1.0);
    return;
#endif
#if DEBUG_SHADOW_FACTOR
    outColour = vec4(vec3(shadowFactor), 1.0);
    return;
#endif
    vec4 textureColor = texture(textureSampler, fragTex);
    vec3 albedo = textureColor.rgb;
    if (dot(albedo, vec3(1.0)) < 0.05)
        albedo = max(fragColor, vec3(0.4));
    outColour = vec4(albedo * lightContribution, textureColor.a);
}
