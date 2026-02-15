#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTex;
layout(location = 4) in vec3 worldPos;
layout(location = 5) in vec3 worldNormal;

layout(set = 1, binding=0) uniform sampler2D textureSampler;

layout(set = 0, binding = 1) uniform	 LightUBO
{
	vec4 position;
	vec4 color;
	vec4 direction;

} light;

layout(location = 0) out vec4 outColour;   
 
void main() {

	
    vec3 N = normalize(worldNormal); 

    
	vec3 L_vector = light.position.xyz - worldPos;
    vec3 L = normalize(L_vector);

    
    float intensity = max(0.0, dot(N, -L));
    

    
    vec3 ambient = 0.1 * light.color.rgb; 

    
    vec3 diffuse = light.color.rgb * intensity ;
    vec3 lightContribution = ambient + diffuse;

    
    vec4 textureColor = texture(textureSampler, fragTex);
    
    
    outColour = vec4(textureColor.rgb * lightContribution, textureColor.a);

}