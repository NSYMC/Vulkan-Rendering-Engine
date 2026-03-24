#version 450
// Fullscreen triangle: no vertex buffer, generate NDC positions from vertex index
layout(location = 0) out vec2 outNDC;

void main() {
	// Single triangle covering NDC: (-1,-1), (3,-1), (-1,3)
	vec2 ndc = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
	outNDC = ndc;
	gl_Position = vec4(ndc, 1.0, 1.0);
}
