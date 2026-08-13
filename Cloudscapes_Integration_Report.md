# Integration Report: Real-Time Cloudscapes with Volumetric Raymarching

**Source Blog Post:** [Maxime Heckel - Real-time dreamy Cloudscapes with Volumetric Raymarching](https://blog.maximeheckel.com/posts/real-time-cloudscapes-with-volumetric-raymarching/)

**Target Engine:** `c:\VulkanProject\atmosphere_bac_src\` — Vulkan atmospheric scattering + volumetric cloud engine (Bruneton & Neyret 2008)

**Date:** May 27, 2026

---

## 1. Your Engine's Current State — Summary

### Architecture Overview

Your engine (`atmosphere_bac_src/`) is a sophisticated Vulkan-based atmospheric renderer built around the **Bruneton & Neyret (2008)** atmospheric scattering model. It uses:

- **CMake** build system with `glslc` for shader compilation to SPIR-V
- **Vulkan 1.0** API via GLFW, with validation layers in debug builds
- **GLM** for math, **ImGui** for debug UI, **stb_image/tinyexr** for asset loading
- **C++17** with a flat class hierarchy (`Application` → `Renderer` → Vulkan wrappers)

### Rendering Pipeline (4-stage)

| Stage | What Happens | Key Shaders |
|-------|-------------|-------------|
| **Compute** | Bruneton LUT generation (transmittance, multiscatter, skyview, aerial perspective) | `transmittanceLUT.glsl`, `multiscatteringLUT.glsl`, `skyviewLUT.glsl`, `aerialPerspectiveLUT.glsl` |
| **Subpass 0** | Terrain (heightmap displacement, diffuse lighting, atmospheric transmittance) | `terrain.vert` + `terrain.frag` |
| **Subpass 1** | Far sky (SkyViewLUT lookup + sun bloom) | `screen_triangle.vert` + `draw_far_sky.frag` |
| **Subpass 2** | **Volumetric clouds** (Worley noise, raymarching, light marching, phase function) | `screen_triangle.vert` + `draw_clouds.frag` |
| **Subpass 3** | Aerial perspective fog | `draw_AE_perspective.frag` |
| **PostProcess** | Auto-exposure histogram + tonemapping (Reinhard/ACES/Uchimura/Lottes) + sRGB | `histogram_generate.glsl`, `histogram_sum.glsl`, `final_composition.frag` |

All intermediate rendering targets an **HDR offscreen framebuffer** (`R32G32B32A32_SFLOAT` color + `D32_SFLOAT` depth), then tonemapped down to the swapchain image.

### Existing Cloud System (`draw_clouds.frag`)

Your cloud system is **already quite advanced** and implements many techniques from the blog post. Here's exactly what you have:

```glsl
float sampleDensity(vec3 samplePos, float distFactor)
{
    const float baseScale = 1.0/1000.0;
    vec3 uvw = samplePos * baseScale * cloudsParameters.cloudsScale * vec3(1.0, 1.0, 1.0);
    // ... height gradient computation ...
    vec4 shapeNoise = texture(worleyNoiseSampler, uvw);
    vec4 normalizedShapeWeights = normalize(cloudsParameters.shapeNoiseWeights); 
    float shapeFBM = dot(shapeNoise, normalizedShapeWeights);
    float baseShapeDensity = shapeFBM - min(cloudsParameters.densityOffset + distFactor, 1.0);
    if(baseShapeDensity > 0.0)
    {
        vec3 detailSamplePos = uvw * cloudsParameters.detailScale;
        vec4 detailNoise = texture(worleyNoiseDetailSampler, detailSamplePos);
        // ... detail erosion ...
        float cloudDensity = baseShapeDensity - (1.0 - detailFBM) * detailErodeWeight * cloudsParameters.detailNoiseMultiplier;
        return cloudDensity * cloudsParameters.densityMultiplier * 4.9 * heightGradient;
    }
    return 0.0;
}
```

Key features already present:

- **3D Worley noise** with 4-channel FBM (shape + detail), GPU-generated via compute shaders
- **Height gradient** for cloud layer tapering
- **Detail erosion** (detail noise eats into the shape where shape is weak)
- **Beer's law** for transmittance: `exp(-density * integrationStep * lightAbsThroughCloud)`
- **Beer's powder approximation** (non-physical aesthetic term): `powderTransmittanceIncOverIntStep`
- **Light marching** (`getCloudTransAlongRay`): secondary raymarch from each sample toward the sun
- **Henyey-Greenstein double-lobe phase function** (`phase()` and `miePhaseFunctionDHG()`)
- **Ray-sphere intersection** for planetary cloud layer entry/exit (`getRayCloudLayerInfo`)
- **Depth-buffer awareness**: raymarch stops at terrain intersections
- **Atmospheric transmittance integration**: samples the Bruneton transmittance LUT per cloud sample
- **Alpha blending** with proper accumulation

---

## 2. Blog Post Techniques — Gap Analysis

Here is every technique from Maxime Heckel's post, mapped against your engine:

| # | Technique | Blog Implementation | Your Engine Status | Priority |
|---|-----------|-------------------|-------------------|----------|
| 1 | Volumetric Raymarching Loop | Constant step size, invert SDF (density > 0 = inside) | **Already implemented** ✓ | — |
| 2 | 3D Noise + FBM | 2D slice noise from 3D position + FBM | **Already implemented** ✓ (3D Worley noise, superior) | — |
| 3 | Directional Derivative Lighting | `clamp((scene(p) - scene(p + 0.3*sunDir)) / 0.3, 0, 1)` | **Already implemented** ✓ (full light marching, superior) | — |
| 4 | Beer's Law | `exp(-dist * absorption)` | **Already implemented** ✓ | — |
| 5 | Nested Light March | Secondary raymarch from sample → sun | **Already implemented** ✓ (`getCloudTransAlongRay`) | — |
| 6 | Henyey-Greenstein Phase | Single/double-lobe HG | **Already implemented** ✓ (double-lobe + bonus term) | — |
| 7 | Beer's Powder | Non-physical aesthetic absorption boost | **Already implemented** ✓ (`powderTransmittanceIncOverIntStep`) | — |
| 8 | **Blue Noise Dithering** | Blue noise texture offsets raymarch start, eliminates banding at low step counts | **NOT IMPLEMENTED** ✗ Uses `fract(sin(dot(...)))` hash noise instead | 🔴 HIGH |
| 9 | **Bicubic Upscaling** | Render at 0.5x resolution, upscale with bicubic filter as post-process pass | **NOT IMPLEMENTED** ✗ Always renders at native resolution | 🔴 HIGH |
| 10 | **Temporal Blue Noise** | `fract(blueNoise + float(uFrame%32) / sqrt(0.5))` to smooth dithering | **NOT IMPLEMENTED** ✗ | 🟡 MEDIUM |
| 11 | Morphing Cloud Shapes | SDF mixing (`mix(s1, s2, t)`, `smoothmin`) between torus/cross/sphere shapes | **NOT IMPLEMENTED** ✗ | 🟢 LOW |
| 12 | Cloud Merging | `min()` and `smoothmin()` of two SDFs | **NOT IMPLEMENTED** ✗ | 🟢 LOW |

---

## 3. Recommended Integration Plan

### Priority 1: Blue Noise Dithering (Highest Impact)

**Problem:** Your cloud shader uses a basic pseudorandom offset for dithering:

```glsl
/* TODO: This probably should be blueNoise */
float offset = fract(sin(dot(inUV, vec2(12.9898, 78.233))) * 43758.5453) * integrationLength/70.0;
```

This hash-based noise creates visible banding at lower sample counts and doesn't distribute error evenly. Blue noise has fewer low-frequency patterns and is less visible to the human eye.

**Implementation Steps:**

1. **Generate/load a blue noise texture** (e.g., 64×64 RGBA8 or R8). Blue noise textures are readily available online or can be generated offline. Store it as a `VulkanImage` with `VK_FORMAT_R8_UNORM`.

2. **Add a new descriptor binding** to the clouds pipeline. Currently the `WorleyNoise` descriptor set layout (set=4) has bindings 0 (shape noise) and 1 (detail noise). You can add binding 2 for the blue noise texture. Or create a new descriptor set.

   New additions in `renderer.cpp`:
   ```cpp
   // In createAttachments or loadAssets:
   frameSharedImages["BlueNoise"] = std::make_unique<VulkanImage>(
       vDevice, "assets/textures/blue_noise.png"); // or .exr

   // In createDescriptorSetLayout: add a new layout or extend WorleyNoise layout
   // New binding 2 in set=4 for blue noise sampler2D
   ```

3. **Modify `draw_clouds.frag`** shader:

   ```glsl
   // Add to shader:
   layout (set = 4, binding = 2) uniform sampler2D blueNoiseSampler;
   
   // Replace the hash-based offset in main():
   float blueNoise = texture(blueNoiseSampler, gl_FragCoord.xy / 64.0).r;
   float offset = fract(blueNoise + float(uFrame % 32) / sqrt(0.5));
   
   // Use offset to perturb initial raymarch position:
   vec3 startPosition = cameraPosition + (distanceToCloudBB + offset * MARCH_SIZE) * cameraRayWorld;
   ```

4. **Add frame counter** to a uniform buffer (or reuse `commonParameters.time` modulo a cycle).

**Expected benefit:** 2-3x reduction in required `sampleCount` (from ~100 down to 30-40) with equivalent visual quality, or dramatically better quality at the same sample count. The temporal component smooths remaining noise over frames.

---

### Priority 2: Bicubic Upscaling / Render Scale

**Problem:** Clouds are rendered at native resolution (1920×1080), which is computationally expensive. The blog demonstrates that rendering at 0.5x resolution with bicubic upscaling produces nearly identical quality.

**Implementation Steps:**

1. **Create a lower-resolution offscreen target** for clouds only:
   ```cpp
   // In createAttachments():
   uint32_t cloudWidth = vSwapChain->swapChainExtent.width / 2;   // 960
   uint32_t cloudHeight = vSwapChain->swapChainExtent.height / 2;  // 540
   
   perFrameData[i].images["CloudsHalfRes"] = std::make_unique<VulkanImage>(
       vDevice, cloudWidth, cloudHeight, 1, VK_SAMPLE_COUNT_1_BIT,
       VK_FORMAT_R16G16B16A16_SFLOAT, ...);
   ```

2. **Render clouds to half-res target** in a separate render pass (or use `VK_EXT_fragment_density_map` if available). This requires extracting the clouds subpass into its own render pass.

3. **Create a bicubic upscaling fragment shader** as a new post-process pass. N8Programs' implementation from Shadertoy:

   ```glsl
   // bicubic_upscale.frag
   #version 450
   layout (location = 0) out vec4 outColor;
   layout (location = 0) in vec2 inUV;
   layout (set = 0, binding = 0) uniform sampler2D inputTex;
   
   vec4 cubic(float v) {
       vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
       vec4 s = n * n * n;
       float x = s.x;
       float y = s.y - 4.0 * s.x;
       float z = s.z - 4.0 * s.y + 6.0 * s.x;
       float w = 6.0 - x - y - z;
       return vec4(x, y, z, w) / 6.0;
   }
   
   vec4 textureBicubic(sampler2D tex, vec2 uv, vec2 texSize) {
       vec2 texelSize = 1.0 / texSize;
       vec2 f = fract(uv * texSize);
       vec4 xCubic = cubic(f.x);
       vec4 yCubic = cubic(f.y);
       vec4 c = vec4(uv - f * texelSize, uv + (1.0 - f) * texelSize);
       vec4 s = vec4(xCubic.xz + xCubic.yw, yCubic.xz + yCubic.yw);
       vec4 offset = c + vec4(xCubic.yw, yCubic.yw) / s;
       offset /= texSize.xxyy;
       vec4 sample0 = texture(tex, offset.xz);
       vec4 sample1 = texture(tex, offset.yz);
       vec4 sample2 = texture(tex, offset.xw);
       vec4 sample3 = texture(tex, offset.yw);
       float sx = s.x / (s.x + s.y);
       float sy = s.z / (s.z + s.w);
       return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
   }
   
   void main() {
       vec2 texSize = vec2(textureSize(inputTex, 0));
       outColor = textureBicubic(inputTex, inUV, texSize);
   }
   ```

4. **Composite the upscaled clouds** back into the HDR backbuffer using additive blending in a new subpass or the final composition pass.

**Expected benefit:** ~4x reduction in cloud fragment shader invocations (1/4 the pixels), with minimal quality loss due to the soft nature of clouds. The bicubic filter smooths remaining noise better than bilinear.

**Alternative simpler approach:** Instead of a dedicated bicubic pass, you can use hardware bilinear filtering on the upsampled texture. Less quality than bicubic but zero implementation cost — just use `vkCmdBlitImage` with `VK_FILTER_LINEAR`.

---

### Priority 3: Temporal Blue Noise Smoothing

**Problem:** Even with blue noise dithering, individual frames can show visible noise patterns, especially in static scenes.

**Implementation:**
- Add a uniform `uint frameIndex` that increments each frame (modulo some period)
- In the shader: `float offset = fract(blueNoise + float(frameIndex % 32) / sqrt(0.5));`
- This cycles through different noise patterns each frame, averaging out over time with a TAA-like effect (even without explicit TAA, the human eye does the temporal averaging)

Add to your uniform buffer structure:
```cpp
// In buffer_defines.hpp or a new struct:
struct CommonParametersBuffer {
    // ... existing fields ...
    alignas(4) uint32_t frameIndex;
};
```

In the shader:
```glsl
float temporalOffset = float(commonParameters.frameIndex % 32) / sqrt(0.5);
float offset = fract(blueNoise + temporalOffset);
```

---

### Priority 4 (Optional): Morphing Cloud Shapes & Artistic SDFs

The blog demonstrates morphing clouds between torus, cross, sphere, and capsule shapes using `mix()` with a time-parameterized interpolation. This uses the same volumetric raymarching infrastructure you already have — only the SDF changes.

**Concept:** Instead of (or in addition to) a single planetary cloud layer, add artistic cloud objects:

```glsl
// In a new or modified scene() function:
float sdCloudShape(vec3 p) {
    float torus = sdTorus(p, vec2(1.3, 0.9));
    float cross = sdCross(p * 2.0, 0.6);
    float sphere = sdSphere(p, 1.5);
    
    float t = mod(nextStep(uTime, 3.0, 1.2), 4.0);
    float d = mix(torus, cross, clamp(t, 0.0, 1.0));
    d = mix(d, sphere, clamp(t - 1.0, 0.0, 1.0));
    // ... chain more shapes ...
    
    float f = fbm(p);  // your existing noise
    return -d + f;     // invert for volumetric: positive density inside
}
```

This requires:
- Adding SDF primitives (torus, cross, capsule) to your GLSL utility functions
- A new uniform for shape parameters or morph time
- Optionally: a new pipeline/render pass for "artistic clouds" separate from the planetary cloud layer

---

## 4. Implementation Roadmap

### Phase A: Blue Noise Dithering (1-2 days)

1. Add blue noise texture asset and loading
2. Modify `descriptorLayouts` to include blue noise sampler for the clouds pipeline
3. Update `draw_clouds.frag` to sample blue noise and use temporal offset
4. Reduce `sampleCount` from ~100 to ~40, observe quality parity
5. Tune dithering offset magnitude

### Phase B: Render Scale + Upscaling (2-3 days)

1. Extract clouds subpass into its own render pass with half-resolution framebuffer
2. Implement bicubic upscale fragment shader
3. Add upscale pass to the post-processing pipeline (before or integrated into `final_composition.frag`)
4. Add a UI slider in ImGui to select render scale (1.0x, 0.5x, 0.25x)
5. Validate quality at different scales

### Phase C: Polish & Performance (1-2 days)

1. Add frame index uniform for temporal blue noise
2. Tune `sampleCount`, `sampleCountToSun`, `marchSize` with blue noise active
3. Profile with Vulkan timestamps (your existing querry pool)
4. Document performance/quality trade-offs

### Phase D (Optional): Artistic Clouds (2-3 days)

1. Add SDF primitives to `common_func.glsl`
2. Create `draw_artistic_clouds.frag` or extend existing
3. Add morphing uniform parameters
4. Create ImGui controls for shape selection and morph speed

---

## 5. Code-Level Integration Details

### New Files Needed

| File | Purpose |
|------|---------|
| `assets/textures/blue_noise.png` | 64×64 blue noise texture |
| `shaders/bicubic_upscale.frag` | Bicubic filter for upscaling |
| `shaders/sdf_primitives.glsl` | SDF shapes (torus, cross, capsule) as GLSL include |
| (optional) `shaders/draw_artistic_clouds.frag` | Artistic cloud objects |

### Modified Files

| File | Changes |
|------|---------|
| `renderer.hpp` | New pipeline members: `bicubicUpscalePipeline`, optional `artisticCloudsPipeline`; new images: `CloudsHalfRes`, `BlueNoise` |
| `renderer.cpp` | New `createAttachments`, `createPipelines`, `createDescriptorSetLayout`, `createCommandBuffers` entries; blue noise texture loading; render scale parameter |
| `buffer_defines.hpp` | Add `frameIndex` to `UniformBufferObject` |
| `draw_clouds.frag` | Blue noise dithering, temporal offset, render-to-half-res target support |
| `common_func.glsl` | SDF primitives (if doing artistic clouds) |
| `imgui_impl.cpp` | UI controls for render scale, blue noise toggle, artistic cloud params |
| `CMakeLists.txt` | New shader source files in compilation list |

### Descriptor Set Layout Changes

For the blue noise approach, the least disruptive change is to add a binding to the existing `WorleyNoise` descriptor set (set=4):

```
set=4, binding=0: sampler3D worleyNoiseSampler       (existing)
set=4, binding=1: sampler3D worleyNoiseDetailSampler  (existing)
set=4, binding=2: sampler2D blueNoiseSampler          (NEW)
```

For bicubic upscaling, a new descriptor set layout is needed:
```
set=N, binding=0: sampler2D halfResCloudsInput
```

---

## 6. Performance Projections

Based on the blog's results and your engine's current parameters (sampleCount=60-120, sampleCountToSun=4-5):

| Configuration | Cloud Pass Cost | Quality |
|--------------|----------------|---------|
| Current (no dithering, native res, 100 samples) | Baseline (100%) | Good, some banding |
| + Blue noise, 40 samples | ~40% of baseline | Equivalent or better |
| + Half-res rendering | ~10% of baseline | Nearly identical (bicubic) |
| Combined | **~10% of baseline** | Nearly identical |

Your engine already uses timestamp queries (querryPool with 30 timestamps), so you can precisely measure these improvements.

---

## 7. Key Differences to Note

Your engine operates at **planetary scale** (the camera orbits a planet with `bottom_radius` ~6360 km in world-space units, with `cameraScale = 0.1` conversion). The blog's scenes are in **local space**. This means:

- Your raymarching uses `cameraScale` to convert between camera and world coordinates
- Cloud boundaries are defined in planetary altitude (km above surface)
- Ray-sphere intersection handles the curved planetary geometry
- The techniques map directly; just be mindful of scale when setting `MARCH_SIZE` and dithering offsets

The blog's `MARCH_SIZE = 0.08` in local units translates differently in your planetary coordinate system. Your equivalent is dynamically computed via `integrationLength / cloudsParameters.sampleCount`.

---

## Summary

Your engine is impressively advanced — it already implements ~70% of the blog's techniques at AAA quality (Bruneton atmosphere, 3D Worley noise, full light marching, dual-lobe HG phase function). The remaining 30% are performance and polish improvements:

1. **Blue noise dithering** (replaces hash noise, enables 2-3x fewer raymarch steps)
2. **Bicubic upscaling** (renders clouds at lower resolution, ~4x pixel savings)
3. **Temporal noise cycling** (smooths remaining noise across frames)
4. **Artistic SDF clouds** (optional creative feature)

The first two alone could reduce cloud rendering cost by ~90% while maintaining or improving visual quality. All techniques are well-documented, have reference GLSL implementations, and fit into your existing Vulkan pipeline architecture with minimal structural changes.
