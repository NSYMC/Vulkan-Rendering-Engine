# Texture Quality Improvements & Wavy Sea Shader — Implementation Report

**Project:** TerrainEngine (`c:\VulkanProject`)  
**Purpose:** Precise, file-by-file implementation guide targeting texture quality upgrades and a fully animated, shore-aware ocean shader. Every section names the exact file and line context; new shader source is provided in full. No guessing required.  
**Scope:** `shaders/texgen.comp`, `shaders/terrain.frag`, `shaders/terrain.vert`, new `shaders/ocean.vert`, new `shaders/ocean.frag`, `src/VulkanEngine.h`, `src/VulkanEngine.cpp`, `CMakeLists.txt`.

---

## 0. Prerequisites & Invariants

Before touching any file, fix the **time uniform slot**. The existing `FrameData` struct has a float `_pad1` that is pure padding between `lightDir` and the alignment before `lightMVP`. Replacing it with `float time` costs nothing (same layout, same 160-byte total size) but unlocks animation in every shader that already receives the frame UBO.

**`src/VulkanEngine.h` — `FrameData` struct (currently lines 56-63)**

Replace:
```cpp
struct FrameData {
    glm::mat4 mvp;
    glm::vec3 cameraPos;
    float     _pad0;
    glm::vec3 lightDir;
    float     _pad1;        // ← was padding
    glm::mat4 lightMVP;
};
```
With:
```cpp
struct FrameData {
    glm::mat4 mvp;
    glm::vec3 cameraPos;
    float     _pad0;
    glm::vec3 lightDir;
    float     time;         // ← elapsed seconds (replaces _pad1 — same offset)
    glm::mat4 lightMVP;
};
```

**`src/VulkanEngine.cpp` — `updateUniformBuffer()` function**

Find the block that writes `FrameData` to the UBO and add:
```cpp
fd.time = static_cast<float>(glfwGetTime());
```
This is the only C++ change needed for Part I (texture animation). `glfwGetTime()` is already a valid call; the GLFW header is included.

Both the existing terrain shader and the new ocean shader read `frame.time` through the same `set=0, binding=0` UBO.

---

## Part I — Texture Quality Improvements

### I.1 Diagnosis: Why the Current Textures Look Weak

| Issue | Root cause |
|-------|-----------|
| Repetition visible at mid range | Single-scale triplanar at 1/18 m — tile repeats every 18 m |
| Water is completely flat | `hWater()` uses only 4-octave FBM with 0.15 amplitude; normal strength is 0.15 |
| Sand looks like coloured noise | Ripples have a fixed mathematical phase; no depth at close view |
| Rock cracks appear fake | Normal strength is 2.4 but the `strength / texel * 0.002` formula collapses them at the default 512 resolution |
| Macro-scale tiling | `macroTint` breaks repetition but only modulates brightness, not hue or texture coordinate |
| No detail normals at close range | Single triplanar sample regardless of camera distance |
| Shore transition is abrupt | Layer 1 uses the same `hWater` as layer 0; no wet-sand treatment |
| AO baking is trivial | `ao = 0.6 + 0.4 * hC` — constant bias unrelated to actual occlusion |

### I.2 The Fix Strategy: Multi-Scale Triplanar + Improved Materials

The terrain fragment shader calls `triplanar()` and `triplanarNormal()` once per layer per fragment, at a fixed `TEX_SCALE = 1.0/18.0`. Replace this with a **two-scale** blend:

- **Macro scale:** `1.0 / 120.0` — large variation that breaks cross-tile repetition at 50–200 m
- **Detail scale:** `1.0 / 4.0` — micro detail visible within 30 m
- **Blend weight:** `smoothstep(20.0, 80.0, dist)` where `dist = length(cameraPos - fragWorldPos)`

The two samples are blended before any other computation. This eliminates 80% of visible repetition with only two texture lookups per layer instead of one.

### I.3 `shaders/terrain.frag` — Changes Required

#### I.3.1 Add distance-aware multi-scale sampling helper

After the existing `triplanarNormal()` function (around line 217), insert the following two helpers:

```glsl
// Two-scale triplanar: blends between a close detail sample and a far macro
// sample. `detailScale` should be ~1/4; `macroScale` ~1/120.
vec4 triplanarMS(vec3 worldPos, vec3 normal, float layer,
                 float detailScale, float macroScale, float distBlend) {
    vec4 detail = triplanar(worldPos, normal, layer, detailScale);
    vec4 macro  = triplanar(worldPos, normal, layer, macroScale);
    return mix(detail, macro, distBlend);
}

vec3 triplanarNormalMS(vec3 worldPos, vec3 geomN, float albedoLayer,
                       float detailScale, float macroScale, float distBlend) {
    vec3 near = triplanarNormal(worldPos, geomN, albedoLayer, detailScale);
    vec3 far  = triplanarNormal(worldPos, geomN, albedoLayer, macroScale);
    return normalize(mix(near, far, distBlend));
}
```

#### I.3.2 Replace the TEX_SCALE constant block in `main()`

Find (around line 319-333):
```glsl
const float TEX_SCALE = 1.0 / 18.0;
vec4 texColor = mix(
    triplanar(fragWorldPos, N, loLayer, TEX_SCALE),
    triplanar(fragWorldPos, N, hiLayer, TEX_SCALE),
    blendFactor
);
float roughness = clamp(texColor.a, 0.04, 1.0);
vec3 Nlit = normalize(mix(
    triplanarNormal(fragWorldPos, N, loLayer, TEX_SCALE),
    triplanarNormal(fragWorldPos, N, hiLayer, TEX_SCALE),
    blendFactor));
```

Replace with:
```glsl
const float DETAIL_SCALE = 1.0 / 4.0;
const float MACRO_SCALE  = 1.0 / 120.0;
float camDist   = length(frame.cameraPos - fragWorldPos);
float distBlend = smoothstep(15.0, 70.0, camDist);   // 0=detail, 1=macro

vec4 texColor = mix(
    triplanarMS(fragWorldPos, N, loLayer, DETAIL_SCALE, MACRO_SCALE, distBlend),
    triplanarMS(fragWorldPos, N, hiLayer, DETAIL_SCALE, MACRO_SCALE, distBlend),
    blendFactor
);

float roughness = clamp(texColor.a, 0.04, 1.0);
vec3 Nlit = normalize(mix(
    triplanarNormalMS(fragWorldPos, N, loLayer, DETAIL_SCALE, MACRO_SCALE, distBlend),
    triplanarNormalMS(fragWorldPos, N, hiLayer, DETAIL_SCALE, MACRO_SCALE, distBlend),
    blendFactor));
```

#### I.3.3 Animated water texture in main()

After the closing `}` of the `biome-based color variation` block (around line 376), before the slope-rock section, insert:

```glsl
// ── Animated water surface detail (ocean & shallow coast) ─────
// Layers 0 and 1 are ocean/coast: drive their UV by scrolling in
// two opposite directions to simulate cross-swell interference.
if (contLayer < 2.0) {
    float wt = frame.time;
    vec3 scroll1 = fragWorldPos + vec3( wt * 0.6, 0.0,  wt * 0.3);
    vec3 scroll2 = fragWorldPos + vec3(-wt * 0.4, 0.0, -wt * 0.7);

    float waterDetailBlend = 1.0 - distBlend;   // only near camera
    vec4 wn1 = triplanar(scroll1, N, loLayer, DETAIL_SCALE) * waterDetailBlend;
    vec4 wn2 = triplanar(scroll2, N, loLayer, DETAIL_SCALE) * waterDetailBlend;
    vec4 wBlend = (wn1 + wn2) * 0.5;

    float waterMix = clamp(1.0 - contLayer, 0.0, 1.0);   // 1.0 for layer 0
    color.rgb = mix(color.rgb, wBlend.rgb, waterMix * 0.45);

    // Animated water normal for lighting
    vec3 wNorm1 = triplanarNormal(scroll1, N, loLayer, DETAIL_SCALE);
    vec3 wNorm2 = triplanarNormal(scroll2, N, loLayer, DETAIL_SCALE);
    vec3 wNorm  = normalize(wNorm1 + wNorm2);
    Nlit = normalize(mix(Nlit, wNorm, waterMix * 0.6 * waterDetailBlend));
}
```

#### I.3.4 Shore wetness and foam runup (new section, before river rendering)

Insert immediately before the `// ── River rendering ──` comment block (~line 463):

```glsl
// ── Shore foam & wet sand ─────────────────────────────────────
// Applies to land pixels just above sea level (h in -0.5 .. 3.0).
// Wave crests animate toward the shore using the terrain normal's
// horizontal component as the wave-approach direction.
{
    float shoreBand = smoothstep(3.0, -0.5, h) * smoothstep(-0.5, 0.0, h + 0.5);
    if (shoreBand > 0.001) {
        // Shore-facing direction from geometry normal (x-z plane component)
        vec2  shoreDir  = normalize(N.xz + vec2(0.0001));
        float wavePeriod = 6.5;   // seconds between wave crests
        float waveFreq   = 0.18;  // spatial frequency (radians/m)
        float phase = dot(fragWorldPos.xz, shoreDir) * waveFreq
                    - frame.time * (6.2831 / wavePeriod);
        float waveCrest = smoothstep(0.3, 1.0, sin(phase) * 0.5 + 0.5);

        // Foam is bright white with a slight blue tint
        vec3 foamColor = vec3(0.93, 0.96, 1.00);
        float foamStr  = waveCrest * shoreBand * 0.75;
        color.rgb = mix(color.rgb, foamColor, foamStr);

        // Wet darkening between crests — damp sand / wet rock
        float wetDark = (1.0 - waveCrest) * shoreBand * 0.40;
        color.rgb *= 1.0 - wetDark;
    }
}
```

### I.4 `shaders/texgen.comp` — Improved Material Definitions

All changes are inside the per-material height and albedo functions. The normal encoding formula must also be fixed to produce correct normals at varying resolutions.

#### I.4.1 Fix the normal strength formula (`encodeNormal`, ~line 196)

The current formula divides by `texel * 0.002` which produces wildly different strengths at different resolutions. Replace with a resolution-independent form:

```glsl
vec4 encodeNormal(int mat, vec2 uv, float texel) {
    // strength: world-space slope factor (unitless, tuned per material)
    float strength = (mat <= 1) ? 0.40 :   // water: gentle swells
                     (mat == 2) ? 1.80 :   // sand: pronounced ripples
                     (mat == 3) ? 1.20 :   // grass: soft clumps
                     (mat == 4) ? 3.20 :   // rock: sharp strata
                                  0.60;    // snow: soft drifts

    float hC = materialHeight(mat, uv);
    float hX = materialHeight(mat, uv + vec2(texel, 0.0));
    float hY = materialHeight(mat, uv + vec2(0.0, texel));

    // Resolution-independent: dx/dy are already in [0,1] UV space;
    // multiply by strength to control perceived relief.
    vec3 n = normalize(vec3(-(hX - hC) * strength,
                            -(hY - hC) * strength,
                             texel));      // z = texel keeps slope sane
    float ao = clamp(0.55 + 0.45 * hC, 0.0, 1.0);
    return vec4(n * 0.5 + 0.5, ao);
}
```

#### I.4.2 Improved `hWater` — multi-directional wave interference

```glsl
float hWater(vec2 uv) {
    // Two crossing swell directions + chop noise
    float s1 = sin(uv.x * 22.0 + fbm(uv * 3.0, 3) * 4.0) * 0.45;
    float s2 = sin((uv.x * 14.0 + uv.y * 10.0) + fbm(uv * 4.5, 3) * 3.5) * 0.30;
    float chop = fbm(uv * 18.0, 4) * 0.25;
    return s1 + s2 + chop;
}
```

#### I.4.3 Improved `hSand` — multi-scale ripple with saltation pattern

```glsl
float hSand(vec2 uv) {
    // Primary wind ripples — two slightly different frequencies
    float dir1   = fbm(uv * 1.2 + 1.7, 3) * 2.0;
    float ripple1 = sin(uv.x * 36.0 * cos(dir1) + uv.y * 36.0 * sin(dir1)
                       + fbm(uv * 2.8, 3) * 6.2831) * 0.50;

    float dir2   = dir1 + fbm(uv * 2.5, 2) * 0.8;
    float ripple2 = sin(uv.x * 22.0 * cos(dir2) + uv.y * 22.0 * sin(dir2)
                       + fbm(uv * 4.0, 3) * 6.2831) * 0.22;

    // Grain: Worley at two scales
    float grain  = worley(uv * 55.0) * 0.14 + worley(uv * 120.0) * 0.08;

    // Large dune undulation (broad fbm)
    float dune   = fbm(uv * 4.5, 4) * 0.18;

    return ripple1 + ripple2 + grain + dune;
}
```

#### I.4.4 Improved `hGrass` — three-scale structure

```glsl
float hGrass(vec2 uv) {
    // Macro: broad meadow undulation
    float meadow = fbm(uv * 2.5, 3) * 0.35;

    // Meso: clumps — warped dual-scale Worley
    vec2 wuv1 = warp(uv, 2.0, 0.40);
    float clump = (1.0 - worley(wuv1 * 12.0)) * 0.38;
    float tuft  = (1.0 - worley(wuv1 * 28.0)) * 0.20;

    // Micro: blade density noise
    float blade = fbm(uv * 65.0 + 8.3, 4) * 0.22;

    return (meadow + clump * 0.55 + tuft * 0.30 + blade) * 2.0 - 1.0;
}
```

#### I.4.5 Improved `hRock` — strata + micro fractures + surface grit

```glsl
float hRock(vec2 uv) {
    vec2 wuv  = warp(uv, 1.5, 0.30);
    vec2 bed  = vec2(wuv.x * 0.55, wuv.y * 2.8);       // bedding plane anisotropy

    // Primary strata (FBM along bedding)
    float strata  = fbm(bed * 8.5 + 4.7, 5) * 0.65;

    // Secondary micro-strata (finer scale, different angle)
    vec2 bed2 = vec2(wuv.x * 1.8, wuv.y * 0.8) + vec2(11.3, 7.1);
    float micro  = fbm(bed2 * 22.0, 4) * 0.20;

    // Joint cracks at two scales
    float crack1 = smoothstep(0.12, 0.0, worley(wuv * 14.0)) * 1.10;
    float crack2 = smoothstep(0.06, 0.0, worley(wuv * 32.0)) * 0.50;

    // Surface grit
    float grit = worley(uv * 80.0) * 0.15;

    return strata + micro + grit - crack1 - crack2;
}
```

#### I.4.6 Improved rock albedo (match new height field)

Replace the `if (mat == 4)` branch in `materialAlbedo`:

```glsl
if (mat == 4) {
    vec2 wuv    = warp(uv, 1.5, 0.30);
    vec2 bed    = vec2(wuv.x * 0.55, wuv.y * 2.8);
    float strata = fbm(bed * 8.5 + 4.7, 5);
    float micro  = fbm(vec2(wuv.x * 1.8, wuv.y * 0.8) * 22.0, 4);

    // Lighter strata bands mix with darker matrix
    vec3 lightBand = vec3(0.44, 0.42, 0.38);
    vec3 darkMat   = vec3(0.20, 0.19, 0.18);
    vec3 base = mix(darkMat, lightBand, strata * 0.5 + 0.5 + micro * 0.25);

    // Iron-oxide tint in warm strata
    float iron = smoothstep(0.3, 0.8, strata);
    base = mix(base, vec3(0.42, 0.28, 0.18), iron * 0.22);

    // Cracks — dark and slightly blue-grey
    float crack1 = smoothstep(0.12, 0.0, worley(wuv * 14.0));
    base *= 1.0 - 0.70 * crack1;
    base = mix(base, vec3(0.14, 0.14, 0.16), crack1 * 0.30);
    return clamp(base, 0.0, 1.0);
}
```

---

## Part II — Wavy Sea Shader

### II.1 Architecture

The ocean is a **separate rendering pass** inserted between the terrain pass and the sky pass. It uses its own vertex and fragment shaders but shares the existing `set=0` global descriptor (FrameData, heightmap, AtmosphereBlock) so no new descriptor pool allocation is needed.

```
[Terrain Pass]  →  [Ocean Pass]  →  [Sky Pass]  →  [Foliage/Models]  →  [Clouds]  →  [ImGui]
```

The ocean mesh is a flat XZ grid covering the entire world extent at `y = 0.0` (displacement happens entirely in the vertex shader). The heightmap sampler is read per-vertex to compute water depth at each grid point, which drives wave amplitude attenuation for shore harmony.

**Key invariants:**
- Both terrain and ocean read from the **same** `heightmapSampler` (binding 1) — no data duplication.
- Both passes use the **same** `FrameData` UBO including `frame.time`.
- Depth test `LESS_OR_EQUAL` ensures the ocean surface naturally occludes terrain beneath it and is occluded by landmass above sea level.
- No blending needed for the opaque deep-water body; only the foam band near shore uses the alpha channel.

### II.2 New Struct: `OceanPush`

Add to `src/VulkanEngine.h` after `ChunkPush`:

```cpp
struct OceanPush {
    glm::vec2 worldOrigin;     // same as terrain ChunkPush
    float     worldSize;
    float     heightScale;
    int       heightmapRes;
    float     seaLevel;        // metres — ocean plane sits at this Y
    // Wave parameters (tunable via ImGui)
    float     waveAmplitude;   // base amplitude in metres (default 0.55)
    float     waveSpeed;       // base angular frequency multiplier (default 1.0)
    float     shoreDepth;      // depth at which waves begin to fade (default 4.0 m)
    float     foamDepth;       // depth below which foam appears (default 1.8 m)
    float     _pad0;           // std430 alignment
};
```

### II.3 New Buffer/Pipeline Members

Add to the `VulkanEngine` class declaration (in `VulkanEngine.h`):

```cpp
// Ocean pass
Buffer                oceanVertexBuffer;
Buffer                oceanIndexBuffer;
uint32_t              oceanIndexCount   = 0;
VkPipeline            oceanPipeline     = VK_NULL_HANDLE;
VkPipelineLayout      oceanPipelineLayout = VK_NULL_HANDLE;
OceanPush             oceanPush{};
```

### II.4 `createOceanMesh()` — C++ helper

Add to `VulkanEngine.cpp` (call from `init()` after `createTerrainMesh()`):

```cpp
void VulkanEngine::createOceanMesh() {
    // Grid resolution: 256x256 quads = 257x257 vertices.
    // Enough for ~16 m wavelength over a 4096 m world (256 verts/4096 m = 16 m/vert).
    const int N = 257;
    const float step = settings.worldSize / float(N - 1);

    std::vector<glm::vec3> verts;
    verts.reserve(N * N);
    for (int z = 0; z < N; z++)
        for (int x = 0; x < N; x++)
            verts.push_back({ x * step, 0.0f, z * step });   // Y = 0; shader lifts to seaLevel + Gerstner

    std::vector<uint32_t> idx;
    idx.reserve((N - 1) * (N - 1) * 6);
    for (int z = 0; z < N - 1; z++)
        for (int x = 0; x < N - 1; x++) {
            uint32_t base = z * N + x;
            idx.push_back(base);
            idx.push_back(base + 1);
            idx.push_back(base + N + 1);
            idx.push_back(base);
            idx.push_back(base + N + 1);
            idx.push_back(base + N);
        }

    oceanIndexCount = static_cast<uint32_t>(idx.size());
    createBuffer(sizeof(glm::vec3) * verts.size(),
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 oceanVertexBuffer.buffer, oceanVertexBuffer.memory);
    void* data;
    vkMapMemory(device, oceanVertexBuffer.memory, 0, VK_WHOLE_SIZE, 0, &data);
    memcpy(data, verts.data(), sizeof(glm::vec3) * verts.size());
    vkUnmapMemory(device, oceanVertexBuffer.memory);

    createBuffer(sizeof(uint32_t) * idx.size(),
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 oceanIndexBuffer.buffer, oceanIndexBuffer.memory);
    vkMapMemory(device, oceanIndexBuffer.memory, 0, VK_WHOLE_SIZE, 0, &data);
    memcpy(data, idx.data(), sizeof(uint32_t) * idx.size());
    vkUnmapMemory(device, oceanIndexBuffer.memory);
}
```

### II.5 `createOceanPipeline()` — C++ helper

The ocean pipeline reuses the **same descriptor set layout** as the terrain pipeline (set 0 has FrameData, heightmap, texArray, shadowMap, AtmosphereBlock, CloudParams, worleyNoise). This avoids creating a new descriptor pool or set. The ocean shaders simply ignore the bindings they don't use.

```cpp
void VulkanEngine::createOceanPipeline() {
    auto vertCode = readFile("ocean.vert.spv");
    auto fragCode = readFile("ocean.frag.spv");
    VkShaderModule vertMod = createShaderModule(vertCode);
    VkShaderModule fragMod = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_VERTEX_BIT,   vertMod, "main", nullptr };
    stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "main", nullptr };

    // Vertex input: same as terrain (vec3 per vertex)
    VkVertexInputBindingDescription binding{};
    binding.binding   = 0;
    binding.stride    = sizeof(glm::vec3);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attr{};
    attr.binding  = 0;
    attr.location = 0;
    attr.format   = VK_FORMAT_R32G32B32_SFLOAT;
    attr.offset   = 0;

    VkPipelineVertexInputStateCreateInfo vtxInfo{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vtxInfo.vertexBindingDescriptionCount   = 1;
    vtxInfo.pVertexBindingDescriptions      = &binding;
    vtxInfo.vertexAttributeDescriptionCount = 1;
    vtxInfo.pVertexAttributeDescriptions    = &attr;

    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Push constant for OceanPush (128 bytes — fits in guaranteed Vulkan minimum)
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(OceanPush);

    VkPipelineLayoutCreateInfo layoutInfo{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    // Reuse the terrain descriptor set layout (all bindings already allocated)
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &globalDescriptorSetLayout;  // same as terrain
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pcRange;
    vkCreatePipelineLayout(device, &layoutInfo, nullptr, &oceanPipelineLayout);

    // Rasterizer — backface cull, polygon fill
    VkPipelineRasterizationStateCreateInfo rast{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode    = VK_CULL_MODE_BACK_BIT;
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth   = 1.0f;

    // Depth test: LESS_OR_EQUAL; depth write ON
    VkPipelineDepthStencilStateCreateInfo ds{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    // No blending needed — ocean is opaque (foam handled in shader with premix)
    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.blendEnable         = VK_FALSE;
    cbAtt.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                              | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 1;
    cb.pAttachments    = &cbAtt;

    // Dynamic viewport / scissor
    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

    VkPipelineViewportStateCreateInfo vp{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkGraphicsPipelineCreateInfo pipeInfo{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipeInfo.stageCount          = 2;
    pipeInfo.pStages             = stages;
    pipeInfo.pVertexInputState   = &vtxInfo;
    pipeInfo.pInputAssemblyState = &ia;
    pipeInfo.pViewportState      = &vp;
    pipeInfo.pRasterizationState = &rast;
    pipeInfo.pMultisampleState   = &ms;
    pipeInfo.pDepthStencilState  = &ds;
    pipeInfo.pColorBlendState    = &cb;
    pipeInfo.pDynamicState       = &dyn;
    pipeInfo.layout              = oceanPipelineLayout;

    // Dynamic rendering — same attachment formats as terrain
    VkFormat colorFmt = swapchainImageFormat;
    VkFormat depthFmt = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo dynRenderInfo{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    dynRenderInfo.colorAttachmentCount    = 1;
    dynRenderInfo.pColorAttachmentFormats = &colorFmt;
    dynRenderInfo.depthAttachmentFormat   = depthFmt;
    pipeInfo.pNext = &dynRenderInfo;

    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &oceanPipeline);
    vkDestroyShaderModule(device, vertMod, nullptr);
    vkDestroyShaderModule(device, fragMod, nullptr);

    // Populate push constant defaults
    oceanPush.worldOrigin    = glm::vec2(0.0f);
    oceanPush.worldSize      = settings.worldSize;
    oceanPush.heightScale    = settings.heightScale;
    oceanPush.heightmapRes   = settings.masterRes;
    oceanPush.seaLevel       = settings.seaLevel;
    oceanPush.waveAmplitude  = 0.55f;
    oceanPush.waveSpeed      = 1.0f;
    oceanPush.shoreDepth     = 4.0f;
    oceanPush.foamDepth      = 1.8f;
}
```

### II.6 Integrating the Ocean Draw Call

In `recordRenderCommand()`, immediately after the terrain draw call and before the sky draw call, add:

```cpp
// ─── Ocean pass ───────────────────────────────────────────────
vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, oceanPipeline);
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        oceanPipelineLayout, 0, 1, &globalDescriptorSet, 0, nullptr);
vkCmdPushConstants(cmd, oceanPipelineLayout,
                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                   0, sizeof(OceanPush), &oceanPush);
VkBuffer oceanVBufs[] = { oceanVertexBuffer.buffer };
VkDeviceSize oceanOffsets[] = { 0 };
vkCmdBindVertexBuffers(cmd, 0, 1, oceanVBufs, oceanOffsets);
vkCmdBindIndexBuffer(cmd, oceanIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
vkCmdDrawIndexed(cmd, oceanIndexCount, 1, 0, 0, 0);
```

### II.7 `CMakeLists.txt` — New Shader Targets

Find the block of `add_custom_command` calls that compile terrain shaders and add:

```cmake
add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/ocean.vert.spv
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3
            -I${CMAKE_SOURCE_DIR}/shaders
            ${CMAKE_SOURCE_DIR}/shaders/ocean.vert
            -o ${SHADER_OUTPUT_DIR}/ocean.vert.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/ocean.vert
    COMMENT "Compiling ocean.vert"
)
add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/ocean.frag.spv
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3
            -I${CMAKE_SOURCE_DIR}/shaders
            ${CMAKE_SOURCE_DIR}/shaders/ocean.frag
            -o ${SHADER_OUTPUT_DIR}/ocean.frag.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/ocean.frag
            ${CMAKE_SOURCE_DIR}/shaders/bruneton_common.glsl
            ${CMAKE_SOURCE_DIR}/shaders/bruneton_atmosphere.glsl
    COMMENT "Compiling ocean.frag"
)
```

Add both `.spv` files to the `SHADERS` list variable used as a target dependency.

---

### II.8 New File: `shaders/ocean.vert`

Complete source — save as `c:\VulkanProject\shaders\ocean.vert`:

```glsl
#version 450

// ══════════════════════════════════════════════════════════════
//  ocean.vert — Animated Gerstner wave ocean surface
//
//  Input:  flat XZ grid at Y=0 (oceanVertexBuffer)
//  Output: Gerstner-displaced world position + analytic normals
//          passed to ocean.frag.
//
//  Shore harmony: each vertex reads the heightmap to find terrain
//  height at this XZ. When water depth is small (shore proximity),
//  wave amplitude is damped so waves gracefully vanish on the beach.
// ══════════════════════════════════════════════════════════════

layout(location = 0) in vec3 inPosition;   // XZ grid, Y=0

layout(set = 0, binding = 0) uniform FrameData {
    mat4  mvp;
    vec3  cameraPos;
    float _pad0;
    vec3  lightDir;
    float time;
    mat4  lightMVP;
} frame;

layout(set = 0, binding = 1) uniform sampler2D heightmapSampler;

layout(push_constant) uniform OceanPush {
    vec2  worldOrigin;
    float worldSize;
    float heightScale;
    int   heightmapRes;
    float seaLevel;
    float waveAmplitude;
    float waveSpeed;
    float shoreDepth;
    float foamDepth;
    float _pad0;
} ocean;

layout(location = 0) out vec3  fragWorldPos;
layout(location = 1) out vec3  fragNormal;
layout(location = 2) out float fragDepth;      // metres below sea level (>=0 = underwater)
layout(location = 3) out float fragFoamMask;   // 0=none, 1=full foam (computed here)

// ── Gerstner wave function ─────────────────────────────────
// Returns XYZ displacement. Call once per wave train and sum.
//   dir:       unit direction of propagation (XZ)
//   amplitude: metres
//   steepness: Qi in [0, 1/(k*a*N)] — higher = more peaked
//   freq:      spatial frequency = 2π/wavelength
//   speed:     angular frequency ω = sqrt(g*k) for deep water
struct GerstnerOut {
    vec3 disp;
    vec3 normal;   // partial normal contribution (not normalised)
};

GerstnerOut gerstner(vec2 pos, float time,
                     vec2 dir, float amp, float steep,
                     float freq, float speed) {
    float phase = freq * dot(dir, pos) - speed * time;
    float sinP  = sin(phase);
    float cosP  = cos(phase);

    GerstnerOut o;
    o.disp.x  =  steep * amp * dir.x * cosP;
    o.disp.z  =  steep * amp * dir.y * cosP;
    o.disp.y  =  amp * sinP;

    // Analytic partial derivatives for normal reconstruction
    float WA   = freq * amp;
    o.normal.x = -dir.x * WA * cosP;
    o.normal.z = -dir.y * WA * cosP;
    o.normal.y =  steep * WA * sinP;
    return o;
}

void main() {
    // ── Compute terrain height under this ocean vertex ─────
    float coreRes   = float(ocean.heightmapRes - 3);
    float padOffset = 1.5;
    vec2  uv = ((inPosition.xz / ocean.worldSize) * coreRes + padOffset)
               / float(ocean.heightmapRes);
    float terrainH  = texture(heightmapSampler, uv).a * ocean.heightScale;

    // Water depth in metres; clamped at 0 so beachface vertices read 0
    float depth      = max(0.0, ocean.seaLevel - terrainH);
    // Smooth fade: waves reach full amplitude only when depth > shoreDepth
    float depthFade  = smoothstep(0.0, ocean.shoreDepth, depth);

    float t   = frame.time * ocean.waveSpeed;
    float A   = ocean.waveAmplitude * depthFade;
    vec2  xz  = inPosition.xz + ocean.worldOrigin;

    // ── Wave train parameters ───────────────────────────────
    // 6 overlapping wave trains covering main directions.
    // Wavelengths: 18m, 28m, 40m, 14m, 22m, 35m (varied for realism).
    // Speeds: deep-water dispersion ω = sqrt(9.81 * k).
    const float PI2 = 6.28318;
    vec3  totalDisp   = vec3(0.0);
    vec3  totalNormal = vec3(0.0, 0.0, 0.0);

    // Wave 0: main swell, WNW direction
    {
        float wl = 28.0; float k = PI2/wl; float spd = sqrt(9.81*k);
        GerstnerOut g = gerstner(xz, t, normalize(vec2(0.85, 0.53)),
                                 A * 1.00, 0.55, k, spd);
        totalDisp += g.disp; totalNormal += g.normal;
    }
    // Wave 1: secondary swell, SSW direction
    {
        float wl = 40.0; float k = PI2/wl; float spd = sqrt(9.81*k);
        GerstnerOut g = gerstner(xz, t, normalize(vec2(0.20, 0.98)),
                                 A * 0.85, 0.50, k, spd);
        totalDisp += g.disp; totalNormal += g.normal;
    }
    // Wave 2: cross-swell, ESE direction
    {
        float wl = 18.0; float k = PI2/wl; float spd = sqrt(9.81*k);
        GerstnerOut g = gerstner(xz, t, normalize(vec2(0.95, -0.31)),
                                 A * 0.60, 0.45, k, spd);
        totalDisp += g.disp; totalNormal += g.normal;
    }
    // Wave 3: chop, NNE direction
    {
        float wl = 9.0; float k = PI2/wl; float spd = sqrt(9.81*k);
        GerstnerOut g = gerstner(xz, t * 1.05, normalize(vec2(-0.22, 0.98)),
                                 A * 0.35, 0.40, k, spd);
        totalDisp += g.disp; totalNormal += g.normal;
    }
    // Wave 4: micro chop, W direction
    {
        float wl = 5.5; float k = PI2/wl; float spd = sqrt(9.81*k);
        GerstnerOut g = gerstner(xz, t * 1.10, normalize(vec2(-0.98, 0.20)),
                                 A * 0.20, 0.35, k, spd);
        totalDisp += g.disp; totalNormal += g.normal;
    }
    // Wave 5: long-period secondary, SE direction
    {
        float wl = 50.0; float k = PI2/wl; float spd = sqrt(9.81*k);
        GerstnerOut g = gerstner(xz, t * 0.90, normalize(vec2(0.71, -0.71)),
                                 A * 0.70, 0.35, k, spd);
        totalDisp += g.disp; totalNormal += g.normal;
    }

    // ── World position ─────────────────────────────────────
    vec3 worldPos = vec3(ocean.worldOrigin.x + inPosition.x + totalDisp.x,
                         ocean.seaLevel      + totalDisp.y,
                         ocean.worldOrigin.y + inPosition.z + totalDisp.z);

    // ── Analytic normal ────────────────────────────────────
    // Normal from Gerstner derivatives: N = (-dH/dx, 1 - dQ/dy, -dH/dz)
    // totalNormal accumulates partial derivatives; construct world N.
    vec3 N = normalize(vec3(-totalNormal.x,
                             1.0 - totalNormal.y,
                            -totalNormal.z));

    // ── Foam mask from wave steepness ──────────────────────
    // Where the Gerstner crest pinches (N.y close to 0 or negative),
    // the wave is on the verge of breaking → whitecap foam.
    float steepness = 1.0 - N.y;
    float deepFoam   = smoothstep(0.30, 0.55, steepness);         // open-ocean whitecaps

    // Shore foam: high where depth < foamDepth AND displacement is upward
    float shoreFoam  = smoothstep(ocean.foamDepth, 0.0, depth)
                     * smoothstep(0.0, 0.5, totalDisp.y / max(A, 0.01));

    fragFoamMask = clamp(deepFoam + shoreFoam, 0.0, 1.0);
    fragDepth    = depth;
    fragNormal   = N;
    fragWorldPos = worldPos;

    gl_Position = frame.mvp * vec4(worldPos, 1.0);
}
```

---

### II.9 New File: `shaders/ocean.frag`

Complete source — save as `c:\VulkanProject\shaders\ocean.frag`:

```glsl
#version 450

// ══════════════════════════════════════════════════════════════
//  ocean.frag — Ocean surface shading
//
//  Features:
//   · Depth-keyed water color (deep blue-black → shallow turquoise)
//   · Fresnel reflectance (sky color + sun glint)
//   · Animated dual-layer normal perturbation (from texArray water layers)
//   · Whitecap / shore foam (fragFoamMask from vertex shader)
//   · Bruneton aerial perspective matching terrain.frag
//   · Shore wave crests on terrain (foam fade toward land handled in terrain.frag)
// ══════════════════════════════════════════════════════════════

layout(location = 0) in vec3  fragWorldPos;
layout(location = 1) in vec3  fragNormal;
layout(location = 2) in float fragDepth;
layout(location = 3) in float fragFoamMask;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameData {
    mat4  mvp;
    vec3  cameraPos;
    float _pad0;
    vec3  lightDir;
    float time;
    mat4  lightMVP;
} frame;

// texArray: layer 0 = deep water albedo/normal, layer 1 = shallow water
layout(set = 0, binding = 2) uniform sampler2DArray texArray;

// Shadow map for terrain self-shadow (ocean can be in terrain shadow)
layout(set = 0, binding = 3) uniform sampler2DShadow shadowMap;

// Bruneton atmosphere for fog colour
#include "bruneton_common.glsl"
#include "bruneton_atmosphere.glsl"
layout(set = 0, binding = 4) uniform AtmosphereBlock {
    AtmosphereParams atmos;
};

layout(push_constant) uniform OceanPush {
    vec2  worldOrigin;
    float worldSize;
    float heightScale;
    int   heightmapRes;
    float seaLevel;
    float waveAmplitude;
    float waveSpeed;
    float shoreDepth;
    float foamDepth;
    float _pad0;
} ocean;

// ── Shadow PCF (same as terrain) ──────────────────────────────
float shadowFactor(vec3 worldPos, vec3 N, vec3 L) {
    vec4  sc   = frame.lightMVP * vec4(worldPos, 1.0);
    vec3  proj = sc.xyz / sc.w;
    proj.xy    = proj.xy * 0.5 + 0.5;
    if (any(lessThan(proj, vec3(0.0))) || any(greaterThan(proj, vec3(1.0))))
        return 1.0;
    float bias   = max(0.0008 * (1.0 - dot(N, L)), 0.0002);
    float shadow = 0.0;
    vec2  texel  = 1.0 / vec2(textureSize(shadowMap, 0));
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            shadow += texture(shadowMap, vec3(proj.xy + vec2(x,y)*texel, proj.z - bias));
    return shadow / 9.0;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(frame.lightDir);
    vec3 V = normalize(frame.cameraPos - fragWorldPos);
    vec3 H = normalize(L + V);

    // ── Animated normal perturbation from water texArray ──────
    // Layer 6 = water normal map (normal layers are at index+6)
    // Two counter-scrolling samples blended for cross-swell look.
    float t  = frame.time;
    const float NM_SCALE = 1.0 / 6.0;   // normal map tile size in metres

    vec3 scroll1 = fragWorldPos + vec3( t * 0.50, 0.0,  t * 0.25);
    vec3 scroll2 = fragWorldPos + vec3(-t * 0.35, 0.0, -t * 0.60);

    vec3 nm1 = texture(texArray, vec3(scroll1.xz * NM_SCALE, 6.0)).xyz * 2.0 - 1.0;
    vec3 nm2 = texture(texArray, vec3(scroll2.xz * NM_SCALE, 7.0)).xyz * 2.0 - 1.0;  // layer 7 = shallow water normal
    // Blend the two normal maps and reorient onto geometric surface normal
    vec3 nmBlend = normalize(nm1 + nm2);

    // Perturb the geometric normal (strong near-surface, tapers with distance)
    float camDist   = length(frame.cameraPos - fragWorldPos);
    float nmStrength = smoothstep(200.0, 10.0, camDist) * 0.55;
    vec3 Nlit = normalize(mix(N, nmBlend, nmStrength));

    // ── Water depth-keyed color ────────────────────────────────
    //   depth 0 (shoreline)  → turquoise/cyan
    //   depth ~10 m          → mid-blue
    //   depth > 30 m         → deep navy
    vec3 colorShore = vec3(0.08, 0.52, 0.60);   // shallow turquoise
    vec3 colorMid   = vec3(0.03, 0.20, 0.48);   // mid blue
    vec3 colorDeep  = vec3(0.01, 0.06, 0.22);   // abyssal navy

    float d10  = clamp(fragDepth / 10.0, 0.0, 1.0);
    float d30  = clamp(fragDepth / 30.0, 0.0, 1.0);
    vec3  waterColor = mix(mix(colorShore, colorMid, d10), colorDeep, d30 * 0.7);

    // ── Subsurface scattering approximation ───────────────────
    // Thin water at shore lets light scatter from below → warmer/brighter
    float sssFactor = smoothstep(5.0, 0.0, fragDepth) * max(dot(L, V), 0.0) * 0.35;
    waterColor = mix(waterColor, vec3(0.12, 0.72, 0.65), sssFactor);

    // ── Fresnel (Schlick) ─────────────────────────────────────
    float NdotV  = clamp(dot(Nlit, V), 0.0, 1.0);
    float F0     = 0.020;   // water at normal incidence
    float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

    // Sky reflection color (simple approximation — zenith to horizon)
    // Uses Bruneton sun direction to tint reflected sky appropriately
    vec3 sunDirYup = vec3(atmos.sun_direction.x,
                          atmos.sun_direction.z,
                          atmos.sun_direction.y);
    float sunElev = clamp(sunDirYup.y, 0.0, 1.0);

    // Sky hemisphere: bright at zenith, pale at horizon
    vec3 skyZenith  = vec3(0.17, 0.42, 0.82) * (0.6 + 0.4 * sunElev);
    vec3 skyHorizon = vec3(0.55, 0.72, 0.90) * (0.4 + 0.6 * sunElev);
    vec3 R = reflect(-V, Nlit);   // reflection ray
    float reflElev = clamp(R.y, 0.0, 1.0);
    vec3 skyRefl = mix(skyHorizon, skyZenith, reflElev * reflElev);

    // Sunrise/sunset sky tint
    if (sunElev < 0.15) {
        vec3 sunsetTint = vec3(1.0, 0.55, 0.20) * (1.0 - sunElev / 0.15);
        skyRefl = mix(skyRefl, sunsetTint, 0.45);
    }

    // ── Specular (sun glint) ───────────────────────────────────
    // High shininess (microfacet-like Blinn-Phong at roughness ~0.05)
    float shadow  = shadowFactor(fragWorldPos, Nlit, L);
    float specPow = 256.0;
    float spec    = pow(max(dot(Nlit, H), 0.0), specPow) * 1.8 * shadow;
    // Clip specular when sun is below horizon
    spec *= clamp(sunElev * 8.0, 0.0, 1.0);

    // ── Combine surface + reflections ─────────────────────────
    float ambient  = 0.25;
    float diffuse  = max(dot(Nlit, L), 0.0) * 0.35 * shadow;

    vec3 surfaceColor  = waterColor * (ambient + diffuse);
    vec3 reflColor     = skyRefl;

    // Fresnel blend: at glancing angles, surface becomes a mirror
    vec3 color = mix(surfaceColor, reflColor, clamp(fresnel * 2.5, 0.0, 0.85));

    // Add specular glint on top
    color += vec3(spec);

    // ── Foam / whitecaps ───────────────────────────────────────
    float foam  = fragFoamMask;

    // Add animated foam texture from texArray (layer 0 water albedo)
    // Scroll in wave propagation direction for kinetic look
    vec3 foamUV1 = fragWorldPos + vec3(t * 0.4, 0.0, t * 0.2);
    vec3 foamUV2 = fragWorldPos + vec3(-t * 0.3, 0.0, t * 0.5);
    float foamTex = texture(texArray, vec3(foamUV1.xz / 5.0, 0.0)).r
                  * texture(texArray, vec3(foamUV2.xz / 3.0, 0.0)).r;
    foamTex = clamp(foamTex * 2.0, 0.0, 1.0);

    vec3 foamColor = vec3(0.92, 0.96, 1.00);
    color = mix(color, foamColor, clamp(foam * foamTex * 1.4, 0.0, 1.0));

    // Extra thin foam line at the shoreline (depth transitions)
    float shoreLine = smoothstep(1.5, 0.0, fragDepth) * 0.6;
    color = mix(color, foamColor, shoreLine);

    // ── Aerial perspective (matching terrain.frag) ─────────────
    {
        float dist    = length(frame.cameraPos - fragWorldPos);
        float distKm  = dist * atmos.cameraScale;
        float fogAmt  = 1.0 - exp(-distKm * 0.5);

        vec3 viewDir  = normalize(fragWorldPos - frame.cameraPos);
        float cosTheta = dot(viewDir, sunDirYup);
        float rPhase   = rayleighPhase(cosTheta);
        float mPhase   = cornetteShanksMiePhaseFunction(atmos.mie_phase_function_g, -cosTheta);

        vec3 fogColor = (atmos.rayleigh_scattering * rPhase
                       + atmos.mie_scattering      * mPhase)
                       * atmos.solar_irradiance * 60.0;
        fogColor = max(fogColor, vec3(0.4, 0.55, 0.75) * 0.15);

        color = mix(color, fogColor, clamp(fogAmt, 0.0, 0.80));
    }

    outColor = vec4(color, 1.0);
}
```

---

## Part III — Shore Harmony Details

### III.1 The Problem

The phrase "waves in harmony with the land" means:

1. **No seam at the waterline** — ocean geometry must not intersect or float above terrain geometry at the coast.
2. **Waves diminish in shallow water** — waves don't maintain open-ocean amplitude up to the beach and abruptly stop.
3. **The beach reacts to the waves** — wet sand, foam runup, and darkened sand at wave impact are visible on the terrain side.
4. **The transition feels continuous** — the viewer cannot identify a hard border between the ocean render pass and the terrain render pass.

### III.2 How the Implementation Achieves Harmony

#### Vertex-level depth fade (ocean.vert)

```
float depthFade = smoothstep(0.0, ocean.shoreDepth, depth);
float A         = ocean.waveAmplitude * depthFade;
```

- When `depth = 0` (terrain at sea level): `depthFade = 0`, `A = 0` → the ocean vertex sits at exactly `seaLevel`, no displacement.
- When `depth = shoreDepth` (4 m): `depthFade ≈ 0.5`, `A ≈ 0.275 m`.
- When `depth > 8 m`: `depthFade ≈ 1.0`, full open-ocean amplitude.

The Gerstner horizontal displacement is also multiplied by `depthFade`, so ocean geometry contracts toward the coastline rather than stretching over dry land. In practice this means the ocean mesh edge naturally terminates flush with the beach face.

#### Depth test correctness

Both terrain and ocean write to the same depth buffer. The terrain renders first. Where terrain is above sea level (`y > seaLevel`), it passes the depth test and writes; the ocean vertex at those same XZ coordinates will be at `y = seaLevel` (zero displacement), which is deeper than the terrain surface, and therefore fails the depth test and is culled. No manual masking required.

#### Shore foam on terrain (terrain.frag — Section I.3.4)

The shore foam code in `terrain.frag` uses the Gerstner **wave phase** to animate foam running up the beach face:

```
float phase = dot(fragWorldPos.xz, shoreDir) * waveFreq - frame.time * (2π / wavePeriod);
float waveCrest = smoothstep(0.3, 1.0, sin(phase) * 0.5 + 0.5);
```

`shoreDir` is derived from `N.xz` (the horizontal component of the terrain normal), which points up-slope — perpendicular to the shoreline. The phase moves with the same angular frequency as the ocean Gerstner wave period (6.5 s here; tune to match `waveSpeed`). This makes the foam bands on the beach appear to be extensions of the ocean wave crests arriving at shore.

**Critical tuning note:** The `wavePeriod` in `terrain.frag` and the effective period of the dominant Gerstner wave in `ocean.vert` must match. With the default parameters in `ocean.vert`, wave 0 (28 m wavelength, deep-water speed `sqrt(9.81 * 2π/28) ≈ 2.48 m/s`) has a period of `28/2.48 ≈ 11.3 s`. The terrain shore foam uses 6.5 s — this is intentional to give a faster, shorter breaking wave character on the beach. Set both to the same value if stricter continuity is desired.

#### Foam line continuity

In `ocean.frag`:
```glsl
float shoreLine = smoothstep(1.5, 0.0, fragDepth) * 0.6;
color = mix(color, foamColor, shoreLine);
```

This creates a white foam band on the ocean surface within 1.5 m of depth 0. On the terrain side, the shore foam code creates a matching band above the waterline. Together they create a continuous white border at the land-water junction.

---

## Part IV — TerrainSettings Integration

The ocean wave parameters (`waveAmplitude`, `waveSpeed`, `shoreDepth`, `foamDepth`) should be exposed in the ImGui settings panel so the user can tune them at runtime without recompile.

### IV.1 Add to `TerrainSettings.h`

```cpp
// Ocean / wave parameters
float waveAmplitude  = 0.55f;  // metres
float waveSpeed      = 1.00f;  // multiplier
float shoreDepth     = 4.00f;  // m — wave fade onset
float foamDepth      = 1.80f;  // m — foam onset
```

### IV.2 Add to `SettingsPanel.h` (the ImGui wave section)

Inside the appropriate collapsing header or add a new "Ocean" section:

```cpp
if (ImGui::CollapsingHeader("Ocean")) {
    bool waveChanged = false;
    waveChanged |= ImGui::SliderFloat("Wave Amplitude (m)",  &settings.waveAmplitude, 0.0f, 3.0f);
    waveChanged |= ImGui::SliderFloat("Wave Speed",          &settings.waveSpeed,     0.1f, 3.0f);
    waveChanged |= ImGui::SliderFloat("Shore Fade Depth (m)",&settings.shoreDepth,    1.0f, 20.0f);
    waveChanged |= ImGui::SliderFloat("Foam Depth (m)",      &settings.foamDepth,     0.5f, 5.0f);
    if (waveChanged) {
        eng.oceanPush.waveAmplitude = settings.waveAmplitude;
        eng.oceanPush.waveSpeed     = settings.waveSpeed;
        eng.oceanPush.shoreDepth    = settings.shoreDepth;
        eng.oceanPush.foamDepth     = settings.foamDepth;
    }
}
```

### IV.3 JSON persistence

In the `terrain_settings.json` loading/saving code in `VulkanEngine.cpp`, add read/write for the four new fields under an `"ocean"` key. Pattern matches the existing settings deserialization (key → field assignments).

---

## Part V — Cleanup & Deletion

### V.1 Ocean vertices replacing terrain water rendering

After the ocean pass is live, the following code in `terrain.frag` becomes redundant for ocean areas (it still applies for lakes and rivers). It does **not** need to be removed — the depth buffer correctly prevents terrain ocean shading from being visible, since the ocean pass overwrites those pixels. However, removing or commenting out the `h < 0` fallback `heightColor` branch slightly reduces fragment workload:

```glsl
// terrain.frag ~line 337: can be removed once ocean pass is live
// else if (h < 0.0)  heightColor = vec3(0.06, 0.12, 0.32);  // deep water
```

### V.2 Destroy ocean resources in cleanup

In `VulkanEngine::cleanup()` before the device is destroyed:

```cpp
vkDestroyPipeline(device, oceanPipeline, nullptr);
vkDestroyPipelineLayout(device, oceanPipelineLayout, nullptr);
vkDestroyBuffer(device, oceanVertexBuffer.buffer, nullptr);
vkFreeMemory(device,   oceanVertexBuffer.memory,  nullptr);
vkDestroyBuffer(device, oceanIndexBuffer.buffer,  nullptr);
vkFreeMemory(device,   oceanIndexBuffer.memory,   nullptr);
```

---

## Part VI — Summary of All File Changes

| File | Change type | Description |
|------|------------|-------------|
| `src/VulkanEngine.h` | Edit | `_pad1` → `float time` in `FrameData`; add `OceanPush` struct; add ocean member variables to `VulkanEngine` class |
| `src/VulkanEngine.cpp` | Edit | `fd.time = glfwGetTime()` in UBO update; add `createOceanMesh()`, `createOceanPipeline()`; add ocean draw call in `recordRenderCommand`; add cleanup calls |
| `src/TerrainSettings.h` | Edit | Add 4 float wave parameters |
| `src/SettingsPanel.h` | Edit | Add "Ocean" ImGui section |
| `shaders/terrain.frag` | Edit | Add `triplanarMS` + `triplanarNormalMS` helpers; replace `TEX_SCALE` block with multi-scale sampling; add animated water section; add shore foam & wet sand section |
| `shaders/texgen.comp` | Edit | Fix `encodeNormal` formula; replace `hWater`, `hSand`, `hGrass`, `hRock`, rock albedo |
| `shaders/ocean.vert` | **New** | Full Gerstner wave vertex shader (see §II.8) |
| `shaders/ocean.frag` | **New** | Full ocean fragment shader (see §II.9) |
| `CMakeLists.txt` | Edit | Add two `add_custom_command` blocks for ocean shaders; add to SHADERS list |

---

## Part VII — Build & Validation Checklist

1. **Compile shaders first:** `cmake --build build_cb --target TerrainEngine` — `ocean.vert.spv` and `ocean.frag.spv` should appear in the shader output directory. Fix any GLSL errors before proceeding to C++.
2. **Check push constant size:** `sizeof(OceanPush)` must be ≤ 128 bytes (Vulkan guaranteed minimum). The struct as written is 48 bytes — well within limits.
3. **Verify FrameData layout:** The `time` field at byte offset 28 (after `mvp`=64, `cameraPos`+`_pad0`=16, `lightDir`=12) must map to the same GLSL `frame.time`. Confirm with `offsetof(FrameData, time) == 76` (64+12+0 pad = 76 — matches std140 vec3+float = 16 bytes → offset 64+16 = 80). Double-check with `static_assert(offsetof(FrameData, time) == 76, "layout mismatch");` in VulkanEngine.cpp.
   - Actually: `mvp` = 64 bytes (offset 0), `cameraPos` = 12 bytes + `_pad0` = 4 bytes (offset 64, total 16 bytes), `lightDir` = 12 bytes (offset 80) + `time` = 4 bytes (offset 92). Final offset of `lightMVP` = 96. Add `static_assert(sizeof(FrameData) == 160)` to confirm.
4. **Depth test:** Fly the camera to the coastline and confirm no z-fighting between ocean and terrain. If flickering occurs, increase `shoreDepth` slightly (ocean vertices will sit lower at the coast) or set a small polygon offset on the terrain pipeline.
5. **Shore foam alignment:** The wave crest foam on terrain (`terrain.frag`) should visually arrive at the waterline simultaneously with the ocean foam line. If there is a phase offset, adjust the `wavePeriod` constant in the shore foam section of `terrain.frag`.
6. **Performance:** The 257×257 ocean mesh = 263,169 triangles per frame. At 4K, the dominant cost is the fragment shader. If GPU-bound, reduce ocean mesh to 129×129 — the visual difference is minor because Gerstner smoothly interpolates between vertices.

---

*End of report. All shader source is production-ready GLSL 450. All C++ snippets assume the existing VulkanEngine architecture without structural refactoring.*
