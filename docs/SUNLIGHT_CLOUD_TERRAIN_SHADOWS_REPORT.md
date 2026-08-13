# TerrainEngine - Sunlight, Cloud, and Terrain Shadow Report

**Purpose:** This document is a hand-off report for a stronger coding model. The task is to make sunlight and volumetric clouds create believable shadows on the terrain. This report explains the current state of the project, the relevant rendering systems, the likely failure points, and concrete prompts that can be given to an AI model to implement the feature safely.

**Primary goal:** terrain should receive:

- Direct sunlight and terrain self-shadowing from the sun.
- Moving cloud shadows projected onto the terrain.
- Lighting that stays consistent with the runtime sun controls, Bruneton sky, and cloud raymarch pass.

**Important note:** the project already has a terrain shadow map and volumetric cloud renderer. Do not start from scratch. The right implementation is to repair/align the existing sunlight path first, then add a low-cost cloud transmittance term to the terrain shader.

---

## 1. Current Project State

### Active Engine

The active executable is `TerrainEngine` from the root `CMakeLists.txt`.

Relevant source:

- `src/VulkanEngine.cpp`
- `src/VulkanEngine.h`
- `src/main.cpp`
- `src/TerrainSettings.h`
- `shaders/terrain.vert`
- `shaders/terrain.frag`
- `shaders/shadow.vert`
- `shaders/bruneton_sky.frag`
- `shaders/bruneton_clouds.frag`
- `shaders/cloud_noise_gen.comp`
- `shaders/bruneton_*`

There is also an older/reference renderer in `atmosphere_bac_src/`. It is useful as background, but the root `src/` and root `shaders/` path is the current engine. The existing `Cloudscapes_Integration_Report.md` describes the earlier porting work and confirms that the cloud code was adapted from `atmosphere_bac_src`.

### Build and Shader Pipeline

The root `CMakeLists.txt` compiles all active shaders with `glslc --target-env=vulkan1.3` and copies SPIR-V files next to the executable.

Relevant shaders already included in the build:

- Terrain: `heightmap.comp`, `terrain.vert`, `terrain.frag`
- Sun shadow map: `shadow.vert`
- Atmosphere: `bruneton_transmittance.comp`, `bruneton_multiscatter.comp`, `bruneton_skyview.comp`, `bruneton_sky.vert`, `bruneton_sky.frag`
- Clouds: `cloud_noise_gen.comp`, `bruneton_clouds.vert`, `bruneton_clouds.frag`

Any new shader required for cloud shadow precomputation must be added to `CMakeLists.txt` using `add_shader(TerrainEngine ${SHADER_SRC_DIR}/...)`.

---

## 2. Terrain Rendering and Existing Sun Shadows

### Terrain Data

The active terrain system is regional/infinite by default:

- `TerrainSettings::infiniteTerrainEnabled = true`
- `regionWorldSize = 4096.0f` meters
- `regionRes = 512`
- `loadedRegionRadius = 1`, so a 3x3 region neighborhood is normally loaded

`VulkanEngine::generateTerrainRegion()` creates CPU-side region height data using SimpleHydrology-style OpenSimplex noise, hydrology, airfield carving, and terrain statistics. `uploadTerrainRegionTexture()` uploads this data into the GPU heightmap image used by terrain rendering.

`TerrainRegion` stores:

- `worldOrigin`
- `worldSize`
- `resolution`
- `heights`
- `flow`
- GPU `image`
- graphics `descriptorSet`
- shadow `shadowDescriptorSet`
- min/max height

The terrain vertex shader, `shaders/terrain.vert`, samples the heightmap and outputs:

- `fragWorldPos`
- `fragNormal`
- `fragSlope`
- `fragHumidity`
- `fragRiverMask`

The heightmap channels used by terrain rendering are:

- `.a` = height
- `.g` = humidity
- `.b` = river mask

### Terrain Material and Lighting

`shaders/terrain.frag` performs:

- triplanar terrain texture array sampling
- biome color fallback
- humidity tinting
- slope-based rock exposure
- river rendering
- Blinn-Phong style direct lighting
- shadow map lookup
- aerial perspective approximation using Bruneton atmosphere data

Current terrain lighting flow:

1. `FrameData.lightDir` is read by `terrain.frag`.
2. `shadowFactor(worldPos, N, L)` transforms world position by `FrameData.lightMVP`.
3. It samples `sampler2DShadow shadowMap`.
4. The final direct lighting is:

```glsl
color.rgb = color.rgb * ambient + (color.rgb * diffuse + specular * vec3(1.0)) * shadow;
```

This means the current sun shadow factor only affects diffuse/specular direct light. Ambient/fog remains visible.

### Existing Shadow Map

The engine already creates:

- `shadowMap`: `D32_SFLOAT`, `2048 x 2048`
- `shadowSampler`: compare sampler with `VK_COMPARE_OP_LESS_OR_EQUAL`
- `shadowUniformBuffer`: `mat4 lightMVP`
- `shadowPipeline`: depth-only terrain rendering using `shaders/shadow.vert`

The shadow pass happens in `VulkanEngine::recordRenderCommand()` before main terrain rendering:

1. Recompute skyview LUT.
2. Transition shadow map to `DEPTH_ATTACHMENT_OPTIMAL`.
3. Begin depth-only dynamic rendering at `SHADOW_MAP_SIZE`.
4. Draw terrain regions through `shadowPipeline`.
5. Transition shadow map to `DEPTH_READ_ONLY_OPTIMAL`.
6. Begin main rendering and draw terrain using the shadow map.

### Critical Existing Problem

The runtime sun and the terrain shadow light direction are not unified.

The atmosphere/sky sun uses `settings.sunElevation` and `settings.sunAzimuth`:

- `createAtmosphereUBO()`
- `updateAtmosphereUBO()`
- `src/main.cpp` arrow keys edit these settings at runtime

But `updateUniformBuffer()` currently sets:

```cpp
frame.lightDir = glm::normalize(glm::vec3(0.5f, 1.0f, 0.3f));
```

The shadow pass also computes:

```cpp
glm::vec3 lightDir = glm::normalize(glm::vec3(0.5f, 1.0f, 0.3f));
```

This means terrain lighting and shadow casting do not follow the visible sun. The first implementation priority must be to derive `FrameData.lightDir`, `FrameData.lightMVP`, and the shadow pass light frustum from the same `settings.sunElevation/settings.sunAzimuth` vector used by the Bruneton atmosphere.

---

## 3. Atmosphere and Sun Direction

The project uses a Bruneton/Hillaire-style atmosphere pipeline:

- `AtmosphereParams` in `src/VulkanEngine.h`
- `createAtmosphereUBO()`
- `updateAtmosphereUBO()`
- `createBrunetonLUTs()`
- `runBrunetonStaticLUTs()`
- `dispatchSkyviewLUT()`
- `bruneton_sky.frag`

The atmosphere convention is:

- Engine world is Y-up, meters.
- Bruneton math is Z-up, kilometers.
- `atmosphereParams.cameraScale = 0.001f`, converting engine meters to kilometers.
- Engine sun direction is converted to Bruneton with:

```cpp
atmosphereParams.sun_direction = glm::vec3(sunYup.x, sunYup.z, sunYup.y);
```

The shared source of truth for the sun should be:

```cpp
glm::vec3 sunYup = glm::normalize(glm::vec3(
    cos(glm::radians(settings.sunAzimuth)) * cos(glm::radians(settings.sunElevation)),
    sin(glm::radians(settings.sunElevation)),
    sin(glm::radians(settings.sunAzimuth)) * cos(glm::radians(settings.sunElevation))
));
```

Use this vector for:

- `FrameData.lightDir`
- shadow map light camera
- terrain direct lighting
- optional cloud shadow projection

Use the swizzled Bruneton version only for atmosphere/cloud shaders that expect Z-up.

---

## 4. Volumetric Clouds

### Existing Cloud Renderer

Cloud rendering is already advanced and active.

`src/VulkanEngine.h` documents the system:

- `cloud_noise_gen.comp` fills `cloudNoise3D`, a `128^3 RGBA16F` 3D Worley/Perlin FBM volume.
- `blueNoise2D` is a `64 x 64 R8` texture uploaded from CPU.
- `CloudParams` controls density, bounds, phase, sample counts, light absorption, and wind.
- `bruneton_clouds.frag` draws a full-screen raymarch pass.

`bruneton_clouds.frag` already implements:

- density sampling from a 3D noise volume
- cloud layer as a spherical shell
- primary raymarch from the camera
- secondary light march from cloud samples toward the sun
- Beer's law extinction
- powder darkening
- double Henyey-Greenstein phase
- blue noise start offset
- transmittance LUT sampling
- depth-buffer truncation so clouds stop behind terrain
- premultiplied alpha compositing over terrain/sky/foliage

### Current Cloud Render Order

`recordRenderCommand()` order:

1. Recompute skyview LUT.
2. Render terrain into the shadow map.
3. Render main terrain.
4. Render sky for far-depth pixels.
5. Render foliage.
6. Render aircraft.
7. End main dynamic rendering.
8. Transition depth to `DEPTH_READ_ONLY_OPTIMAL`.
9. `updateCloudParamsUBO()`.
10. `drawCloudsPass()`.
11. Restore depth layout.
12. Render ImGui.
13. Present.

Clouds are composited after terrain. This is good for visible clouds, but it means clouds do not currently affect terrain lighting. The cloud pass knows how much light the clouds receive from the sun, but the terrain pass does not know how much sun is blocked by clouds above each terrain point.

### Current Missing Feature

There is no cloud shadow term in `terrain.frag`.

The cloud density exists only in `bruneton_clouds.frag`. Terrain lighting only sees:

- terrain self-shadow map
- fixed ambient
- direct diffuse/specular
- approximate fog

Therefore cloud shadows on terrain must be implemented either by:

1. Sampling the same cloud density function in `terrain.frag` along the ray from terrain point to sun.
2. Precomputing a screen/world-space cloud shadow texture and sampling it in `terrain.frag`.
3. Precomputing a lower-resolution world-space cloud transmittance map centered on loaded terrain regions.

For this project, option 1 is the best first implementation because it reuses the existing cloud parameters and noise texture without a new render pass. Option 3 is the best long-term implementation if performance becomes a problem.

---

## 5. Recommended Implementation Strategy

### Priority 1: Unify Sun Direction

Create one helper in `VulkanEngine.cpp` or `VulkanEngine.h`:

```cpp
glm::vec3 VulkanEngine::currentSunDirectionYUp() const;
```

It should return the normalized engine-space Y-up direction computed from `settings.sunElevation` and `settings.sunAzimuth`.

Then use it in:

- `createAtmosphereUBO()`
- `updateAtmosphereUBO()`
- `updateUniformBuffer()`
- shadow pass light frustum setup inside `recordRenderCommand()`

Acceptance criteria:

- Arrow-key sun movement changes the sky, terrain lighting, and terrain shadow direction together.
- Low sun angles produce long terrain shadows in the expected direction.
- The terrain's bright side faces the visible sun disc.

### Priority 2: Fix Shadow Map Robustness Before Cloud Shadows

The current shadow system is single-map orthographic shadowing over the loaded regional neighborhood. It is acceptable for a first pass, but the model must check for these issues:

- `shadow.vert` uses `uv = inPosition.xz / chunk.worldSize`, while `terrain.vert` uses a padded sampling formula for regional heightmaps. This can cause mismatch at region edges.
- `shadowFactor()` does not guard against projected coordinates outside `[0,1]`. Sampling outside the shadow map may clamp to edges and create false shadows.
- `FrameData.lightMVP` and the shadow pass light MVP are calculated in two places. They must not diverge.
- Only terrain casts terrain shadows. Foliage and aircraft currently do not cast into the shadow map.
- Shadow coverage uses one 2048 map over up to a 3x3 loaded region area, so resolution may be coarse.

Minimum fixes:

- Make `shadow.vert` use the same heightmap UV convention as `terrain.vert`.
- Return `1.0` from `shadowFactor()` if projected XY is outside `[0,1]` or projected Z is outside a useful range.
- Compute the light matrix once per frame and share it between the shadow UBO and `FrameData`.

### Priority 3: Add Terrain Cloud Shadow Sampling

The first cloud-shadow implementation should add a lightweight cloud transmittance function to `terrain.frag`:

1. Bind cloud-related resources to the terrain descriptor set:
   - `CloudParams` UBO
   - `cloudNoise3D` sampler
   - optional blue-noise texture if dithering terrain shadow samples
2. Add the same cloud density helper used by `bruneton_clouds.frag`, or move shared GLSL functions into an include file such as `shaders/cloud_common.glsl`.
3. In `terrain.frag`, from each terrain fragment:
   - Convert `fragWorldPos` from engine Y-up meters to Bruneton Z-up kilometers.
   - March along the sun direction through the cloud layer.
   - Accumulate cloud density.
   - Convert density to sunlight transmittance with Beer's law.
   - Multiply only the direct sunlight term by this transmittance.

The final terrain lighting should conceptually become:

```glsl
float terrainShadow = shadowFactor(fragWorldPos, N, L);
float cloudShadow = cloudSunTransmittanceAtTerrain(fragWorldPos);
float directVisibility = terrainShadow * cloudShadow;

color.rgb = color.rgb * ambient
          + (color.rgb * diffuse + specular * vec3(1.0)) * directVisibility;
```

Do not darken ambient, aerial perspective, or river reflection fully. Clouds block direct sunlight, not all light.

### Priority 4: Make Cloud Shadows Cheap and Stable

Terrain fragments are numerous, and a full cloud raymarch per terrain pixel can be expensive. Start with low sample counts:

- 6-12 cloud-shadow samples from terrain point toward sun.
- Early out when ray misses the cloud layer.
- Early out when accumulated optical depth reaches a dark threshold.
- Use a deterministic world-space dither, not screen-space-only noise, to avoid crawling on camera movement.

Recommended cloud-shadow behavior:

- Soft broad shadows, not razor-sharp hard silhouettes.
- Strength controlled by density and sun elevation.
- At high sun: shadows are compact and close under clouds.
- At low sun: shadows stretch far, but should fade with distance to avoid horizon artifacts.

### Priority 5: Long-Term Cloud Shadow Map

If direct terrain-fragment sampling is too slow, implement a cloud shadow texture:

- Low resolution, e.g. `512 x 512` or `1024 x 1024`.
- World-space coverage matching the loaded terrain/shadow coverage.
- Centered on the same region center as the terrain shadow map.
- Computed once per frame or every few frames by compute shader.
- Stores cloud sun transmittance in `R8_UNORM` or `R16_SFLOAT`.
- Sampled by `terrain.frag` using world XZ coordinates.

This is likely the best final architecture for a large terrain renderer because it decouples cloud marching cost from terrain pixel count.

---

## 6. Descriptor and Pipeline Changes

### Terrain Descriptor Set Today

The active terrain descriptor set has at least these bindings:

- binding 0: `FrameData` UBO
- binding 1: heightmap sampler
- binding 2: terrain texture array
- binding 3: shadow map sampler
- binding 4: atmosphere UBO

This matches `shaders/terrain.vert` and `shaders/terrain.frag`.

### Required Additions for Direct Cloud Shadow Sampling

Add bindings to the terrain descriptor set:

- binding 5: `CloudParams` UBO
- binding 6: `sampler3D cloudNoise3D`

Optional:

- binding 7: `sampler2D blueNoise2D`

Then update:

- `VulkanEngine::createDescriptorLayouts()`
- descriptor pool sizing
- `allocateTerrainRegionDescriptorSet()`
- `createGlobalTerrainDescriptorSet()`
- swapchain/cleanup paths if descriptor layouts are recreated
- `shaders/terrain.frag`

Be careful: `CloudParams` lives in `VulkanEngine` and is currently written before `drawCloudsPass()`. If the terrain pass samples cloud parameters, update `cloudParams` before terrain rendering, not after the main render pass. Move `updateCloudParamsUBO()` to before terrain draw, or split wind/frame updates so terrain and cloud pass use the same cloud state in the same frame.

---

## 7. Correctness Details

### Coordinate Systems

Engine:

- Y-up
- meters
- terrain world position: `vec3(x, y, z)`

Bruneton/cloud math:

- Z-up
- kilometers
- conversion:

```glsl
vec3 toBruneton(vec3 v) { return vec3(v.x, v.z, v.y); }
```

For a terrain point:

```glsl
vec3 terrainKm = toBruneton(fragWorldPos) * atmos.cameraScale;
```

For the sun:

- `atmos.sun_direction` is already Bruneton Z-up.
- `frame.lightDir` should remain engine Y-up.

### Cloud Layer

`CloudParams` stores:

- `minBounds`: cloud-layer floor in km above surface
- `maxBounds`: cloud-layer ceiling in km above surface
- `cloudsScale`
- `densityOffset`
- `densityMultiplier`
- absorption settings

The cloud shadow ray starts at the terrain point and marches toward `atmos.sun_direction` in Bruneton space. It should only integrate within the cloud shell. Reuse the cloud shader's `rayCloudLayer()` logic where possible.

### Physical Interpretation

Terrain receives direct sunlight attenuated by:

```text
terrain self-shadow visibility * cloud optical transmittance * atmospheric transmittance approximation
```

The project currently approximates atmospheric terrain fog after direct lighting. Do not try to fully redesign atmospheric terrain shading in the first pass. Keep the existing fog and only add cloud attenuation to direct sun.

### Shadow Strength

Cloud shadow should be:

- `1.0` when no cloud lies between terrain and sun.
- near `0.3-0.7` under typical clouds.
- rarely fully black, because atmospheric skylight and multi-scattering still illuminate terrain.

Use a tunable floor:

```glsl
float cloudVisibility = mix(minCloudSunVisibility, 1.0, exp(-opticalDepth * absorption));
```

Suggested default:

- `minCloudSunVisibility = 0.35`
- absorption tied to `clouds.lightAbsTowardsSun` or a new terrain-specific scalar

---

## 8. AI Prompts for Implementation

### Prompt 1 - Read and Confirm Current Rendering Paths

Use this prompt first:

> Thoroughly inspect `src/VulkanEngine.cpp`, `src/VulkanEngine.h`, `src/TerrainSettings.h`, `src/main.cpp`, `shaders/terrain.vert`, `shaders/terrain.frag`, `shaders/shadow.vert`, and `shaders/bruneton_clouds.frag`. Confirm the active render order, terrain descriptor bindings, sun direction sources, shadow map lifecycle, and cloud resource bindings. Do not modify code yet. Return a concise implementation plan for unifying sun direction and adding cloud shadows to terrain.

### Prompt 2 - Unify Sun Direction

> Implement a single engine-space Y-up sun direction helper based on `settings.sunElevation` and `settings.sunAzimuth`. Use it for `FrameData.lightDir`, shadow-map light camera setup, `FrameData.lightMVP`, and atmosphere UBO swizzling. Remove duplicated hardcoded `glm::vec3(0.5f, 1.0f, 0.3f)` lighting vectors. Ensure runtime arrow-key sun controls update visible sky, terrain direct lighting, and terrain shadow direction consistently.

Acceptance criteria:

- Search confirms no hardcoded terrain/shadow sun vector remains except fallback/default settings.
- `FrameData.lightDir` matches the engine-space version of the atmosphere sun.
- Shadow pass and terrain shader use the same `lightMVP`.

### Prompt 3 - Repair Terrain Shadow Map Accuracy

> Audit and fix terrain shadow mapping. Make `shaders/shadow.vert` sample region heightmaps using the same padded UV convention as `shaders/terrain.vert`. Add robust out-of-bounds handling to `shadowFactor()` in `shaders/terrain.frag` so fragments outside the shadow projection are treated as lit. Keep the existing 3x3 PCF and depth bias unless a validation issue requires a minimal adjustment. Do not add cascaded shadows yet.

Acceptance criteria:

- Terrain shadows align with rendered terrain geometry.
- Region edges do not create false dark bands.
- Low sun angles do not black out the whole terrain.

### Prompt 4 - Share Cloud Density Code

> Refactor cloud density helpers from `shaders/bruneton_clouds.frag` into a shared GLSL include, for example `shaders/cloud_common.glsl`, without changing visual cloud output. The shared code should expose density sampling and cloud-layer ray intersection functions that can be used by both `bruneton_clouds.frag` and `terrain.frag`. Keep coordinate assumptions explicit: Bruneton Z-up, kilometers.

Acceptance criteria:

- Existing clouds render the same after refactor.
- The shared include compiles under both shaders.
- No descriptor binding numbers change in the cloud draw pipeline unless necessary.

### Prompt 5 - Add Direct Cloud Shadows to Terrain

> Add cloud shadowing to `shaders/terrain.frag` by sampling the existing cloud noise volume and cloud parameters along the sun ray from each terrain fragment. Extend the terrain descriptor layout to bind `CloudParams` and `cloudNoise3D`. Move `updateCloudParamsUBO()` so terrain and clouds use the same wind/frame state. Multiply only terrain direct diffuse/specular sunlight by the resulting cloud transmittance; keep ambient, fog, and sky contribution visible.

Acceptance criteria:

- Moving clouds cast moving soft shadows on terrain.
- Shadows follow `settings.sunElevation/settings.sunAzimuth`.
- Cloud shadows are absent when cloud density along the sun ray is zero.
- Terrain is not fully black under clouds.
- Frame time remains acceptable with low sample count defaults.

### Prompt 6 - Add Debug Controls

> Add debug controls for cloud shadows. Provide at least an enable/disable toggle and a cloud-shadow strength or minimum visibility parameter. Add an optional debug overlay mode or temporary visualization that displays cloud shadow transmittance on the terrain, so correctness can be verified without guessing from final lighting.

Suggested settings:

- `enableCloudShadows`
- `cloudShadowSampleCount`
- `cloudShadowStrength`
- `cloudShadowMinVisibility`

### Prompt 7 - Optional Cloud Shadow Texture

Use this only after the direct sampling version works:

> Replace per-fragment terrain cloud-shadow marching with a low-resolution world-space cloud shadow texture. Add a compute shader that writes cloud sun transmittance over the loaded terrain neighborhood. Center coverage on the same world center as the shadow map. Sample this texture in `terrain.frag` by world XZ. Preserve the direct sampling implementation only as a debug/reference path if useful.

Acceptance criteria:

- Terrain cloud shadows look similar to direct sampling.
- Performance is better or more stable.
- Shadow texture updates with wind and sun changes.

---

## 9. Testing Plan

### Build Tests

Run the normal CMake build after shader edits. Confirm all new GLSL includes and SPIR-V targets compile.

### Visual Tests

Use runtime controls:

- Arrow keys change sun elevation and azimuth.
- F7 saves sun position.
- F8 reloads sun position.

Test scenes:

1. High sun, broken clouds: cloud shadows should sit near clouds' ground projection.
2. Low sun, broken clouds: cloud shadows should stretch along the opposite direction of sunlight.
3. No clouds or low density: terrain should look like baseline sun+shadow rendering.
4. Thick clouds: direct sun should dim, but ambient/fog should preserve visibility.
5. Mountain terrain: terrain self-shadows and cloud shadows should multiply without double-black artifacts.
6. Region boundaries: no obvious shadow seams across loaded regions.

### Debug Tests

Add or use overlays to verify:

- `FrameData.lightDir`
- `atmosphereParams.sun_direction`
- `lightMVP` coverage
- cloud shadow transmittance
- shadow map projection bounds

### Regression Risks

- Descriptor binding mismatch between C++ and GLSL.
- Updating `cloudParams` after terrain pass, causing terrain cloud shadows to lag clouds by one frame.
- Mixing engine Y-up vectors with Bruneton Z-up vectors.
- Sampling heightmaps differently in terrain and shadow passes.
- Making clouds visually correct but physically unrelated to terrain shadows.
- Excessive raymarch cost in `terrain.frag`.

---

## 10. Recommended Final Architecture

The best final design is:

1. A single sun direction helper in engine Y-up space.
2. Atmosphere receives the swizzled Bruneton Z-up sun direction.
3. Terrain, shadow map, foliage lighting, aircraft lighting, and cloud-shadow projection use the same engine-space sun.
4. Terrain self-shadow remains a normal depth shadow map.
5. Cloud shadows are represented as sunlight transmittance, not geometry shadows.
6. First implementation samples cloud density in `terrain.frag`.
7. Optimized implementation precomputes a world-space cloud shadow texture.

This keeps the system aligned with the current renderer instead of bolting on a separate lighting model.

---

## 11. Definition of Done

The feature is complete when:

- The visible sun, terrain light direction, terrain self-shadows, cloud lighting, and cloud shadows all agree.
- Cloud shadows move with cloud wind.
- Sun azimuth/elevation changes alter both terrain shadows and cloud-shadow projection.
- Terrain under clouds is dimmed but not crushed to black.
- Existing terrain, sky, clouds, foliage, aircraft, and ImGui passes still render in the expected order.
- Shader compilation and the normal build succeed.
- The implementation includes at least one debug path for verifying cloud-shadow transmittance.
