# Procedural Texture Generation System — Design Report

**Project:** TerrainEngine (`c:\VulkanProject`)  
**Purpose:** Professional design for a procedural texture system that is consistent, high-quality, and sensitive to biome, height, slope, humidity, and hydrology — aligned with the existing Vulkan compute + regional terrain architecture.  
**Scope:** Analysis and architecture only (no code changes).

---

## 1. Executive Summary

TerrainEngine already has the skeleton of a procedural texture pipeline: `texgen.comp` generates six albedo layers into a `sampler2DArray`, and `terrain.frag` selects and blends them using height, slope, humidity, and river masks via triplanar mapping. The system is **not yet professional-grade** because:

- Textures are **albedo-only**, generated **once at startup**, and **decoupled from world regen** (F6 rebuilds height but not the texture array).
- **Humidity-driven biome variation is implemented in the shader but never fed** — the active CPU upload path writes `G = 0` on every heightmap pixel.
- There is **no formal biome model** — only six height bands with hard-coded thresholds.
- **No normal/roughness/AO maps**, no mipmaps, no cache, and no artist override path.

A professional system should treat procedural textures as a **first-class material pipeline** tied to the same seed, world-space coordinates, and geomorphology that drive height and hydrology. The natural evolution is not a greenfield rewrite but a **layered upgrade** of `texgen.comp`, heightmap data channels, and `terrain.frag`, optionally extending toward lightweight terrain PBR consistent with the aircraft `model.frag` path.

---

## 2. Current State (Baseline Audit)

### 2.1 What exists today

| Component | Location | Role |
|-----------|----------|------|
| Albedo texgen | `shaders/texgen.comp` | 6 layers: deep/shallow water, sand, grass, rock, snow — Perlin fBm, seed via push constant |
| Texture upload | `VulkanEngine::createTextureArray()` | Init-only compute dispatch; `texRes` default 512 |
| Material selection | `shaders/terrain.frag` | Height bands + `biomeNoise()` + triplanar blend + slope rock overlay |
| Height data | `VulkanEngine::generateTerrainRegion()` | OpenSimplex2 fBm + ridge + hydrology carve |
| Humidity (unused) | `shaders/heightmap.comp` → `humidityField()` | Exists on legacy GPU path; not on active CPU path |
| Geomorph analysis | `src/DebugAnalyzer.h` | 10-class `GeomorphClass` — analysis only, not rendering |
| Reference | `atmosphere_bac_src/shaders/terrain.frag` | Older diffuse/normal sampler pattern |

Current `texgen.comp` layers are simple fBm color ramps:

```glsl
vec3 deepWater(vec2 uv) {
    float n = fbm(uv * 4.0 + 3.7, 4);
    return mix(vec3(0.03, 0.10, 0.25), vec3(0.05, 0.15, 0.38), n * 0.5 + 0.5);
}
// ... sandTex, grassTex, rockTex, snowTex — similar fBm + tint
```

Fragment shader height bands (with noise perturbation):

```glsl
if (effectiveH < 0.0)        contLayer = 0.0;
else if (effectiveH < 2.0)   contLayer = 1.0 + effectiveH / 2.0;
else if (effectiveH < 5.0)   contLayer = 2.0 + (effectiveH - 2.0) / 3.0;
else if (effectiveH < 28.0)  contLayer = 3.0 + (effectiveH - 5.0) / 23.0;
else if (effectiveH < 44.0)  contLayer = 4.0 + (effectiveH - 28.0) / 16.0;
else                          contLayer = 5.0;
```

### 2.2 Critical gaps

1. **Seed/lifecycle mismatch** — `regenerateTerrain()` does not call `createTextureArray()`; textures can disagree with regenerated height.
2. **Humidity channel dead** — shader logic for dry grass, desert push, snow-line shift is inert.
3. **No cross-map coherence** — albedo cracks in rock do not appear in normal/roughness.
4. **World-space vs UV-space mismatch** — texgen uses normalized `[0,1]²` UV; terrain samples via triplanar world coords at `TEX_SCALE = 0.02`. This works but makes tile frequency independent of `regionWorldSize` (4096 m).
5. **No mip chain** — `maxLod = 0` causes shimmer at distance and grazing angles.
6. **Lighting model ceiling** — Blinn-Phong on albedo-only caps perceived quality regardless of texture detail.

---

## 3. Design Goals & Quality Bar

### 3.1 Functional requirements

| Requirement | Definition |
|-------------|------------|
| **Consistent** | Same `worldSeed` + settings → identical textures across sessions, regions, and regen |
| **High-quality** | No obvious tiling at 50–200 m viewing distance; believable macro/meso/micro detail; stable under Bruneton lighting + cloud shadows |
| **Height-sensitive** | Snow line, rock exposure, coastal sand, and shelf tones respond to elevation and local relief |
| **Biome-sensitive** | Distinct surface families (arid, temperate, alpine, riparian, coastal) driven by climate proxies, not height alone |
| **Hydrology-aware** | Rivers, banks, and wet zones tint or swap materials coherently |
| **Performance-safe** | Fits regional streaming; no per-frame full-world regen |

### 3.2 Visual quality targets

- **Macro (100 m–4 km):** Large-scale color variation via climate/biome fields, not repeated tile seams.
- **Meso (1–100 m):** Material-specific structure (grass clumps, rock strata, sand ripples).
- **Micro (<1 m):** High-frequency normal/detail without albedo noise soup.
- **Transitions:** Soft, organic boundaries (already started with `biomeNoise()`); extend to humidity and slope.
- **Coherence:** Albedo hue, roughness, and normal direction agree (wet = darker + smoother; rock = rough + high normal variance).

---

## 4. Recommended Architecture

### 4.1 Three-tier material model

Separate **what surface it is** from **how it looks at this pixel**:

```
┌─────────────────────────────────────────────────────────────┐
│  Tier 1: Surface Classification (CPU or low-res GPU map)   │
│  Inputs: height, slope, humidity, discharge, distance-to-   │
│          water, latitude proxy, worldSeed                    │
│  Output: BiomeWeights[8] + MaterialMask (river, beach, …)   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier 2: Procedural Material Library (offline GPU bake)     │
│  Per material: Albedo, Normal, Roughness, (optional AO)     │
│  Tileable, seed-stable, authorable parameters               │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier 3: Runtime Compositing (terrain.frag)                 │
│  Triplanar sample + weight blend + slope/height modifiers   │
│  + river/airfield overlays + lighting/fog/shadows           │
└─────────────────────────────────────────────────────────────┘
```

This matches TerrainEngine’s existing **compute-then-sample** pattern (`heightmap.comp` → sample in draw) and avoids per-pixel procedural cost in the fragment shader.

### 4.2 Formal biome taxonomy (recommended)

Replace implicit height bands with **8–12 named biomes**, each mapped to one or more material presets. Align with `DebugAnalyzer::GeomorphClass` where sensible:

| Biome ID | Name | Primary drivers | Base materials |
|----------|------|-----------------|----------------|
| 0 | Deep ocean | h < −8 m | Deep water |
| 1 | Shallow sea / shelf | −8 to 0 m | Shallow water, sand shelf |
| 2 | Coastal beach | 0–3 m, low slope, near coast | Sand, wet sand |
| 3 | Arid lowland | low humidity, h 2–15 m | Dry grass, scrub, bare soil |
| 4 | Temperate grassland | mid humidity, gentle slope | Grass, soil patches |
| 5 | Forest / lush | high humidity, moderate slope | Dark grass, moss, leaf litter |
| 6 | Alpine rock | h > 28 m or high slope | Rock, scree |
| 7 | Snow / ice | h > snow line (humidity-adjusted) | Snow, ice crust |
| 8 | Riparian | river mask > 0 | Wet soil, reeds, bank mud |
| 9 | Wetland | high humidity + low slope + near water | Marsh grass, mud |

**Biome weights** should be computed on a **low-resolution climate grid** (e.g. 128×128 per 4096 m region) during `generateTerrainRegion()`, stored in a compact GPU texture (e.g. RGBA8 with 4 biome weights per texel, two textures for 8 biomes), and sampled in `terrain.frag`. Height and slope remain **continuous modifiers** on top, not the sole selector.

### 4.3 Material library structure

Each material preset is a **data record** (JSON + GPU arrays), not hard-coded GLSL switches:

```json
{
  "id": "temperate_grass",
  "albedoLayer": 3,
  "normalLayer": 3,
  "roughnessLayer": 3,
  "baseColor": [0.14, 0.42, 0.12],
  "colorVariation": 0.15,
  "macroScale": 0.004,
  "mesoScale": 0.04,
  "microScale": 0.25,
  "roughnessBase": 0.85,
  "roughnessVariation": 0.12,
  "normalStrength": 1.0,
  "heightBlendRange": [5, 28],
  "humidityBias": [0.3, 0.8],
  "slopeRockBias": 0.0
}
```

**Generation:** one compute shader family (`texgen.comp` → split or modular includes) dispatches **layer groups**:

- Array 0: Albedo (RGBA8 sRGB or linear — pick one pipeline-wide)
- Array 1: Normal (RG8 or RGBA8, tangent-space encoded)
- Array 2: Roughness (+ optional AO in A channel)

Push constants / SSBO: material index, seed offset, scales, palette vectors.

---

## 5. Procedural Generation Techniques (Per Material)

### 5.1 Shared noise foundation

Standardize on **one noise library** shared between CPU and GPU:

| Function | Use |
|----------|-----|
| **Gradient noise (Perlin/Simplex)** | Macro color variation, humidity, biome boundaries |
| **Worley / cellular** | Rock grain, sand grain, soil clumps |
| **fBm with fixed lacunarity** | Organic variation; match existing `fbm()` in `texgen.comp` and `heightmap.comp` |
| **Domain warp** | Break up repetition; already in `humidityField()` |
| **Multi-scale composite** | `macro * 0.5 + meso * 0.35 + micro * 0.15` — avoid single-scale fBm |

**Seed derivation (critical for consistency):**

```
globalSeed     = settings.activeWorldSeed()
materialSeed   = hash(globalSeed, materialId)
worldNoiseSeed = hash(globalSeed, floor(worldX/regionSize), floor(worldZ/regionSize))
```

Use the same hash on CPU (region gen) and GPU (texgen) so climate and texture tiles agree.

### 5.2 Material-specific recipes

| Material | Albedo technique | Normal | Roughness |
|----------|------------------|--------|-----------|
| **Deep water** | Low-freq color gradient + subtle caustic-like fBm | Flat | Very low (~0.05) |
| **Shallow water** | Shore-color blend mask (for shader) | Flat | Low |
| **Sand** | Worley dots + directional ripple fBm (wind proxy) | Ripple normal from height of ripple field | 0.75–0.9 |
| **Grass** | Dual-scale: clump Worley + fine fBm; hue shift by humidity uniform | Short blade noise from meso height | 0.8–0.95 |
| **Rock** | Strata (anisotropic fBm along fake bedding angle) + crack Worley | Strong derived from composite height | 0.7–0.95, cracks darker/rougher |
| **Snow** | Soft fBm + sparkle sparse high-freq | Soft bumps | 0.3–0.6 (wet vs dry) |
| **Scree** | Rock palette + high Worley | High variance | 0.9+ |
| **Mud / riparian** | Low-sat brown-green; smooth fBm | Soft | 0.2–0.4 when wet |

**Normal generation:** build a **scalar height field** `H(u,v)` per material from the same noise stack used for albedo, then:

`normal = normalize(vec3(-dH/du, -dH/dv, 1.0))` → encode to `[0,1]`.

This guarantees albedo–normal coherence (cracks dark and indented).

### 5.3 Anti-tiling strategy

At 4096 m regions with 512 height samples, visible tile repeat at `TEX_SCALE=0.02` is ~50 m — acceptable for grass, obvious for rock.

Recommended mitigations (combine at least two):

1. **World-space macro tint** — low-freq noise in `terrain.frag` modulates albedo per biome (already partially done via `biomeNoise` on height; extend to color).
2. **Stochastic triplanar blending** — per-world-cell rotation/offset hash (small angle) on triplanar weights.
3. **Texture bombing** — 3–5 weighted samples per axis with random offsets from cell hash (costly; use for hero materials only).
4. **Increase tile physical size** — tie `TEX_SCALE` to `regionWorldSize / texRes` so one tile ≈ 8–16 m, not 50 m.
5. **Generate larger tiles** — 1024² or 2048² with mips; memory cost is modest for 6–12 layers × 3 maps.

---

## 6. Height & Biome Integration

### 6.1 Wire humidity (highest-impact quick win)

Port `humidityField()` from `heightmap.comp` to CPU in `generateTerrainRegion()`:

- Input: world XZ, `activeWorldSeed()`
- Output: `[0,1]` per heightmap cell → write to **G channel** (already consumed by `terrain.vert` / `terrain.frag`)
- Boost humidity near rivers using `MasterHydrology::FlowResult` (mirror GPU logic in `heightmap.comp` lines 379–387)

This immediately activates existing shader paths for dry/lush grass, desert push, and snow-line shift.

### 6.2 Height-sensitive rules (refined)

Keep height as a **continuous influence**, not discrete bands:

| Phenomenon | Rule |
|------------|------|
| Snow line | `snowLine = baseSnow + (humidity - 0.5) * snowHumidityCoeff + latitudeProxy` |
| Tree line | Cap forest biome weight above elevation threshold |
| Rock exposure | `rockWeight = smoothstep(slopeMin, slopeMax, slope)` — already present; drive roughness too |
| Coastal sand | `beachWeight = (1 - smoothstep(0, 3, h)) * (1 - slope) * nearCoastMask` |
| Alpine | Boost rock/scree above hypsometric percentile (from region stats) |

**Normalize thresholds to settings:** derive band edges from `terrainTotalHeight`, `seaLevel`, and `seaFraction` instead of hard-coded 28 m / 44 m so JSON tuning does not break visuals.

### 6.3 Biome blending function

Replace single `contLayer` with weighted mix:

```
color = Σ weight_i * sampleMaterial(biome_i)
Σ weight_i = 1
```

Use **smooth maximum** or **softmax** over biome suitability scores to avoid hard seams:

```
score_i = suitability(height, slope, humidity, discharge, distWater, biome_i)
weight_i = exp(k * score_i) / Σ exp(k * score_j)
```

`k ≈ 4–8` gives crisp but smooth transitions. Perturb scores with `biomeNoise(worldXZ)` at 0.0008 and 0.015 scales (reuse existing function).

### 6.4 Hydrology integration

| Mask value | Visual treatment |
|------------|------------------|
| Channel (≥ 0.5) | Keep procedural water in shader; optionally swap shore material |
| Bank (0.05–0.5) | Blend riparian material; increase roughness wetness |
| High discharge | Wider riparian band via weight boost |

River rendering in `terrain.frag` is already strong (fresnel, specular); procedural textures should **underpaint** banks with mud/reeds rather than override channel water.

---

## 7. Consistency & Determinism

### 7.1 Single source of truth

| Data | Authority | Consumers |
|------|-----------|-----------|
| `worldSeed` | `TerrainSettings` | Height, humidity, texgen, vegetation |
| Region origin | `TerrainRegion::worldOrigin` | Triplanar, climate grid, physics |
| Material tiles | GPU texgen from seed | All regions (world-space sampling) |
| Biome weights | CPU per region | Optional low-res GPU texture |

### 7.2 Lifecycle hooks

| Event | Required action |
|-------|-----------------|
| App init | Generate material arrays (all maps + mips) |
| F6 regen / seed change | Regenerate material arrays **and** height |
| `texRes` change | Destroy/recreate arrays + samplers |
| Settings hot-reload | Debounced regen for texture-affecting params only |

### 7.3 Optional bake cache

For faster startup and artist review:

- Cache key: `hash(worldSeed, texRes, materialPresetVersion)`
- Format: KTX2 (supports array layers + mips) beside `terrain_fingerprint.json`
- Fallback: regen on cache miss

---

## 8. Rendering Integration

### 8.1 Descriptor layout extension

Current terrain set (binding 2 = albedo array). Proposed:

| Binding | Resource |
|---------|----------|
| 2 | `sampler2DArray albedoArray` |
| 7 | `sampler2DArray normalArray` |
| 8 | `sampler2DArray roughnessArray` |
| 9 | `sampler2D biomeWeightMap` (optional per-region) |

Keep bindings 3–6 (shadow, atmosphere, clouds) unchanged.

### 8.2 Lighting upgrade path

**Phase A (minimal):** Use roughness to modulate specular exponent in existing Blinn-Phong.

**Phase B (recommended):** GGX-style diffuse+spec with normal mapping; energy levels need not match aircraft PBR exactly but should feel coherent at altitude.

**Phase C (optional):** Shared sun/atmosphere with `model.frag`; terrain roughness typically higher than aircraft paint.

Triplanar normal mapping requires **blending tangent-space normals in world space** (or reoriented triplanar normals) — standard technique; budget extra ALU in `terrain.frag`.

### 8.3 Mipmaps

Generate full mip chains for all arrays (`vkCmdBlitImage` or compute downsample). Use **anisotropic filtering** on terrain sampler (currently likely 1x). This is one of the cheapest quality wins.

---

## 9. Authoring & Tuning Workflow

### 9.1 Data-driven presets

Add `material_presets.json` (alongside `terrain_settings.json`):

- Biome → material mapping
- Color palettes per biome
- Noise scales relative to metres (not UV)
- Snow line / desert humidity thresholds

Expose in `SettingsPanel.h`:

- `texRes`
- Material regen button
- Biome preview overlay (reuse `DebugAnalyzer` geomorph colors)
- Split-screen: procedural vs fallback albedo

### 9.2 Validation suite

Automated captures (fixed camera, fixed seed):

1. **Tile detect** — FFT or autocorrelation on flat grass plateau
2. **Seam test** — triplanar cube render of each material
3. **Transition test** — cross-section from ocean → peak
4. **Regen consistency** — hash of texture arrays before/after identical regen
5. **Performance** — texgen dispatch time vs `texRes`

Embed summaries in `terrain_analysis.json` schema v2.2 under `"procedural_textures"`.

### 9.3 Artist hybrid path

Allow **optional PNG overrides** per material layer (stb_image path already exists in `Model.cpp`):

- If `textures/overrides/grass_albedo.png` present → upload instead of compute for that layer
- Procedural normal/roughness still derived or overridden independently

This supports iteration without recompiling shaders.

---

## 10. Implementation Roadmap

### Phase 0 — Fix foundation (1–2 days)

- Regenerate texture arrays on seed/regen
- CPU `humidityField` → heightmap G channel
- Normalize height bands to `TerrainSettings`
- Add mips + anisotropic sampling
- Expose `texRes` in settings UI

**Outcome:** Existing system works as intended; humidity and regen consistency fixed.

### Phase 1 — Material quality (1 week)

- Refactor `texgen.comp` into include modules per material
- Add normal + roughness arrays with coherent generation
- Worley + strata for rock/sand; dual-scale grass
- Tie `TEX_SCALE` to physical metres
- Modulate specular by roughness in `terrain.frag`

**Outcome:** Visibly richer ground at all altitudes.

### Phase 2 — Biome system (1–2 weeks)

- Define biome enum + suitability functions on CPU
- Low-res biome weight texture per region
- Replace `contLayer` ladder with weighted biome compositing
- Align debug overlay with `GeomorphClass` colors

**Outcome:** Deserts, forests, and alpine zones distinct at same elevation.

### Phase 3 — Polish & pipeline (1 week)

- KTX2 cache
- JSON material presets
- Validation captures + schema extension
- Optional artist overrides

**Outcome:** Production-ready, tunable, reproducible pipeline.

### Phase 4 — Extended scope (optional)

- Foliage texture atlas (currently flat cones in `foliage.frag`)
- Detail normal distance blend (high-res near camera)
- Parallax or relief mapping on rock near ground

---

## 11. Performance Budget

| Operation | Cost | Frequency |
|-----------|------|-----------|
| Full texgen 512² × 6 × 3 maps | ~1–5 ms GPU | Init / regen |
| Full texgen 1024² × 12 × 3 maps | ~10–30 ms GPU | Init / regen |
| Biome weight grid 128² / region | ~0.1 ms CPU | Region load |
| Extra triplanar samples in frag | +3–6 texture fetches | Per pixel |

At 512–1024 tex resolution, **offline bake at init/regen is correct** — do not move full procedural generation to per-pixel fragment work.

Regional streaming stays efficient because materials are **world-space tileable**; only optional biome weight maps are per-region.

---

## 12. Risk Register

| Risk | Mitigation |
|------|------------|
| Triplanar normal artifacts | World-space normal blending; reduce normal strength on steep slopes |
| sRGB vs linear mismatch | Document pipeline; one `VK_FORMAT` policy for albedo |
| Descriptor set churn | Material arrays are global; only biome map is per-region |
| Visual clash with Bruneton sky | Tune roughness/specular for outdoor scale; test at 30° sun |
| Scope creep to full PBR | Phase A/B sufficient for terrain; aircraft stays separate |

---

## 13. Key File Index

| Concern | Primary paths |
|---------|---------------|
| Settings | `src/TerrainSettings.h`, `terrain_settings.json` |
| Texgen | `shaders/texgen.comp`, `VulkanEngine::createTextureArray()` |
| Terrain materials | `shaders/terrain.frag`, `shaders/terrain.vert` |
| Humidity (reference) | `shaders/heightmap.comp` (`humidityField`) |
| World gen | `src/VulkanEngine.cpp` (`generateTerrainRegion`, `regenerateTerrain`) |
| Hydrology | `src/MasterHydrology.h` |
| Geomorph analysis | `src/DebugAnalyzer.h` |
| Related docs | `docs/SUNLIGHT_CLOUD_TERRAIN_SHADOWS_REPORT.md`, `docs/AIRCRAFT_PHYSICS_ENGINE_REPORT.md` |

---

## 14. Summary Recommendation

Build a **data-driven procedural material library** on top of the existing `texgen.comp` + `terrain.frag` architecture, not a replacement renderer. The highest-leverage sequence is:

1. **Activate humidity** and **sync texgen with world regen**
2. **Add normal + roughness arrays** with shared noise fields
3. **Introduce formal biome weights** decoupled from pure height bands
4. **Mips, physical scale, and cache** for stability and polish

That path respects TerrainEngine’s compute-then-sample design, regional infinite terrain, hydrology, and Bruneton atmosphere — and closes the gap between the current six-layer fBm albedo and a professional, biome-aware ground appearance suitable for low-altitude flight simulation.
