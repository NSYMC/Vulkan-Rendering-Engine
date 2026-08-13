# Vulkan Terrain Engine

A Windows-first Vulkan 1.3 research renderer that combines procedural terrain, hydrology, physically based atmosphere and clouds, aircraft simulation, and an interactive scene editor in a single C++20 application.

The project has grown beyond a basic Vulkan sandbox. It now generates a finite world from a reproducible seed, derives rivers and water flow, renders atmosphere and volumetric clouds, simulates rigid bodies with Jolt Physics, and provides both aerodynamic visualization and an autonomous airport-to-airport aircraft demonstration.

## Highlights

- Procedural finite terrain with configurable seeds, erosion-oriented data, biome coloring, foliage, ocean rendering, and multiple terrain debug views
- CPU-side master hydrology for flow accumulation, rivers, lakes, terrain queries, and vegetation placement
- Vulkan 1.3 rendering with dynamic rendering, GPU-generated terrain textures, shadow mapping, and GPU timing diagnostics
- Bruneton/Hillaire-style atmospheric scattering using transmittance, multi-scattering, and sky-view lookup tables
- Volumetric clouds with procedurally generated 3D noise, temporal accumulation, compositing, and terrain cloud shadows
- Jolt Physics integration for terrain collision and dynamic rigid bodies
- Aircraft aerodynamics, control surfaces, panel-force visualization, streamlines, and airflow particle visualization
- Autonomous aircraft and a cinematic two-airport mission workflow
- FBX loading through `ufbx`, PBR material support, built-in primitive assets, and an ImGui/ImGuizmo scene editor
- JSON-based runtime configuration through `terrain_settings.json`

## Requirements

- Windows 10 or 11, 64-bit
- A Vulkan 1.3-capable GPU and current graphics driver
- [Vulkan SDK](https://vulkan.lunarg.com/) with `glslc`
- CMake 3.21 or newer
- A C++20 compiler, such as Visual Studio 2022 or a recent MinGW-w64 toolchain
- Git, because CMake fetches GLM, GLFW, and Jolt Physics during configuration

## Build

From a Developer PowerShell or another shell where the compiler and Vulkan SDK are available:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

CMake compiles the GLSL shaders to SPIR-V and copies both the generated shaders and `terrain_settings.json` beside the executable.

For a Visual Studio multi-configuration build, run:

```powershell
.\build\Release\TerrainEngine.exe
```

For a single-configuration generator such as MinGW Makefiles or Ninja, the executable is normally located at:

```powershell
.\build\TerrainEngine.exe
```

The included `run.bat` is a convenience script for the original local MinGW setup. If your compiler or Vulkan SDK is installed elsewhere, use the portable CMake commands above or adjust the paths in that script.

## Controls

| Input | Action |
| --- | --- |
| Left mouse drag | Rotate the free camera or aircraft-follow camera |
| Mouse wheel | Zoom |
| `W` / `A` / `S` / `D` | Move the free camera |
| `Q` / `E` | Move down / up |
| Left `Shift` | Increase camera movement speed |
| `P` | Spawn a physics sphere in front of the camera |
| `F1` | Disable the terrain debug overlay |
| `F2` | Show the height heatmap |
| `F3` | Show the slope map |
| `F4` | Show the land/water mask |
| `F5` | Run terrain analysis and write the generated report |
| `F6` | Restart with a new random world seed |
| Arrow keys | Change sun elevation and azimuth |
| `F7` / `F8` | Save / reload the runtime sun position |
| `F9` | Toggle terrain cloud shadows |
| `F10` | Cycle the cloud-shadow debug view |
| `Esc` | Exit |

Most rendering, terrain, physics, atmosphere, cloud, vegetation, aircraft, and editor options are also available from the in-application ImGui panel.

## Configuration

`terrain_settings.json` is the central runtime configuration file. It contains settings for:

- world seed, dimensions, height scale, sea level, and terrain noise;
- hydrology, erosion, rivers, lakes, and biome thresholds;
- camera movement and input sensitivity;
- sunlight, atmosphere, volumetric clouds, and shadow quality;
- vegetation density and placement;
- physics and aircraft behavior;
- aerodynamic visualization and autonomous mission controls.

Generated diagnostics such as `terrain_analysis.json`, `last_seed.txt`, runtime logs, build folders, binaries, and screenshots are intentionally excluded from version control.

## Optional aircraft asset

The public repository does not bundle the large third-party aircraft archive or 4K texture set from the local development workspace. The engine remains usable without them and falls back to built-in debug geometry.

To use a properly licensed aircraft model, place it at:

```text
models/uploads_files_6285295_cessna172lowpoly.fbx
```

Additional `.fbx` files placed directly under `models/` are discovered by the scene editor. See `models/README.md` for texture naming details.

## Project layout

```text
.
|-- CMakeLists.txt                 Build configuration and dependencies
|-- terrain_settings.json         Runtime world and renderer configuration
|-- src/                          Engine, simulation, editor, and vendored source
|-- shaders/                      Graphics and compute shaders
|-- docs/                         Focused implementation reports
|-- models/                       Optional user-supplied FBX assets
|-- dem_analyzer.py               Terrain/DEM analysis utility
|-- tune_terrain.py               Terrain parameter tuning utility
`-- run.bat                       Local Windows build-and-run helper
```

## Technical reports

- `docs/AERODYNAMIC_SIMULATION_AND_AIRFLOW_VISUALIZATION_REPORT.md`
- `docs/AIRCRAFT_PHYSICS_ENGINE_REPORT.md`
- `docs/PROCEDURAL_TEXTURE_GENERATION_REPORT.md`
- `docs/SUNLIGHT_CLOUD_TERRAIN_SHADOWS_REPORT.md`
- `docs/TEXTURE_QUALITY_AND_WAVY_SEA_SHADER_REPORT.md`
- `Cloudscapes_Integration_Report.md`

## Third-party components

GLM, GLFW, and Jolt Physics are fetched by CMake. Dear ImGui, ImGuizmo, FastNoiseLite, `stb_image`, and `ufbx` are kept in the source tree for the integration used by the engine. Their respective upstream license terms continue to apply.

## Status

This is an active experimental renderer rather than a production engine. GPU support, driver behavior, and performance vary by system; validation layers and the included diagnostic views are the best starting point when investigating rendering issues.
