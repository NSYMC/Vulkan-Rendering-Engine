[README.md](https://github.com/user-attachments/files/26203270/README.md)
# VulkanApp

A small Vulkan scene / rendering sandbox: glTF loading, directional shadow mapping, basic camera controls, and a few in-app panels.

I put this together while learning Vulkan; the layout is meant to look like a real project, not a single-file tutorial.

## What’s in here

- Vulkan render path (validation layers in Debug builds)
- Shadow map for a directional light
- glTF / GLB via Assimp
- GLFW for window + input, GLM for math
- Simple panels for scene / models / lights

## Requirements

- Windows 10 or 11 (x64)
- Visual Studio 2022 with the C++ desktop workload
- [Vulkan SDK](https://vulkan.lunarg.com/)
- GLFW
- Assimp

The include and library paths in `VulkanApp/VulkanApp.vcxproj` are set up for a specific folder layout on my machine. If yours differs, update **Additional Include Directories** and **Additional Library Directories** in the project settings.

## Build and run

1. Clone the repo.
2. Open `VulkanApp.sln` in Visual Studio.
3. Fix dependency paths if needed.
4. Compile the shaders (needs `glslc` on your PATH, e.g. from the Vulkan SDK):

```bat
cd VulkanApp\Shaders
compile_shaders.bat
```

5. Build **x64** (Debug or Release) and run.

If you skip the shader step, shader module creation will likely fail at startup.

## Controls

- Camera: mouse + keyboard (see `Camera` / `Input`).
- With the presentation demo flag `sunumGolgeDemo` enabled in `Kaynak.cpp`, shadow debug shortcuts:
  - **F5** — normal shading
  - **F6** — shadow UV / debug sampling
  - **F7** — hard shadow look
  - **F8** — depth contrast

Toggle `sunumGolgeDemo` in `Kaynak.cpp` to switch between the minimal floor + demo model setup and the full Earth / Moon scene.

## Layout

- `VulkanApp/` — C++ sources
- `VulkanApp/Shaders/` — GLSL + `compile_shaders.bat`
- `VulkanApp/EarthObject`, `MoonObject`, `Models`, `Scene`, `UIObjects` — bundled assets

## Third-party assets

`license.txt` files next to models are kept on purpose. If you redistribute those assets, respect the original terms.

## Contributing

Issues and PRs are welcome. A short note on *what* changed and *why* helps. Small, focused changes are easier to review than large refactors.
