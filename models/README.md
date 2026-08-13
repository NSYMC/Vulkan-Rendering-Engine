# Optional models

This directory is intentionally source-only in the public repository. Add only assets that you have permission to redistribute.

The aircraft loader first looks for:

```text
models/uploads_files_6285295_cessna172lowpoly.fbx
```

If that file is missing, TerrainEngine uses its built-in debug aircraft geometry. Other `.fbx` files placed directly in this directory are discovered automatically and added to the scene editor's asset registry.

For the local Cessna material workflow, the loader recognizes the following optional PBR texture names next to the FBX or in a nearby texture directory:

```text
CSSNA172_SKIN02_BaseColor.png
CSSNA172_SKIN02_Metallic.png
CSSNA172_SKIN02_Roughness.png
```

Large archives and 4K texture sets are ignored by Git to keep the repository lightweight and to avoid republishing third-party assets without explicit licensing information.
