// ══════════════════════════════════════════════════════════════
//  Model — FBX mesh + PBR material loader (ufbx + stb_image)
//
//  Loads a binary/ASCII FBX into CPU-side vertex/index buffers split
//  into per-material submeshes, and resolves each material's PBR
//  texture maps (base color / metallic / roughness / normal) to files
//  on disk. The renderer (VulkanEngine) uploads these to the GPU.
// ══════════════════════════════════════════════════════════════
#pragma once

#include <glm/glm.hpp>
#include <cstdint>
#include <string>
#include <vector>

struct ModelVertex {
    glm::vec3 pos{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec2 uv{0.0f};
    glm::vec4 tangent{1.0f, 0.0f, 0.0f, 1.0f};  // xyz + handedness in w
};

struct ModelMaterial {
    std::string name;
    std::string baseColorPath;   // empty => use factor / default texture
    std::string metallicPath;
    std::string roughnessPath;
    std::string normalPath;
    glm::vec4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
    float metallicFactor  = 1.0f;
    float roughnessFactor = 1.0f;
};

struct ModelSubmesh {
    uint32_t indexOffset = 0;
    uint32_t indexCount  = 0;
    int      materialIndex = -1;   // index into ModelData::materials (-1 = none)
};

struct ModelData {
    std::vector<ModelVertex>   vertices;
    std::vector<uint32_t>      indices;
    std::vector<ModelSubmesh>  submeshes;
    std::vector<ModelMaterial> materials;
    glm::vec3 boundsMin{0.0f};
    glm::vec3 boundsMax{0.0f};
    bool valid = false;
};

// Decoded 8-bit RGBA image (row-major, top-left origin).
struct ImageRGBA8 {
    int width  = 0;
    int height = 0;
    std::vector<uint8_t> pixels;  // width*height*4
    bool valid() const { return width > 0 && height > 0 && !pixels.empty(); }
};

namespace ModelLoader {

// Load an FBX file into a merged, triangulated, per-material model.
// Returns ModelData with valid=false on failure (reason logged to stdout).
ModelData loadFBX(const std::string& path);

// Decode an image (PNG/JPG/TGA/...) to RGBA8, optionally box-downsampled so
// neither dimension exceeds maxDim (0 = no limit). Returns invalid on failure.
ImageRGBA8 loadImageRGBA8(const std::string& path, int maxDim = 0);

} // namespace ModelLoader
