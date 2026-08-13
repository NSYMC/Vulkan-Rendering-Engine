// ══════════════════════════════════════════════════════════════
//  Model — FBX + texture loading implementation
// ══════════════════════════════════════════════════════════════
#include "Model.h"

#include "third_party/ufbx.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_TGA
#include "third_party/stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace {

glm::vec3 toGlm(ufbx_vec3 v) { return glm::vec3((float)v.x, (float)v.y, (float)v.z); }
glm::vec2 toGlm(ufbx_vec2 v) { return glm::vec2((float)v.x, (float)v.y); }

// Try to resolve a texture referenced by the FBX to an actual file on disk.
// FBX paths are frequently stale (absolute paths from the artist's machine),
// so we fall back to searching by basename in a few likely directories.
std::string resolveTexturePath(const ufbx_texture* tex,
                               const fs::path& fbxDir,
                               const std::vector<fs::path>& searchDirs) {
    if (!tex) return {};

    auto tryPath = [](const fs::path& p) -> std::string {
        std::error_code ec;
        if (!p.empty() && fs::exists(p, ec) && fs::is_regular_file(p, ec))
            return p.string();
        return {};
    };

    // 1) ufbx-resolved filename, then relative/absolute as given.
    auto sv = [](const ufbx_string& s) { return std::string(s.data ? s.data : "", s.length); };
    for (const std::string& cand : { sv(tex->filename), sv(tex->relative_filename), sv(tex->absolute_filename) }) {
        if (cand.empty()) continue;
        std::string hit = tryPath(cand);
        if (!hit.empty()) return hit;
        // Relative to the FBX directory.
        hit = tryPath(fbxDir / fs::path(cand));
        if (!hit.empty()) return hit;
    }

    // 2) Search by basename in the candidate directories.
    std::string base;
    {
        std::string rel = sv(tex->relative_filename);
        std::string abs = sv(tex->absolute_filename);
        std::string fn  = sv(tex->filename);
        std::string pick = !fn.empty() ? fn : (!rel.empty() ? rel : abs);
        if (!pick.empty()) base = fs::path(pick).filename().string();
    }
    if (!base.empty()) {
        for (const fs::path& dir : searchDirs) {
            std::string hit = tryPath(dir / base);
            if (!hit.empty()) return hit;
        }
    }
    return {};
}

// Given a found base-color path like ".../CSSNA172_SKIN0_BaseColor.png", derive
// the sibling map (Metallic/Roughness/Normal) by swapping the suffix token.
std::string deriveSiblingMap(const std::string& baseColorPath, const char* token) {
    if (baseColorPath.empty()) return {};
    fs::path p(baseColorPath);
    std::string stem = p.stem().string();   // e.g. CSSNA172_SKIN0_BaseColor
    const std::string needle = "BaseColor";
    auto pos = stem.find(needle);
    if (pos == std::string::npos) return {};
    stem.replace(pos, needle.size(), token);
    fs::path cand = p.parent_path() / (stem + p.extension().string());
    std::error_code ec;
    if (fs::exists(cand, ec)) return cand.string();
    return {};
}

} // namespace

namespace ModelLoader {

ModelData loadFBX(const std::string& path) {
    ModelData out;

    ufbx_load_opts opts = {};
    opts.target_axes = ufbx_axes_right_handed_y_up;  // +Y up, right-handed
    opts.target_unit_meters = 1.0f;                  // normalise to metres
    opts.generate_missing_normals = true;
    opts.space_conversion = UFBX_SPACE_CONVERSION_TRANSFORM_ROOT;

    ufbx_error error;
    ufbx_scene* scene = ufbx_load_file(path.c_str(), &opts, &error);
    if (!scene) {
        std::cout << "  [model] FBX load FAILED: " << error.description.data << "\n";
        return out;
    }

    const fs::path fbxDir = fs::path(path).parent_path();
    std::vector<fs::path> searchDirs;
    searchDirs.push_back(fbxDir);
    {   // include subdirectories of the FBX folder (e.g. the "4K ..." texture set)
        std::error_code ec;
        for (auto it = fs::directory_iterator(fbxDir, ec);
             !ec && it != fs::directory_iterator(); ++it) {
            if (it->is_directory(ec)) searchDirs.push_back(it->path());
        }
    }

    // ── Materials: build a global list, mapping ufbx_material* -> index. ──
    std::vector<const ufbx_material*> matPtrs;
    auto materialIndex = [&](const ufbx_material* m) -> int {
        if (!m) return -1;
        for (size_t i = 0; i < matPtrs.size(); ++i)
            if (matPtrs[i] == m) return (int)i;
        matPtrs.push_back(m);
        ModelMaterial mm;
        mm.name = std::string(m->name.data ? m->name.data : "", m->name.length);

        const ufbx_material_pbr_maps& pbr = m->pbr;
        if (pbr.base_color.has_value) {
            ufbx_vec4 c = pbr.base_color.value_vec4;
            mm.baseColorFactor = glm::vec4((float)c.x, (float)c.y, (float)c.z, (float)c.w);
        }
        if (pbr.metalness.has_value)  mm.metallicFactor  = (float)pbr.metalness.value_real;
        if (pbr.roughness.has_value)  mm.roughnessFactor = (float)pbr.roughness.value_real;

        mm.baseColorPath = resolveTexturePath(pbr.base_color.texture, fbxDir, searchDirs);
        mm.metallicPath  = resolveTexturePath(pbr.metalness.texture,  fbxDir, searchDirs);
        mm.roughnessPath = resolveTexturePath(pbr.roughness.texture,  fbxDir, searchDirs);
        mm.normalPath    = resolveTexturePath(pbr.normal_map.texture, fbxDir, searchDirs);

        // If the FBX texture refs were stale, derive siblings from base color.
        if (!mm.baseColorPath.empty()) {
            if (mm.metallicPath.empty())  mm.metallicPath  = deriveSiblingMap(mm.baseColorPath, "Metallic");
            if (mm.roughnessPath.empty()) mm.roughnessPath = deriveSiblingMap(mm.baseColorPath, "Roughness");
            if (mm.normalPath.empty())    mm.normalPath    = deriveSiblingMap(mm.baseColorPath, "Normal");
        }
        out.materials.push_back(std::move(mm));
        return (int)matPtrs.size() - 1;
    };

    // ── Geometry: iterate nodes, bake world transform, triangulate. ──
    glm::vec3 bbMin( 1e30f), bbMax(-1e30f);

    for (size_t ni = 0; ni < scene->nodes.count; ++ni) {
        const ufbx_node* node = scene->nodes.data[ni];
        if (!node || !node->mesh) continue;
        const ufbx_mesh* mesh = node->mesh;
        const ufbx_matrix geoToWorld = node->geometry_to_world;

        std::vector<uint32_t> triBuf(mesh->max_face_triangles * 3 + 6);

        for (size_t pi = 0; pi < mesh->material_parts.count; ++pi) {
            const ufbx_mesh_part& part = mesh->material_parts.data[pi];
            if (part.num_triangles == 0) continue;

            const ufbx_material* mat =
                (pi < mesh->materials.count) ? mesh->materials.data[pi] : nullptr;
            const int matIdx = materialIndex(mat);

            ModelSubmesh sub;
            sub.indexOffset = (uint32_t)out.indices.size();
            sub.materialIndex = matIdx;

            for (size_t fi = 0; fi < part.face_indices.count; ++fi) {
                const ufbx_face face = mesh->faces.data[part.face_indices.data[fi]];
                const uint32_t numTris =
                    ufbx_triangulate_face(triBuf.data(), triBuf.size(), mesh, face);

                for (uint32_t t = 0; t < numTris * 3; ++t) {
                    const uint32_t ix = triBuf[t];
                    ModelVertex v;
                    ufbx_vec3 p = ufbx_get_vertex_vec3(&mesh->vertex_position, ix);
                    p = ufbx_transform_position(&geoToWorld, p);
                    v.pos = toGlm(p);

                    if (mesh->vertex_normal.exists) {
                        ufbx_vec3 n = ufbx_get_vertex_vec3(&mesh->vertex_normal, ix);
                        n = ufbx_transform_direction(&geoToWorld, n);
                        v.normal = glm::normalize(toGlm(n) + glm::vec3(1e-8f));
                    }
                    if (mesh->vertex_uv.exists) {
                        glm::vec2 uv = toGlm(ufbx_get_vertex_vec2(&mesh->vertex_uv, ix));
                        v.uv = glm::vec2(uv.x, 1.0f - uv.y);  // FBX bottom-left -> Vulkan top-left
                    }

                    bbMin = glm::min(bbMin, v.pos);
                    bbMax = glm::max(bbMax, v.pos);

                    out.indices.push_back((uint32_t)out.vertices.size());
                    out.vertices.push_back(v);
                }
            }

            sub.indexCount = (uint32_t)out.indices.size() - sub.indexOffset;
            if (sub.indexCount > 0) out.submeshes.push_back(sub);
        }
    }

    ufbx_free_scene(scene);

    if (out.vertices.empty()) {
        std::cout << "  [model] FBX had no triangles\n";
        return out;
    }

    // ── Canonicalize orientation from the bounding box. ──────────────────
    //   An aircraft is flattest vertically (smallest extent => up axis) and
    //   its wingspan is wider than its length, so of the two horizontal axes
    //   the SMALLER extent is the fuselage (forward). We rotate the mesh so
    //   forward => +X, up => +Y, right => +Z (engine flight convention).
    //   NOTE: nose vs tail (±X) can't be told from a box; if it ends up
    //   backwards, apply a 180° model yaw offset in settings.
    {
        const glm::vec3 ext = bbMax - bbMin;
        int up = 0;
        if (ext.y <= ext.x && ext.y <= ext.z) up = 1;
        else if (ext.z <= ext.x && ext.z <= ext.y) up = 2;
        else up = 0;
        const int h0 = (up + 1) % 3, h1 = (up + 2) % 3;
        const float e0 = (&ext.x)[h0], e1 = (&ext.x)[h1];
        const int fwd = (e0 <= e1) ? h0 : h1;   // smaller horizontal = fuselage

        glm::vec3 fwdSrc(0.0f), upSrc(0.0f);
        (&fwdSrc.x)[fwd] = 1.0f;
        (&upSrc.x)[up]   = 1.0f;
        const glm::vec3 rightSrc = glm::cross(fwdSrc, upSrc);  // right-handed

        const bool alreadyCanonical = (fwd == 0 && up == 1);
        if (!alreadyCanonical) {
            auto remap = [&](const glm::vec3& v) {
                return glm::vec3(glm::dot(v, fwdSrc), glm::dot(v, upSrc), glm::dot(v, rightSrc));
            };
            glm::vec3 nbMin( 1e30f), nbMax(-1e30f);
            for (ModelVertex& v : out.vertices) {
                v.pos    = remap(v.pos);
                v.normal = glm::normalize(remap(v.normal) + glm::vec3(1e-8f));
                nbMin = glm::min(nbMin, v.pos);
                nbMax = glm::max(nbMax, v.pos);
            }
            bbMin = nbMin; bbMax = nbMax;
            std::cout << "  [model] re-oriented: fwd-axis=" << fwd << " up-axis=" << up << " -> nose +X\n";
        }
    }

    // ── Per-triangle tangents (UV-derivative method), averaged per vertex. ──
    {
        std::vector<glm::vec3> tan(out.vertices.size(), glm::vec3(0.0f));
        for (size_t i = 0; i + 2 < out.indices.size(); i += 3) {
            const uint32_t i0 = out.indices[i], i1 = out.indices[i + 1], i2 = out.indices[i + 2];
            const glm::vec3& p0 = out.vertices[i0].pos;
            const glm::vec3 e1 = out.vertices[i1].pos - p0;
            const glm::vec3 e2 = out.vertices[i2].pos - p0;
            const glm::vec2 d1 = out.vertices[i1].uv - out.vertices[i0].uv;
            const glm::vec2 d2 = out.vertices[i2].uv - out.vertices[i0].uv;
            const float denom = d1.x * d2.y - d2.x * d1.y;
            const float r = (std::abs(denom) > 1e-8f) ? (1.0f / denom) : 0.0f;
            const glm::vec3 t = (e1 * d2.y - e2 * d1.y) * r;
            tan[i0] += t; tan[i1] += t; tan[i2] += t;
        }
        for (size_t i = 0; i < out.vertices.size(); ++i) {
            const glm::vec3 n = out.vertices[i].normal;
            glm::vec3 t = tan[i] - n * glm::dot(n, tan[i]);  // Gram-Schmidt
            if (glm::dot(t, t) < 1e-12f) t = glm::vec3(1.0f, 0.0f, 0.0f);
            t = glm::normalize(t);
            out.vertices[i].tangent = glm::vec4(t, 1.0f);
        }
    }

    // ── Livery: force the SKIN02 PBR set on every material (per request).
    //    SKIN02 ships BaseColor/Metallic/Roughness (no normal map -> flat). ──
    {
        auto findByBasename = [&](const std::string& name) -> std::string {
            std::error_code ec;
            for (const fs::path& dir : searchDirs) {
                fs::path cand = dir / name;
                if (fs::exists(cand, ec) && fs::is_regular_file(cand, ec))
                    return cand.string();
            }
            return {};
        };
        const std::string skinBase  = findByBasename("CSSNA172_SKIN02_BaseColor.png");
        if (!skinBase.empty()) {
            const std::string skinMetal = findByBasename("CSSNA172_SKIN02_Metallic.png");
            const std::string skinRough = findByBasename("CSSNA172_SKIN02_Roughness.png");
            if (out.materials.empty()) {
                ModelMaterial mm; mm.name = "SKIN02";
                out.materials.push_back(mm);
                for (ModelSubmesh& s : out.submeshes)
                    if (s.materialIndex < 0) s.materialIndex = 0;
            }
            for (ModelMaterial& m : out.materials) {
                m.baseColorFactor = glm::vec4(1.0f);
                m.baseColorPath = skinBase;
                m.metallicPath  = skinMetal;
                m.roughnessPath = skinRough;
                m.normalPath.clear();           // SKIN02 has no normal map
            }
            std::cout << "  [model] livery forced -> SKIN02 PBR set\n";
        } else {
            std::cout << "  [model] WARNING: SKIN02 textures not found; keeping FBX materials\n";
        }
    }

    out.boundsMin = bbMin;
    out.boundsMax = bbMax;
    out.valid = true;

    std::cout << "  [model] FBX loaded: verts=" << out.vertices.size()
              << " tris=" << (out.indices.size() / 3)
              << " submeshes=" << out.submeshes.size()
              << " materials=" << out.materials.size()
              << " bounds=(" << bbMin.x << "," << bbMin.y << "," << bbMin.z
              << ")..(" << bbMax.x << "," << bbMax.y << "," << bbMax.z << ")\n";
    for (const ModelMaterial& m : out.materials) {
        std::cout << "    [mat] '" << m.name << "'"
                  << " base="  << (m.baseColorPath.empty() ? "-" : fs::path(m.baseColorPath).filename().string())
                  << " metal=" << (m.metallicPath.empty()  ? "-" : fs::path(m.metallicPath).filename().string())
                  << " rough=" << (m.roughnessPath.empty() ? "-" : fs::path(m.roughnessPath).filename().string())
                  << " norm="  << (m.normalPath.empty()    ? "-" : fs::path(m.normalPath).filename().string())
                  << "\n";
    }
    return out;
}

ImageRGBA8 loadImageRGBA8(const std::string& path, int maxDim) {
    ImageRGBA8 img;
    int w = 0, h = 0, ch = 0;
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &ch, 4);
    if (!data) {
        std::cout << "  [model] texture load FAILED: " << path
                  << " (" << stbi_failure_reason() << ")\n";
        return img;
    }

    // Optional box-filter downsample to keep VRAM / upload time reasonable.
    int dstW = w, dstH = h;
    if (maxDim > 0 && (w > maxDim || h > maxDim)) {
        const float s = (float)maxDim / (float)std::max(w, h);
        dstW = std::max(1, (int)std::lround(w * s));
        dstH = std::max(1, (int)std::lround(h * s));
    }

    img.width = dstW;
    img.height = dstH;
    img.pixels.resize((size_t)dstW * dstH * 4);

    if (dstW == w && dstH == h) {
        std::memcpy(img.pixels.data(), data, img.pixels.size());
    } else {
        // Simple averaged box downsample.
        for (int y = 0; y < dstH; ++y) {
            const int sy0 = (int)((int64_t)y * h / dstH);
            const int sy1 = std::max(sy0 + 1, (int)((int64_t)(y + 1) * h / dstH));
            for (int x = 0; x < dstW; ++x) {
                const int sx0 = (int)((int64_t)x * w / dstW);
                const int sx1 = std::max(sx0 + 1, (int)((int64_t)(x + 1) * w / dstW));
                uint32_t acc[4] = {0, 0, 0, 0};
                uint32_t count = 0;
                for (int sy = sy0; sy < sy1; ++sy)
                    for (int sx = sx0; sx < sx1; ++sx) {
                        const stbi_uc* px = data + ((size_t)sy * w + sx) * 4;
                        acc[0] += px[0]; acc[1] += px[1]; acc[2] += px[2]; acc[3] += px[3];
                        ++count;
                    }
                uint8_t* dst = img.pixels.data() + ((size_t)y * dstW + x) * 4;
                for (int c = 0; c < 4; ++c) dst[c] = (uint8_t)(acc[c] / std::max(1u, count));
            }
        }
    }

    stbi_image_free(data);
    return img;
}

} // namespace ModelLoader
