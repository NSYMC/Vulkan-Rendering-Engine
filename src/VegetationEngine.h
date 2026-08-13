#pragma once
// ══════════════════════════════════════════════════════════════
//  VegetationEngine.h — SimpleHydrology vegetation system
//
//  Adapted from weigert/SimpleHydrology source/vegetation.h
//
//  Implements Plant particles with root density that interact
//  with the erosion system:
//    - Plants spawn based on slope, discharge, and height
//    - Plants grow over time toward max size
//    - Roots stabilize soil: reduce deposition rate in erosion
//    - Plants die from flooding (high discharge), altitude, or age
//    - Vegetation spreads via random dispersal
//
//  Key difference from SimpleHydrology:
//    Works on the fixed-resolution master grid (R×R) instead of
//    a quadtree, compatible with the existing chunk-based pipeline.
// ══════════════════════════════════════════════════════════════

#include <glm/glm.hpp>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace VegetationEngine {

// ────────────────────────────────────────────────────────────────
//  Forward declarations
// ────────────────────────────────────────────────────────────────

class MasterWorld;

// ────────────────────────────────────────────────────────────────
//  MasterWorld — interface for vegetation to query terrain
//
//  Provides height, normal, and discharge queries that the
//  Plant system needs. Also maintains the rootDensity map
//  that feeds back into erosion.
// ══════════════════════════════════════════════════════════════

class MasterWorld {
public:
    std::vector<float> height;         // R×R heightmap
    std::vector<float> discharge;      // accumulated flow
    std::vector<float> momentumX;      // flow momentum x
    std::vector<float> momentumY;      // flow momentum y
    std::vector<float> rootDensity;    // root stabilization per cell

    uint32_t R;
    float    worldSize;
    float    cellSize;

    MasterWorld() : R(0), worldSize(0), cellSize(0) {}

    void init(uint32_t res, float ws,
              const std::vector<float>& hm,
              const std::vector<float>& disc,
              const std::vector<float>& mx,
              const std::vector<float>& my)
    {
        R = res;
        worldSize = ws;
        cellSize = ws / (float)R;
        height = hm;
        discharge = disc;
        momentumX = mx;
        momentumY = my;
        rootDensity.assign(R * R, 0.0f);
    }

    // Direct cell access
    float& at(int x, int y) { return height[y * R + x]; }
    const float& at(int x, int y) const { return height[y * R + x]; }

    float& rootAt(int x, int y) { return rootDensity[y * R + x]; }
    const float& rootAt(int x, int y) const { return rootDensity[y * R + x]; }

    float discAt(int x, int y) const { return discharge[y * R + x]; }

    // Height at world position (bilinear)
    float sampleHeight(glm::vec2 worldPos) const {
        float u = worldPos.x / worldSize;
        float v = worldPos.y / worldSize;
        u = glm::clamp(u, 0.f, 1.f - 1.f / R);
        v = glm::clamp(v, 0.f, 1.f - 1.f / R);
        float fx = u * (R - 1);
        float fy = v * (R - 1);
        int x0 = (int)fx, y0 = (int)fy;
        int x1 = std::min(x0 + 1, (int)R - 1);
        int y1 = std::min(y0 + 1, (int)R - 1);
        float tx = fx - (float)x0, ty = fy - (float)y0;
        return (height[y0 * R + x0] * (1.f - tx) + height[y0 * R + x1] * tx) * (1.f - ty)
             + (height[y1 * R + x0] * (1.f - tx) + height[y1 * R + x1] * tx) * ty;
    }

    // Normal at integer cell
    glm::vec3 normal(int x, int y) const {
        float hl = height[y * R + std::max(0, x - 1)];
        float hr = height[y * R + std::min((int)R - 1, x + 1)];
        float hd = height[std::max(0, y - 1) * R + x];
        float hu = height[std::min((int)R - 1, y + 1) * R + x];
        glm::vec3 n(-(hr - hl) / (2.0f * cellSize),
                     1.0f,
                    -(hu - hd) / (2.0f * cellSize));
        return glm::normalize(n);
    }

    // Normal at world position (bilinear of cell normals)
    glm::vec3 normalAt(glm::vec2 worldPos) const {
        float fx = worldPos.x / cellSize;
        float fy = worldPos.y / cellSize;
        int x0 = (int)glm::clamp(fx, 1.f, (float)R - 2.f);
        int y0 = (int)glm::clamp(fy, 1.f, (float)R - 2.f);
        int x1 = std::min(x0 + 1, (int)R - 1);
        int y1 = std::min(y0 + 1, (int)R - 1);
        float tx = fx - (float)x0, ty = fy - (float)y0;
        return glm::normalize(
            normal(x0, y0) * (1.f - tx) * (1.f - ty) +
            normal(x1, y0) * tx * (1.f - ty) +
            normal(x0, y1) * (1.f - tx) * ty +
            normal(x1, y1) * tx * ty);
    }

    // Discharge at world position (used by Plant::spawn/die)
    float sampleDischarge(glm::vec2 worldPos) const {
        float fx = worldPos.x / cellSize;
        float fy = worldPos.y / cellSize;
        int x = (int)glm::clamp(fx, 0.f, (float)R - 1.f);
        int y = (int)glm::clamp(fy, 0.f, (float)R - 1.f);
        // Apply erf smoothing matching SimpleHydrology node::discharge()
        return std::erf(0.4f * discharge[y * R + x]);
    }

    float heightNormalized(glm::vec2 worldPos) const {
        return sampleHeight(worldPos);
    }

    bool isLand(int x, int y) const {
        return height[y * R + x] > -5.0f;  // above deep ocean
    }

    bool isOcean(glm::vec2 worldPos) const {
        return sampleHeight(worldPos) < -5.0f;
    }
};

// ══════════════════════════════════════════════════════════════
//  Plant — individual plant particle
//
//  Mirrors SimpleHydrology's Plant struct exactly:
//    pos          world-space position
//    size         current plant size (0 → maxSize)
//    maxSize      maximum achievable size
//    growRate     logistic growth rate
//    maxSteep     maximum terrain slope (dot product threshold)
//    maxDischarge maximum water flow for survival
//    maxTreeHeight maximum elevation for tree growth
// ══════════════════════════════════════════════════════════════

struct Plant {
    glm::vec2 pos;
    float     size = 0.0f;

    static float maxSize;
    static float growRate;
    static float maxSteep;
    static float maxDischarge;
    static float maxTreeHeight;
    static float rootRadius;   // root influence radius in cells

    // Methods (defined below after MasterWorld is complete)
    bool grow();
    bool die(MasterWorld& world);
    static bool canSpawn(glm::vec2 pos, MasterWorld& world);
    void root(MasterWorld& world, float factor);
};

// Default parameters (same as SimpleHydrology defaults)
inline float Plant::maxSize        = 1.5f;
inline float Plant::growRate       = 0.05f;
inline float Plant::maxSteep       = 0.8f;   // dot(normal, up) min (cos of ~37°)
inline float Plant::maxDischarge   = 0.3f;
inline float Plant::maxTreeHeight  = 0.8f;   // normalized height
inline float Plant::rootRadius     = 2.0f;

// ────────────────────────────────────────────────────────────────
//  Plant::grow() — logistic growth toward maxSize
//
//  SimpleHydrology original:
//    void Plant::grow(){
//      size += growRate*(maxSize-size);
//    }
// ────────────────────────────────────────────────────────────────

inline bool Plant::grow() {
    size += growRate * (maxSize - size);
    return size >= maxSize * 0.99f;  // returns true when fully grown
}

// ────────────────────────────────────────────────────────────────
//  Plant::die() — death conditions
//
//  SimpleHydrology original:
//    bool Plant::die(){
//      if(World::map.discharge(pos) >= Plant::maxDischarge) return true;
//      if(World::map.height(pos) >= Plant::maxTreeHeight) return true;
//      if(rand()%1000 == 0) return true;
//      return false;
//    }
// ────────────────────────────────────────────────────────────────

inline bool Plant::die(MasterWorld& world) {
    // Drowned by high discharge (flooding)
    if (world.sampleDischarge(pos) >= maxDischarge) return true;

    // Above tree line
    if (world.heightNormalized(pos) >= maxTreeHeight) return true;

    // Underwater (sea level change)
    if (world.isOcean(pos)) return true;

    // Random death (aging)
    // Using a simple LCG for portability
    static uint32_t lcg = 1;
    lcg = lcg * 1103515245u + 12345u;
    if ((lcg % 1000) == 0) return true;

    return false;
}

// ────────────────────────────────────────────────────────────────
//  Plant::canSpawn() — spawn eligibility check
//
//  SimpleHydrology original:
//    bool Plant::spawn(vec2 pos){
//      if(World::map.discharge(pos) >= Plant::maxDischarge) return false;
//      glm::vec3 n = World::map.normal(pos);
//      if(n.y < Plant::maxSteep) return false;
//      if(World::map.height(pos) >= Plant::maxTreeHeight) return false;
//      return true;
//    }
// ────────────────────────────────────────────────────────────────

inline bool Plant::canSpawn(glm::vec2 pos, MasterWorld& world) {
    // No spawning in water
    if (world.isOcean(pos)) return false;

    // Too much discharge (flooded area)
    if (world.sampleDischarge(pos) >= maxDischarge) return false;

    // Too steep
    glm::vec3 n = world.normalAt(pos);
    if (n.y < maxSteep) return false;

    // Above tree line
    if (world.heightNormalized(pos) >= maxTreeHeight) return false;

    return true;
}

// ────────────────────────────────────────────────────────────────
//  Plant::root() — apply root density to surrounding cells
//
//  SimpleHydrology original applies a cross-shaped pattern:
//    center: 1.0f, cardinals: 0.6f, diagonals: 0.4f
//
//  This stabilizes soil and reduces erosion deposition.
// ────────────────────────────────────────────────────────────────

inline void Plant::root(MasterWorld& world, float factor) {
    // Convert world position to cell coordinates
    float fx = pos.x / world.cellSize;
    float fy = pos.y / world.cellSize;
    int cx = (int)glm::clamp(fx, 2.f, (float)world.R - 3.f);
    int cy = (int)glm::clamp(fy, 2.f, (float)world.R - 3.f);

    // Apply cross-shaped root density pattern matching SimpleHydrology
    auto addRoot = [&](int dx, int dy, float weight) {
        int nx = cx + dx;
        int ny = cy + dy;
        if (nx < 0 || nx >= (int)world.R || ny < 0 || ny >= (int)world.R) return;
        world.rootAt(nx, ny) = glm::clamp(
            world.rootAt(nx, ny) + factor * weight,
            0.0f, 1.0f);
    };

    addRoot( 0,  0, 1.0f);   // center
    addRoot( 1,  0, 0.6f);   // cardinal neighbors
    addRoot(-1,  0, 0.6f);
    addRoot( 0,  1, 0.6f);
    addRoot( 0, -1, 0.6f);
    addRoot(-1, -1, 0.4f);   // diagonal neighbors
    addRoot( 1, -1, 0.4f);
    addRoot(-1,  1, 0.4f);
    addRoot( 1,  1, 0.4f);
}

// ══════════════════════════════════════════════════════════════
//  Vegetation — container of all plants
//
//  SimpleHydrology original:
//    struct Vegetation {
//      static std::vector<Plant> plants;
//      static bool grow();
//    };
//
//  Vegetation::grow() does two things per call:
//    1. Attempts to spawn one new plant at random position
//    2. Iterates over all plants: grow, kill expired, spread
// ══════════════════════════════════════════════════════════════

struct Vegetation {
    std::vector<Plant> plants;

    uint32_t seed;            // RNG seed for reproducibility
    size_t   maxPlants = 500;  // population cap

    // RNG state
    std::mt19937 rng;

    Vegetation(uint32_t s = 42) : seed(s), rng(s) {}

    float rand01() {
        return std::uniform_real_distribution<float>(0.f, 1.f)(rng);
    }

    int randInt(int max) {
        return std::uniform_int_distribution<int>(0, max - 1)(rng);
    }

    // ── spawnOne(): try to spawn a plant at random position ──
    bool spawnOne(MasterWorld& world, uint32_t R) {
        if (plants.size() >= maxPlants) return false;

        int x = randInt((int)R);
        int y = randInt((int)R);
        glm::vec2 pos((float)x * world.cellSize + world.cellSize * 0.5f,
                      (float)y * world.cellSize + world.cellSize * 0.5f);

        if (Plant::canSpawn(pos, world)) {
            plants.emplace_back();
            plants.back().pos = pos;
            plants.back().size = 0.0f;
            plants.back().root(world, 1.0f);
            return true;
        }
        return false;
    }

    // ── grow(): full vegetation update cycle ─────────────────
    bool grow(MasterWorld& world, uint32_t R) {
        // ── 1. Spawn attempt ──────────────────────────────────
        spawnOne(world, R);

        // ── 2. Iterate over plants ────────────────────────────
        for (int i = 0; i < (int)plants.size(); i++) {
            // Grow
            plants[i].grow();

            // Check death
            if (plants[i].die(world)) {
                plants[i].root(world, -1.0f);  // remove root influence
                plants.erase(plants.begin() + i);
                i--;
                continue;
            }

            // Random chance to spread (1 in 20, matching SimpleHydrology)
            if (randInt(20) != 0)
                continue;

            // Find new position in random direction (max 4 cells away)
            glm::vec2 npos = plants[i].pos
                           + glm::vec2((float)(randInt(9) - 4),
                                       (float)(randInt(9) - 4)) * world.cellSize;

            // Check bounds
            if (npos.x < 0 || npos.x >= world.worldSize ||
                npos.y < 0 || npos.y >= world.worldSize)
                continue;

            // Check discharge at new position
            if (world.sampleDischarge(npos) >= Plant::maxDischarge)
                continue;

            // Root density check (don't place too close to existing roots)
            float fx = npos.x / world.cellSize;
            float fy = npos.y / world.cellSize;
            int cx = (int)glm::clamp(fx, 0.f, (float)world.R - 1.f);
            int cy = (int)glm::clamp(fy, 0.f, (float)world.R - 1.f);
            if (world.rootAt(cx, cy) >= 0.5f)
                continue;

            // Slope check
            glm::vec3 n = world.normalAt(npos);
            if (n.y <= Plant::maxSteep)
                continue;

            // Population cap
            if (plants.size() >= maxPlants)
                continue;

            // Spawn new plant at spread position
            plants.emplace_back();
            plants.back().pos = npos;
            plants.back().size = 0.0f;
            plants.back().root(world, 1.0f);
        }

        return !plants.empty();
    }

    // ── clearRootDensity(): remove all root influence ────────
    void clearRootDensity(MasterWorld& world) {
        for (int i = 0; i < (int)plants.size(); i++) {
            plants[i].root(world, -1.0f);
        }
        std::fill(world.rootDensity.begin(), world.rootDensity.end(), 0.0f);
    }

    // ── applyAllRoots(): recalculate root density from plants ─
    void applyAllRoots(MasterWorld& world) {
        std::fill(world.rootDensity.begin(), world.rootDensity.end(), 0.0f);
        for (auto& p : plants) {
            p.root(world, 1.0f);
        }
    }

    size_t count() const { return plants.size(); }
    const std::vector<Plant>& getPlants() const { return plants; }
};

} // namespace VegetationEngine
