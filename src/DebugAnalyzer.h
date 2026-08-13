#pragma once
// ══════════════════════════════════════════════════════════════
//  DebugAnalyzer.h — comprehensive terrain topology & statistics
//  analyser with structured AI-ingestible output.
//
//  Designed for two consumers:
//  1. Human developers —  console summaries, debug overlays, stats
//  2. Future AI models —  JSON with semantic keys + natural language
//
//  Analysis passes (ordered by dependency):
//   1. Global statistics       min/max/mean/stddev/median/histogram
//   2. Slope computation       finite-difference gradient → rad + histogram
//   3. Peak detection          local maxima with prominence filtering
//   4. Basin detection         local minima (ocean deeps)
//   5. Continental mass CCL    connected-component labelling above sea level
//   6. Landmass shape analysis compactness, PCA aspect ratio, convexity, perimeter
//   7. Geomorph classification terrain type per cell (mountain/plateau/hill/plain/…)
//   8. Tectonic boundary class convergent/divergent/transform per boundary segment
//   9. D8 flow accumulation    flow direction + accumulation count + river extraction
//  10. Cross-section profiling elevation profile along arbitrary line segments
//  11. Ridge connectivity      flood-fill tracking along high boundary cells
//
//  Outputs:
//   • JSON report  →  terrain_analysis.json  (AI-ingestible)
//   • Natural language summary  →  console + embedded in JSON
//   • Debug overlay texture     →  RGBA8 for GPU visualization
//
//  Dependencies: glm, <vector>, <string>, <fstream>, <cmath>, <sstream>
// ══════════════════════════════════════════════════════════════

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <functional>
#include <chrono>
#include <iomanip>

namespace DebugAnalyzer {

// ────────────────────────────────────────────────────────────────
//  Constants
// ────────────────────────────────────────────────────────────────
constexpr const char* SCHEMA_VERSION = "2.1.0";
constexpr int HISTOGRAM_BINS = 30;
constexpr int SLOPE_BINS     = 12;
constexpr int HYPSO_BINS     = 20;   // hypsometric curve bins

// ────────────────────────────────────────────────────────────────
//  Enums
// ────────────────────────────────────────────────────────────────

enum class GeomorphClass : uint8_t {
    DEEP_OCEAN     = 0,  // < -8m
    OCEAN_BASIN    = 1,  // -8m to -1m
    SHALLOW_WATER  = 2,  // -1m to 0m
    COASTAL_PLAIN  = 3,  // 0m to 3m, flat
    LOWLAND        = 4,  // 0m to 6m, flat to gentle
    HILL           = 5,  // moderate elevation, moderate slope
    PLATEAU        = 6,  // high elevation, low slope
    MOUNTAIN       = 7,  // high elevation, high slope
    PEAK           = 8,  // very high, local maximum
    TRENCH         = 9   // narrow deep depression near convergent boundary
};

enum class BoundaryClass : uint8_t {
    NONE            = 0,
    CONVERGENT_CC   = 1,  // continental-continental collision
    CONVERGENT_OC   = 2,  // oceanic subducting under continental
    CONVERGENT_OO   = 3,  // oceanic-oceanic collision
    DIVERGENT_OCEAN = 4,  // mid-ocean ridge
    DIVERGENT_RIFT  = 5,  // continental rift
    TRANSFORM       = 6,
    PASSIVE_MARGIN  = 7   // continent-ocean boundary, no active motion
};

enum class DebugOverlayMode : int {
    NONE               = 0,
    HEIGHT_HEATMAP     = 1,
    SLOPE_MAP          = 2,
    PLATE_BOUNDARIES   = 3,
    LAND_WATER_MASK    = 4,
    MOUNTAIN_PEAKS     = 5,
    RIDGE_LINES        = 6,
    GEOMORPHOLOGY      = 7,
    FLOW_ACCUMULATION  = 8
};

// ────────────────────────────────────────────────────────────────
//  Data structures
// ────────────────────────────────────────────────────────────────

struct PeakInfo {
    glm::ivec2 pixel;
    glm::vec2  worldPos;
    float      height;
    float      prominence;     // height above highest saddle to a higher peak
    int        parentLandmass;
    int        parentPlate;
};

struct BasinInfo {
    glm::ivec2 pixel;
    glm::vec2  worldPos;
    float      depth;           // positive = below sea level
    float      volume;          // approximate filled volume
    int        cellCount;       // connected basin size
};

struct LandmassShape {
    float compactness;           // 4π·area/perimeter²  (circle = 1.0)
    float circularity;           // perimeter / (2√(π·area))  (circle = 1.0)
    float aspectRatio;           // major/minor PCA axis ratio
    float convexity;             // area / convex-hull-area estimate
    float elongationAngleDeg;    // principal axis orientation (0-180°)
    float perimeterCells;
    float perimeterWorld;
    glm::vec2 boundingMin;       // world-space AABB
    glm::vec2 boundingMax;
    float     coastlineLength;   // land-cells bordering water cells
};

struct ContinentalMass {
    int         id;
    glm::ivec2  centroidPx;
    glm::vec2   centroidWorld;
    float       areaCells;
    float       areaWorld;
    float       minHeight, maxHeight, meanHeight, stdDevHeight;
    int         peakCount;
    LandmassShape shape;
    std::vector<int> boundaryPlateIds;

    // Hypsometric breakdown (fraction of cells in each elevation band)
    std::vector<float> hypsometry;

    // Geomorphology breakdown
    int geomCount[10];  // indexed by GeomorphClass

    // River potential
    int   maxFlowAccumulation;
    float drainageDensity;       // river cells / area

    // Description
    std::string aiDescription;
};

struct RidgeSegment {
    std::vector<glm::ivec2> cells;
    float    meanHeight;
    float    lengthWorld;
    glm::vec2 startWorld, endWorld;
    int      plateA, plateB;
    BoundaryClass boundaryType;
};

struct TectonicBoundary {
    int      plateA, plateB;
    BoundaryClass classification;
    float    lengthWorld;
    float    meanHeight;
    float    meanConvergenceRate;  // if convergent
    int      cellCount;
};

struct PlateStatistics {
    int    id;
    float  areaCells;
    float  areaWorld;
    float  meanHeight;
    float  meanVelocity;
    float  continentalFraction;   // % crust that is continental
    int    landmassCount;         // how many landmasses touch this plate
    int    peakCount;
};

struct FlowNetwork {
    std::vector<int>   flowDirection;      // D8 encoding (0-7 cardinal, -1 = sink)
    std::vector<int>   flowAccumulation;   // number of upstream cells draining here
    std::vector<bool>  isRiver;            // accumulation > threshold
    int                maxAccumulation;
    float              riverDensity;       // river cells / total cells
    std::vector<std::vector<glm::ivec2>> riverSegments;  // connected river paths
};

struct CrossSection {
    std::string  label;
    glm::vec2    startWorld, endWorld;
    struct Sample { float distance; float height; float slope; int landmassId; };
    std::vector<Sample> samples;
    float minHeight, maxHeight;
    int   landmassCrossings;
};

struct AnalysisResult {
    // ── Metadata ───
    std::string schemaVersion = SCHEMA_VERSION;
    std::string timestamp;
    uint32_t    heightmapRes;
    float       worldSize;
    float       seaThreshold;
    int         totalCells;

    // ── Global statistics ───
    float minHeight, maxHeight;
    float meanHeight, medianHeight, stdDev;
    float landFraction, oceanFraction;

    // ── Height histogram (HISTOGRAM_BINS bins) ───
    struct HistBin { float low, high; int count; };
    std::vector<HistBin> heightHistogram;

    // ── Slope statistics ───
    float slopeMean, slopeMax;       // radians
    float slopeMeanDeg, slopeMaxDeg; // degrees
    std::vector<float> slopeHistogram;  // SLOPE_BINS bins

    // ── Topological features ───
    std::vector<PeakInfo>         peaks;
    std::vector<BasinInfo>        basins;
    std::vector<RidgeSegment>     ridges;
    std::vector<ContinentalMass>  landmasses;
    std::vector<TectonicBoundary> boundaries;
    std::vector<PlateStatistics>  plateStats;

    // ── Terrain classification ───
    std::vector<GeomorphClass>    geomMap;
    std::unordered_map<int, int>  globalGeomCounts;  // GeomorphClass → cell count

    // ── Flow network ───
    FlowNetwork flowNet;

    // ── Cross-sections ───
    std::vector<CrossSection> crossSections;

    // ── Natural language description ───
    std::string aiNaturalLanguageSummary;
};

// ────────────────────────────────────────────────────────────────
//  Internal helpers
// ────────────────────────────────────────────────────────────────

namespace {
    // 8-directional neighbours (E, NE, N, NW, W, SW, S, SE)
    const int DX8[8] = { 1,  1,  0, -1, -1, -1,  0,  1 };
    const int DY8[8] = { 0,  1,  1,  1,  0, -1, -1, -1 };

    // 4-directional neighbours (E, N, W, S) — cardinal only
    const int DX4[4] = { 1, 0, -1, 0 };
    const int DY4[4] = { 0, 1,  0, -1 };

    // D8 flow-direction encoding: index = downhill neighbour, -1 = pit
    // Order: E=0, NE=1, N=2, NW=3, W=4, SW=5, S=6, SE=7

    struct CellId { int x, y; };

    // ── Simple convex-hull area via monotone chain ─────────
    // Returns area of convex hull for a set of 2D points (world coords)
    float convexHullArea2D(const std::vector<glm::vec2>& pts) {
        if (pts.size() < 3) return 0.f;
        auto cross = [](glm::vec2 o, glm::vec2 a, glm::vec2 b) {
            return (a.x-o.x)*(b.y-o.y) - (a.y-o.y)*(b.x-o.x);
        };
        auto sorted = pts;
        std::sort(sorted.begin(), sorted.end(),
            [](glm::vec2 a, glm::vec2 b) { return a.x<b.x || (a.x==b.x && a.y<b.y); });
        std::vector<glm::vec2> hull(2*sorted.size());
        int k = 0;
        for (int pass = 0; pass < 2; pass++) {
            int start = k;
            for (auto& p : sorted) {
                while (k - start >= 2 && cross(hull[k-2], hull[k-1], p) <= 0) k--;
                hull[k++] = p;
            }
            k--;
            std::reverse(sorted.begin(), sorted.end());
        }
        hull.resize(k);
        float area = 0.f;
        for (size_t i = 0; i < hull.size(); i++) {
            auto& a = hull[i];
            auto& b = hull[(i+1)%hull.size()];
            area += a.x*b.y - b.x*a.y;
        }
        return std::abs(area) * 0.5f;
    }

    // ── Approximate perimeter of shape defined by cell set ──
    float estimatePerimeter(const std::vector<glm::ivec2>& cells, int W, int H,
                            const std::vector<bool>& isLandMask) {
        // Perimeter ≈ number of land cells with at least one water neighbor
        int perimeterCells = 0;
        for (auto& c : cells) {
            for (int n = 0; n < 4; n++) {
                int nx = c.x + DX4[n], ny = c.y + DY4[n];
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) { perimeterCells++; break; }
                if (!isLandMask[ny * W + nx]) { perimeterCells++; break; }
            }
        }
        return (float)perimeterCells;
    }
}

// ══════════════════════════════════════════════════════════════
//  D8 Flow-direction encoding helpers
// ══════════════════════════════════════════════════════════════

namespace {
    // Given a heightmap and cell index, compute the D8 steepest-descent direction.
    // Returns -1 if pit (no downhill neighbour).
    int computeD8Dir(const std::vector<float>& hm, int x, int y, int W) {
        float h = hm[y*W + x];
        float maxDrop = 0.f;
        int bestDir = -1;
        for (int d = 0; d < 8; d++) {
            int nx = x + DX8[d], ny = y + DY8[d];
            if (nx < 0 || nx >= W || ny < 0 || ny >= W) continue;
            float nh = hm[ny*W + nx];
            float drop = h - nh;
            if (drop > maxDrop) {
                maxDrop = drop;
                bestDir = d;
            }
        }
        return bestDir;
    }

    // D8 accumulation via iterative upstream counting (memoized).
    // Uses an explicit stack to avoid stack overflow on long flow chains.
    int flowAccumulate(int idx, const std::vector<int>& flowDir, int W,
                        std::vector<int>& accum, std::vector<int>& visited) {
        if (visited[idx]) return accum[idx];

        // Walk downhill, pushing cells onto an explicit path stack
        std::vector<int> path;
        path.reserve(64);
        int cur = idx;

        while (true) {
            visited[cur] = 1;
            path.push_back(cur);

            int x = cur % W, y = cur / W;
            int dir = flowDir[cur];
            if (dir < 0) { accum[cur] = 1; break; }

            int nx = x + DX8[dir], ny = y + DY8[dir];
            if (nx < 0 || nx >= W || ny < 0 || ny >= W) { accum[cur] = 1; break; }

            int nidx = ny * W + nx;
            if (visited[nidx]) {
                // Downstream already resolved — use its value
                accum[cur] = 1 + accum[nidx];
                break;
            }
            cur = nidx;
        }

        // Unwind: compute accumulation for all upstream cells on the path
        for (int i = (int)path.size() - 2; i >= 0; i--) {
            int cell = path[i];
            int x = cell % W, y = cell / W;
            int dir = flowDir[cell];
            if (dir < 0) continue;
            int nx = x + DX8[dir], ny = y + DY8[dir];
            if (nx < 0 || nx >= W || ny < 0 || ny >= W) continue;
            int nidx = ny * W + nx;
            accum[cell] = 1 + accum[nidx];
        }

        return accum[idx];
    }

    // Extract river segments by tracing cells with high accumulation
    std::vector<std::vector<glm::ivec2>> extractRiverSegments(
        const std::vector<int>& flowDir,
        const std::vector<int>& accum,
        int W, int threshold) {

        std::vector<std::vector<glm::ivec2>> segments;
        std::vector<bool> used(accum.size(), false);

        for (int y = 0; y < W; y++) {
            for (int x = 0; x < W; x++) {
                int idx = y * W + x;
                if (used[idx] || accum[idx] < threshold) continue;

                // Start of a river segment: trace downstream
                std::vector<glm::ivec2> seg;
                int cx = x, cy = y;
                while (true) {
                    int ci = cy * W + cx;
                    if (used[ci] || ci < 0 || ci >= (int)accum.size()) break;
                    used[ci] = true;
                    seg.push_back(glm::ivec2(cx, cy));
                    int dir = flowDir[ci];
                    if (dir < 0) break;
                    cx += DX8[dir]; cy += DY8[dir];
                    if (cx < 0 || cx >= W || cy < 0 || cy >= W) break;
                    int ni = cy * W + cx;
                    if (accum[ni] < threshold) break;
                }
                if (seg.size() >= 3) segments.push_back(seg);
            }
        }
        return segments;
    }
}

// ══════════════════════════════════════════════════════════════
//  MAIN ANALYSIS ENTRY POINT
// ══════════════════════════════════════════════════════════════

inline void analyze(const std::vector<float>& heightmap,
                     uint32_t R, float worldSize,
                     AnalysisResult& out,
                     const std::vector<int>* plateMap = nullptr,
                     const std::vector<uint8_t>* crustTypeMap = nullptr,
                     float seaThreshold = 0.0f)
{
    const int W = (int)R, H = (int)R;
    const int total = W * H;
    const float worldTexel = worldSize / (float)R;

    out.heightmapRes = R;
    out.worldSize = worldSize;
    out.seaThreshold = seaThreshold;
    out.totalCells = total;

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ts; ts << std::put_time(std::localtime(&t), "%Y-%m-%dT%H:%M:%S");
    out.timestamp = ts.str();

    // ──────────────────────────────────────────────────────────
    //  PASS 1: Global statistics
    // ──────────────────────────────────────────────────────────
    std::cout << "  [analyze] pass 1/11: global statistics..." << std::endl;
    {
        float sum = 0.f, sumSq = 0.f;
        out.minHeight = 1e30f; out.maxHeight = -1e30f;
        std::vector<float> sorted = heightmap;
        std::sort(sorted.begin(), sorted.end());
        out.medianHeight = sorted[total / 2];

        int landCells = 0;
        for (float h : heightmap) {
            out.minHeight = std::min(out.minHeight, h);
            out.maxHeight = std::max(out.maxHeight, h);
            sum += h; sumSq += h * h;
            if (h > seaThreshold) landCells++;
        }
        out.meanHeight = sum / (float)total;
        out.stdDev = std::sqrt(sumSq / (float)total - out.meanHeight * out.meanHeight);
        out.landFraction = (float)landCells / (float)total;
        out.oceanFraction = 1.f - out.landFraction;

        // Height histogram
        float hRange = out.maxHeight - out.minHeight;
        if (hRange < 0.001f) hRange = 1.f;
        out.heightHistogram.resize(HISTOGRAM_BINS);
        for (int i = 0; i < HISTOGRAM_BINS; i++) {
            out.heightHistogram[i].low  = out.minHeight + hRange * (float)i / HISTOGRAM_BINS;
            out.heightHistogram[i].high = out.minHeight + hRange * (float)(i+1) / HISTOGRAM_BINS;
            out.heightHistogram[i].count = 0;
        }
        for (float h : heightmap) {
            int bin = (int)((h - out.minHeight) / hRange * HISTOGRAM_BINS);
            bin = std::min(bin, HISTOGRAM_BINS - 1);
            out.heightHistogram[bin].count++;
        }
    }

    // ──────────────────────────────────────────────────────────
    //  PASS 2: Slope computation
    // ──────────────────────────────────────────────────────────
    std::cout << "  [analyze] pass 2/11: slope computation..." << std::endl;
    std::vector<float> slopes(total, 0.f);
    {
        float totalSlope = 0.f;
        out.slopeMax = 0.f;
        for (int y = 1; y < H-1; y++) {
            for (int x = 1; x < W-1; x++) {
                float hL = heightmap[y*W + (x-1)];
                float hR = heightmap[y*W + (x+1)];
                float hD = heightmap[(y-1)*W + x];
                float hU = heightmap[(y+1)*W + x];
                float dx = (hR - hL) / (2.f * worldTexel);
                float dy = (hU - hD) / (2.f * worldTexel);
                float slope = std::atan(std::sqrt(dx*dx + dy*dy));
                slopes[y*W + x] = slope;
                totalSlope += slope;
                out.slopeMax = std::max(out.slopeMax, slope);
            }
        }
        out.slopeMean = totalSlope / (float)((W-2)*(H-2));
        out.slopeMeanDeg = out.slopeMean * 180.f / glm::pi<float>();
        out.slopeMaxDeg  = out.slopeMax  * 180.f / glm::pi<float>();

        out.slopeHistogram.resize(SLOPE_BINS, 0.f);
        for (int i = 0; i < total; i++) {
            int bin = (int)(slopes[i] / (glm::half_pi<float>() + 0.001f) * SLOPE_BINS);
            bin = std::min(bin, SLOPE_BINS - 1);
            out.slopeHistogram[bin]++;
        }
    }

    // ──────────────────────────────────────────────────────────
    //  PASS 3: Peak detection
    // ──────────────────────────────────────────────────────────
    std::cout << "  [analyze] pass 3/11: peak detection..." << std::endl;
    {
        const float peakMinHeight = std::max(out.medianHeight + out.stdDev * 1.5f, seaThreshold + 1.f);
        const float peakMinProminence = out.stdDev * 0.5f;

        struct PeakCandidate { int x, y; float h; };
        std::vector<PeakCandidate> candidates;

        for (int y = 3; y < H-3; y++) {
            for (int x = 3; x < W-3; x++) {
                float h = heightmap[y*W + x];
                if (h < peakMinHeight) continue;

                // 5×5 local max check
                bool isMax = true;
                for (int dy = -2; dy <= 2 && isMax; dy++)
                    for (int dx = -2; dx <= 2 && isMax; dx++)
                        if (heightmap[(y+dy)*W + (x+dx)] > h)
                            isMax = false;
                if (!isMax) continue;

                // Prominence check (7×7 window)
                float minSurround = 1e30f;
                for (int dy = -3; dy <= 3; dy++)
                    for (int dx = -3; dx <= 3; dx++)
                        if (dx != 0 || dy != 0)
                            minSurround = std::min(minSurround, heightmap[(y+dy)*W + (x+dx)]);
                if (h - minSurround < peakMinProminence) continue;

                candidates.push_back({x, y, h});
            }
        }
        // Sort descending
        std::sort(candidates.begin(), candidates.end(),
            [](const PeakCandidate& a, const PeakCandidate& b) { return a.h > b.h; });

        for (auto& c : candidates) {
            PeakInfo pk;
            pk.pixel = glm::ivec2(c.x, c.y);
            pk.worldPos = glm::vec2((float)c.x / R * worldSize, (float)c.y / R * worldSize);
            pk.height = c.h;
            pk.prominence = c.h; // will be refined
            pk.parentLandmass = -1;
            pk.parentPlate = (plateMap ? (*plateMap)[c.y * W + c.x] : -1);

            // Refine prominence: find highest saddle to a higher peak (simplified)
            float saddle = 1e30f;
            for (int dy = -5; dy <= 5; dy++)
                for (int dx = -5; dx <= 5; dx++)
                    if (abs(dx)+abs(dy) > 3) {  // don't check immediate neighbourhood
                        int sx = c.x + dx, sy = c.y + dy;
                        if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;
                        saddle = std::min(saddle, heightmap[sy*W + sx]);
                    }
            pk.prominence = c.h - saddle;

            out.peaks.push_back(pk);
        }
    }

    // ──────────────────────────────────────────────────────────
    //  PASS 4: Basin detection (ocean deeps)
    // ──────────────────────────────────────────────────────────
    std::cout << "  [analyze] pass 4/11: basin detection..." << std::endl;
    {
        const float basinMaxHeight = seaThreshold - out.stdDev * 0.3f;
        for (int y = 1; y < H-1; y++) {
            for (int x = 1; x < W-1; x++) {
                float h = heightmap[y*W + x];
                if (h > basinMaxHeight) continue;
                bool isLocalMin = true;
                for (int n = 0; n < 8 && isLocalMin; n++)
                    if (heightmap[(y+DY8[n])*W + (x+DX8[n])] < h)
                        isLocalMin = false;
                if (!isLocalMin) continue;

                BasinInfo b;
                b.pixel = glm::ivec2(x, y);
                b.worldPos = glm::vec2((float)x / R * worldSize, (float)y / R * worldSize);
                b.depth = -(h - seaThreshold);
                b.volume = 0.f;
                b.cellCount = 1;
                out.basins.push_back(b);
            }
        }
    }

    // ──────────────────────────────────────────────────────────
    //  PASS 5: Continental mass detection (CCL)
    // ──────────────────────────────────────────────────────────
    std::cout << "  [analyze] pass 5/11: continental mass CCL..." << std::endl;
    std::vector<int> landmassLabels(total, -1);
    std::vector<bool> isLandMask(total, false);
    for (int i = 0; i < total; i++) isLandMask[i] = (heightmap[i] > seaThreshold);

    {
        int nextLabel = 0;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                if (!isLandMask[y*W + x]) continue;
                if (landmassLabels[y*W + x] >= 0) continue;

                // Flood fill
                std::queue<CellId> q;
                std::vector<glm::ivec2> massCells;
                q.push({x, y});
                landmassLabels[y*W + x] = nextLabel;
                massCells.push_back(glm::ivec2(x, y));

                ContinentalMass mass;
                mass.id = nextLabel;
                mass.minHeight = 1e30f; mass.maxHeight = -1e30f;
                mass.areaCells = 0.0f; mass.meanHeight = 0.0f;
                mass.peakCount = 0;
                memset(mass.geomCount, 0, sizeof(mass.geomCount));
                float cx = 0.f, cy = 0.f;
                std::unordered_set<int> boundaryPlates;

                while (!q.empty()) {
                    auto c = q.front(); q.pop();
                    float h = heightmap[c.y*W + c.x];
                    mass.minHeight = std::min(mass.minHeight, h);
                    mass.maxHeight = std::max(mass.maxHeight, h);
                    mass.meanHeight += h;
                    mass.areaCells++;
                    cx += (float)c.x; cy += (float)c.y;

                    if (plateMap) {
                        int myPlate = (*plateMap)[c.y*W + c.x];
                        for (int n = 0; n < 8; n++) {
                            int nx = c.x + DX8[n], ny = c.y + DY8[n];
                            if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                                int np = (*plateMap)[ny*W + nx];
                                if (np != myPlate) boundaryPlates.insert(np);
                            }
                        }
                    }

                    for (int n = 0; n < 4; n++) {
                        int nx = c.x + DX4[n], ny = c.y + DY4[n];
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H &&
                            isLandMask[ny*W + nx] &&
                            landmassLabels[ny*W + nx] < 0) {
                            landmassLabels[ny*W + nx] = nextLabel;
                            q.push({nx, ny});
                            massCells.push_back(glm::ivec2(nx, ny));
                        }
                    }
                }

                mass.meanHeight /= mass.areaCells;
                mass.centroidPx = glm::ivec2((int)(cx/mass.areaCells), (int)(cy/mass.areaCells));
                mass.centroidWorld = glm::vec2(mass.centroidPx.x / (float)R * worldSize,
                                               mass.centroidPx.y / (float)R * worldSize);
                mass.areaWorld = mass.areaCells * worldTexel * worldTexel;
                mass.boundaryPlateIds.assign(boundaryPlates.begin(), boundaryPlates.end());

                // Compute stdDev for this landmass
                float varSum = 0.f;
                for (auto& c : massCells) {
                    float h = heightmap[c.y*W + c.x];
                    varSum += (h - mass.meanHeight) * (h - mass.meanHeight);
                }
                mass.stdDevHeight = std::sqrt(varSum / mass.areaCells);

                // Landmass shape analysis
                {
                    LandmassShape& shape = mass.shape;
                    shape.perimeterCells = estimatePerimeter(massCells, W, H, isLandMask);
                    shape.perimeterWorld = shape.perimeterCells * worldTexel;

                    float area = mass.areaCells;
                    float perim = shape.perimeterCells;
                    shape.compactness = (perim > 0.f) ? (4.f * glm::pi<float>() * area) / (perim * perim) : 0.f;
                    shape.circularity = (area > 0.f) ? perim / (2.f * std::sqrt(glm::pi<float>() * area)) : 0.f;

                    // Convert cell coords to world coords for hull
                    std::vector<glm::vec2> worldPts;
                    worldPts.reserve(massCells.size());
                    for (auto& c : massCells)
                        worldPts.push_back(glm::vec2(c.x * worldTexel, c.y * worldTexel));
                    float hullArea = convexHullArea2D(worldPts);
                    shape.convexity = (hullArea > 0.f) ? (area * worldTexel * worldTexel) / hullArea : 0.f;

                    // PCA for aspect ratio and orientation
                    glm::vec2 mean(0.f);
                    for (auto& p : worldPts) mean += p;
                    mean /= (float)worldPts.size();
                    float covXX = 0.f, covXY = 0.f, covYY = 0.f;
                    for (auto& p : worldPts) {
                        float dx = p.x - mean.x, dy = p.y - mean.y;
                        covXX += dx * dx; covXY += dx * dy; covYY += dy * dy;
                    }
                    covXX /= (float)worldPts.size();
                    covXY /= (float)worldPts.size();
                    covYY /= (float)worldPts.size();
                    // Eigenvalues of 2×2 covariance matrix
                    float trace = covXX + covYY;
                    float det = covXX * covYY - covXY * covXY;
                    float disc = std::sqrt(std::max(0.f, trace*trace - 4.f*det));
                    float lambda1 = (trace + disc) * 0.5f;
                    float lambda2 = (trace - disc) * 0.5f;
                    shape.aspectRatio = (lambda2 > 1e-6f) ? std::sqrt(lambda1 / lambda2) : 1.f;
                    shape.elongationAngleDeg = std::atan2(2.f*covXY, covXX - covYY) * 0.5f * 180.f / glm::pi<float>();

                    // Bounding box
                    shape.boundingMin = glm::vec2(1e30f);
                    shape.boundingMax = glm::vec2(-1e30f);
                    for (auto& p : worldPts) {
                        shape.boundingMin = glm::min(shape.boundingMin, p);
                        shape.boundingMax = glm::max(shape.boundingMax, p);
                    }

                    // Coastline length: land cells with at least one water neighbor
                    int coastCells = 0;
                    for (auto& c : massCells) {
                        for (int n = 0; n < 4; n++) {
                            int nx = c.x + DX4[n], ny = c.y + DY4[n];
                            if (nx < 0 || nx >= W || ny < 0 || ny >= H) { coastCells++; break; }
                            if (!isLandMask[ny*W + nx]) { coastCells++; break; }
                        }
                    }
                    shape.coastlineLength = (float)coastCells * worldTexel;
                }

                // Count peaks in this landmass
                for (auto& pk : out.peaks) {
                    if (pk.parentLandmass < 0 && landmassLabels[pk.pixel.y*W + pk.pixel.x] == nextLabel) {
                        pk.parentLandmass = nextLabel;
                        mass.peakCount++;
                    }
                }

                // Hypsometry (elevation distribution within landmass)
                mass.hypsometry.resize(HYPSO_BINS, 0.f);
                if (mass.maxHeight > mass.minHeight) {
                    for (auto& c : massCells) {
                        float h = heightmap[c.y*W + c.x];
                        int bin = (int)((h - mass.minHeight) / (mass.maxHeight - mass.minHeight) * HYPSO_BINS);
                        bin = std::min(bin, HYPSO_BINS - 1);
                        mass.hypsometry[bin]++;
                    }
                    for (auto& v : mass.hypsometry) v /= mass.areaCells;
                }

                out.landmasses.push_back(mass);
                nextLabel++;
            }
        }

        // Sort landmasses by area descending
        std::sort(out.landmasses.begin(), out.landmasses.end(),
            [](const ContinentalMass& a, const ContinentalMass& b) { return a.areaCells > b.areaCells; });
    }

    // ──────────────────────────────────────────────────────────
    //  PASS 6: Geomorphological classification per cell
    // ──────────────────────────────────────────────────────────
    std::cout << "  [analyze] pass 6/11: geomorph classification..." << std::endl;
    out.geomMap.resize(total, GeomorphClass::DEEP_OCEAN);
    for (int i = 0; i < 10; i++) out.globalGeomCounts[i] = 0;
    {
        const float deepOceanMax    = -8.f;
        const float oceanBasinMax   = -1.f;
        const float shallowWaterMax = seaThreshold;
        const float coastalPlainMax = 3.f;
        const float lowlandMax      = 6.f;
        const float mountainMinSlope = 25.f * glm::pi<float>() / 180.f; // 25 degrees
        const float plateauMinHeight = out.meanHeight + out.stdDev;
        const float hillMinSlope     = 10.f * glm::pi<float>() / 180.f;
        const float plateauMaxSlope  = 8.f  * glm::pi<float>() / 180.f;

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float h = heightmap[y*W + x];
                float s = slopes[y*W + x];
                GeomorphClass cls;

                if (h <= deepOceanMax)
                    cls = GeomorphClass::DEEP_OCEAN;
                else if (h <= oceanBasinMax)
                    cls = GeomorphClass::OCEAN_BASIN;
                else if (h <= shallowWaterMax)
                    cls = GeomorphClass::SHALLOW_WATER;
                else if (h <= coastalPlainMax && s < hillMinSlope)
                    cls = GeomorphClass::COASTAL_PLAIN;
                else if (h <= lowlandMax && s < hillMinSlope)
                    cls = GeomorphClass::LOWLAND;
                else if (h >= plateauMinHeight && s < plateauMaxSlope)
                    cls = GeomorphClass::PLATEAU;
                else if (h >= plateauMinHeight && s >= mountainMinSlope)
                    cls = GeomorphClass::MOUNTAIN;
                else if (s >= hillMinSlope)
                    cls = GeomorphClass::HILL;
                else
                    cls = GeomorphClass::LOWLAND;

                out.geomMap[y*W + x] = cls;
                out.globalGeomCounts[(int)cls]++;

                // Update landmass geom counts
                int lmId = landmassLabels[y*W + x];
                if (lmId >= 0 && lmId < (int)out.landmasses.size())
                    out.landmasses[lmId].geomCount[(int)cls]++;
            }
        }

        // Trench detection: narrow deep cells near convergent boundaries
        if (plateMap) {
            for (int y = 2; y < H-2; y++) {
                for (int x = 2; x < W-2; x++) {
                    // Check for narrow depression near boundary
                    float h = heightmap[y*W + x];
                    if (h > oceanBasinMax) continue;
                    int myPlate = (*plateMap)[y*W + x];
                    bool nearBoundary = false;
                    for (int n = 0; n < 8; n++)
                        if ((*plateMap)[(y+DY8[n])*W + (x+DX8[n])] != myPlate) {
                            nearBoundary = true; break;
                        }
                    if (!nearBoundary) continue;
                    // Check if depth is significantly lower than neighbours
                    float avgNeighbor = 0.f;
                    int count = 0;
                    for (int n = 0; n < 8; n++) {
                        avgNeighbor += heightmap[(y+DY8[n])*W + (x+DX8[n])];
                        count++;
                    }
                    avgNeighbor /= count;
                    if (h < avgNeighbor - 2.f) {
                        out.geomMap[y*W + x] = GeomorphClass::TRENCH;
                        out.globalGeomCounts[(int)GeomorphClass::TRENCH]++;
                    }
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────
    //  PASS 7: Tectonic boundary classification
    // ──────────────────────────────────────────────────────────
    std::cout << "  [analyze] pass 7/11: tectonic boundaries..." << std::endl;
    if (plateMap && crustTypeMap) {
        // Build per-plate statistics first
        {
            int N = 0;
            for (int i = 0; i < total; i++) N = std::max(N, (*plateMap)[i] + 1);
            std::vector<float> plateCellCount(N, 0.f);
            std::vector<float> plateHeightSum(N, 0.f);
            std::vector<float> plateContCells(N, 0.f);

            for (int i = 0; i < total; i++) {
                int p = (*plateMap)[i];
                if (p < 0) continue;
                plateCellCount[p]++;
                plateHeightSum[p] += heightmap[i];
                if (crustTypeMap && (*crustTypeMap)[i] == 0) // continental
                    plateContCells[p]++;
            }
            out.plateStats.resize(N);
            for (int p = 0; p < N; p++) {
                out.plateStats[p].id = p;
                out.plateStats[p].areaCells = plateCellCount[p];
                out.plateStats[p].areaWorld = plateCellCount[p] * worldTexel * worldTexel;
                out.plateStats[p].meanHeight = plateCellCount[p] > 0 ?
                    plateHeightSum[p] / plateCellCount[p] : 0.f;
                out.plateStats[p].continentalFraction = plateCellCount[p] > 0 ?
                    plateContCells[p] / plateCellCount[p] : 0.f;
                out.plateStats[p].meanVelocity = 0.f; // not available without plate velocity data
                out.plateStats[p].landmassCount = 0;
                out.plateStats[p].peakCount = 0;
            }

            // Count landmasses per plate
            for (auto& lm : out.landmasses) {
                for (int pid : lm.boundaryPlateIds)
                    if (pid >= 0 && pid < N)
                        out.plateStats[pid].landmassCount++;
            }

            // Count peaks per plate
            for (auto& pk : out.peaks) {
                if (pk.parentPlate >= 0 && pk.parentPlate < N)
                    out.plateStats[pk.parentPlate].peakCount++;
            }
        }

        // Classify boundaries between plates
        std::unordered_map<uint64_t, BoundaryClass> boundaryTypes;
        std::unordered_map<uint64_t, int> boundaryCellCounts;
        std::unordered_map<uint64_t, float> boundaryHeightSums;

        auto pairKey = [](int a, int b) -> uint64_t {
            if (a > b) std::swap(a, b);
            return (uint64_t)a | ((uint64_t)b << 32);
        };

        for (int y = 1; y < H-1; y++) {
            for (int x = 1; x < W-1; x++) {
                int myPlate = (*plateMap)[y*W + x];
                if (myPlate < 0) continue;

                for (int n = 0; n < 8; n++) {
                    int nx = x + DX8[n], ny = y + DY8[n];
                    if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                    int nbPlate = (*plateMap)[ny*W + nx];
                    if (nbPlate < 0 || nbPlate == myPlate) continue;

                    uint64_t key = pairKey(myPlate, nbPlate);

                    // Heuristic classification based on crust types and heights
                    auto& ct = boundaryTypes[key];
                    if (ct == BoundaryClass::NONE) {
                        uint8_t cA = (*crustTypeMap)[y*W + x];
                        uint8_t cB = (*crustTypeMap)[ny*W + nx];
                        float hA = heightmap[y*W + x];
                        float hB = heightmap[ny*W + nx];

                        if (cA == 0 && cB == 0) { // both continental
                            if (hA > seaThreshold + 4.f || hB > seaThreshold + 4.f)
                                ct = BoundaryClass::CONVERGENT_CC;
                            else if (hA < seaThreshold - 2.f && hB < seaThreshold - 2.f)
                                ct = BoundaryClass::DIVERGENT_RIFT;
                            else
                                ct = BoundaryClass::TRANSFORM;
                        }
                        else if ((cA == 0 && cB == 1) || (cA == 1 && cB == 0)) {
                            // Continental-oceanic: convergent if mountains on cont side
                            int contIdx = (cA == 0) ? (y*W+x) : (ny*W+nx);
                            if (heightmap[contIdx] > seaThreshold + 3.f)
                                ct = BoundaryClass::CONVERGENT_OC;
                            else
                                ct = BoundaryClass::PASSIVE_MARGIN;
                        }
                        else { // both oceanic
                            if (hA > seaThreshold - 3.f || hB > seaThreshold - 3.f)
                                ct = BoundaryClass::DIVERGENT_OCEAN;
                            else if (hA < seaThreshold - 4.f || hB < seaThreshold - 4.f)
                                ct = BoundaryClass::CONVERGENT_OO;
                            else
                                ct = BoundaryClass::TRANSFORM;
                        }
                    }
                    boundaryCellCounts[key]++;
                    boundaryHeightSums[key] += heightmap[y*W + x];
                }
            }
        }

        for (auto& [key, cls] : boundaryTypes) {
            if (cls == BoundaryClass::NONE) continue;
            int a = (int)(key & 0xFFFFFFFF);
            int b = (int)(key >> 32);
            TectonicBoundary tb;
            tb.plateA = a; tb.plateB = b;
            tb.classification = cls;
            tb.cellCount = boundaryCellCounts[key];
            tb.lengthWorld = (float)tb.cellCount * worldTexel;
            tb.meanHeight = tb.cellCount > 0 ? boundaryHeightSums[key] / tb.cellCount : 0.f;
            tb.meanConvergenceRate = 0.f;
            out.boundaries.push_back(tb);
        }
    }

    // ──────────────────────────────────────────────────────────
    //  PASS 8: Ridge detection
    // ──────────────────────────────────────────────────────────
    std::cout << "  [analyze] pass 8/11: ridge detection..." << std::endl;
    if (plateMap && !out.landmasses.empty()) {
        const float ridgeMinHeight = std::max(out.medianHeight + out.stdDev, seaThreshold + 2.f);
        std::vector<bool> visited(total, false);

        for (int y = 1; y < H-1; y++) {
            for (int x = 1; x < W-1; x++) {
                int idx = y*W + x;
                if (visited[idx]) continue;
                float h = heightmap[idx];
                if (h < ridgeMinHeight) continue;

                int myPlate = (*plateMap)[idx];
                bool nearBoundary = false;
                for (int n = 0; n < 8; n++)
                    if ((*plateMap)[(y+DY8[n])*W + (x+DX8[n])] != myPlate) {
                        nearBoundary = true; break;
                    }
                if (!nearBoundary) continue;

                // Flood-fill
                std::queue<CellId> q;
                q.push({x, y});
                visited[idx] = true;

                RidgeSegment ridge;
                ridge.plateA = myPlate;
                ridge.plateB = -1;
                ridge.boundaryType = BoundaryClass::NONE;

                while (!q.empty()) {
                    auto c = q.front(); q.pop();
                    ridge.cells.push_back(glm::ivec2(c.x, c.y));
                    ridge.meanHeight += heightmap[c.y*W + c.x];

                    if (ridge.plateB < 0) {
                        for (int n = 0; n < 8; n++) {
                            int np = (*plateMap)[(c.y+DY8[n])*W + (c.x+DX8[n])];
                            if (np != myPlate) { ridge.plateB = np; break; }
                        }
                    }

                    for (int n = 0; n < 4; n++) {
                        int nx = c.x + DX4[n], ny = c.y + DY4[n];
                        if (nx >= 1 && nx < W-1 && ny >= 1 && ny < H-1 &&
                            !visited[ny*W + nx] &&
                            heightmap[ny*W + nx] >= ridgeMinHeight) {
                            visited[ny*W + nx] = true;
                            q.push({nx, ny});
                        }
                    }
                }

                if (ridge.cells.size() >= 8) {
                    ridge.meanHeight /= (float)ridge.cells.size();
                    ridge.lengthWorld = (float)ridge.cells.size() * worldTexel;

                    int fx = ridge.cells.front().x, fy = ridge.cells.front().y;
                    int bx = ridge.cells.back().x, by = ridge.cells.back().y;
                    ridge.startWorld = glm::vec2(fx * worldTexel, fy * worldTexel);
                    ridge.endWorld   = glm::vec2(bx * worldTexel, by * worldTexel);

                    out.ridges.push_back(ridge);
                }
            }
        }
        std::sort(out.ridges.begin(), out.ridges.end(),
            [](const RidgeSegment& a, const RidgeSegment& b) { return a.cells.size() > b.cells.size(); });
    }

    // ──────────────────────────────────────────────────────────
    //  PASS 9: D8 flow accumulation & river extraction
    // ──────────────────────────────────────────────────────────
    std::cout << "  [analyze] pass 9/11: D8 flow accumulation..." << std::endl;
    {
        out.flowNet.flowDirection.resize(total, -1);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                out.flowNet.flowDirection[y*W + x] = computeD8Dir(heightmap, x, y, W);

        out.flowNet.flowAccumulation.resize(total, 0);
        std::vector<int> visited(total, 0);
        out.flowNet.maxAccumulation = 0;
        for (int i = 0; i < total; i++) {
            int acc = flowAccumulate(i, out.flowNet.flowDirection, W,
                                      out.flowNet.flowAccumulation, visited);
            out.flowNet.maxAccumulation = std::max(out.flowNet.maxAccumulation, acc);
        }

        const int riverThreshold = std::max(50, (int)(total * 0.0002f));
        out.flowNet.isRiver.resize(total, false);
        int riverCellCount = 0;
        for (int i = 0; i < total; i++) {
            if (out.flowNet.flowAccumulation[i] >= riverThreshold) {
                out.flowNet.isRiver[i] = true;
                riverCellCount++;
            }
        }
        out.flowNet.riverDensity = (float)riverCellCount / (float)total;
        out.flowNet.riverSegments = extractRiverSegments(
            out.flowNet.flowDirection, out.flowNet.flowAccumulation, W, riverThreshold);

        // Update landmass river data
        for (auto& lm : out.landmasses) {
            lm.maxFlowAccumulation = 0;
            int riverCells = 0;
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    if (landmassLabels[y*W + x] == lm.id) {
                        int idx = y*W + x;
                        lm.maxFlowAccumulation = std::max(lm.maxFlowAccumulation,
                            out.flowNet.flowAccumulation[idx]);
                        if (out.flowNet.isRiver[idx]) riverCells++;
                    }
            lm.drainageDensity = lm.areaCells > 0 ? (float)riverCells / lm.areaCells : 0.f;
        }
    }

    // ──────────────────────────────────────────────────────────
    //  PASS 10: Cross-section profiles
    // ──────────────────────────────────────────────────────────
    std::cout << "  [analyze] pass 10/11: cross-section profiles..." << std::endl;
    {
        // Generate cross-sections across each major landmass
        for (size_t li = 0; li < std::min(out.landmasses.size(), (size_t)3); li++) {
            auto& lm = out.landmasses[li];

            // Horizontal cross-section through centroid
            {
                CrossSection cs;
                cs.label = "landmass_" + std::to_string(lm.id) + "_horizontal";
                cs.startWorld = glm::vec2(0, lm.centroidWorld.y);
                cs.endWorld   = glm::vec2(worldSize, lm.centroidWorld.y);
                cs.minHeight = 1e30f; cs.maxHeight = -1e30f;
                cs.landmassCrossings = 0;

                int steps = 256;
                for (int i = 0; i <= steps; i++) {
                    float t = (float)i / steps;
                    float wx = cs.startWorld.x + t * (cs.endWorld.x - cs.startWorld.x);
                    float wz = cs.startWorld.y;
                    int px = (int)(wx / worldSize * R);
                    int py = (int)(wz / worldSize * R);
                    px = glm::clamp(px, 0, (int)R-1);
                    py = glm::clamp(py, 0, (int)R-1);

                    CrossSection::Sample s;
                    s.distance = t * worldSize;
                    s.height = heightmap[py*R + px];
                    s.slope = slopes[py*R + px];
                    s.landmassId = landmassLabels[py*R + px];
                    cs.samples.push_back(s);
                    cs.minHeight = std::min(cs.minHeight, s.height);
                    cs.maxHeight = std::max(cs.maxHeight, s.height);
                }

                // Count landmass crossings
                int prevLm = -1;
                for (auto& s : cs.samples) {
                    if (s.landmassId >= 0 && s.landmassId != prevLm) {
                        cs.landmassCrossings++;
                        prevLm = s.landmassId;
                    } else if (s.landmassId < 0) {
                        prevLm = -1;
                    }
                }

                out.crossSections.push_back(cs);
            }

            // Vertical cross-section through centroid
            {
                CrossSection cs;
                cs.label = "landmass_" + std::to_string(lm.id) + "_vertical";
                cs.startWorld = glm::vec2(lm.centroidWorld.x, 0);
                cs.endWorld   = glm::vec2(lm.centroidWorld.x, worldSize);
                cs.minHeight = 1e30f; cs.maxHeight = -1e30f;
                cs.landmassCrossings = 0;

                int steps = 256;
                for (int i = 0; i <= steps; i++) {
                    float t = (float)i / steps;
                    float wx = cs.startWorld.x;
                    float wz = cs.startWorld.y + t * (cs.endWorld.y - cs.startWorld.y);
                    int px = (int)(wx / worldSize * R);
                    int py = (int)(wz / worldSize * R);
                    px = glm::clamp(px, 0, (int)R-1);
                    py = glm::clamp(py, 0, (int)R-1);

                    CrossSection::Sample s;
                    s.distance = t * worldSize;
                    s.height = heightmap[py*R + px];
                    s.slope = slopes[py*R + px];
                    s.landmassId = landmassLabels[py*R + px];
                    cs.samples.push_back(s);
                    cs.minHeight = std::min(cs.minHeight, s.height);
                    cs.maxHeight = std::max(cs.maxHeight, s.height);
                }

                int prevLm = -1;
                for (auto& s : cs.samples) {
                    if (s.landmassId >= 0 && s.landmassId != prevLm) {
                        cs.landmassCrossings++;
                        prevLm = s.landmassId;
                    } else if (s.landmassId < 0) {
                        prevLm = -1;
                    }
                }

                out.crossSections.push_back(cs);
            }
        }
    }

    // ──────────────────────────────────────────────────────────
    //  PASS 11: AI natural language summary
    // ──────────────────────────────────────────────────────────
    std::cout << "  [analyze] pass 11/11: AI language summary..." << std::endl;
    {
        std::ostringstream nl;
        nl << "This " << worldSize << "x" << worldSize << " world contains ";
        int majorLandmasses = 0;
        for (auto& lm : out.landmasses)
            if (lm.areaCells > total * 0.005f) majorLandmasses++;

        if (majorLandmasses == 0) {
            nl << "no significant continental landmasses (land fraction = "
               << (out.landFraction * 100.f) << "%). ";
        } else if (majorLandmasses == 1) {
            nl << "a single major continental landmass covering approximately "
               << (out.landmasses[0].areaCells / (float)total * 100.f)
               << "% of the surface. ";
        } else {
            nl << majorLandmasses << " major continental landmasses covering approximately "
               << (out.landFraction * 100.f) << "% of the total surface. The largest continent spans "
               << (out.landmasses[0].areaCells / (float)total * 100.f) << "% of the world area. ";
        }

        nl << "Elevation ranges from " << out.minHeight << "m (deepest trench) to "
           << out.maxHeight << "m (highest peak) with mean " << out.meanHeight
           << "m and standard deviation " << out.stdDev << "m. ";

        // Describe largest landmass
        if (!out.landmasses.empty()) {
            auto& lm = out.landmasses[0];
            nl << "The primary landmass has " << lm.peakCount << " mountain peaks";

            if (lm.shape.compactness > 0.5f)
                nl << ", a compact circular shape (compactness=" << lm.shape.compactness << ")";
            else if (lm.shape.compactness > 0.25f)
                nl << ", a moderately irregular shape (compactness=" << lm.shape.compactness << ")";
            else
                nl << ", a highly irregular or elongated shape (compactness=" << lm.shape.compactness
                   << ", aspect ratio " << lm.shape.aspectRatio << ")";

            int mountainCells = lm.geomCount[(int)GeomorphClass::MOUNTAIN];
            float mountainFrac = mountainCells / lm.areaCells;
            if (mountainFrac > 0.15f)
                nl << ", with extensive mountainous terrain (" << (mountainFrac*100.f) << "% of area). ";
            else if (mountainFrac > 0.05f)
                nl << ", with moderate mountain coverage (" << (mountainFrac*100.f) << "% of area). ";
            else
                nl << ", with limited mountainous terrain. ";

            nl << "Mean elevation of this landmass is " << lm.meanHeight << "m (max " << lm.maxHeight << "m). ";

            if (lm.drainageDensity > 0.01f)
                nl << "The drainage network is well-developed (drainage density " << lm.drainageDensity
                   << ", max flow accumulation " << lm.maxFlowAccumulation << "). ";
            else
                nl << "Drainage development is limited. ";
        }

        // Boundary analysis
        if (!out.boundaries.empty()) {
            int convCount = 0, divCount = 0, transCount = 0;
            for (auto& b : out.boundaries) {
                switch (b.classification) {
                    case BoundaryClass::CONVERGENT_CC:
                    case BoundaryClass::CONVERGENT_OC:
                    case BoundaryClass::CONVERGENT_OO: convCount++; break;
                    case BoundaryClass::DIVERGENT_OCEAN:
                    case BoundaryClass::DIVERGENT_RIFT: divCount++; break;
                    case BoundaryClass::TRANSFORM: transCount++; break;
                    default: break;
                }
            }
            nl << "Tectonic analysis identified " << out.boundaries.size() << " plate boundaries: "
               << convCount << " convergent, " << divCount << " divergent, "
               << transCount << " transform/passive. ";
        }

        // Overall assessment
        if (out.ridges.size() >= 3 && out.peaks.size() >= 10) {
            nl << "The terrain exhibits geologically coherent mountain-building patterns with "
               << out.ridges.size() << " distinct ridge segments aligned with plate boundaries.";
        } else if (out.peaks.size() >= 5) {
            nl << "Peaks exist (" << out.peaks.size() << " detected) but ridge connectivity is limited, "
               << "suggesting diffuse or weak tectonic orogeny.";
        } else {
            nl << "Mountain formation appears limited; tectonic forces may need strengthening "
               << "for more dramatic topography.";
        }

        out.aiNaturalLanguageSummary = nl.str();

        // Per-landmass AI descriptions
        for (auto& lm : out.landmasses) {
            std::ostringstream ld;
            ld << "Continental mass #" << lm.id << ": covers " << lm.areaWorld << " world-units² ("
               << (lm.areaCells/total*100.f) << "% of world). ";
            ld << "Shape is ";
            if (lm.shape.compactness > 0.5f) ld << "compact/circular";
            else if (lm.shape.compactness > 0.25f) ld << "moderately irregular";
            else ld << "elongated (aspect ratio " << lm.shape.aspectRatio << ")";
            ld << " with coastline length " << lm.shape.coastlineLength << "m. ";
            ld << "Elevation: " << lm.minHeight << "m to " << lm.maxHeight << "m (mean " << lm.meanHeight << "m). ";
            ld << lm.peakCount << " major peaks. ";
            ld << "Touches " << lm.boundaryPlateIds.size() << " tectonic plates. ";
            int mtn = lm.geomCount[(int)GeomorphClass::MOUNTAIN];
            int plat = lm.geomCount[(int)GeomorphClass::PLATEAU];
            int plains = lm.geomCount[(int)GeomorphClass::LOWLAND] + lm.geomCount[(int)GeomorphClass::COASTAL_PLAIN];
            int hills = lm.geomCount[(int)GeomorphClass::HILL];
            ld << "Terrain: " << mtn << " mountain cells, " << hills << " hill cells, "
               << plat << " plateau cells, " << plains << " plain/lowland cells.";
            lm.aiDescription = ld.str();
        }
    }
    std::cout << "  [analyze] analysis complete, writing JSON report..." << std::endl;
}

// ══════════════════════════════════════════════════════════════
//  JSON EXPORT — structured for AI model ingestion
// ══════════════════════════════════════════════════════════════

inline void saveJsonReport(const AnalysisResult& result,
                           const std::string& filepath)
{
    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "[DebugAnalyzer] ERROR: failed to open " << filepath << " for writing\n";
        return;
    }

    // ── Helper lambdas ──
    auto jf  = [&f](const char* key, float val) { f << "\"" << key << "\": " << val; };
    auto jfs = [&f](const char* key, float val, bool last) {
        f << "\"" << key << "\": " << val;
        if (!last) f << ",";
        f << "\n";
    };
    auto jis = [&f](const char* key, int val, bool last) {
        f << "\"" << key << "\": " << val;
        if (!last) f << ",";
        f << "\n";
    };
    auto jss = [&f](const char* key, const std::string& val, bool last) {
        f << "\"" << key << "\": \"" << val << "\"";
        if (!last) f << ",";
        f << "\n";
    };
    auto jvs = [&f](const char* key, const std::vector<float>& vals, bool last) {
        f << "\"" << key << "\": [";
        for (size_t i = 0; i < vals.size(); i++) {
            f << vals[i];
            if (i < vals.size()-1) f << ", ";
        }
        f << "]";
        if (!last) f << ",";
        f << "\n";
    };

    // ── Header ──
    f << "{\n";
    jss("schema_version", result.schemaVersion, false);
    f << "  \"analysis_type\": \"terrain_topology_deep\",\n";
    jss("timestamp", result.timestamp, false);
    f << "  \"metadata\": {\n";
    jis("  heightmap_resolution", result.heightmapRes, false);
    jfs("  world_size", result.worldSize, false);
    jfs("  sea_threshold", result.seaThreshold, false);
    jis("  total_cells", result.totalCells, true);
    f << "  },\n";

    // ── Natural language summary ──
    jss("natural_language_summary", result.aiNaturalLanguageSummary, false);

    // ── Global stats ──
    f << "  \"global_statistics\": {\n";
    jfs("    min_height", result.minHeight, false);
    jfs("    max_height", result.maxHeight, false);
    jfs("    mean_height", result.meanHeight, false);
    jfs("    median_height", result.medianHeight, false);
    jfs("    std_dev", result.stdDev, false);
    jfs("    land_fraction", result.landFraction, false);
    jfs("    ocean_fraction", result.oceanFraction, true);
    f << "  },\n";

    // ── Height distribution ──
    f << "  \"height_distribution\": [\n";
    for (size_t i = 0; i < result.heightHistogram.size(); i++) {
        auto& bin = result.heightHistogram[i];
        f << "    {\"bin\": " << i
          << ", \"low_m\": " << bin.low
          << ", \"high_m\": " << bin.high
          << ", \"count\": " << bin.count << "}";
        if (i < result.heightHistogram.size()-1) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    // ── Slope stats ──
    f << "  \"slope_analysis\": {\n";
    jfs("    mean_radians", result.slopeMean, false);
    jfs("    max_radians", result.slopeMax, false);
    jfs("    mean_degrees", result.slopeMeanDeg, false);
    jfs("    max_degrees", result.slopeMaxDeg, false);
    jvs("    histogram", result.slopeHistogram, true);
    f << "  },\n";

    // ── Geomorphology breakdown ──
    f << "  \"geomorphology\": {\n";
    const char* geomNames[] = {
        "deep_ocean", "ocean_basin", "shallow_water", "coastal_plain",
        "lowland", "hill", "plateau", "mountain", "peak", "trench"
    };
    for (int i = 0; i < 10; i++) {
        auto it = result.globalGeomCounts.find(i);
        int count = (it != result.globalGeomCounts.end()) ? it->second : 0;
        f << "    \"" << geomNames[i] << "\": " << count;
        if (i < 9) f << ",";
        f << "\n";
    }
    f << "  },\n";

    // ── Mountain peaks (top 30) ──
    f << "  \"mountain_peaks\": [\n";
    size_t peakLimit = std::min(result.peaks.size(), (size_t)30);
    for (size_t i = 0; i < peakLimit; i++) {
        auto& pk = result.peaks[i];
        f << "    {\"rank\": " << i
          << ", \"world_x\": " << pk.worldPos.x
          << ", \"world_z\": " << pk.worldPos.y
          << ", \"height_m\": " << pk.height
          << ", \"prominence_m\": " << pk.prominence
          << ", \"landmass_id\": " << pk.parentLandmass
          << ", \"plate_id\": " << pk.parentPlate
          << ", \"pixel_x\": " << pk.pixel.x
          << ", \"pixel_y\": " << pk.pixel.y << "}";
        if (i < peakLimit-1) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    // ── Continental masses ──
    f << "  \"continental_masses\": [\n";
    for (size_t i = 0; i < result.landmasses.size(); i++) {
        auto& lm = result.landmasses[i];
        f << "    {\n";
        jis("      \"id\"", lm.id, false);
        jfs("      \"area_cells\"", lm.areaCells, false);
        jfs("      \"area_world_units2\"", lm.areaWorld, false);
        jfs("      \"centroid_world_x\"", lm.centroidWorld.x, false);
        jfs("      \"centroid_world_z\"", lm.centroidWorld.y, false);
        jfs("      \"mean_height_m\"", lm.meanHeight, false);
        jfs("      \"stddev_height_m\"", lm.stdDevHeight, false);
        jfs("      \"min_height_m\"", lm.minHeight, false);
        jfs("      \"max_height_m\"", lm.maxHeight, false);
        jis("      \"peak_count\"", lm.peakCount, false);
        f << "      \"shape\": {\n";
        jfs("        \"compactness\"", lm.shape.compactness, false);
        jfs("        \"circularity\"", lm.shape.circularity, false);
        jfs("        \"aspect_ratio\"", lm.shape.aspectRatio, false);
        jfs("        \"convexity\"", lm.shape.convexity, false);
        jfs("        \"elongation_angle_deg\"", lm.shape.elongationAngleDeg, false);
        jfs("        \"perimeter_world_m\"", lm.shape.perimeterWorld, false);
        jfs("        \"coastline_length_m\"", lm.shape.coastlineLength, true);
        f << "      },\n";
        f << "      \"geomorphology_breakdown\": {\n";
        for (int g = 0; g < 10; g++) {
            f << "        \"" << geomNames[g] << "\": " << lm.geomCount[g];
            if (g < 9) f << ",";
            f << "\n";
        }
        f << "      },\n";
        jfs("      \"drainage_density\"", lm.drainageDensity, false);
        jis("      \"max_flow_accumulation\"", lm.maxFlowAccumulation, false);
        f << "      \"boundary_plate_ids\": [";
        for (size_t j = 0; j < lm.boundaryPlateIds.size(); j++) {
            f << lm.boundaryPlateIds[j];
            if (j < lm.boundaryPlateIds.size()-1) f << ", ";
        }
        f << "],\n";
        f << "      \"hypsometry\": [";
        for (size_t hi = 0; hi < lm.hypsometry.size(); hi++) {
            f << lm.hypsometry[hi];
            if (hi < lm.hypsometry.size()-1) f << ", ";
        }
        f << "],\n";
        jss("      \"ai_description\"", lm.aiDescription, true);
        f << "    }";
        if (i < result.landmasses.size()-1) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    // ── Tectonic boundaries ──
    f << "  \"tectonic_boundaries\": [\n";
    for (size_t i = 0; i < result.boundaries.size(); i++) {
        auto& tb = result.boundaries[i];
        const char* btypeStr = "unknown";
        switch (tb.classification) {
            case BoundaryClass::CONVERGENT_CC:   btypeStr = "convergent_continental_collision"; break;
            case BoundaryClass::CONVERGENT_OC:   btypeStr = "convergent_oceanic_subduction"; break;
            case BoundaryClass::CONVERGENT_OO:   btypeStr = "convergent_oceanic_collision"; break;
            case BoundaryClass::DIVERGENT_OCEAN: btypeStr = "divergent_mid_ocean_ridge"; break;
            case BoundaryClass::DIVERGENT_RIFT:  btypeStr = "divergent_continental_rift"; break;
            case BoundaryClass::TRANSFORM:       btypeStr = "transform"; break;
            case BoundaryClass::PASSIVE_MARGIN:  btypeStr = "passive_margin"; break;
            default: break;
        }
        f << "    {\"plate_a\": " << tb.plateA
          << ", \"plate_b\": " << tb.plateB
          << ", \"type\": \"" << btypeStr << "\""
          << ", \"length_m\": " << tb.lengthWorld
          << ", \"mean_height_m\": " << tb.meanHeight
          << ", \"cell_count\": " << tb.cellCount << "}";
        if (i < result.boundaries.size()-1) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    // ── Plate statistics ──
    f << "  \"plate_statistics\": [\n";
    for (size_t i = 0; i < result.plateStats.size(); i++) {
        auto& ps = result.plateStats[i];
        f << "    {\"plate_id\": " << ps.id
          << ", \"area_world_m2\": " << ps.areaWorld
          << ", \"mean_height_m\": " << ps.meanHeight
          << ", \"continental_fraction\": " << ps.continentalFraction
          << ", \"landmass_count\": " << ps.landmassCount
          << ", \"peak_count\": " << ps.peakCount << "}";
        if (i < result.plateStats.size()-1) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    // ── Mountain ridges ──
    f << "  \"mountain_ridges\": [\n";
    for (size_t i = 0; i < result.ridges.size(); i++) {
        auto& r = result.ridges[i];
        f << "    {\"id\": " << i
          << ", \"cell_count\": " << r.cells.size()
          << ", \"length_world_m\": " << r.lengthWorld
          << ", \"mean_height_m\": " << r.meanHeight
          << ", \"plate_a\": " << r.plateA
          << ", \"plate_b\": " << r.plateB
          << ", \"start_world_x\": " << r.startWorld.x
          << ", \"start_world_z\": " << r.startWorld.y
          << ", \"end_world_x\": " << r.endWorld.x
          << ", \"end_world_z\": " << r.endWorld.y << "}";
        if (i < result.ridges.size()-1) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    // ── Flow network summary ──
    f << "  \"flow_network\": {\n";
    jis("    max_accumulation", result.flowNet.maxAccumulation, false);
    jfs("    river_density", result.flowNet.riverDensity, false);
    jis("    river_segment_count", (int)result.flowNet.riverSegments.size(), true);
    f << "  },\n";

    // ── Cross-sections ──
    f << "  \"cross_sections\": [\n";
    for (size_t ci = 0; ci < result.crossSections.size(); ci++) {
        auto& cs = result.crossSections[ci];
        f << "    {\n";
        jss("      \"label\"", cs.label, false);
        jfs("      \"min_height_m\"", cs.minHeight, false);
        jfs("      \"max_height_m\"", cs.maxHeight, false);
        jis("      \"landmass_crossings\"", cs.landmassCrossings, false);
        f << "      \"profile\": [\n";
        // Downsample to ~64 samples for JSON size
        int stride = std::max(1, (int)cs.samples.size() / 64);
        for (size_t si = 0; si < cs.samples.size(); si += stride) {
            auto& s = cs.samples[si];
            f << "        {\"dist_m\": " << s.distance
              << ", \"height_m\": " << s.height
              << ", \"landmass_id\": " << s.landmassId << "}";
            if (si + stride < cs.samples.size()) f << ",";
            f << "\n";
        }
        f << "      ]\n";
        f << "    }";
        if (ci < result.crossSections.size()-1) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    // ── Basins ──
    f << "  \"ocean_basins\": [\n";
    for (size_t i = 0; i < std::min(result.basins.size(), (size_t)15); i++) {
        auto& b = result.basins[i];
        f << "    {\"id\": " << i
          << ", \"world_x\": " << b.worldPos.x
          << ", \"world_z\": " << b.worldPos.y
          << ", \"depth_m\": " << b.depth
          << ", \"volume\": " << b.volume << "}";
        if (i < std::min(result.basins.size(),(size_t)15)-1) f << ",";
        f << "\n";
    }
    f << "  ]\n";

    f << "}\n";
    f.close();

    std::cout << "\n[DebugAnalyzer] ==========================================\n";
    std::cout << "[DebugAnalyzer] Report written to: " << filepath << "\n";
    std::cout << "[DebugAnalyzer] " << result.peaks.size() << " peaks | "
              << result.landmasses.size() << " landmasses | "
              << result.ridges.size() << " ridges | "
              << result.boundaries.size() << " boundaries | "
              << result.basins.size() << " basins\n";
    std::cout << "[DebugAnalyzer] " << result.flowNet.riverSegments.size() << " river segments | "
              << result.crossSections.size() << " cross-sections\n";
    std::cout << "[DebugAnalyzer] Natural language summary:\n";
    std::cout << "[DebugAnalyzer]   " << result.aiNaturalLanguageSummary << "\n";
    std::cout << "[DebugAnalyzer] ==========================================\n\n";
}

// ══════════════════════════════════════════════════════════════
//  DEBUG TEXTURE GENERATION — for GPU visualization overlays
// ══════════════════════════════════════════════════════════════

inline std::vector<uint8_t> generateDebugOverlay(
    const std::vector<float>& heightmap,
    uint32_t R,
    DebugOverlayMode mode,
    const std::vector<int>* plateMap = nullptr,
    const std::vector<uint8_t>* crustTypeMap = nullptr,
    const std::vector<GeomorphClass>* geomMap = nullptr,
    const std::vector<int>* flowAccum = nullptr,
    int flowMaxAccum = 1)
{
    std::vector<uint8_t> rgba(R * R * 4, 0);
    const int W = (int)R, H = (int)R;

    float hMin = 1e30f, hMax = -1e30f;
    for (float h : heightmap) {
        hMin = std::min(hMin, h);
        hMax = std::max(hMax, h);
    }

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float h = heightmap[y*W + x];
            int idx = (y*W + x) * 4;
            uint8_t r = 0, g = 0, b = 0, a = 255;

            switch (mode) {
            case DebugOverlayMode::HEIGHT_HEATMAP: {
                float t = (h - hMin) / (hMax - hMin + 0.001f);
                t = glm::clamp(t, 0.f, 1.f);
                // Blue → Cyan → Green → Yellow → Red
                if      (t < 0.25f) { r = 0;          g = (uint8_t)(t*4.f*255); b = 200; }
                else if (t < 0.50f) { r = 0;          g = 200; b = (uint8_t)((1.f-(t-0.25f)*4.f)*200); }
                else if (t < 0.75f) { r = (uint8_t)((t-0.5f)*4.f*255); g = 200; b = 0; }
                else                { r = 255; g = (uint8_t)((1.f-(t-0.75f)*4.f)*200); b = 0; }
                break;
            }
            case DebugOverlayMode::SLOPE_MAP: {
                if (x > 0 && x < W-1 && y > 0 && y < H-1) {
                    float dx = (heightmap[y*W+(x+1)] - heightmap[y*W+(x-1)]) / 2.f;
                    float dy = (heightmap[(y+1)*W+x] - heightmap[(y-1)*W+x]) / 2.f;
                    float slope = std::atan(std::sqrt(dx*dx + dy*dy));
                    float t = slope / glm::half_pi<float>();
                    r = g = b = (uint8_t)(t * 255.f);
                }
                break;
            }
            case DebugOverlayMode::PLATE_BOUNDARIES: {
                if (plateMap) {
                    int myPlate = (*plateMap)[y*W + x];
                    bool isBoundary = false;
                    for (int n = 0; n < 8; n++) {
                        int nx = x+DX8[n], ny = y+DY8[n];
                        if (nx >=0 && nx<W && ny>=0 && ny<H &&
                            (*plateMap)[ny*W+nx] != myPlate) {
                            isBoundary = true; break;
                        }
                    }
                    if (isBoundary) {
                        r = 255; g = 50; b = 50; a = 220;
                    } else if (crustTypeMap && (*crustTypeMap)[y*W+x] == 0) {
                        // Continental
                        float t = (h - hMin) / (hMax - hMin + 0.001f);
                        r = (uint8_t)(180 * (0.5f + t*0.5f));
                        g = (uint8_t)(150 * (0.5f + t*0.5f));
                        b = (uint8_t)(100 * (0.5f + t*0.5f));
                    } else {
                        // Oceanic
                        r = 40; g = 70; b = 140;
                    }
                }
                break;
            }
            case DebugOverlayMode::LAND_WATER_MASK: {
                if (h > 0.f) { r = 40; g = 160; b = 40; }  // land: green
                else          { r = 20; g = 60; b = 180; }  // ocean: blue
                break;
            }
            case DebugOverlayMode::MOUNTAIN_PEAKS: {
                if (h > hMin + (hMax-hMin)*0.65f) {
                    r = 255; g = 255; b = 255;
                } else {
                    float t = (h - hMin) / (hMax - hMin + 0.001f);
                    r = g = b = (uint8_t)(t * 120.f);
                }
                break;
            }
            case DebugOverlayMode::RIDGE_LINES: {
                // Simplified: highlight high+steep cells
                if (x > 0 && x < W-1 && y > 0 && y < H-1) {
                    float dx = (heightmap[y*W+(x+1)] - heightmap[y*W+(x-1)]) / 2.f;
                    float dy = (heightmap[(y+1)*W+x] - heightmap[(y-1)*W+x]) / 2.f;
                    float slp = std::atan(std::sqrt(dx*dx + dy*dy));
                    if (h > hMin + (hMax-hMin)*0.5f && slp > 0.3f) {
                        r = 255; g = 200; b = 50;
                    } else {
                        float t = (h - hMin) / (hMax - hMin + 0.001f);
                        r = g = b = (uint8_t)(t * 80.f);
                    }
                }
                break;
            }
            case DebugOverlayMode::GEOMORPHOLOGY: {
                if (geomMap) {
                    GeomorphClass gc = (*geomMap)[y*W + x];
                    switch (gc) {
                        case GeomorphClass::DEEP_OCEAN:    r=10;  g=30;  b=100; break;  // dark blue
                        case GeomorphClass::OCEAN_BASIN:   r=30;  g=80;  b=160; break;  // medium blue
                        case GeomorphClass::SHALLOW_WATER: r=60;  g=140; b=200; break;  // light blue
                        case GeomorphClass::COASTAL_PLAIN: r=180; g=200; b=120; break;  // pale yellow-green
                        case GeomorphClass::LOWLAND:       r=50;  g=160; b=50;  break;  // green
                        case GeomorphClass::HILL:          r=120; g=130; b=60;  break;  // olive
                        case GeomorphClass::PLATEAU:       r=160; g=140; b=80;  break;  // tan
                        case GeomorphClass::MOUNTAIN:      r=130; g=110; b=80;  break;  // brown
                        case GeomorphClass::PEAK:          r=240; g=240; b=240; break;  // white
                        case GeomorphClass::TRENCH:        r=80;  g=0;   b=80;  break;  // dark purple
                    }
                }
                break;
            }
            case DebugOverlayMode::FLOW_ACCUMULATION: {
                if (flowAccum && flowMaxAccum > 0) {
                    int acc = (*flowAccum)[y*W + x];
                    float t = std::log(1.f + (float)acc) / std::log(1.f + (float)flowMaxAccum);
                    t = glm::clamp(t, 0.f, 1.f);
                    // Dark → Bright blue for rivers
                    if (acc > flowMaxAccum * 0.01f) {
                        r = (uint8_t)(t * 80.f);
                        g = (uint8_t)(t * 140.f);
                        b = (uint8_t)(t * 255.f);
                    } else {
                        r = g = b = 20;
                    }
                }
                break;
            }
            default:
                break;
            }

            rgba[idx+0] = r;
            rgba[idx+1] = g;
            rgba[idx+2] = b;
            rgba[idx+3] = a;
        }
    }
    return rgba;
}

} // namespace DebugAnalyzer
