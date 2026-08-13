#pragma once
// ══════════════════════════════════════════════════════════════
//  MasterHydrology.h — world-scale D-infinity flow + river network
//
//  Implements:
//   1. Priority-flood depression filling (Wang & Liu 2006) — critical
//      for avoiding the "maxAccumulation=44" problem where almost
//      every land cell is a local sink.
//   2. D-infinity (D∞) flow direction computation (Tarboton 1997)
//   3. Flow accumulation by processing cells in descending order
//   4. River extraction via threshold on accumulation
//
//  Outputs (per-cell, R = resolution):
//    flowDir[]  — float in [0, 2π), or NaN for ocean/pit
//    flowAccum[] — int, upstream contributing cell count
//    isRiver[]  — bool, accumulation > threshold
//    riverWidth[] — float, derived from accumulation
//
//  Dependencies: <vector>, <cmath>, <algorithm>, <queue>
// ══════════════════════════════════════════════════════════════

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <utility>
#include <cstdint>
#include <functional>

namespace MasterHydrology {

// ────────────────────────────────────────────────────────────────
//  D8 neighbour table
// ────────────────────────────────────────────────────────────────
// 0:E  1:SE  2:S  3:SW  4:W  5:NW  6:N  7:NE
const int DX8[8] = { 1,  1,  0, -1, -1, -1,  0,  1 };
const int DY8[8] = { 0,  1,  1,  1,  0, -1, -1, -1 };
const float DDIST[8] = { 1.0f, 1.41421356f, 1.0f, 1.41421356f,
                          1.0f, 1.41421356f, 1.0f, 1.41421356f };
const float DAZIMUTH[8] = {
    0.0f,
    0.785398163f,
    1.570796327f,
    2.356194490f,
    3.141592654f,
    3.926990817f,
    4.712388980f,
    5.497787144f
};

struct FlowResult {
    std::vector<float>  flowDirection;
    std::vector<float>  flowFraction[2];
    std::vector<int>    flowNeighbor[2];
    std::vector<int>    flowAccumulation;
    std::vector<bool>   isRiver;
    std::vector<float>  riverWidth;
    // Continuous river intensity in [0,1]. Unlike the binary isRiver flag this
    // ramps up smoothly around the discharge threshold so banks fade out
    // instead of terminating in a hard 1-cell edge.
    std::vector<float>  riverStrength;
    // Lake support: waterLevel is the depression-fill (spill) surface produced
    // by priority flooding; where it rises meaningfully above the natural
    // terrain the cell is submerged and flagged as a lake with a flat surface.
    std::vector<float>  waterLevel;
    std::vector<float>  lakeDepth;
    std::vector<bool>   isLake;
    int                 maxAccumulation;
    int                 totalLandCells;

    void allocate(uint32_t R) {
        size_t N = (size_t)R * R;
        flowDirection.assign(N, NAN);
        flowFraction[0].assign(N, 0.0f);
        flowFraction[1].assign(N, 0.0f);
        flowNeighbor[0].assign(N, -1);
        flowNeighbor[1].assign(N, -1);
        flowAccumulation.assign(N, 0);
        isRiver.assign(N, false);
        riverWidth.assign(N, 0.0f);
        riverStrength.assign(N, 0.0f);
        waterLevel.assign(N, 0.0f);
        lakeDepth.assign(N, 0.0f);
        isLake.assign(N, false);
        maxAccumulation = 0;
        totalLandCells  = 0;
    }
};

// ══════════════════════════════════════════════════════════════
//  1.  Priority-flood depression filling (Wang & Liu 2006)
//
//  Works on a COPY of the heightmap passed in.  The caller is
//  responsible for passing a copy if the original should be kept.
//
//  Algorithm:
//   - Initialise an open set with all border cells and all ocean
//     cells (height ≤ seaLevel).
//   - Process the open set in ascending height order (min-heap).
//   - For each open cell, examine its 8 neighbours:
//       if neighbour already open → skip
//       if neighbour is ocean → enqueue at its natural height
//       if neighbour is land  → enqueue at max(natural, parent+ε)
//   - This guarantees every land cell has a drainage path to the
//     ocean or the map border, eliminating interior pits.
// ══════════════════════════════════════════════════════════════
inline void fillDepressions(
    std::vector<float>& heightmap,
    uint32_t R,
    float seaLevel)
{
    const float EPS = 1e-4f;
    int W   = (int)R;
    int total = W * W;

    std::vector<bool> open(total, false);

    // min-heap: (height, flat_index)
    using HI = std::pair<float, int>;
    std::priority_queue<HI, std::vector<HI>, std::greater<HI>> pq;

    auto enqueue = [&](int idx) {
        if (!open[idx]) {
            open[idx] = true;
            pq.push({heightmap[idx], idx});
        }
    };

    // Seed: border cells and ocean cells
    for (int y = 0; y < W; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x;
            if (x == 0 || x == W-1 || y == 0 || y == W-1 ||
                heightmap[idx] <= seaLevel)
            {
                enqueue(idx);
            }
        }
    }

    // Flood
    while (!pq.empty()) {
        auto [ch, cidx] = pq.top(); pq.pop();
        int cx = cidx % W, cy = cidx / W;

        for (int n = 0; n < 8; n++) {
            int nx = cx + DX8[n];
            int ny = cy + DY8[n];
            if (nx < 0 || nx >= W || ny < 0 || ny >= W) continue;
            int nidx = ny * W + nx;
            if (open[nidx]) continue;

            open[nidx] = true;
            if (heightmap[nidx] <= seaLevel) {
                // Ocean cell — keep its original depth as-is
                pq.push({heightmap[nidx], nidx});
            } else {
                // Land cell — raise if needed so it can drain through parent
                float filled = std::max(heightmap[nidx], ch + EPS);
                heightmap[nidx] = filled;
                pq.push({filled, nidx});
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════
//  2.  D-infinity flow direction (Tarboton 1997)
// ══════════════════════════════════════════════════════════════
inline void computeFlowDirections(
    const std::vector<float>& heightmap,
    uint32_t R,
    float seaLevel,
    FlowResult& out)
{
    out.allocate(R);
    int W = (int)R;

    for (int y = 1; y < (int)R - 1; y++) {
        for (int x = 1; x < (int)R - 1; x++) {
            int idx = y * W + x;
            float h = heightmap[idx];

            if (h <= seaLevel) {
                // Ocean / below-sea sink
                continue;
            }

            out.totalLandCells++;

            float bestGradient = -1e30f;
            float bestAzimuth  =  0.0f;
            int   bestN1 = -1, bestN2 = -1;
            float bestFrac1 = 0.5f;

            for (int f = 0; f < 8; f++) {
                int n1 = f;
                int n2 = (f + 1) % 8;

                int nx1 = x + DX8[n1], ny1 = y + DY8[n1];
                int nx2 = x + DX8[n2], ny2 = y + DY8[n2];

                if (nx1 < 0 || nx1 >= W || ny1 < 0 || ny1 >= (int)R) continue;
                if (nx2 < 0 || nx2 >= W || ny2 < 0 || ny2 >= (int)R) continue;

                float h1 = heightmap[ny1 * W + nx1];
                float h2 = heightmap[ny2 * W + nx2];

                if (h1 >= h && h2 >= h) continue;   // no downhill facet

                float e1 = h - h1;
                float e2 = h - h2;

                float a1 = DAZIMUTH[n1];
                float a2 = DAZIMUTH[n2];
                float cos1 = std::cos(a1), sin1 = std::sin(a1);
                float cos2 = std::cos(a2), sin2 = std::sin(a2);

                float det = cos1 * sin2 - cos2 * sin1;
                if (std::abs(det) < 1e-10f) continue;

                float gx = (e1 * sin2 - e2 * sin1) / det;
                float gy = (cos1 * e2 - cos2 * e1) / det;
                float gradMag = std::sqrt(gx * gx + gy * gy);

                float proj1 = gx * cos1 + gy * sin1;
                float proj2 = gx * cos2 + gy * sin2;
                if (proj1 < -0.001f && proj2 < -0.001f) continue;

                float gradAzimuth = std::atan2(gy, gx);
                if (gradAzimuth < 0.0f) gradAzimuth += 6.283185307f;

                // Check sector containment
                float a1n = a1, a2n = a2;
                if (a2n < a1n) a2n += 6.283185307f;
                float gan = gradAzimuth;
                if (gan < a1n) gan += 6.283185307f;
                if (gan < a1n - 0.001f || gan > a2n + 0.001f) continue;

                if (gradMag > bestGradient) {
                    bestGradient = gradMag;
                    bestAzimuth  = gradAzimuth;
                    bestN1 = n1; bestN2 = n2;
                    float span = a2n - a1n;
                    bestFrac1 = (span > 0.001f)
                              ? std::max(0.0f, std::min(1.0f,
                                    1.0f - (gan - a1n) / span))
                              : 0.5f;
                }
            }

            if (bestGradient <= 0.0f) {
                out.totalLandCells--;   // flat-top / no outflow
                continue;
            }

            out.flowDirection[idx]    = bestAzimuth;
            out.flowNeighbor[0][idx]  = bestN1;
            out.flowNeighbor[1][idx]  = bestN2;
            out.flowFraction[0][idx]  = bestFrac1;
            out.flowFraction[1][idx]  = 1.0f - bestFrac1;
        }
    }
}

// ══════════════════════════════════════════════════════════════
//  3.  Flow accumulation (descending elevation order)
// ══════════════════════════════════════════════════════════════
inline void computeFlowAccumulation(
    const std::vector<float>& heightmap,
    uint32_t R,
    FlowResult& flow)
{
    int W = (int)R;
    int total = W * W;

    std::fill(flow.flowAccumulation.begin(), flow.flowAccumulation.end(), 1);

    // Sort descending by height
    std::vector<int> order(total);
    for (int i = 0; i < total; i++) order[i] = i;
    std::sort(order.begin(), order.end(),
        [&](int a, int b) { return heightmap[a] > heightmap[b]; });

    for (int idx : order) {
        int acc = flow.flowAccumulation[idx];
        if (acc <= 0) continue;

        int n1 = flow.flowNeighbor[0][idx];
        if (n1 < 0) continue;   // no outflow

        int x = idx % W, y = idx / W;

        // Primary neighbour
        {
            int nx = x + DX8[n1], ny = y + DY8[n1];
            if (nx >= 0 && nx < W && ny >= 0 && ny < W) {
                int nidx = ny * W + nx;
                flow.flowAccumulation[nidx] +=
                    (int)(acc * flow.flowFraction[0][idx] + 0.5f);
            }
        }
        // Secondary neighbour
        int n2 = flow.flowNeighbor[1][idx];
        if (n2 >= 0 && n2 != n1) {
            int nx = x + DX8[n2], ny = y + DY8[n2];
            if (nx >= 0 && nx < W && ny >= 0 && ny < W) {
                int nidx = ny * W + nx;
                flow.flowAccumulation[nidx] +=
                    (int)(acc * flow.flowFraction[1][idx] + 0.5f);
            }
        }
    }

    flow.maxAccumulation = *std::max_element(
        flow.flowAccumulation.begin(), flow.flowAccumulation.end());
}

// ══════════════════════════════════════════════════════════════
//  4.  River extraction
// ══════════════════════════════════════════════════════════════
inline void extractRivers(
    FlowResult& flow,
    uint32_t R,
    float accumFraction,
    int minCells = 50)
{
    int total = (int)R * (int)R;
    int threshold = std::max(minCells,
        (int)(accumFraction * (float)std::max(1, flow.maxAccumulation)));

    // A river "begins" to register a little below the hard threshold so the
    // transition from dry ground → bank → channel is gradual rather than a
    // step. Cells below onset are dry; cells at/above threshold are full
    // channels; cells in between are partial (banks / braided edges).
    const float onset = 0.45f * (float)threshold;
    const float span  = std::max(1.0f, (float)threshold - onset);

    for (int i = 0; i < total; i++) {
        int acc = flow.flowAccumulation[i];
        bool hasFlow = !std::isnan(flow.flowDirection[i]);

        float strength = 0.0f;
        if (hasFlow && acc > onset)
            strength = std::min(1.0f, ((float)acc - onset) / span);
        flow.riverStrength[i] = strength;

        if (acc >= threshold && hasFlow) {
            flow.isRiver[i]    = true;
            float norm         = (float)acc / (float)std::max(1, flow.maxAccumulation);
            flow.riverWidth[i] = 0.4f + 3.0f * std::sqrt(norm);
        } else {
            flow.isRiver[i]    = false;
            flow.riverWidth[i] = 0.0f;
        }
    }
}

// ══════════════════════════════════════════════════════════════
//  4b.  Lake extraction
//
//  Priority flooding raises every interior pit up to its lowest spill
//  (pour-point) elevation. That filled surface IS a natural water table:
//  wherever it sits meaningfully above the original terrain, the basin
//  would hold standing water. We record those cells as lakes with a flat
//  surface at the spill level, and clear any river flag there so a basin
//  reads as one coherent body of water instead of a knot of isolated
//  high-accumulation channel cells.
//
//   waterLevel  — copy of the depression-filled surface (spill height)
//   original    — the un-filled terrain heights
//   minLakeDepth — minimum fill thickness (metres) to qualify as a lake
// ══════════════════════════════════════════════════════════════
inline void extractLakes(
    FlowResult& flow,
    const std::vector<float>& waterLevel,
    const std::vector<float>& original,
    uint32_t R,
    float seaLevel,
    float minLakeDepth)
{
    int total = (int)R * (int)R;
    flow.waterLevel = waterLevel;
    for (int i = 0; i < total; i++) {
        float depth = waterLevel[i] - original[i];
        // Only land basins (above sea level) become lakes; the ocean is
        // handled separately by the sea-level surface.
        bool submerged = depth > minLakeDepth && original[i] > seaLevel;
        flow.isLake[i]   = submerged;
        flow.lakeDepth[i] = submerged ? depth : 0.0f;
        if (submerged) {
            // A lake surface is still water, not a flowing channel.
            flow.isRiver[i]      = false;
            flow.riverStrength[i] = 1.0f;
        }
    }
}

// ══════════════════════════════════════════════════════════════
//  Master pipeline
//   heightmap → fill depressions → flow dirs → accumulation → rivers
// ══════════════════════════════════════════════════════════════
inline void computeMasterHydrology(
    const std::vector<float>& heightmap,
    uint32_t R,
    float seaLevel,
    float riverAccumFraction,
    FlowResult& out,
    int riverMinCells = 50,
    float minLakeDepth = 0.8f)
{
    // Work on a copy so the original heightmap (used for rendering) is untouched.
    std::vector<float> workMap = heightmap;
    fillDepressions(workMap, R, seaLevel);
    // After flooding, workMap is the spill-level water table — capture it
    // before flow routing (which only reads workMap) so it survives the
    // allocate() inside computeFlowDirections().
    std::vector<float> filled = workMap;

    computeFlowDirections(workMap, R, seaLevel, out);
    computeFlowAccumulation(workMap, R, out);
    extractRivers(out, R, riverAccumFraction, riverMinCells);
    extractLakes(out, filled, heightmap, R, seaLevel, minLakeDepth);
}

} // namespace MasterHydrology
