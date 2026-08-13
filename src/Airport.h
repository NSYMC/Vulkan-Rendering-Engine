// ══════════════════════════════════════════════════════════════
//  Airport — logical airport layout (runway, taxiways, apron, gates)
//
//  Pure data + geometry math (no Vulkan, no physics). The engine uses
//  this to (1) flatten a concrete pad into the heightmap, (2) build the
//  rendered concrete surfaces + markings, and (3) drive the aircraft's
//  taxi / takeoff / landing route. Buildings are placed by the user;
//  we only lay out the paved surfaces and expose navigation points.
//
//  Local runway frame: u = along the runway axis (heading), v = across
//  it (to the "side" where the parallel taxiway + apron live).
// ══════════════════════════════════════════════════════════════
#pragma once

#include <glm/glm.hpp>
#include <cmath>
#include <vector>

// One flat, axis-oriented paved rectangle or painted marking. The renderer
// turns each of these into two triangles at (pad elevation + yOffset).
struct AirportQuad {
    glm::vec2 center;      // world XZ
    float     halfLen;     // half-size along heading (m)
    float     halfWid;     // half-size across heading (m)
    float     headingRad;  // orientation
    glm::vec3 color;       // flat surface colour
    float     yOffset;     // height above the pad surface (m), for layering
};

// A parking position on the apron (nose heading = parked orientation).
struct AirportGate {
    glm::vec2 position;
    float     headingRad;
};

struct Airport {
    glm::vec2 center{0.0f};    // pad centre, world XZ (m)
    float elevation  = 0.0f;   // flattened pad elevation (m)
    float headingRad = 0.0f;   // runway axis bearing

    // Dimensions (m) — scaled-down but realistically proportioned.
    float runwayLength  = 900.0f;
    float runwayWidth   = 45.0f;
    float taxiwayWidth  = 23.0f;
    float taxiwayOffset = 95.0f;   // runway CL → parallel taxiway CL
    float apronHalfLen  = 80.0f;   // apron size along runway axis
    float apronHalfWid  = 55.0f;   // apron size across runway axis
    int   gateCount     = 4;

    std::vector<AirportGate> gates;

    // ── Local frame ──────────────────────────────────────────
    glm::vec2 dir()  const { return glm::vec2(std::cos(headingRad), std::sin(headingRad)); }
    glm::vec2 side() const { glm::vec2 d = dir(); return glm::vec2(-d.y, d.x); }   // +90°
    glm::vec2 local(float u, float v) const { return center + dir() * u + side() * v; }

    float halfRunway() const { return runwayLength * 0.5f; }
    float apronCenterV() const { return taxiwayOffset + taxiwayWidth * 0.5f + apronHalfWid; }
    float apronCenterU() const { return -runwayLength * 0.20f; }

    // Runway threshold points (the two ends of the runway centerline).
    // "negEnd" is at u = -L/2, "posEnd" at u = +L/2.
    glm::vec2 negEnd() const { return local(-halfRunway() + 30.0f, 0.0f); }
    glm::vec2 posEnd() const { return local( halfRunway() - 30.0f, 0.0f); }

    // ── Layout build ─────────────────────────────────────────
    // Call once after center/elevation/heading/dims are set.
    void rebuild() {
        gates.clear();
        const float v = apronCenterV();
        const float spanU = apronHalfLen * 1.4f;
        const int n = gateCount < 1 ? 1 : gateCount;
        for (int i = 0; i < n; ++i) {
            float t = (n == 1) ? 0.5f : (float)i / (float)(n - 1);
            float u = apronCenterU() + (t - 0.5f) * spanU;
            AirportGate g;
            g.position  = local(u, v - apronHalfWid * 0.45f);
            g.headingRad = headingRad + 1.5707963f;   // nose faces away from taxiway
            gates.push_back(g);
        }
    }

    // ── Concrete surfaces + markings → coloured quads ────────
    std::vector<AirportQuad> buildSurfaceQuads() const {
        std::vector<AirportQuad> q;
        const glm::vec3 cRunway (0.165f, 0.170f, 0.180f);  // dark asphalt
        const glm::vec3 cTaxi   (0.205f, 0.205f, 0.215f);  // lighter asphalt
        const glm::vec3 cApron  (0.330f, 0.330f, 0.340f);  // concrete
        const glm::vec3 cWhite  (0.880f, 0.885f, 0.890f);
        const glm::vec3 cYellow (0.840f, 0.720f, 0.090f);

        const float SURF = 0.06f;   // base surface lift above terrain pad
        const float MARK = 0.10f;   // markings lift above surface

        auto rect = [&](float u, float v, float hl, float hw, glm::vec3 c, float y) {
            q.push_back({ local(u, v), hl, hw, headingRad, c, y });
        };

        // ── Apron (concrete slab) ──
        rect(apronCenterU(), apronCenterV(), apronHalfLen, apronHalfWid, cApron, SURF);

        // ── Parallel taxiway (full length) + two end connectors ──
        rect(0.0f, taxiwayOffset, halfRunway(), taxiwayWidth * 0.5f, cTaxi, SURF);
        // connector from taxiway to apron
        rect(apronCenterU(), (taxiwayOffset + apronCenterV()) * 0.5f,
             taxiwayWidth * 0.5f, std::abs(apronCenterV() - taxiwayOffset) * 0.5f, cTaxi, SURF);
        // end connectors joining runway ends to the taxiway
        for (float endU : { -halfRunway() + taxiwayWidth * 0.5f, halfRunway() - taxiwayWidth * 0.5f }) {
            rect(endU, taxiwayOffset * 0.5f, taxiwayWidth * 0.5f, taxiwayOffset * 0.5f, cTaxi, SURF);
        }

        // ── Runway slab ──
        rect(0.0f, 0.0f, halfRunway(), runwayWidth * 0.5f, cRunway, SURF);

        // ── Runway edge lines (white) ──
        for (float s : { -1.0f, 1.0f }) {
            rect(0.0f, s * (runwayWidth * 0.5f - 0.9f), halfRunway() - 4.0f, 0.45f, cWhite, MARK);
        }

        // ── Runway centerline dashes (white) ──
        {
            const float dash = 14.0f, gap = 12.0f;
            const float usable = runwayLength - 120.0f;
            int nd = (int)(usable / (dash + gap));
            for (int i = 0; i < nd; ++i) {
                float u = -usable * 0.5f + (dash + gap) * i + dash * 0.5f;
                rect(u, 0.0f, dash * 0.5f, 0.45f, cWhite, MARK);
            }
        }

        // ── Threshold "piano key" bars at both ends (white) ──
        for (float endSign : { -1.0f, 1.0f }) {
            float baseU = endSign * (halfRunway() - 22.0f);
            for (int k = -3; k <= 3; ++k) {
                rect(baseU, k * 3.4f, 9.0f, 1.1f, cWhite, MARK);
            }
        }

        // ── Taxiway centerline (yellow, continuous) ──
        rect(0.0f, taxiwayOffset, halfRunway() - 4.0f, 0.40f, cYellow, MARK);
        rect(apronCenterU(), (taxiwayOffset + apronCenterV()) * 0.5f,
             0.40f, std::abs(apronCenterV() - taxiwayOffset) * 0.5f, cYellow, MARK);
        for (float endU : { -halfRunway() + taxiwayWidth * 0.5f, halfRunway() - taxiwayWidth * 0.5f }) {
            rect(endU, taxiwayOffset * 0.5f, 0.40f, taxiwayOffset * 0.5f, cYellow, MARK);
        }

        // ── Gate parking markings (yellow stub per gate) ──
        for (const AirportGate& g : gates) {
            // a short lead-in stub pointing toward the apron edge
            glm::vec2 d(std::cos(g.headingRad), std::sin(g.headingRad));
            glm::vec2 c = g.position - d * 6.0f;
            float ang = g.headingRad;
            q.push_back({ c, 7.0f, 0.4f, ang, cYellow, MARK });
        }

        return q;
    }
};

// Build a logical pair of airports laid out along their connecting axis.
// A and B share the runway axis bearing (A→B). Elevations are filled in
// later from the terrain. worldSize is the square world extent (m).
inline void layoutDemoAirports(float worldSize, Airport& a, Airport& b) {
    // SW departure and NE arrival, deliberately offset so the leg is not a
    // straight east-west hop — the demo flight arcs across the terrain mass.
    const float inset = worldSize * 0.16f;
    a.center = glm::vec2(inset, inset * 1.35f);
    b.center = glm::vec2(worldSize - inset, worldSize - inset * 0.85f);

    glm::vec2 ab = b.center - a.center;
    float bearing = std::atan2(ab.y, ab.x);
    a.headingRad = bearing;
    b.headingRad = bearing;   // shared axis; operational direction is per-leg

    a.rebuild();
    b.rebuild();
}
