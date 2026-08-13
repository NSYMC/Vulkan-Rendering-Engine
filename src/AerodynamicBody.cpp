#include "AerodynamicBody.h"
#include <glm/gtc/constants.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

static constexpr float kDeg2Rad = 0.01745329252f;
static constexpr float kRad2Deg = 57.2957795f;
static constexpr float kPi      = glm::pi<float>();

// ── Thin airfoil + stall model: {Cl, Cd} for a 2D section at AoA (radians) ──
static glm::vec2 airfoilCoeffs(float aoaRad, float stallAoARad,
                                float liftSlope, float cd0) {
    float cl;
    float aoaAbs = std::abs(aoaRad);
    float clPeak = liftSlope * stallAoARad;

    if (aoaAbs <= stallAoARad) {
        cl = liftSlope * aoaRad;
    } else {
        // Smooth post-stall falloff (Kirchhoff-style approximation)
        float t = std::clamp((aoaAbs - stallAoARad) / stallAoARad, 0.0f, 1.0f);
        float sign = (aoaRad >= 0.0f) ? 1.0f : -1.0f;
        cl = sign * clPeak * (1.0f - 0.7f * t * (2.0f - t));
    }

    float cd = cd0 + 0.008f * cl * cl;   // quadratic profile polar
    return {cl, cd};
}

void AerodynamicBody::buildFromHalfExtents(float halfLength, float halfSpan, float halfHeight,
                                            float /*totalMassKg*/, float wingArea,
                                            float stallAoADeg, float liftCurveSlope) {
    halfLength_   = halfLength;
    wingspan_     = std::max(0.5f, halfSpan * 2.0f);
    refArea_      = std::max(0.5f, wingArea);
    panels_.clear();

    const float cd0_wing  = 0.015f;
    const float cd0_stab  = 0.018f;
    const float cd0_fuse  = 0.025f;
    const int   NWING     = 6;    // strips per wing
    const int   NSTAB     = 3;    // strips per stabilizer

    // ── Main wing: left and right, NWING strips each ──────────
    float wingChord = refArea_ / wingspan_;   // mean chord (keeps total area = wingArea)
    float stripSpan = halfSpan / float(NWING);
    float stripArea = wingChord * stripSpan;

    for (int side : {-1, 1}) {
        for (int i = 0; i < NWING; i++) {
            float zCenter = side * ((i + 0.5f) * stripSpan);
            AeroPanel p;
            p.center      = glm::vec3(0.25f * wingChord, 0.0f, zCenter);
            p.normal      = glm::vec3(0.0f, 1.0f, 0.0f);
            p.chordDir    = glm::vec3(-1.0f, 0.0f, 0.0f);
            p.spanDir     = glm::vec3(0.0f, 0.0f, float(side));
            p.area        = stripArea;
            p.chord       = wingChord;
            p.span        = stripSpan;
            p.stallAoADeg = stallAoADeg;
            p.liftSlope   = liftCurveSlope;
            p.cd0         = cd0_wing;
            p.groupID     = 0;
            panels_.push_back(p);
        }
    }

    // ── Horizontal stabilizer ──────────────────────────────────
    float stabChord = wingChord * 0.4f;
    float stabSpan  = halfSpan * 0.35f;
    float stabStripSpan = stabSpan / float(NSTAB);
    float stabStripArea = stabChord * stabStripSpan;
    float tailArm   = -halfLength * 0.85f;

    for (int side : {-1, 1}) {
        for (int i = 0; i < NSTAB; i++) {
            float zCenter = side * ((i + 0.5f) * stabStripSpan);
            AeroPanel p;
            p.center      = glm::vec3(tailArm + 0.25f * stabChord, 0.0f, zCenter);
            p.normal      = glm::vec3(0.0f, 1.0f, 0.0f);
            p.chordDir    = glm::vec3(-1.0f, 0.0f, 0.0f);
            p.spanDir     = glm::vec3(0.0f, 0.0f, float(side));
            p.area        = stabStripArea;
            p.chord       = stabChord;
            p.span        = stabStripSpan;
            p.stallAoADeg = 12.0f;
            p.liftSlope   = liftCurveSlope * 0.8f;
            p.cd0         = cd0_stab;
            p.groupID     = 1;
            panels_.push_back(p);
        }
    }

    // ── Vertical fin (drag + yaw) ──────────────────────────────
    {
        float finChord = wingChord * 0.55f;
        float finSpan  = halfHeight * 1.8f;
        for (int side : {-1, 1}) {
            AeroPanel p;
            p.center      = glm::vec3(tailArm, finSpan * 0.5f, float(side) * 0.01f);
            p.normal      = glm::vec3(0.0f, 0.0f, float(side));
            p.chordDir    = glm::vec3(-1.0f, 0.0f, 0.0f);
            p.spanDir     = glm::vec3(0.0f, 1.0f, 0.0f);
            p.area        = finChord * finSpan * 0.5f;
            p.chord       = finChord;
            p.span        = finSpan * 0.5f;
            p.stallAoADeg = 16.0f;
            p.liftSlope   = 4.5f;
            p.cd0         = 0.020f;
            p.groupID     = 2;
            panels_.push_back(p);
        }
    }

    // ── Fuselage (4 axis-aligned drag panels) ─────────────────
    {
        float fArea = halfLength * halfHeight;
        glm::vec3 fuseNormals[4] = {
            {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f}, {0.0f,  0.0f, -1.0f}
        };
        for (auto& fn : fuseNormals) {
            AeroPanel p;
            p.center      = glm::vec3(0.0f);
            p.normal      = fn;
            p.chordDir    = glm::vec3(-1.0f, 0.0f, 0.0f);
            p.spanDir     = (std::abs(fn.y) > 0.5f)
                            ? glm::vec3(0.0f, 0.0f, 1.0f)
                            : glm::vec3(0.0f, 1.0f, 0.0f);
            p.area        = fArea;
            p.chord       = halfLength;
            p.span        = halfHeight;
            p.stallAoADeg = 90.0f;
            p.liftSlope   = 2.0f;
            p.cd0         = cd0_fuse;
            p.groupID     = 3;
            panels_.push_back(p);
        }
    }

    finalize();
    std::cout << "  [aero] built " << panels_.size() << " panels from half-extents"
              << " span=" << wingspan_ << "m refArea=" << refArea_ << "m^2\n";
}

void AerodynamicBody::finalize() {
    float ar = wingspan_ * wingspan_ / refArea_;
    inducedDragK_ = 1.0f / (kPi * oswaldEfficiency * std::max(0.5f, ar));
}

AeroResult AerodynamicBody::compute(const glm::vec3& vAirBody, float airDensity) {
    AeroResult result{};
    float V = glm::length(vAirBody);
    if (V < 0.5f) return result;

    const float qDyn = 0.5f * airDensity * V * V;
    const glm::vec3 vDir = vAirBody / V;

    float totalCl   = 0.0f;
    float totalArea = 0.0f;

    for (AeroPanel& panel : panels_) {
        float vChord  = glm::dot(vDir, panel.chordDir);
        float vNormal = glm::dot(vDir, panel.normal);

        float localAoA = std::atan2(-vNormal, vChord);
        panel.localAoADeg = localAoA * kRad2Deg;

        float stallRad = panel.stallAoADeg * kDeg2Rad;
        glm::vec2 coeffs = airfoilCoeffs(localAoA, stallRad, panel.liftSlope, panel.cd0);
        float cl = coeffs.x, cdProf = coeffs.y;
        panel.Cl = cl;
        panel.Cd = cdProf;
        panel.Cp = -cl;

        glm::vec3 liftDir = panel.normal - glm::dot(panel.normal, vDir) * vDir;
        float liftDirLen = glm::length(liftDir);
        liftDir = (liftDirLen > 1e-4f) ? liftDir / liftDirLen : glm::vec3(0.0f);
        glm::vec3 dragDir = -vDir;

        float panelLift = qDyn * panel.area * cl;
        float panelDrag = qDyn * panel.area * cdProf;

        glm::vec3 panelForce = liftDir * panelLift + dragDir * panelDrag;
        result.force     += panelForce;
        result.totalLift += panelLift;
        result.totalDrag += panelDrag;

        result.moment += glm::cross(panel.center, panelForce);

        totalCl   += cl * panel.area;
        totalArea += panel.area;
    }

    float CLtotal = (totalArea > 0.0f) ? totalCl / totalArea : 0.0f;
    float CDi     = inducedDragK_ * CLtotal * CLtotal;
    float inducedDragForce = qDyn * refArea_ * CDi;
    result.force     -= vDir * inducedDragForce;
    result.totalDrag += inducedDragForce;

    result.avgCl     = CLtotal;
    result.avgAoADeg = std::atan2(-glm::dot(vDir, glm::vec3(0.0f, 1.0f, 0.0f)),
                                   glm::dot(vDir, glm::vec3(1.0f, 0.0f, 0.0f))) * kRad2Deg;
    return result;
}
