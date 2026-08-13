#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <string>

// One panel in the blade-element strip theory.
// All coordinates are in the aircraft body frame (+X = nose, +Y = up, +Z = right).
struct AeroPanel {
    glm::vec3  center{0.0f};        // Panel centroid (body frame, metres from CG)
    glm::vec3  normal{0.0f,1.0f,0.0f}; // Outward unit normal (body frame)
    glm::vec3  chordDir{-1.0f,0.0f,0.0f}; // Chordwise direction (= -airstream at a=0)
    glm::vec3  spanDir{0.0f,0.0f,1.0f};   // Spanwise direction (quarter-chord line)
    float      area = 0.0f;         // Panel area (m^2)
    float      chord = 0.0f;        // Panel chord length (m)
    float      span = 0.0f;         // Panel span (m)
    float      stallAoADeg = 15.0f; // Panel stall angle of attack (degrees)
    float      liftSlope = 5.4f;    // CL per radian (local 2D)
    float      cd0 = 0.02f;         // Profile drag coefficient at zero lift
    float      dcl_dflap = 0.0f;    // dCL/dflap (0 if not a control surface)
    float      flapDeflection = 0.0f; // Current flap deflection (radians)
    int        groupID = 0;         // 0=mainWing, 1=hStab, 2=vFin, 3=fuselage

    // Computed per-step
    float      Cl = 0.0f;
    float      Cd = 0.0f;
    float      Cp = 0.0f;
    float      localAoADeg = 0.0f;
};

// Result of one aerodynamic step
struct AeroResult {
    glm::vec3 force{0.0f};      // Net aerodynamic force (body frame, Newtons)
    glm::vec3 moment{0.0f};     // Net moment about CG (body frame, N·m)
    float     totalLift = 0.0f; // Scalar lift (N)
    float     totalDrag = 0.0f; // Scalar drag (N)
    float     avgCl = 0.0f;     // Area-weighted average Cl
    float     avgAoADeg = 0.0f; // Average AoA (deg)
};

class AerodynamicBody {
public:
    // Build panel layout automatically from half-extents. Suitable for the
    // Cessna and any box-like aircraft.
    void buildFromHalfExtents(float halfLength, float halfSpan, float halfHeight,
                               float totalMassKg,
                               float wingArea,        // m^2 total reference area
                               float stallAoADeg,
                               float liftCurveSlope);

    // Given freestream velocity in the body frame and air density, return
    // forces/moments. Also updates each panel's Cl, Cd, Cp fields.
    AeroResult compute(const glm::vec3& vAirBody, float airDensity);

    const std::vector<AeroPanel>& panels() const { return panels_; }
    std::vector<AeroPanel>&       panels()       { return panels_; }
    bool empty() const { return panels_.empty(); }

    float wingSpan()      const { return wingspan_; }
    float referenceArea() const { return refArea_; }
    float meanChord()     const { return (wingspan_ > 1e-3f) ? refArea_ / wingspan_ : 1.0f; }

    float oswaldEfficiency = 0.85f;

private:
    std::vector<AeroPanel> panels_;
    float wingspan_   = 10.0f;
    float refArea_    = 18.0f;
    float halfLength_ = 6.0f;
    float inducedDragK_ = 0.05f;

    void finalize();
};
