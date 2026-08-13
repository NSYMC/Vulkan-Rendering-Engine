// ══════════════════════════════════════════════════════════════

//  PhysicsWorld — Jolt Physics 3D integration for TerrainEngine

// ══════════════════════════════════════════════════════════════

#pragma once



#include <cstdint>

#include <memory>

#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "AerodynamicBody.h"



enum class PhysicsBodyKind : uint8_t {

    Sphere = 0,

    Box    = 1,

};



struct PhysicsBodyState {

    glm::vec3 position{0.0f};

    glm::vec3 velocity{0.0f};

    glm::vec3 angularVelocity{0.0f};

    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};

    glm::vec3 halfExtents{1.0f};

    float     radius   = 1.0f;

    PhysicsBodyKind kind = PhysicsBodyKind::Sphere;

    bool      kinematic = false;

    bool      active   = false;

};



// One landing-gear leg: raycast spring-damper suspension (report §5.6).

struct GearWheel {

    glm::vec3 localAttach{0.0f};  // attach point in body-local space (metres)

    float restLength = 0.6f;      // suspension travel at full extension (m)

    float radius     = 0.35f;     // wheel radius (m)

    // Live telemetry (updated each physics substep):

    float compression = 0.0f;     // current spring compression (m)

    bool  grounded    = false;    // ray found ground within reach

    float load        = 0.0f;     // last applied suspension force (N)

};



// Flight phases. The first three values are preserved for the legacy two-phase
// flight + existing UI; the rest support the full airport-to-airport mission.
enum class FlightPhase : uint8_t {
    Idle = 0, Takeoff = 1, Cruise = 2,
    Taxi = 3, Climb = 4, Descent = 5, Approach = 6, Flare = 7, Rollout = 8, Parked = 9
};

// Per-frame control command issued by the engine's mission controller. The
// physics step executes it with real forces so the body stays a genuine
// rigid body (suspension, inertia, aero) while following a smooth flight plan.
struct FlightControl {
    enum Mode { Hold = 0, Taxi = 1, TakeoffRoll = 2, Airborne = 3, Rollout = 4 };
    int   mode             = Hold;
    float targetHeadingRad = 0.0f;
    float targetSpeedMps   = 0.0f;   // ground speed (Taxi/Rollout) or airspeed (Airborne)
    float targetAltMSL     = 0.0f;   // airborne altitude hold target
    float climbRateFF      = 0.0f;   // feed-forward vertical rate (m/s); cancels glideslope tracking lag
    float brake            = 0.0f;   // 0..1 wheel braking on the ground
    FlightPhase displayPhase = FlightPhase::Idle;   // for telemetry/HUD only
};

// Aerodynamic + control tuning for the assisted-attitude flight model.
// Translation (thrust/lift/drag/gravity) is fully force-driven through Jolt;
// orientation is steered toward the autopilot target with rate limiting, which
// keeps the model stable and predictable while still letting stalls emerge from
// the airspeed/AoA lift curve.
struct AircraftFlightConfig {
    float wingArea          = 18.0f;     // m^2 reference wing area
    float liftCurveSlope    = 5.4f;      // CL per radian (linear region)
    float stallAoADeg       = 15.0f;     // CL peaks here, then falls off (stall)
    float zeroLiftDragCD0   = 0.028f;    // parasitic drag coefficient
    float inducedDragK      = 0.05f;     // induced drag factor (k*CL^2)
    float maxThrustN        = 16000.0f;  // full-throttle thrust
    float airDensity        = 1.225f;    // kg/m^3 (sea level)
    float takeoffSpeedMps   = 28.0f;     // rotate / liftoff airspeed (Vr)
    float climbAngleDeg     = 9.0f;      // commanded pitch during initial climb
    float maxBankDeg        = 25.0f;     // bank limit in coordinated turns
    float maxPitchDeg       = 18.0f;     // commanded pitch limit in cruise
    float pitchRateGain     = 1.2f;      // P gain: pitch error -> pitch rate (1/s)
    float rollRateGain      = 1.6f;      // P gain: bank error  -> roll rate (1/s)
    float maxPitchRateDeg   = 12.0f;     // realistic pitch rate limit (deg/s)
    float maxRollRateDeg    = 35.0f;     // realistic roll rate limit (deg/s)
    float bankPerHeadingErr  = 1.4f;     // desired bank = err(rad) * this (rad/rad)

    // ── Wind + turbulence (noise-driven) ──
    bool  windEnabled         = true;
    float windSpeedMps        = 5.0f;    // steady wind speed
    float windDirDeg          = 270.0f;  // compass dir the wind blows TOWARD
    float turbulenceIntensity = 0.4f;    // 0 = calm air, 1 = rough
    float turbulenceScale     = 0.7f;    // gust frequency multiplier (~Hz)
    float gustMagnitudeMps    = 7.0f;    // peak gust airspeed at intensity 1
    float buffetRateDeg       = 9.0f;    // peak attitude buffeting (deg/s) at intensity 1
    float altitudeGain      = 0.07f;     // alt error (m) -> target climb rate (m/s)
    float maxClimbRateMps   = 9.0f;
    float pitchPerClimbErr  = 3.0f;      // climb-rate error -> pitch (deg per m/s)
    float speedThrottleGain = 0.03f;     // throttle response to speed error
    float runwayElevationMSL = 0.0f;     // ground height under the runway (MSL)
    float climbToCruiseAglM  = 60.0f;    // AGL at which takeoff -> cruise
};

// Live flight readouts for the HUD / control panel.
struct FlightTelemetry {
    FlightPhase phase        = FlightPhase::Idle;
    float airspeed           = 0.0f;   // m/s
    float altitudeMSL        = 0.0f;   // m
    float altitudeAGL        = 0.0f;   // m above runway elevation
    float verticalSpeed      = 0.0f;   // m/s
    float aoaDeg             = 0.0f;   // angle of attack
    float headingDeg         = 0.0f;   // [0,360)
    float targetHeadingDeg   = 0.0f;
    float targetAltitudeMSL  = 0.0f;
    float throttle           = 0.0f;   // [0,1]
    float pitchDeg           = 0.0f;
    float cmdPitchDeg        = 0.0f;   // commanded (target) pitch attitude
    float bankDeg            = 0.0f;
    bool  onGround           = false;
    bool  active             = false;
    // Panel-model aerodynamic readouts (populated when an AerodynamicBody is set)
    float totalLift          = 0.0f;   // N
    float totalDrag          = 0.0f;   // N
    float avgCl              = 0.0f;   // area-weighted CL
};



class PhysicsWorld {

public:

    PhysicsWorld();

    ~PhysicsWorld();



    PhysicsWorld(const PhysicsWorld&) = delete;

    PhysicsWorld& operator=(const PhysicsWorld&) = delete;



    bool init();

    void shutdown();



    void rebuildTerrain(const std::vector<float>& heights,

                        uint32_t res,

                        float worldSizeMeters,

                        float heightScale,

                        float seaLevel,

                        int meshStride);



    void step(float deltaTime);



    uint32_t spawnSphere(const glm::vec3& position, float radius = 1.0f);

    uint32_t spawnBox(const glm::vec3& position,

                      const glm::quat& rotation,

                      const glm::vec3& halfExtents,

                      bool kinematic = false);

    void setKinematicState(uint32_t bodyIndex,

                           const glm::vec3& position,

                           const glm::quat& rotation,

                           const glm::vec3& velocity);

    void destroyBody(uint32_t bodyIndex);



    void clearDynamicBodies();



    // ── Geared aircraft (dynamic fuselage + raycast spring-damper suspension) ──

    uint32_t spawnAircraftWithGear(const glm::vec3& position,

                                   const glm::quat& rotation,

                                   const glm::vec3& fuselageHalfExtents,

                                   float massKg,

                                   const std::vector<GearWheel>& wheels,

                                   float suspensionStiffness,

                                   float suspensionDamping);

    void clearAircraftGear();

    bool hasAircraftGear() const;

    int  gearWheelCount() const;

    const GearWheel& gearWheel(int i) const;



    // ── Two-phase assisted-attitude flight (drives the geared aircraft) ──

    void setFlightEnabled(bool enabled, const AircraftFlightConfig& config);

    void setFlightConfig(const AircraftFlightConfig& config);

    void setFlightTargets(float targetAltitudeMSL,
                          float targetHeadingRad,
                          float targetSpeedMps);

    // Begin the takeoff roll (Idle -> Takeoff). No-op unless a geared aircraft
    // exists and flight is enabled.
    void requestTakeoff();

    // Force back to Idle (e.g. after respawn / on the runway).
    void resetFlightToIdle();

    bool flightEnabled() const;

    FlightTelemetry flightTelemetry() const;

    // ── Mission control (full airport-to-airport demo) ──
    // When mission mode is on, applyAircraftFlight() executes the engine's
    // per-frame FlightControl command (taxi / takeoff / airborne / rollout)
    // instead of running its own two-phase autopilot.
    void setMissionMode(bool on);
    bool missionMode() const;
    void setFlightControl(const FlightControl& fc);
    // Hard-reset the dynamic aircraft body to a pose (zero velocity). Used to
    // place the plane at a gate at the start of each leg.
    void placeAircraft(const glm::vec3& position, const glm::quat& rotation);

    // ── Shape-based aerodynamics (panel/strip theory) ──
    // When a non-empty body is set, applyAircraftFlight() uses the panel model
    // for forces + moments instead of the single-point lumped model.
    void setAerodynamicBody(const AerodynamicBody& body);
    const AerodynamicBody& getAerodynamicBody() const;



    const std::vector<PhysicsBodyState>& getDynamicBodies() const;



    bool isInitialized() const { return initialized; }

    int  terrainSampleCount() const { return terrainSamples; }

    int  terrainTriangleCount() const { return terrainTris; }



private:

    struct Impl;

    std::unique_ptr<Impl> impl;

    bool initialized = false;

    int  terrainSamples = 0;

    int  terrainTris = 0;

    std::vector<PhysicsBodyState> dynamicBodies;



    void applyAircraftSuspension(float dt);

    void applyAircraftFlight(float dt);

    void applyMissionFlight(float dt);   // mission-mode control law

};


