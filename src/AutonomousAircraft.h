// TEMPORARY: rectangular-prism aircraft — autonomous takeoff, cruise, landing.
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cstdint>
#include <string>

struct TerrainSettings;
class PhysicsWorld;

// Lightweight takeoff/landing endpoint for the temporary autonomous aircraft
// (replaces the removed airfield system). Positions are world XZ + ground
// elevation; halfLength is the usable runway half-extent for roll positioning.
struct FlightPoint {
    glm::vec2 center{0.0f};
    float elevation = 0.0f;
    float headingRad = 0.0f;
    float halfLength = 600.0f;
    bool enabled = false;
};

// Everything the renderer needs to draw the aircraft in world space.
struct AircraftRenderState {
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 halfExtents{6.0f, 1.0f, 1.5f};
    bool      visible = false;
};

enum class AutonomousFlightPhase {
    Idle,
    TakeoffRoll,
    Climb,
    Cruise,
    Descent,
    FinalApproach,
    LandingRollout,
    Parked,
};

class AutonomousAircraft {
public:
    void configure(const FlightPoint& departure,
                   const FlightPoint& destination,
                   const TerrainSettings& settings);

    bool spawn(PhysicsWorld& physics);
    void activateWithoutPhysics();
    void shutdown(PhysicsWorld& physics);

    void update(float deltaTime, PhysicsWorld& physics);

    bool isActive() const { return active; }
    AutonomousFlightPhase phase() const { return phase_; }
    const char* phaseName() const;
    glm::vec3 position() const { return position_; }
    glm::vec3 velocity() const { return velocity_; }
    glm::quat orientation() const;
    float speedMps() const;
    float distanceToDestinationM() const;
    float altitudeAglM() const;       // height above departure-field reference
    float headingErrorDeg() const;    // current heading vs bearing to destination
    uint32_t bodyIndex() const { return bodyIndex_; }
    bool hasPhysicsBody() const { return bodyIndex_ != UINT32_MAX; }
    AircraftRenderState renderState() const;

    // Deferred: once the destination region streams in, update its true elevation.
    void setDestinationElevation(float elevation);

    // Debug: freeze/unfreeze the autopilot state machine.
    void setAutopilotPaused(bool paused) { autopilotPaused_ = paused; }

private:
    void resetToDepartureRunway();
    glm::vec3 runwayWorldPosition(const FlightPoint& field, float alongRunway) const;
    glm::vec3 forwardWorld(float headingRad) const;
    void applyTransform(PhysicsWorld& physics);
    void advancePhase(float dt);

    FlightPoint departure_{};
    FlightPoint destination_{};
    bool configured = false;
    bool active = false;
    uint32_t bodyIndex_ = UINT32_MAX;

    AutonomousFlightPhase phase_ = AutonomousFlightPhase::Idle;
    glm::vec3 position_{0.0f};
    glm::vec3 velocity_{0.0f};
    float headingRad_ = 0.0f;
    float pitchRad_ = 0.0f;
    float groundSpeed_ = 0.0f;
    float phaseTimer_ = 0.0f;
    float repeatCooldown_ = 0.0f;

    glm::vec3 halfExtents_{6.0f, 1.0f, 1.5f};
    float takeoffSpeed_ = 28.0f;
    float cruiseSpeed_ = 55.0f;
    float cruiseAltitudeAgl_ = 120.0f;
    float climbRate_ = 12.0f;
    float descentRange_ = 2500.0f;
    float approachRange_ = 1800.0f;
    float glideSlopeDeg_ = 3.0f;
    bool autoRepeat_ = true;
    float parkedWaitSec_ = 4.0f;
    bool autopilotPaused_ = false;
    float telemetryTimer_ = 0.0f;
};
