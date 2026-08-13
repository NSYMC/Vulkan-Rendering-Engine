#include "AutonomousAircraft.h"
#include "PhysicsWorld.h"
#include "TerrainSettings.h"

#include <glm/gtc/quaternion.hpp>
#include <cmath>
#include <iostream>

namespace {

constexpr float kPi = 3.14159265358979323846f;

float wrapAngle(float a) {
    while (a > kPi) a -= 2.0f * kPi;
    while (a < -kPi) a += 2.0f * kPi;
    return a;
}

float steerHeading(float current, float target, float maxRate, float dt) {
    float diff = wrapAngle(target - current);
    float step = maxRate * dt;
    if (std::abs(diff) <= step) return target;
    return current + std::copysign(step, diff);
}

} // namespace

void AutonomousAircraft::configure(const FlightPoint& departure,
                                   const FlightPoint& destination,
                                   const TerrainSettings& settings) {
    departure_ = departure;
    destination_ = destination;
    configured = departure.enabled && destination.enabled;

    halfExtents_ = glm::vec3(
        std::max(1.0f, settings.aircraftHalfLength),
        std::max(0.25f, settings.aircraftHalfHeight),
        std::max(0.25f, settings.aircraftHalfWidth));

    takeoffSpeed_ = std::max(10.0f, settings.aircraftTakeoffSpeedMps);
    cruiseSpeed_ = std::max(takeoffSpeed_ + 5.0f, settings.aircraftCruiseSpeedMps);
    cruiseAltitudeAgl_ = std::max(30.0f, settings.aircraftCruiseAltitudeAgl);
    climbRate_ = std::max(2.0f, settings.aircraftClimbRateMps);
    descentRange_ = std::max(500.0f, settings.aircraftDestinationDistanceMeters * 0.025f);
    approachRange_ = std::min(descentRange_ * 0.75f, 2000.0f);
    glideSlopeDeg_ = glm::clamp(settings.aircraftGlideSlopeDeg, 2.0f, 6.0f);
    autoRepeat_ = settings.aircraftAutoRepeat;
    parkedWaitSec_ = std::max(1.0f, settings.aircraftParkedWaitSec);

    headingRad_ = departure_.headingRad;

    resetToDepartureRunway();
    phase_ = AutonomousFlightPhase::TakeoffRoll;
    phaseTimer_ = 0.0f;
    telemetryTimer_ = 0.0f;
    autopilotPaused_ = false;
}

bool AutonomousAircraft::spawn(PhysicsWorld& physics) {
    shutdown(physics);
    if (!configured || !physics.isInitialized()) return false;

    const glm::quat rot = orientation();
    bodyIndex_ = physics.spawnBox(position_, rot, halfExtents_, true);
    if (bodyIndex_ == UINT32_MAX) return false;

    active = true;
    applyTransform(physics);
    std::cout << "  [aircraft/temp] spawned bodyIndex=" << bodyIndex_
              << " phase=" << phaseName()
              << " pos=("
              << position_.x << "," << position_.y << "," << position_.z
              << ") dep=(" << departure_.center.x << "," << departure_.center.y
              << ") dest=(" << destination_.center.x << "," << destination_.center.y << ")\n";
    return true;
}

void AutonomousAircraft::activateWithoutPhysics() {
    if (!configured) return;
    bodyIndex_ = UINT32_MAX;
    active = true;
    std::cout << "  [aircraft/temp] running without physics body phase=" << phaseName()
              << " pos=(" << position_.x << "," << position_.y << "," << position_.z << ")\n";
}

void AutonomousAircraft::shutdown(PhysicsWorld& physics) {
    if (active && bodyIndex_ != UINT32_MAX && physics.isInitialized())
        physics.destroyBody(bodyIndex_);
    bodyIndex_ = UINT32_MAX;
    active = false;
    phase_ = AutonomousFlightPhase::Idle;
}

glm::vec3 AutonomousAircraft::runwayWorldPosition(const FlightPoint& field,
                                                  float alongRunway) const {
    const float c = std::cos(field.headingRad);
    const float s = std::sin(field.headingRad);
    return glm::vec3(
        field.center.x + alongRunway * c,
        field.elevation + halfExtents_.y,
        field.center.y + alongRunway * s);
}

glm::vec3 AutonomousAircraft::forwardWorld(float headingRad) const {
    return glm::vec3(std::cos(headingRad), 0.0f, std::sin(headingRad));
}

void AutonomousAircraft::resetToDepartureRunway() {
    const float startAlong = -departure_.halfLength * 0.82f;
    position_ = runwayWorldPosition(departure_, startAlong);
    velocity_ = glm::vec3(0.0f);
    groundSpeed_ = 0.0f;
    pitchRad_ = 0.0f;
    headingRad_ = departure_.headingRad;
}

void AutonomousAircraft::applyTransform(PhysicsWorld& physics) {
    if (!active || bodyIndex_ == UINT32_MAX) return;
    physics.setKinematicState(bodyIndex_, position_, orientation(), velocity_);
}

const char* AutonomousAircraft::phaseName() const {
    switch (phase_) {
    case AutonomousFlightPhase::Idle: return "Idle";
    case AutonomousFlightPhase::TakeoffRoll: return "Takeoff roll";
    case AutonomousFlightPhase::Climb: return "Climb";
    case AutonomousFlightPhase::Cruise: return "Cruise";
    case AutonomousFlightPhase::Descent: return "Descent";
    case AutonomousFlightPhase::FinalApproach: return "Final approach";
    case AutonomousFlightPhase::LandingRollout: return "Landing rollout";
    case AutonomousFlightPhase::Parked: return "Parked";
    default: return "?";
    }
}

float AutonomousAircraft::speedMps() const {
    return glm::length(velocity_);
}

float AutonomousAircraft::distanceToDestinationM() const {
    glm::vec2 d = destination_.center - glm::vec2(position_.x, position_.z);
    return glm::length(d);
}

float AutonomousAircraft::altitudeAglM() const {
    return position_.y - departure_.elevation;
}

float AutonomousAircraft::headingErrorDeg() const {
    glm::vec2 toDest = destination_.center - glm::vec2(position_.x, position_.z);
    if (glm::length(toDest) < 1.0f) return 0.0f;
    float bearing = std::atan2(toDest.y, toDest.x);
    return glm::degrees(std::abs(wrapAngle(bearing - headingRad_)));
}

glm::quat AutonomousAircraft::orientation() const {
    return glm::angleAxis(headingRad_, glm::vec3(0.0f, 1.0f, 0.0f)) *
           glm::angleAxis(pitchRad_, glm::vec3(0.0f, 0.0f, 1.0f));
}

AircraftRenderState AutonomousAircraft::renderState() const {
    AircraftRenderState rs;
    rs.position = position_;
    rs.rotation = orientation();
    rs.halfExtents = halfExtents_;
    rs.visible = active;
    return rs;
}

void AutonomousAircraft::setDestinationElevation(float elevation) {
    destination_.elevation = elevation;
}

void AutonomousAircraft::advancePhase(float dt) {
    phaseTimer_ += dt;

    const glm::vec3 destThreshold =
        runwayWorldPosition(destination_, -destination_.halfLength * 0.82f);
    const glm::vec2 toDest = destination_.center - glm::vec2(position_.x, position_.z);
    const float distHoriz = glm::length(toDest);
    const float targetHeading =
        distHoriz > 1.0f ? std::atan2(toDest.y, toDest.x) : headingRad_;

    const float cruiseAlt =
        departure_.elevation + cruiseAltitudeAgl_;
    const float glideTan = std::tan(glm::radians(glideSlopeDeg_));

    switch (phase_) {
    case AutonomousFlightPhase::TakeoffRoll: {
        headingRad_ = steerHeading(headingRad_, targetHeading, 0.15f, dt);
        groundSpeed_ = glm::mix(groundSpeed_, takeoffSpeed_, dt * 0.35f);
        const glm::vec3 fwd = forwardWorld(headingRad_);
        velocity_ = fwd * groundSpeed_;
        position_ += velocity_ * dt;
        position_.y = departure_.elevation + halfExtents_.y;

        const float rolled =
            glm::length(glm::vec2(position_.x, position_.z) -
                        glm::vec2(departure_.center.x, departure_.center.y));
        if (groundSpeed_ >= takeoffSpeed_ * 0.92f || rolled > 350.0f) {
            phase_ = AutonomousFlightPhase::Climb;
            phaseTimer_ = 0.0f;
        }
        break;
    }
    case AutonomousFlightPhase::Climb: {
        headingRad_ = steerHeading(headingRad_, targetHeading, 0.25f, dt);
        groundSpeed_ = glm::mix(groundSpeed_, cruiseSpeed_, dt * 0.2f);
        pitchRad_ = glm::mix(pitchRad_, glm::radians(12.0f), dt * 0.5f);
        const glm::vec3 fwd = forwardWorld(headingRad_);
        velocity_ = fwd * groundSpeed_;
        velocity_.y = climbRate_;
        position_ += velocity_ * dt;

        if (position_.y >= cruiseAlt - 5.0f) {
            phase_ = AutonomousFlightPhase::Cruise;
            phaseTimer_ = 0.0f;
            pitchRad_ = glm::radians(3.0f);
        }
        break;
    }
    case AutonomousFlightPhase::Cruise: {
        headingRad_ = steerHeading(headingRad_, targetHeading, 0.18f, dt);
        pitchRad_ = glm::mix(pitchRad_, glm::radians(2.0f), dt * 0.4f);
        const glm::vec3 fwd = forwardWorld(headingRad_);
        velocity_ = fwd * cruiseSpeed_;
        position_ += velocity_ * dt;
        position_.y = glm::mix(position_.y, cruiseAlt, dt * 0.5f);
        velocity_.y = (cruiseAlt - position_.y) * 2.0f;

        if (distHoriz < descentRange_)
            phase_ = AutonomousFlightPhase::Descent;
        break;
    }
    case AutonomousFlightPhase::Descent: {
        headingRad_ = steerHeading(headingRad_, targetHeading, 0.22f, dt);
        const float altTarget = std::max(
            destination_.elevation + halfExtents_.y + 2.0f,
            destThreshold.y + distHoriz * glideTan);
        position_.y = glm::mix(position_.y, altTarget, dt * 0.35f);
        const glm::vec3 fwd = forwardWorld(headingRad_);
        velocity_ = fwd * cruiseSpeed_;
        velocity_.y = (altTarget - position_.y) * 1.5f;
        position_ += velocity_ * dt;
        pitchRad_ = glm::mix(pitchRad_, glm::radians(-4.0f), dt * 0.3f);

        if (distHoriz < approachRange_)
            phase_ = AutonomousFlightPhase::FinalApproach;
        break;
    }
    case AutonomousFlightPhase::FinalApproach: {
        headingRad_ = steerHeading(headingRad_, destination_.headingRad, 0.3f, dt);
        const float altTarget =
            destination_.elevation + halfExtents_.y +
            std::max(2.0f, distHoriz * glideTan);
        position_.y = glm::mix(position_.y, altTarget, dt * 0.55f);
        const float approachSpeed = glm::mix(cruiseSpeed_, takeoffSpeed_, 0.45f);
        const glm::vec3 fwd = forwardWorld(headingRad_);
        velocity_ = fwd * approachSpeed;
        velocity_.y = (altTarget - position_.y) * 2.0f;
        position_ += velocity_ * dt;
        pitchRad_ = glm::mix(pitchRad_, glm::radians(-glideSlopeDeg_), dt * 0.4f);

        const float alongDest = (position_.x - destination_.center.x) * std::cos(destination_.headingRad) +
                                (position_.z - destination_.center.y) * std::sin(destination_.headingRad);
        if (position_.y <= destination_.elevation + halfExtents_.y + 1.5f &&
            alongDest >= -destination_.halfLength * 0.82f) {
            phase_ = AutonomousFlightPhase::LandingRollout;
            phaseTimer_ = 0.0f;
            pitchRad_ = 0.0f;
            groundSpeed_ = glm::length(glm::vec2(velocity_.x, velocity_.z));
            position_.y = destination_.elevation + halfExtents_.y;
        }
        break;
    }
    case AutonomousFlightPhase::LandingRollout: {
        headingRad_ = steerHeading(headingRad_, destination_.headingRad, 0.2f, dt);
        groundSpeed_ = glm::max(0.0f, groundSpeed_ - dt * 8.0f);
        const glm::vec3 fwd = forwardWorld(headingRad_);
        velocity_ = fwd * groundSpeed_;
        position_ += velocity_ * dt;
        position_.y = destination_.elevation + halfExtents_.y;

        if (groundSpeed_ < 2.0f) {
            phase_ = AutonomousFlightPhase::Parked;
            phaseTimer_ = 0.0f;
            velocity_ = glm::vec3(0.0f);
        }
        break;
    }
    case AutonomousFlightPhase::Parked: {
        position_ = runwayWorldPosition(destination_, destination_.halfLength * 0.75f);
        velocity_ = glm::vec3(0.0f);
        groundSpeed_ = 0.0f;
        pitchRad_ = 0.0f;
        headingRad_ = destination_.headingRad;

        if (autoRepeat_ && phaseTimer_ >= parkedWaitSec_) {
            resetToDepartureRunway();
            phase_ = AutonomousFlightPhase::TakeoffRoll;
            phaseTimer_ = 0.0f;
            std::cout << "  [aircraft/temp] restarting flight to destination\n";
        }
        break;
    }
    default:
        break;
    }
}

void AutonomousAircraft::update(float deltaTime, PhysicsWorld& physics) {
    if (!active) return;
    deltaTime = glm::clamp(deltaTime, 0.001f, 0.05f);
    if (!autopilotPaused_) {
        advancePhase(deltaTime);
        if (physics.isInitialized() && bodyIndex_ != UINT32_MAX)
            applyTransform(physics);
    }

    // ── 1 Hz console telemetry ────────────────────────────────
    telemetryTimer_ += deltaTime;
    if (telemetryTimer_ >= 1.0f) {
        telemetryTimer_ = 0.0f;
        std::cout << "[aircraft] phase=" << phaseName()
                  << " pos=(" << position_.x << "," << position_.y << "," << position_.z << ")"
                  << " speed=" << speedMps()
                  << " altAGL=" << altitudeAglM()
                  << " dist=" << distanceToDestinationM()
                  << " headingErr=" << headingErrorDeg()
                  << (autopilotPaused_ ? " [PAUSED]" : "")
                  << "\n";
    }
}
