// ══════════════════════════════════════════════════════════════
//  PhysicsWorld — Jolt Physics 3D (terrain heightfield + dynamics)
// ══════════════════════════════════════════════════════════════

#include "PhysicsWorld.h"

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <cstdarg>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <thread>

JPH_SUPPRESS_WARNINGS

using namespace JPH;
using namespace JPH::literals;

namespace PhysicsLayers {
    static constexpr ObjectLayer NON_MOVING = 0;
    static constexpr ObjectLayer MOVING     = 1;
    static constexpr ObjectLayer NUM_LAYERS = 2;
}

class ObjectLayerPairFilterImpl : public ObjectLayerPairFilter {
public:
    bool ShouldCollide(ObjectLayer a, ObjectLayer b) const override {
        switch (a) {
        case PhysicsLayers::NON_MOVING: return b == PhysicsLayers::MOVING;
        case PhysicsLayers::MOVING:     return true;
        default: JPH_ASSERT(false); return false;
        }
    }
};

namespace BroadPhaseLayers {
    static constexpr BroadPhaseLayer NON_MOVING(0);
    static constexpr BroadPhaseLayer MOVING(1);
    static constexpr uint NUM_LAYERS = 2;
}

class BPLayerInterfaceImpl final : public BroadPhaseLayerInterface {
public:
    BPLayerInterfaceImpl() {
        mObjectToBroadPhase[PhysicsLayers::NON_MOVING] = BroadPhaseLayers::NON_MOVING;
        mObjectToBroadPhase[PhysicsLayers::MOVING]     = BroadPhaseLayers::MOVING;
    }
    uint GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::NUM_LAYERS; }
    BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer layer) const override {
        JPH_ASSERT(layer < PhysicsLayers::NUM_LAYERS);
        return mObjectToBroadPhase[layer];
    }
private:
    BroadPhaseLayer mObjectToBroadPhase[PhysicsLayers::NUM_LAYERS];
};

class ObjectVsBroadPhaseLayerFilterImpl : public ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(ObjectLayer layer1, BroadPhaseLayer layer2) const override {
        switch (layer1) {
        case PhysicsLayers::NON_MOVING: return layer2 == BroadPhaseLayers::MOVING;
        case PhysicsLayers::MOVING:     return true;
        default: JPH_ASSERT(false); return false;
        }
    }
};

// Suspension raycasts should only hit static terrain, never the aircraft itself.
class TerrainRayObjectFilter : public ObjectLayerFilter {
public:
    bool ShouldCollide(ObjectLayer layer) const override {
        return layer == PhysicsLayers::NON_MOVING;
    }
};
class TerrainRayBroadPhaseFilter : public BroadPhaseLayerFilter {
public:
    bool ShouldCollide(BroadPhaseLayer layer) const override {
        return layer == BroadPhaseLayers::NON_MOVING;
    }
};

static void TraceCallback(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    std::cout << "[Jolt] " << buf << std::endl;
}

#ifdef JPH_ENABLE_ASSERTS
static bool AssertFailedCallback(const char* expr, const char* msg,
                               const char* file, uint line) {
    std::cerr << "[Jolt assert] " << file << ":" << line << " (" << expr << ") "
              << (msg ? msg : "") << std::endl;
    return true;
}
#endif

static bool isPowerOfTwo(uint32_t v) {
    return v >= 2 && (v & (v - 1)) == 0;
}

struct PhysicsWorld::Impl {
    TempAllocatorImpl*          tempAllocator   = nullptr;
    JobSystemThreadPool*        jobSystem       = nullptr;
    PhysicsSystem*              physicsSystem   = nullptr;
    BPLayerInterfaceImpl        bpLayerInterface;
    ObjectVsBroadPhaseLayerFilterImpl objVsBpFilter;
    ObjectLayerPairFilterImpl   objPairFilter;
    BodyID                      terrainBodyId;
    std::vector<BodyID>         dynamicBodyIds;
    std::vector<float>          terrainSamples;   // kept alive for HeightFieldShape

    // Geared aircraft (single instance) — dynamic fuselage + raycast suspension.
    bool                  hasAircraft     = false;
    BodyID                aircraftBodyId;
    uint32_t              aircraftIndex   = UINT32_MAX;  // index into dynamicBodyIds
    std::vector<GearWheel> wheels;
    float                 suspStiffness   = 40000.0f;
    float                 suspDamping     = 4000.0f;

    // Two-phase assisted-attitude flight model.
    bool                  flightEnabled   = false;
    FlightPhase           flightPhase     = FlightPhase::Idle;
    AircraftFlightConfig  flightCfg;
    float                 targetAltMSL    = 120.0f;
    float                 targetHeadingRad = 0.0f;
    float                 targetSpeedMps  = 55.0f;
    float                 runwayHeadingRad = 0.0f;   // captured at takeoff start
    float                 throttle        = 0.0f;
    float                 flightTime      = 0.0f;    // accumulates for wind/turbulence noise
    float                 aircraftMass    = 1000.0f; // cached from spawn (for ground forces)
    bool                  missionMode     = false;   // engine-driven full mission
    FlightControl         control;                   // current mission command
    FlightTelemetry       telemetry;
    AerodynamicBody       aeroBody;                   // panel model (empty => single-point)

    static constexpr uint cMaxBodies            = 4096;
    static constexpr uint cMaxBodyPairs         = 65536;
    static constexpr uint cMaxContactConstraints = 10240;
};

PhysicsWorld::PhysicsWorld() : impl(std::make_unique<Impl>()) {}

PhysicsWorld::~PhysicsWorld() {
    shutdown();
}

bool PhysicsWorld::init() {
    if (initialized) return true;

    RegisterDefaultAllocator();
    Trace = TraceCallback;
    JPH_IF_ENABLE_ASSERTS(AssertFailed = AssertFailedCallback;)

    Factory::sInstance = new Factory();
    RegisterTypes();

    impl->tempAllocator = new TempAllocatorImpl(32 * 1024 * 1024);
    const int hwThreads = (int)std::thread::hardware_concurrency();
    const int jobThreads = std::max(1, hwThreads > 0 ? hwThreads - 1 : 1);
    impl->jobSystem = new JobSystemThreadPool(
        cMaxPhysicsJobs, cMaxPhysicsBarriers, (uint)jobThreads);

    impl->physicsSystem = new PhysicsSystem();
    impl->physicsSystem->Init(
        Impl::cMaxBodies, 0, Impl::cMaxBodyPairs, Impl::cMaxContactConstraints,
        impl->bpLayerInterface, impl->objVsBpFilter, impl->objPairFilter);

    initialized = true;
    std::cout << "  [physics] Jolt Physics initialized (" << jobThreads << " worker threads)\n";
    return true;
}

void PhysicsWorld::shutdown() {
    if (!initialized) return;

    clearDynamicBodies();

    if (!impl->terrainBodyId.IsInvalid()) {
        BodyInterface& bi = impl->physicsSystem->GetBodyInterface();
        bi.RemoveBody(impl->terrainBodyId);
        bi.DestroyBody(impl->terrainBodyId);
        impl->terrainBodyId = BodyID();
    }
    impl->terrainSamples.clear();

    delete impl->physicsSystem;
    impl->physicsSystem = nullptr;
    delete impl->jobSystem;
    impl->jobSystem = nullptr;
    delete impl->tempAllocator;
    impl->tempAllocator = nullptr;

    UnregisterTypes();
    delete Factory::sInstance;
    Factory::sInstance = nullptr;

    initialized = false;
    terrainSamples = 0;
    terrainTris = 0;
}

void PhysicsWorld::rebuildTerrain(const std::vector<float>& heights,
                                  uint32_t res,
                                  float worldSizeMeters,
                                  float heightScale,
                                  float seaLevel,
                                  int meshStride) {
    if (!initialized || heights.empty() || res < 2) return;

    meshStride = std::max(1, meshStride);

    // Jolt HeightFieldShape is most efficient with a power-of-two sample count.
    uint32_t sampleCount = (res - 1) / (uint32_t)meshStride + 1;
    while (sampleCount > 2 && !isPowerOfTwo(sampleCount)) {
        ++meshStride;
        sampleCount = (res - 1) / (uint32_t)meshStride + 1;
    }
    if (sampleCount < 2) sampleCount = 2;

    impl->terrainSamples.resize((size_t)sampleCount * sampleCount);
    int noCollisionCells = 0;

    for (uint32_t y = 0; y < sampleCount; ++y) {
        for (uint32_t x = 0; x < sampleCount; ++x) {
            const uint32_t sx = std::min(x * (uint32_t)meshStride, res - 1);
            const uint32_t sz = std::min(y * (uint32_t)meshStride, res - 1);
            const float worldY = heights[(size_t)sz * res + sx] * heightScale;

            float& sample = impl->terrainSamples[(size_t)y * sampleCount + x];
            if (worldY < seaLevel) {
                sample = HeightFieldShapeConstants::cNoCollisionValue;
                ++noCollisionCells;
            } else {
                sample = worldY;
            }
        }
    }

    const float cellSize = worldSizeMeters / (float)(sampleCount - 1);
    const Vec3 offset(0.0f, 0.0f, 0.0f);
    const Vec3 scale(cellSize, 1.0f, cellSize);

    HeightFieldShapeSettings hfSettings(
        impl->terrainSamples.data(),
        offset,
        scale,
        sampleCount);

    ShapeSettings::ShapeResult shapeResult = hfSettings.Create();
    if (shapeResult.HasError()) {
        std::cerr << "  [physics] heightfield shape error: "
                  << shapeResult.GetError() << std::endl;
        return;
    }

    BodyInterface& bi = impl->physicsSystem->GetBodyInterface();

    if (!impl->terrainBodyId.IsInvalid()) {
        bi.RemoveBody(impl->terrainBodyId);
        bi.DestroyBody(impl->terrainBodyId);
        impl->terrainBodyId = BodyID();
    }

    BodyCreationSettings terrainSettings(
        shapeResult.Get(),
        RVec3::sZero(),
        Quat::sIdentity(),
        EMotionType::Static,
        PhysicsLayers::NON_MOVING);

    Body* terrainBody = bi.CreateBody(terrainSettings);
    if (!terrainBody) {
        std::cerr << "  [physics] failed to create terrain body\n";
        return;
    }

    impl->terrainBodyId = terrainBody->GetID();
    bi.AddBody(impl->terrainBodyId, EActivation::DontActivate);

    terrainSamples = (int)sampleCount;
    terrainTris = (int)((sampleCount - 1) * (sampleCount - 1) * 2);

    std::cout << "  [physics] heightfield collider: " << sampleCount << "x" << sampleCount
              << " samples, cell=" << cellSize << "m, stride=" << meshStride
              << ", heightScale=" << heightScale
              << ", water holes=" << noCollisionCells << "\n";
}

void PhysicsWorld::step(float deltaTime) {
    if (!initialized || deltaTime <= 0.0f) return;

    // Fixed 1/60 s substeps — up to 8 per frame to limit tunneling.
    const float stepSize = 1.0f / 60.0f;
    int collisionSteps = (int)std::ceil(deltaTime / stepSize);
    collisionSteps = std::clamp(collisionSteps, 1, 8);
    const float subStep = deltaTime / (float)collisionSteps;

    for (int i = 0; i < collisionSteps; ++i) {
        applyAircraftSuspension(subStep);  // spring-damper forces before integration
        applyAircraftFlight(subStep);      // thrust/lift/drag + attitude steering
        impl->physicsSystem->Update(subStep, 1, impl->tempAllocator, impl->jobSystem);
    }

    BodyInterface& bi = impl->physicsSystem->GetBodyInterface();
    dynamicBodies.resize(impl->dynamicBodyIds.size());
    for (size_t i = 0; i < impl->dynamicBodyIds.size(); ++i) {
        const BodyID& id = impl->dynamicBodyIds[i];
        PhysicsBodyState& state = dynamicBodies[i];
        if (id.IsInvalid()) continue;

        RVec3 pos = bi.GetCenterOfMassPosition(id);
        Vec3  vel = bi.GetLinearVelocity(id);
        Vec3  avel = bi.GetAngularVelocity(id);
        Quat  rot = bi.GetRotation(id);
        state.position = glm::vec3((float)pos.GetX(), (float)pos.GetY(), (float)pos.GetZ());
        state.velocity = glm::vec3((float)vel.GetX(), (float)vel.GetY(), (float)vel.GetZ());
        state.angularVelocity = glm::vec3(avel.GetX(), avel.GetY(), avel.GetZ());
        state.rotation = glm::quat(rot.GetW(), rot.GetX(), rot.GetY(), rot.GetZ());
        state.active   = bi.IsActive(id);
    }
}

void PhysicsWorld::applyAircraftSuspension(float /*dt*/) {
    if (!initialized || !impl->hasAircraft || impl->aircraftBodyId.IsInvalid())
        return;

    BodyInterface& bi = impl->physicsSystem->GetBodyInterface();
    const NarrowPhaseQuery& npq = impl->physicsSystem->GetNarrowPhaseQuery();
    const BodyID id = impl->aircraftBodyId;

    const RMat44 transform = bi.GetWorldTransform(id);
    const RVec3  com       = bi.GetCenterOfMassPosition(id);
    const Vec3   linVel    = bi.GetLinearVelocity(id);
    const Vec3   angVel    = bi.GetAngularVelocity(id);
    const Vec3   up(0.0f, 1.0f, 0.0f);

    TerrainRayBroadPhaseFilter bpFilter;
    TerrainRayObjectFilter     objFilter;

    for (GearWheel& w : impl->wheels) {
        const float reach = w.restLength + w.radius;
        const RVec3 worldAttach = transform * Vec3(w.localAttach.x, w.localAttach.y, w.localAttach.z);

        RRayCast ray(worldAttach, Vec3(0.0f, -reach, 0.0f));
        RayCastResult hit;
        bool found = npq.CastRay(ray, hit, bpFilter, objFilter);

        if (!found) {
            w.grounded = false;
            w.compression = 0.0f;
            w.load = 0.0f;
            continue;
        }

        const float hitDist = reach * hit.mFraction;
        float compression = reach - hitDist;
        compression = std::clamp(compression, 0.0f, w.restLength);
        if (compression <= 0.0f) {
            w.grounded = false;
            w.compression = 0.0f;
            w.load = 0.0f;
            continue;
        }

        // Contact point and its world-space velocity (linear + angular).
        const RVec3 contact = worldAttach + Vec3(0.0f, -hitDist, 0.0f);
        const Vec3  r = Vec3(contact - com);
        const Vec3  pointVel = linVel + angVel.Cross(r);
        const float vUp = pointVel.Dot(up);

        float force = impl->suspStiffness * compression - impl->suspDamping * vUp;
        force = std::max(0.0f, force);                 // suspension only pushes
        force = std::min(force, impl->suspStiffness * w.restLength * 4.0f);  // safety cap

        bi.AddForce(id, up * force, contact);

        w.grounded = true;
        w.compression = compression;
        w.load = force;
    }
}

namespace {
inline float wrapToPi(float a) {
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kTwoPi = 2.0f * kPi;
    a = std::fmod(a + kPi, kTwoPi);
    if (a < 0.0f) a += kTwoPi;
    return a - kPi;
}

// Smooth band-limited pseudo-noise in [-1,1] built from a few incommensurate
// sine harmonics (cheap fractal "fBm"). `seed` decorrelates separate channels
// so wind axes / buffeting don't move in lock-step.
inline float turbNoise(float t, float seed) {
    float n = std::sin(t * 1.00f + seed * 12.9898f) * 0.50f
            + std::sin(t * 2.17f + seed * 39.346f) * 0.27f
            + std::sin(t * 4.93f + seed * 73.197f) * 0.15f
            + std::sin(t * 9.71f + seed * 91.553f) * 0.08f;
    return std::clamp(n, -1.0f, 1.0f);
}
} // namespace

// Two-phase assisted-attitude flight model (report §5). Translation is fully
// force-driven (thrust, lift via an AoA lift curve, induced+parasitic drag,
// plus Jolt gravity), so energy state and stalls emerge naturally. Orientation
// is steered toward the autopilot's desired attitude by commanding the body's
// angular velocity with rate limiting — stable and predictable on first run.
void PhysicsWorld::applyAircraftFlight(float dt) {
    if (!initialized || !impl->flightEnabled || !impl->hasAircraft ||
        impl->aircraftBodyId.IsInvalid())
        return;

    if (impl->missionMode) { applyMissionFlight(dt); return; }

    BodyInterface& bi = impl->physicsSystem->GetBodyInterface();
    const BodyID id = impl->aircraftBodyId;
    const AircraftFlightConfig& c = impl->flightCfg;

    const Quat  q   = bi.GetRotation(id);
    const RVec3 com = bi.GetCenterOfMassPosition(id);
    const Vec3  v   = bi.GetLinearVelocity(id);   // ground velocity

    const Vec3 fwd      = q * Vec3(1.0f, 0.0f, 0.0f);
    const Vec3 upBody   = q * Vec3(0.0f, 1.0f, 0.0f);
    const Vec3 rightBody= q * Vec3(0.0f, 0.0f, 1.0f);
    const Vec3 worldUp(0.0f, 1.0f, 0.0f);

    const float kDeg2Rad = 0.01745329252f;
    const float altMSL  = (float)com.GetY();
    const float altAGL  = altMSL - c.runwayElevationMSL;
    const float vSpeed  = v.GetY();               // climb rate is ground-frame

    // ── Wind & turbulence: the aircraft flies relative to a moving air mass. ──
    //   A plane doesn't feel ground velocity, it feels velocity relative to the
    //   air. Steady wind + smooth noise gusts perturb that relative airflow, so
    //   lift / AoA / drag (and thus the whole trajectory) react realistically.
    impl->flightTime += (dt > 0.0f ? dt : 0.0f);
    Vec3 wind(0.0f, 0.0f, 0.0f);
    float turbIntensity = 0.0f;
    if (c.windEnabled) {
        const float wd = c.windDirDeg * kDeg2Rad;
        const Vec3 windBase(std::cos(wd) * c.windSpeedMps, 0.0f, std::sin(wd) * c.windSpeedMps);
        // Turbulence fades out near the ground (ground friction calms the air).
        const float groundFade = std::clamp(altAGL / 30.0f, 0.0f, 1.0f);
        turbIntensity = std::clamp(c.turbulenceIntensity, 0.0f, 1.0f) * (0.25f + 0.75f * groundFade);
        const float tt = impl->flightTime * c.turbulenceScale;
        const Vec3 gust(turbNoise(tt, 1.3f),
                        turbNoise(tt, 5.7f) * 1.2f,   // stronger vertical bumps
                        turbNoise(tt, 9.1f));
        wind = windBase + gust * (turbIntensity * c.gustMagnitudeMps);
    }
    const Vec3 vAir = v - wind;                    // airspeed vector
    const float speed = vAir.Length();             // true airspeed (drives aero)

    // Angle of attack: airflow seen by the wing in the body's pitch plane.
    const float uFwd = vAir.Dot(fwd);
    const float wUp  = vAir.Dot(upBody);
    const float aoa  = (speed > 1.0f) ? std::atan2(-wUp, uFwd) : 0.0f;

    // ── Phase / autopilot command ────────────────────────────────────────
    const float deg2rad = 0.01745329252f;
    const float kG = 9.80665f;
    float desiredHeading = impl->runwayHeadingRad;
    float desiredPitch   = 0.0f;   // radians

    if (impl->flightPhase == FlightPhase::Takeoff) {
        impl->throttle = 1.0f;
        if (speed >= c.takeoffSpeedMps)
            desiredPitch = c.climbAngleDeg * deg2rad;     // rotate & climb out
        if (altAGL >= c.climbToCruiseAglM)
            impl->flightPhase = FlightPhase::Cruise;       // hand off to cruise
    } else if (impl->flightPhase == FlightPhase::Cruise) {
        desiredHeading = impl->targetHeadingRad;
        // Altitude hold -> target climb rate -> commanded pitch.
        const float altErr  = impl->targetAltMSL - altMSL;
        const float desClimb = std::clamp(altErr * c.altitudeGain,
                                          -c.maxClimbRateMps, c.maxClimbRateMps);
        const float climbErr = desClimb - vSpeed;
        desiredPitch = std::clamp(c.pitchPerClimbErr * climbErr * deg2rad,
                                  -c.maxPitchDeg * deg2rad, c.maxPitchDeg * deg2rad);
        // Speed hold on throttle.
        impl->throttle = std::clamp(0.55f + c.speedThrottleGain *
                                    (impl->targetSpeedMps - speed), 0.0f, 1.0f);
    } else {
        impl->throttle = 0.0f;   // Idle: park on the runway
    }

    // ── Aerodynamic + propulsive forces (applied at COM) ─────────────────
    //  When a shape-based AerodynamicBody is present, sum per-panel strip
    //  forces (richer, shape-sensitive, with telemetry); otherwise fall back
    //  to the single-point lumped lift/drag model. In both cases the attitude
    //  is steered by the assisted-attitude block below (SetAngularVelocity),
    //  so we apply translational aero force only — not aero torque.
    Vec3 force(0.0f, 0.0f, 0.0f);
    if (!impl->aeroBody.empty() && speed > 0.5f) {
        // Air velocity in the body frame (Jolt: body->world is q, so world->body is q^-1).
        const Vec3 vAirBody_j = q.Conjugated() * vAir;
        const glm::vec3 vAirBody(vAirBody_j.GetX(), vAirBody_j.GetY(), vAirBody_j.GetZ());
        const AeroResult ar = impl->aeroBody.compute(vAirBody, c.airDensity);
        const Vec3 aeroForceBody(ar.force.x, ar.force.y, ar.force.z);
        force += q * aeroForceBody;                        // body -> world
        impl->telemetry.totalLift = ar.totalLift;
        impl->telemetry.totalDrag = ar.totalDrag;
        impl->telemetry.avgCl     = ar.avgCl;
    } else {
        const float qDyn = 0.5f * c.airDensity * speed * speed;
        const float aoaStall = c.stallAoADeg * deg2rad;
        const float aoaAbs = std::abs(aoa);
        const float clPeak = c.liftCurveSlope * aoaStall;
        float cl;
        if (aoaAbs <= aoaStall) {
            cl = c.liftCurveSlope * aoa;                   // linear region
        } else {
            const float t = std::clamp((aoaAbs - aoaStall) / aoaStall, 0.0f, 1.0f);
            const float sign = (aoa >= 0.0f) ? 1.0f : -1.0f;
            cl = sign * clPeak * (1.0f - 0.6f * t);        // post-stall falloff
        }
        const float cd = c.zeroLiftDragCD0 + c.inducedDragK * cl * cl;
        if (speed > 0.5f) {
            const Vec3 vDir = vAir * (1.0f / speed);
            force += upBody * (qDyn * c.wingArea * cl);    // lift (along body up)
            force += vDir * (-qDyn * c.wingArea * cd);     // drag (opposes relative wind)
        }
        impl->telemetry.totalLift = qDyn * c.wingArea * cl;
        impl->telemetry.totalDrag = qDyn * c.wingArea * cd;
        impl->telemetry.avgCl     = cl;
    }
    force += fwd * (impl->throttle * c.maxThrustN);        // thrust
    bi.AddForce(id, force);                                 // at COM (no torque)

    // ── Assisted attitude: realistic coordinated-turn flight ─────────────
    // A real aircraft does NOT yaw straight to a new heading: it banks, and the
    // horizontal component of lift turns it at rate ψ̇ = g·tan(bank)/V. We
    // reproduce that here — bank is commanded from heading error, and the yaw
    // (heading change) rate is derived from the *current* bank, so turns ease
    // in and out gradually. Roll and pitch are rate-limited like a real plane.
    const float curPitch = std::asin(std::clamp(fwd.GetY(), -1.0f, 1.0f));
    const float curBank  = std::atan2(-rightBody.GetY(), std::max(0.05f, upBody.GetY()));
    const float curHdg   = std::atan2(fwd.GetZ(), fwd.GetX());
    const bool  onGround = (altAGL < 2.0f);

    // Desired bank from heading error (only while airborne; wings level on ground).
    float desiredBank = 0.0f;
    if (!onGround && impl->flightPhase != FlightPhase::Idle) {
        const float hdgErr = wrapToPi(desiredHeading - curHdg);
        desiredBank = std::clamp(hdgErr * c.bankPerHeadingErr,
                                 -c.maxBankDeg * deg2rad, c.maxBankDeg * deg2rad);
    }

    const float maxPitchRate = c.maxPitchRateDeg * deg2rad;
    const float maxRollRate  = c.maxRollRateDeg  * deg2rad;
    const float pitchRate = std::clamp((desiredPitch - curPitch) * c.pitchRateGain,
                                       -maxPitchRate, maxPitchRate);
    const float rollRate  = std::clamp((desiredBank - curBank) * c.rollRateGain,
                                       -maxRollRate, maxRollRate);

    // Coordinated-turn yaw rate from current bank (heading only moves when banked).
    // ω about +worldUp DECREASES heading (ω×v), so negate to bank-right => turn-right.
    float yawRate = 0.0f;
    if (!onGround && speed > 5.0f)
        yawRate = -(kG * std::tan(curBank)) / std::max(speed, 20.0f);

    // Body-axis rates: pitch about body right, roll about body forward, yaw vertical.
    Vec3 omega = rightBody * pitchRate + fwd * rollRate + worldUp * yawRate;

    // Turbulence buffeting: small noisy attitude jitter on top of the autopilot
    // command (the autopilot then fights to recover, just like a real pilot).
    if (!onGround && turbIntensity > 0.0f && speed > 8.0f) {
        const float bt = impl->flightTime * c.turbulenceScale * 2.3f;  // faster than gusts
        const float buf = c.buffetRateDeg * kDeg2Rad * turbIntensity;
        omega += rightBody * (turbNoise(bt, 2.1f) * buf)        // pitch jitter
               + fwd       * (turbNoise(bt, 6.4f) * buf * 1.3f) // roll jitter (most felt)
               + worldUp   * (turbNoise(bt, 8.8f) * buf * 0.6f);// yaw jitter
    }
    bi.SetAngularVelocity(id, omega);

    // ── Telemetry snapshot ───────────────────────────────────────────────
    const float rad2deg = 57.2957795f;
    float hdgDeg = curHdg * rad2deg; if (hdgDeg < 0.0f) hdgDeg += 360.0f;
    float tgtHdgDeg = impl->targetHeadingRad * rad2deg;
    if (tgtHdgDeg < 0.0f) tgtHdgDeg += 360.0f;
    impl->telemetry.phase           = impl->flightPhase;
    impl->telemetry.airspeed        = speed;
    impl->telemetry.altitudeMSL     = altMSL;
    impl->telemetry.altitudeAGL     = altAGL;
    impl->telemetry.verticalSpeed   = vSpeed;
    impl->telemetry.aoaDeg          = aoa * rad2deg;
    impl->telemetry.headingDeg      = hdgDeg;
    impl->telemetry.targetHeadingDeg= tgtHdgDeg;
    impl->telemetry.targetAltitudeMSL = impl->targetAltMSL;
    impl->telemetry.throttle        = impl->throttle;
    impl->telemetry.pitchDeg        = curPitch * rad2deg;
    impl->telemetry.bankDeg         = curBank * rad2deg;
    impl->telemetry.onGround        = (altAGL < 2.0f);
    impl->telemetry.active          = true;
}

// Mission control law: executes the engine's per-frame FlightControl command
// with real forces. Translation is fully force-driven (thrust/aero/gravity +
// raycast suspension on the ground); attitude is steered with rate-limited
// angular velocity. Ground modes (taxi/rollout) use nose-wheel steering and a
// computed longitudinal force so taxi speed is regulated and braking is smooth.
void PhysicsWorld::applyMissionFlight(float dt) {
    BodyInterface& bi = impl->physicsSystem->GetBodyInterface();
    const BodyID id = impl->aircraftBodyId;
    const AircraftFlightConfig& c = impl->flightCfg;
    const FlightControl& fc = impl->control;

    const Quat  q   = bi.GetRotation(id);
    const RVec3 com = bi.GetCenterOfMassPosition(id);
    const Vec3  v   = bi.GetLinearVelocity(id);

    const Vec3 fwd       = q * Vec3(1.0f, 0.0f, 0.0f);
    const Vec3 upBody    = q * Vec3(0.0f, 1.0f, 0.0f);
    const Vec3 rightBody = q * Vec3(0.0f, 0.0f, 1.0f);
    const Vec3 worldUp(0.0f, 1.0f, 0.0f);

    const float deg2rad = 0.01745329252f;
    const float kG = 9.80665f;
    const float altMSL = (float)com.GetY();
    const float altAGL = altMSL - c.runwayElevationMSL;
    const float vSpeed = v.GetY();

    // Horizontal forward direction (for ground propulsion / braking).
    Vec3 fwdHoriz(fwd.GetX(), 0.0f, fwd.GetZ());
    if (fwdHoriz.LengthSq() > 1e-4f) fwdHoriz = fwdHoriz.Normalized();
    Vec3 horiz(v.GetX(), 0.0f, v.GetZ());
    const float groundSpeed = horiz.Length();

    // Wind / airspeed (drives aero).
    impl->flightTime += (dt > 0.0f ? dt : 0.0f);
    Vec3 wind(0.0f, 0.0f, 0.0f);
    float turbIntensity = 0.0f;
    if (c.windEnabled) {
        const float wd = c.windDirDeg * deg2rad;
        const Vec3 windBase(std::cos(wd) * c.windSpeedMps, 0.0f, std::sin(wd) * c.windSpeedMps);
        const float groundFade = std::clamp(altAGL / 30.0f, 0.0f, 1.0f);
        turbIntensity = std::clamp(c.turbulenceIntensity, 0.0f, 1.0f) * (0.25f + 0.75f * groundFade);
        const float tt = impl->flightTime * c.turbulenceScale;
        const Vec3 gust(turbNoise(tt, 1.3f), turbNoise(tt, 5.7f) * 1.2f, turbNoise(tt, 9.1f));
        wind = windBase + gust * (turbIntensity * c.gustMagnitudeMps);
    }
    const Vec3 vAir = v - wind;
    const float speed = vAir.Length();
    const float uFwd = vAir.Dot(fwd);
    const float wUp  = vAir.Dot(upBody);
    const float aoa  = (speed > 1.0f) ? std::atan2(-wUp, uFwd) : 0.0f;

    const float curPitch = std::asin(std::clamp(fwd.GetY(), -1.0f, 1.0f));
    const float curBank  = std::atan2(-rightBody.GetY(), std::max(0.05f, upBody.GetY()));
    const float curHdg   = std::atan2(fwd.GetZ(), fwd.GetX());

    const bool airborne = (fc.mode == FlightControl::Airborne);
    const bool onGroundMode = !airborne;

    // ── Longitudinal control & throttle ──────────────────────────────────
    Vec3 force(0.0f, 0.0f, 0.0f);
    float desiredPitch = 0.0f;
    float desiredHeading = fc.targetHeadingRad;

    if (fc.mode == FlightControl::Taxi) {
        // Regulate ground speed with gentle longitudinal accel/decel.
        const float spdErr = fc.targetSpeedMps - groundSpeed;
        float accel = std::clamp(spdErr * 0.65f, -2.5f, 1.2f);
        force += fwdHoriz * (impl->aircraftMass * accel);
        impl->throttle = std::clamp(accel * 0.2f, 0.0f, 0.35f);
    } else if (fc.mode == FlightControl::TakeoffRoll) {
        impl->throttle = 1.0f;
        force += fwd * (impl->throttle * c.maxThrustN);
        if (speed >= c.takeoffSpeedMps) desiredPitch = c.climbAngleDeg * deg2rad;
    } else if (fc.mode == FlightControl::Airborne) {
        // Pitch for climb rate, power for speed. Command a brisk climb rate and
        // allow a steeper nose-up attitude than cruise so the aircraft actually
        // gains altitude; the speed governor (below) trades throttle to hold the
        // target speed, which keeps AoA well short of the stall.
        const float missionAltGain   = 0.30f;   // tighter tracking → less glideslope lag
        const float missionMaxClimb  = 12.0f;
        const float missionClimbPitch = 13.0f * deg2rad;
        const float altErr = fc.targetAltMSL - altMSL;
        // Feed-forward the glideslope/climb rate so the controller doesn't lag a
        // descending (or climbing) altitude ramp; the proportional term only
        // corrects residual error.
        const float desClimb = std::clamp(fc.climbRateFF + altErr * missionAltGain,
                                          -missionMaxClimb, missionMaxClimb);
        const float climbErr = desClimb - vSpeed;
        desiredPitch = std::clamp(c.pitchPerClimbErr * climbErr * deg2rad,
                                  -c.maxPitchDeg * deg2rad, missionClimbPitch);
        // Speed governor with real two-sided authority: cuts to idle to slow on
        // approach, spools up to climb power when slow. ±10 m/s spans full range.
        impl->throttle = std::clamp(0.5f + 0.05f * (fc.targetSpeedMps - speed), 0.0f, 1.0f);
        force += fwd * (impl->throttle * c.maxThrustN);
    } else if (fc.mode == FlightControl::Rollout) {
        impl->throttle = 0.0f;   // idle; braking handled below
    } else { // Hold
        impl->throttle = 0.0f;
    }

    // ── Aerodynamic lift/drag (same model as the legacy autopilot) ────────
    if (!impl->aeroBody.empty() && speed > 0.5f) {
        const Vec3 vAirBody_j = q.Conjugated() * vAir;
        const glm::vec3 vAirBody(vAirBody_j.GetX(), vAirBody_j.GetY(), vAirBody_j.GetZ());
        const AeroResult ar = impl->aeroBody.compute(vAirBody, c.airDensity);
        force += q * Vec3(ar.force.x, ar.force.y, ar.force.z);
        impl->telemetry.totalLift = ar.totalLift;
        impl->telemetry.totalDrag = ar.totalDrag;
        impl->telemetry.avgCl     = ar.avgCl;
    } else if (speed > 0.5f) {
        const float qDyn = 0.5f * c.airDensity * speed * speed;
        const float aoaStall = c.stallAoADeg * deg2rad;
        const float aoaAbs = std::abs(aoa);
        const float clPeak = c.liftCurveSlope * aoaStall;
        float cl;
        if (aoaAbs <= aoaStall) cl = c.liftCurveSlope * aoa;
        else {
            const float t = std::clamp((aoaAbs - aoaStall) / aoaStall, 0.0f, 1.0f);
            cl = ((aoa >= 0.0f) ? 1.0f : -1.0f) * clPeak * (1.0f - 0.6f * t);
        }
        const float cd = c.zeroLiftDragCD0 + c.inducedDragK * cl * cl;
        const Vec3 vDir = vAir * (1.0f / speed);
        force += upBody * (qDyn * c.wingArea * cl);
        force += vDir * (-qDyn * c.wingArea * cd);
        impl->telemetry.totalLift = qDyn * c.wingArea * cl;
        impl->telemetry.totalDrag = qDyn * c.wingArea * cd;
        impl->telemetry.avgCl     = cl;
    }

    // ── Ground braking / rolling resistance / tire grip ───────────────────
    if (onGroundMode && groundSpeed > 0.05f) {
        Vec3 rightHoriz = worldUp.Cross(fwdHoriz);
        if (rightHoriz.LengthSq() > 1e-4f) rightHoriz = rightHoriz.Normalized();

        // Lateral grip: kill side-slip so the fuselage tracks the nose wheel
        // instead of skidding through corners.
        const float vLat = horiz.Dot(rightHoriz);
        const float muLat = (fc.mode == FlightControl::Taxi) ? 0.90f : 0.75f;
        const float maxLatForce = muLat * impl->aircraftMass * kG;
        const float latAccel = std::clamp(-vLat * 6.5f, -maxLatForce / impl->aircraftMass,
                                                         maxLatForce / impl->aircraftMass);
        force += rightHoriz * (impl->aircraftMass * latAccel);

        const Vec3 hdir = horiz * (1.0f / groundSpeed);
        float decel = 0.4f;                      // baseline rolling resistance
        if (fc.mode == FlightControl::Rollout) decel += 5.0f * std::clamp(fc.brake, 0.0f, 1.0f) + 2.5f;
        else if (fc.mode == FlightControl::Taxi) decel += 0.8f * std::clamp(fc.brake, 0.0f, 1.0f);
        else decel += 4.0f * std::clamp(fc.brake, 0.0f, 1.0f);
        // Don't brake harder than needed to stop within this step.
        const float maxDecel = groundSpeed / std::max(dt, 1e-3f);
        decel = std::min(decel, maxDecel);
        force += hdir * (-impl->aircraftMass * decel);
    }

    bi.AddForce(id, force);

    // ── Attitude control (rate-limited angular velocity) ──────────────────
    const float maxPitchRate = c.maxPitchRateDeg * deg2rad;
    const float maxRollRate  = c.maxRollRateDeg  * deg2rad;

    float pitchRate, rollRate, yawRate = 0.0f;

    if (airborne) {
        // Coordinated-turn flight: bank from heading error, yaw from bank.
        const float hdgErr = wrapToPi(desiredHeading - curHdg);
        const float desiredBank = std::clamp(hdgErr * c.bankPerHeadingErr,
                                             -c.maxBankDeg * deg2rad, c.maxBankDeg * deg2rad);
        pitchRate = std::clamp((desiredPitch - curPitch) * c.pitchRateGain, -maxPitchRate, maxPitchRate);
        rollRate  = std::clamp((desiredBank - curBank) * c.rollRateGain, -maxRollRate, maxRollRate);
        if (speed > 5.0f) yawRate = -(kG * std::tan(curBank)) / std::max(speed, 20.0f);
    } else {
        // Ground / takeoff: keep wings level, steer heading via nose wheel.
        // Note dψ/dt = −ω_y. Taxi uses speed-dependent limits so corners are
        // taken slowly with gradual yaw rather than snap turns that skid.
        const float hdgErr = wrapToPi(desiredHeading - curHdg);
        if (fc.mode == FlightControl::Taxi) {
            const float gs = std::max(groundSpeed, 1.5f);
            const float maxYaw = std::min(0.20f, 1.0f / gs);
            yawRate = -std::clamp(hdgErr * 0.50f, -maxYaw, maxYaw);
        } else {
            yawRate = -std::clamp(hdgErr * 1.0f, -0.45f, 0.45f);
        }
        pitchRate = std::clamp((desiredPitch - curPitch) * c.pitchRateGain, -maxPitchRate, maxPitchRate);
        rollRate  = std::clamp((0.0f - curBank) * 2.0f, -maxRollRate, maxRollRate);
    }

    Vec3 omega = rightBody * pitchRate + fwd * rollRate + worldUp * yawRate;
    if (airborne && turbIntensity > 0.0f && speed > 8.0f) {
        const float bt = impl->flightTime * c.turbulenceScale * 2.3f;
        const float buf = c.buffetRateDeg * deg2rad * turbIntensity;
        omega += rightBody * (turbNoise(bt, 2.1f) * buf)
               + fwd       * (turbNoise(bt, 6.4f) * buf * 1.3f)
               + worldUp   * (turbNoise(bt, 8.8f) * buf * 0.6f);
    }
    bi.SetAngularVelocity(id, omega);

    // ── Telemetry ─────────────────────────────────────────────────────────
    const float rad2deg = 57.2957795f;
    float hdgDeg = curHdg * rad2deg; if (hdgDeg < 0.0f) hdgDeg += 360.0f;
    impl->flightPhase               = fc.displayPhase;
    impl->telemetry.phase           = fc.displayPhase;
    impl->telemetry.airspeed        = speed;
    impl->telemetry.altitudeMSL     = altMSL;
    impl->telemetry.altitudeAGL     = altAGL;
    impl->telemetry.verticalSpeed   = vSpeed;
    impl->telemetry.aoaDeg          = aoa * rad2deg;
    impl->telemetry.headingDeg      = hdgDeg;
    impl->telemetry.targetHeadingDeg= fc.targetHeadingRad * rad2deg;
    impl->telemetry.targetAltitudeMSL = fc.targetAltMSL;
    impl->telemetry.throttle        = impl->throttle;
    impl->telemetry.pitchDeg        = curPitch * rad2deg;
    impl->telemetry.cmdPitchDeg     = desiredPitch * rad2deg;
    impl->telemetry.bankDeg         = curBank * rad2deg;
    impl->telemetry.onGround        = onGroundMode || altAGL < 2.0f;
    impl->telemetry.active          = true;
}

void PhysicsWorld::setFlightEnabled(bool enabled, const AircraftFlightConfig& config) {
    impl->flightEnabled = enabled;
    impl->flightCfg = config;
    impl->flightPhase = FlightPhase::Idle;
    impl->throttle = 0.0f;
    impl->telemetry = FlightTelemetry{};
    impl->telemetry.active = enabled;
}

void PhysicsWorld::setFlightConfig(const AircraftFlightConfig& config) {
    impl->flightCfg = config;
}

void PhysicsWorld::setAerodynamicBody(const AerodynamicBody& body) {
    impl->aeroBody = body;
}

const AerodynamicBody& PhysicsWorld::getAerodynamicBody() const {
    return impl->aeroBody;
}

void PhysicsWorld::setFlightTargets(float targetAltitudeMSL,
                                    float targetHeadingRad,
                                    float targetSpeedMps) {
    impl->targetAltMSL = targetAltitudeMSL;
    impl->targetHeadingRad = targetHeadingRad;
    impl->targetSpeedMps = targetSpeedMps;
}

void PhysicsWorld::requestTakeoff() {
    if (!impl->flightEnabled || !impl->hasAircraft) return;
    if (impl->flightPhase != FlightPhase::Idle) return;
    // Capture the current runway heading so the takeoff roll stays straight.
    if (!impl->aircraftBodyId.IsInvalid()) {
        const Quat q = impl->physicsSystem->GetBodyInterface().GetRotation(impl->aircraftBodyId);
        const Vec3 fwd = q * Vec3(1.0f, 0.0f, 0.0f);
        impl->runwayHeadingRad = std::atan2(fwd.GetZ(), fwd.GetX());
    }
    impl->flightPhase = FlightPhase::Takeoff;
}

void PhysicsWorld::resetFlightToIdle() {
    impl->flightPhase = FlightPhase::Idle;
    impl->throttle = 0.0f;
}

void PhysicsWorld::setMissionMode(bool on) { impl->missionMode = on; }
bool PhysicsWorld::missionMode() const { return impl->missionMode; }
void PhysicsWorld::setFlightControl(const FlightControl& fc) { impl->control = fc; }

void PhysicsWorld::placeAircraft(const glm::vec3& position, const glm::quat& rotation) {
    if (!impl->hasAircraft || impl->aircraftBodyId.IsInvalid() || !initialized) return;
    BodyInterface& bi = impl->physicsSystem->GetBodyInterface();
    bi.SetPositionAndRotation(impl->aircraftBodyId,
                              RVec3(position.x, position.y, position.z),
                              Quat(rotation.x, rotation.y, rotation.z, rotation.w),
                              EActivation::Activate);
    bi.SetLinearVelocity(impl->aircraftBodyId, Vec3::sZero());
    bi.SetAngularVelocity(impl->aircraftBodyId, Vec3::sZero());
}

bool PhysicsWorld::flightEnabled() const { return impl->flightEnabled; }

FlightTelemetry PhysicsWorld::flightTelemetry() const { return impl->telemetry; }

uint32_t PhysicsWorld::spawnSphere(const glm::vec3& position, float radius) {
    if (!initialized) return UINT32_MAX;

    BodyInterface& bi = impl->physicsSystem->GetBodyInterface();

    RefConst<SphereShape> sphereShape = new SphereShape(radius);
    BodyCreationSettings settings(
        sphereShape,
        RVec3(position.x, position.y, position.z),
        Quat::sIdentity(),
        EMotionType::Dynamic,
        PhysicsLayers::MOVING);

    // Continuous collision detection — prevents fast spheres tunneling through terrain.
    settings.mMotionQuality = EMotionQuality::LinearCast;
    settings.mRestitution   = 0.15f;
    settings.mFriction      = 0.6f;
    settings.mLinearDamping = 0.05f;

    BodyID id = bi.CreateAndAddBody(settings, EActivation::Activate);
    if (id.IsInvalid()) return UINT32_MAX;

    impl->dynamicBodyIds.push_back(id);

    PhysicsBodyState state;
    state.position = position;
    state.radius   = radius;
    state.active   = true;
    dynamicBodies.push_back(state);

    return (uint32_t)(dynamicBodies.size() - 1);
}

uint32_t PhysicsWorld::spawnBox(const glm::vec3& position,
                                const glm::quat& rotation,
                                const glm::vec3& halfExtents,
                                bool kinematic) {
    if (!initialized) return UINT32_MAX;

    BodyInterface& bi = impl->physicsSystem->GetBodyInterface();

    RefConst<BoxShape> boxShape = new BoxShape(Vec3(
        halfExtents.x, halfExtents.y, halfExtents.z));

    Quat joltRot(rotation.x, rotation.y, rotation.z, rotation.w);
    BodyCreationSettings settings(
        boxShape,
        RVec3(position.x, position.y, position.z),
        joltRot,
        kinematic ? EMotionType::Kinematic : EMotionType::Dynamic,
        PhysicsLayers::MOVING);

    if (!kinematic) {
        settings.mMotionQuality = EMotionQuality::LinearCast;
        settings.mRestitution   = 0.05f;
        settings.mFriction      = 0.5f;
        settings.mLinearDamping = 0.02f;
        settings.mAngularDamping = 0.05f;
    }

    BodyID id = bi.CreateAndAddBody(settings, EActivation::Activate);
    if (id.IsInvalid()) return UINT32_MAX;

    impl->dynamicBodyIds.push_back(id);

    PhysicsBodyState state;
    state.position    = position;
    state.rotation    = rotation;
    state.halfExtents = halfExtents;
    state.kind        = PhysicsBodyKind::Box;
    state.kinematic   = kinematic;
    state.active      = true;
    dynamicBodies.push_back(state);

    return (uint32_t)(dynamicBodies.size() - 1);
}

uint32_t PhysicsWorld::spawnAircraftWithGear(const glm::vec3& position,
                                             const glm::quat& rotation,
                                             const glm::vec3& fuselageHalfExtents,
                                             float massKg,
                                             const std::vector<GearWheel>& wheels,
                                             float suspensionStiffness,
                                             float suspensionDamping) {
    if (!initialized) return UINT32_MAX;

    clearAircraftGear();

    BodyInterface& bi = impl->physicsSystem->GetBodyInterface();

    RefConst<BoxShape> boxShape = new BoxShape(Vec3(
        fuselageHalfExtents.x, fuselageHalfExtents.y, fuselageHalfExtents.z));

    Quat joltRot(rotation.x, rotation.y, rotation.z, rotation.w);
    BodyCreationSettings settings(
        boxShape,
        RVec3(position.x, position.y, position.z),
        joltRot,
        EMotionType::Dynamic,
        PhysicsLayers::MOVING);

    settings.mMotionQuality   = EMotionQuality::LinearCast;
    settings.mRestitution     = 0.05f;   // fuselage barely bounces; suspension does the work
    settings.mFriction        = 0.7f;
    settings.mLinearDamping    = 0.05f;
    settings.mAngularDamping   = 0.20f;
    settings.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
    settings.mMassPropertiesOverride.mMass = std::max(1.0f, massKg);

    BodyID id = bi.CreateAndAddBody(settings, EActivation::Activate);
    if (id.IsInvalid()) return UINT32_MAX;

    impl->dynamicBodyIds.push_back(id);

    PhysicsBodyState state;
    state.position    = position;
    state.rotation    = rotation;
    state.halfExtents = fuselageHalfExtents;
    state.kind        = PhysicsBodyKind::Box;
    state.kinematic   = false;
    state.active      = true;
    dynamicBodies.push_back(state);

    impl->hasAircraft   = true;
    impl->aircraftBodyId = id;
    impl->aircraftIndex = (uint32_t)(dynamicBodies.size() - 1);
    impl->wheels        = wheels;
    impl->suspStiffness = suspensionStiffness;
    impl->suspDamping   = suspensionDamping;
    impl->aircraftMass  = std::max(1.0f, massKg);

    std::cout << "  [physics] geared aircraft spawned: mass=" << massKg
              << "kg wheels=" << wheels.size()
              << " stiffness=" << suspensionStiffness
              << " damping=" << suspensionDamping
              << " at y=" << position.y << "\n";
    return impl->aircraftIndex;
}

void PhysicsWorld::clearAircraftGear() {
    impl->hasAircraft = false;
    impl->aircraftBodyId = BodyID();
    impl->aircraftIndex = UINT32_MAX;
    impl->wheels.clear();
    impl->flightEnabled = false;
    impl->flightPhase = FlightPhase::Idle;
    impl->throttle = 0.0f;
    impl->telemetry = FlightTelemetry{};
}

bool PhysicsWorld::hasAircraftGear() const { return impl->hasAircraft; }
int  PhysicsWorld::gearWheelCount() const { return (int)impl->wheels.size(); }
const GearWheel& PhysicsWorld::gearWheel(int i) const { return impl->wheels[(size_t)i]; }

void PhysicsWorld::setKinematicState(uint32_t bodyIndex,
                                   const glm::vec3& position,
                                   const glm::quat& rotation,
                                   const glm::vec3& velocity) {
    if (!initialized || bodyIndex >= impl->dynamicBodyIds.size()) return;

    BodyID& id = impl->dynamicBodyIds[bodyIndex];
    if (id.IsInvalid()) return;

    BodyInterface& bi = impl->physicsSystem->GetBodyInterface();
    Quat joltRot(rotation.x, rotation.y, rotation.z, rotation.w);
    bi.MoveKinematic(id,
                     RVec3(position.x, position.y, position.z),
                     joltRot,
                     1.0f / 60.0f);
    bi.SetLinearVelocity(id, Vec3(velocity.x, velocity.y, velocity.z));

    PhysicsBodyState& state = dynamicBodies[bodyIndex];
    state.position = position;
    state.rotation = rotation;
    state.velocity = velocity;
}

void PhysicsWorld::destroyBody(uint32_t bodyIndex) {
    if (!initialized || bodyIndex >= impl->dynamicBodyIds.size()) return;

    BodyInterface& bi = impl->physicsSystem->GetBodyInterface();
    BodyID& id = impl->dynamicBodyIds[bodyIndex];
    if (impl->hasAircraft && id == impl->aircraftBodyId)
        clearAircraftGear();
    if (!id.IsInvalid()) {
        bi.RemoveBody(id);
        bi.DestroyBody(id);
        id = BodyID();
    }

    dynamicBodies[bodyIndex].active = false;
}

void PhysicsWorld::clearDynamicBodies() {
    if (!initialized) return;

    BodyInterface& bi = impl->physicsSystem->GetBodyInterface();
    for (BodyID& id : impl->dynamicBodyIds) {
        if (id.IsInvalid()) continue;
        bi.RemoveBody(id);
        bi.DestroyBody(id);
    }
    impl->dynamicBodyIds.clear();
    dynamicBodies.clear();
    clearAircraftGear();
}

const std::vector<PhysicsBodyState>& PhysicsWorld::getDynamicBodies() const {
    return dynamicBodies;
}
