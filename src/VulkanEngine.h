#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <optional>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <set>
#include <unordered_map>
#include <cstdint>
#include "DebugAnalyzer.h"   // for DebugOverlayMode enum
#include "MasterHydrology.h" // for master-level flow computation
#include "VegetationEngine.h" // SimpleHydrology vegetation system
#include "TerrainSettings.h"
#include "PhysicsWorld.h"
#include "AutonomousAircraft.h"
#include "Airport.h"
#include "Model.h"
#include "Scene.h"

// ══════════════════════════════════════════════════════════════
//  Helper structs
// ══════════════════════════════════════════════════════════════

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    bool isComplete() { return graphics.has_value() && present.has_value(); }
};

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

// ══════════════════════════════════════════════════════════════
//  UBO matching shader FrameData (std140 layout)
// ══════════════════════════════════════════════════════════════

struct FrameData {
    glm::mat4 mvp;
    glm::vec3 cameraPos;
    float     _pad0;
    glm::vec3 lightDir;
    float     time;          // elapsed seconds (replaces _pad1 — same offset)
    glm::mat4 lightMVP;
};

// ─── Bruneton/Hillaire atmosphere parameters ────────────────────
//  Layout matches bruneton_atmosphere.glsl (std140 packing).
//  Mirrors atmosphere_bac_src/source/model/sky_model.hpp's
//  AtmosphereParametersBuffer with two engine-specific tail fields
//  (cameraScale, camera_position) used by the sky pass.
//  All physical quantities are in KILOMETERS (Bruneton convention).
struct AtmosphereParams {
    glm::vec3 solar_irradiance;     float sun_angular_radius;
    glm::vec3 absorption_extinction; float _pad0;
    glm::vec3 rayleigh_scattering;   float mie_phase_function_g;
    glm::vec3 mie_scattering;        float bottom_radius;
    glm::vec3 mie_extinction;        float top_radius;
    glm::vec3 mie_absorption;        float _pad1;
    glm::vec3 ground_albedo;         float _pad2;

    // Each density profile uses two layers; we pack as vec4[3] to
    // match the GLSL struct (std140 vec4 array stride = 16).
    //   layer0 = density_profile[0..3]  (w = -1/scaleHeight for Rayleigh/Mie)
    //   layer1 = density_profile[4..7]  (rest of layer 0 + start of layer 1)
    //   layer2 = density_profile[8..11] (rest of layer 1)
    // The compute shaders index as:
    //   .layers[0] -> rayleigh_density[0..1]  (vec4[0])
    //   .layers[1] -> rayleigh_density[1..2]  (vec4[1])
    // Following the original packing scheme of sky_model.hpp:
    //   we write layers as flat float[12] into 3 vec4s.
    glm::vec4 rayleigh_density[3];
    glm::vec4 mie_density[3];
    glm::vec4 absorption_density[3];

    glm::vec2 TransmittanceTexDimensions;
    glm::vec2 MultiscatteringTexDimensions;

    glm::vec2 SkyViewTexDimensions;
    glm::vec2 _pad3;

    glm::vec4 AEPerspectiveTexDimensions;  // .xyz used

    glm::vec3 sun_direction;     float cameraScale;
    glm::vec3 camera_position;   float _pad4;
};

struct ChunkPush {
    glm::vec2 worldOrigin;
    float     worldSize;
    float     heightScale;
    int       heightmapRes;
    int       debugOverlayMode;   // 0=none, 1=heatmap, 2=slope, 3=land/water, 4=peaks
    float     globalMinHeight;    // for heatmap range
    float     globalMaxHeight;
};

struct OceanPush {
    glm::vec2 worldOrigin;     // same as terrain ChunkPush
    float     worldSize;
    float     heightScale;
    int       heightmapRes;
    float     seaLevel;        // metres — ocean plane sits at this world Y
    float     waveAmplitude;   // base amplitude in metres
    float     waveSpeed;       // base angular frequency multiplier
    float     shoreDepth;      // depth at which waves begin to fade (m)
    float     foamDepth;       // depth below which foam appears (m)
    float     _pad0;           // std430 alignment
};

// ── Airflow particle (SSBO std430 layout, 64 bytes) ──────────
struct AeroParticle {
    glm::vec4 pos;     // xyz = world position, w = age [0,1]
    glm::vec4 vel;     // xyz = velocity m/s,   w = speed (cached)
    glm::vec4 color;   // xyz = rgb,             w = alpha
    glm::vec4 trail;   // xyz = prev world pos,  w = birth phase [0,1]
};

// ── Aerodynamic global UBO (std140 safe) ─────────────────────
struct AeroGlobalUBO {
    glm::mat4  bodyToWorld;
    glm::mat4  worldToBody;
    glm::vec4  freestreamWorld;    // xyz=V_inf world,   w=|V_inf|
    glm::vec4  freestreamBody;     // xyz=V_inf body,    w=AoA deg
    glm::vec4  aircraftDims;       // x=halfSpan, y=halfFuseLen, z=fuseRadius, w=time
    glm::vec4  vortexParams;       // x=Gamma_0, y=halfSpan, z=coreRadius, w=0
    glm::vec4  sourceParams;       // x=Q, y=halfFuseLen, z=fuseR, w=0
    glm::vec4  stabParams;         // x=Gamma_stab, y=halfSpanStab, z=coreR, w=0
    glm::vec4  particleConfig;     // x=lifetime, y=lateralRadius, z=boxHalfLen, w=N
    glm::ivec4 flags;              // x=numParticles, y=colorMode, z=resetAll, w=frame
    glm::vec4  simVolume;          // xyz=simulation centre (camera.pos world), w=reserved
};

// ── Aerodynamic surface overlay UBO ──────────────────────────
struct AeroSurfaceUBO {
    glm::vec4  freestreamBody;     // xyz=V_inf body frame, w=qDyn
    glm::vec4  visualConfig;       // x=opacity, y=Cp_min, z=Cp_max, w=lightingStrength
    glm::vec4  surfParams;         // x=halfLength, y=aoaDeg, z/w=unused
};

// ── Per-panel visualization data (SSBO, 64 bytes each) ───────
// Uploaded from AerodynamicBody::panels() each frame, so every
// field here is the actual physics-computed value, not an approx.
struct PanelVizData {
    glm::vec4 center;     // xyz = world-space centroid, w = Cp (from physics)
    glm::vec4 chordHalf;  // xyz = world-space half-chord vector (len = chord*0.5)
    glm::vec4 spanHalf;   // xyz = world-space half-span vector  (len = span*0.5)
    glm::vec4 meta;       // x = liftN, y = groupID, z = localAoADeg, w = Cl
};

// ── CPU-integrated streamline vertex (packed, 20 bytes) ──────
// Streamlines are integrated at 0.2 m step size on the CPU so
// the tight curvature near the leading edge is fully resolved.
// Rendered as LINE_LIST (explicit segment pairs, no prim-restart).
struct StreamlineVertex {
    float x, y, z;   // world position
    float speed;      // local flow speed m/s (drives colour)
    float t;          // [0,1] normalised arc-length (drives animation phase)
};

struct HeightmapPush {
    glm::vec2 worldOrigin;
    float     worldSize;
    int32_t   resolution;
    float     masterWorldSize;    // Phase 8: world extent for master UV calculation
    float     masterTexelScale;   // Phase 8: 1.0/masterTectonicRes for UV offset
    float     seaLevel;           // master sea level for river detection
    float     riverThreshold;     // fraction of land cells for river qualification
    int32_t   masterResolution;   // master map resolution for hydrology sampling
};

struct HydroPush {
    int32_t   resolution;
    float     seaLevel;          // cells below this drain to ocean sink
    float     riverThreshold;    // fraction of total cells to qualify as river
    int32_t   totalCells;        // land cells in chunk
    float     maxCarveDepth;     // maximum river channel depth
    float     riverWidth;        // channel half-width in cells
};

// ══════════════════════════════════════════════════════════════
//  Camera
// ══════════════════════════════════════════════════════════════

class Camera {
public:
    glm::vec3 pos    = glm::vec3(256.0f, 100.0f, 256.0f);  // center of 512 world
    float     yaw    = -45.0f;
    float     pitch  = -30.0f;
    float     dist   = 60.0f;

    glm::mat4 view() const {
        glm::vec3 target = pos + glm::vec3(
            cos(glm::radians(yaw)) * cos(glm::radians(pitch)),
            sin(glm::radians(pitch)),
            sin(glm::radians(yaw)) * cos(glm::radians(pitch))
        );
        glm::vec3 eye = target - target * (dist / glm::length(target - pos));
        // Orbit: eye rotates around target
        glm::vec3 dir = glm::normalize(glm::vec3(
            cos(glm::radians(yaw)) * cos(glm::radians(pitch)),
            sin(glm::radians(pitch)),
            sin(glm::radians(yaw)) * cos(glm::radians(pitch))
        ));
        glm::vec3 eyePos = pos - dir * dist;
        return glm::lookAt(eyePos, pos, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::vec3 eyePos() const {
        glm::vec3 dir = glm::normalize(glm::vec3(
            cos(glm::radians(yaw)) * cos(glm::radians(pitch)),
            sin(glm::radians(pitch)),
            sin(glm::radians(yaw)) * cos(glm::radians(pitch))
        ));
        return pos - dir * dist;
    }
};

// ══════════════════════════════════════════════════════════════
//  Frustum culling (Phase 3)
// ══════════════════════════════════════════════════════════════

struct FrustumPlane {
    glm::vec3 normal;
    float     distance;
};

struct Frustum {
    FrustumPlane planes[6];
};

Frustum extractFrustum(const glm::mat4& projView);
bool isAabbInFrustum(const Frustum& f, const glm::vec3& min, const glm::vec3& max);

// ══════════════════════════════════════════════════════════════
//  Chunk LOD system (Phase 6)
// ══════════════════════════════════════════════════════════════

struct ChunkLOD {
    uint32_t firstIndex;
    uint32_t indexCount;
    float    minDistance;  // camera distance threshold for this LOD
};

// ══════════════════════════════════════════════════════════════
//  VulkanEngine
// ══════════════════════════════════════════════════════════════

class VulkanEngine {
public:
    void init(GLFWwindow* window, const TerrainSettings& cfg);
    void draw();
    void cleanup();
    void onResize();
    void runTerrainAnalysis();  // Phase 8: debug terrain topology analysis
    Camera& getCamera() { return camera; }
    const TerrainSettings& getSettings() const { return settings; }
    TerrainSettings& getSettingsMutable() { return settings; }

    void stepPhysics(float deltaTime);
    PhysicsWorld& getPhysics() { return physics; }
    const PhysicsWorld& getPhysics() const { return physics; }
    const AutonomousAircraft& getAutonomousAircraft() const { return autonomousAircraft; }
    float sampleWorldHeight(float worldX, float worldZ);

    // Unified aircraft render/camera state (autopilot OR drop-test OR flight body).
    AircraftRenderState aircraftRenderState() const;
    glm::vec3 aircraftLinearVelocity() const;
    bool isDropTestActive() const { return dropTestActive; }
    void startAircraftDropTest();

    // ── Two-phase flight (takeoff + cruise) ──
    void startAircraftFlight();          // spawn geared aircraft on the runway, idle
    void requestAircraftTakeoff();       // Idle -> Takeoff (GUI button)
    bool isFlightModeActive() const { return flightActive; }
    FlightTelemetry currentFlightTelemetry() const { return physics.flightTelemetry(); }
    float* flightTargetAltMSLPtr()     { return &flightTargetAltMSL; }
    float* flightTargetHeadingDegPtr() { return &flightTargetHeadingDeg; }
    float* flightTargetSpeedPtr()      { return &flightTargetSpeedMps; }

    // ─── Debug overlay system ──────────────────────────────────
    // F1-F4 cycle through overlay modes; F5 runs deep analysis
    void setDebugOverlay(DebugAnalyzer::DebugOverlayMode mode);
    void cycleDebugOverlay();
    int  getDebugOverlayMode() const { return (int)debugOverlayMode; }

    // ─── Cloud-shadow runtime controls ─────────────────────────
    void toggleCloudShadows();         // F9: enable/disable cloud shadows on terrain
    void cycleCloudShadowDebug();      // F10: toggle transmittance debug overlay
    float getGlobalMinHeight() const { return globalMinHeight; }
    float getGlobalMaxHeight() const { return globalMaxHeight; }

private:
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    GLFWwindow* window = nullptr;
    Camera camera;
    TerrainSettings settings;
    PhysicsWorld physics;

    // Vulkan core
    VkInstance               instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR             surface = VK_NULL_HANDLE;
    VkPhysicalDevice         physicalDevice = VK_NULL_HANDLE;
    VkDevice                 device = VK_NULL_HANDLE;
    VkQueue                  graphicsQueue = VK_NULL_HANDLE;
    VkQueue                  presentQueue = VK_NULL_HANDLE;
    uint32_t                 graphicsFamily = 0;
    uint32_t                 presentFamily = 0;

    // Swapchain
    VkSwapchainKHR           swapchain = VK_NULL_HANDLE;
    std::vector<VkImage>     swapchainImages;
    std::vector<VkImageView> swapchainViews;
    VkFormat                 swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D               swapchainExtent = {0, 0};

    // Depth
    Image depth;

    // Heightmap sampler for the single terrain texture
    VkSampler heightmapSampler = VK_NULL_HANDLE;

    // Master world data (CPU-side, for hydrology + debug analysis)
    // SimpleHydrology integration: quadtree-based world replaces monolithic generators.
    // Initial terrain is generated via fBm noise (matching SimpleHydrology's approach).
    std::vector<float> masterHeightData;   // CPU heightmap for hydrology/debug analysis
    uint32_t worldSeed = 0;               // persisted seed for consistent procedural queries

    // Master heightmap GPU texture — feeds chunk rendering
    // Populated by SimpleHydrology erosion pipeline (fBm noise + particle erosion)
    Image      masterHeightmapImage;
    VkSampler  masterHeightmapSampler = VK_NULL_HANDLE;

    // Single global terrain descriptor set for the finite world.
    VkDescriptorSet globalTerrainDescSet = VK_NULL_HANDLE;
    VkDescriptorSet globalTerrainDescSets[MAX_FRAMES_IN_FLIGHT]{};

    // CPU-side hydrology for the single world (rivers/lakes/flow) used by
    // foliage placement, debug analysis, and the heightmap texture bake.
    MasterHydrology::FlowResult masterFlowData;
    float      masterSeaLevel = 0.0f;  // sea level for flow computation (set from settings)

    // Light-space view-projection for terrain shadows, computed once
    // per frame in updateUniformBuffer() and reused by the shadow pass
    // so the depth render and the terrain lookup never diverge.
    glm::mat4 currentLightMVP{1.0f};
    AutonomousAircraft autonomousAircraft;

    // Landing-gear drop test
    bool     dropTestActive = false;
    uint32_t dropTestBodyIndex = UINT32_MAX;
    float    dropTelemetryTimer = 0.0f;
    void stepDropTestTelemetry(float deltaTime);

    // Two-phase flight (takeoff + cruise)
    bool     flightActive = false;
    uint32_t flightBodyIndex = UINT32_MAX;
    float    flightTargetAltMSL = 150.0f;     // GUI: target altitude (MSL)
    float    flightTargetHeadingDeg = 0.0f;   // GUI: target heading
    float    flightTargetSpeedMps = 55.0f;    // GUI: target cruise speed
    bool     flightTargetsInit = false;       // seeded from runway on first spawn
    float    flightTelemetryTimer = 0.0f;
    float    flightRunwayElevMSL = 0.0f;      // cached for live config rebuilds
    AircraftFlightConfig buildFlightConfig(float runwayElevationMSL) const;

    // ── Cinematic demo mission (airport-to-airport showcase) ──
    enum class MissionPhase {
        Idle, TaxiOut, Lineup, TakeoffRoll, Climb, Cruise,
        Descent, Approach, Flare, Rollout, TaxiIn, Park
    };
    bool         missionActive = false;
    MissionPhase missionPhase  = MissionPhase::Idle;
    int          missionFrom   = 0;          // departure airport index
    int          missionTo     = 1;          // destination airport index
    bool         missionTowardPosEnd = true; // operational runway direction this leg
    std::vector<glm::vec2> missionPath;      // active taxi waypoints (world XZ)
    std::vector<glm::vec2> missionFlightPath; // scenic cruise / approach waypoints
    size_t       missionWaypoint = 0;
    size_t       missionFlightWaypoint = 0;
    float        missionFlightPathLen = 0.0f;
    float        missionHoldTimer = 0.0f;
    float        missionPhaseTimer = 0.0f;
    glm::vec3    missionAircraftHalf{6.0f, 1.0f, 1.5f};
    void  startDemoMission();
    void  stopDemoMission();
    void  updateDemoMission(float dt);
    void  buildDepartureTaxi();
    void  buildArrivalTaxi();
    void  buildScenicFlightPath();
    const char* missionPhaseName() const;

    // ── Visible landing-gear struts (driven by suspension telemetry) ──
    Buffer    gearVertexBuffer;
    void*     gearVertexMapped = nullptr;
    Buffer    gearIndexBuffer;
    uint32_t  gearIndexCount = 0;
    uint32_t  gearVertexCapacity = 0;
    bool      gearResourcesReady = false;
    void createGearResources();
    void updateGearMesh();

    // Vegetation system (SimpleHydrology vegetation.h integration)
    VegetationEngine::Vegetation   vegetation;
    VegetationEngine::MasterWorld  vegetationWorld;  // CPU-side terrain queries for plants
    int         vegetationUpdateCounter = 0;

    // Debug overlay (Phase 8 enhanced)
    DebugAnalyzer::DebugOverlayMode debugOverlayMode = DebugAnalyzer::DebugOverlayMode::NONE;
    Image      debugOverlayImage;          // RGBA8 debug overlay texture (GPU)
    VkSampler  debugOverlaySampler = VK_NULL_HANDLE;
    float      globalMinHeight = 0.0f;     // cached world min/max for heatmap shader
    float      globalMaxHeight = 1.0f;
    void uploadDebugOverlayTexture(DebugAnalyzer::DebugOverlayMode mode);

    // Shadow map (Phase 7)
    Image      shadowMap;
    VkSampler  shadowSampler = VK_NULL_HANDLE;
    Buffer     shadowUniformBuffer;
    void*      shadowUniformMapped = nullptr;
    const uint32_t SHADOW_MAP_SIZE = 2048;

    // Shadow pipeline
    VkPipelineLayout      shadowPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            shadowPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout shadowDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      shadowDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet       shadowDescriptorSet = VK_NULL_HANDLE;

    // Foliage pipeline (Phase 7)
    VkPipelineLayout foliagePipelineLayout = VK_NULL_HANDLE;
    VkPipeline       foliagePipeline = VK_NULL_HANDLE;
    Buffer           foliageVertexBuffer;
    Buffer           foliageIndexBuffer;
    uint32_t         foliageIndexCount = 0;
    // Global foliage instance buffers (vegetation system → GPU)
    Buffer           globalFoliageInstanceBuffer;
    Buffer           globalFoliageColorBuffer;
    uint32_t         globalFoliageCount = 0;

    // TEMPORARY: debug aircraft mesh (fuselage + wings + tail), bright colored.
    VkPipelineLayout aircraftPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       aircraftPipeline = VK_NULL_HANDLE;
    Buffer           aircraftVertexBuffer;
    Buffer           aircraftIndexBuffer;
    uint32_t         aircraftIndexCount = 0;

    // ── Airports (flat concrete pads + runway/taxiway/apron markings) ──
    //  Two logical airports laid out along their connecting axis. The plane
    //  taxis out of one, flies to the other, and lands. Buildings are placed
    //  by the user; we only build the paved surfaces + nav points.
    Airport          airports[2];
    bool             airportsValid = false;
    VkPipelineLayout airportPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       airportPipeline = VK_NULL_HANDLE;   // colored, depth-biased, no cull
    Buffer           airportVertexBuffer;
    Buffer           airportIndexBuffer;
    uint32_t         airportIndexCount = 0;
    void layoutAirports();                 // place the two airports for the demo
    void flattenAirportPads(std::vector<float>& heights, uint32_t res) const;
    void createAirportPipeline();
    void buildAirportMesh();               // (re)build the paved-surface geometry

    // ── Textured model assets (PBR: base/metallic/roughness/normal) ──
    struct ModelMaterialGPU {
        Image base, metal, rough, normalMap;   // per-material textures (or defaults)
        bool  ownBase=false, ownMetal=false, ownRough=false, ownNormal=false;
        Buffer ubo;                              // material factors / flags
        VkDescriptorSet descriptor = VK_NULL_HANDLE;
    };
    struct ModelSubmeshGPU { uint32_t indexOffset=0, indexCount=0; int materialIndex=-1; };

    // A self-contained GPU upload of one model asset (its own buffers, materials,
    // and descriptor pool). Shared resources (pipeline, layout, sampler, default
    // textures) live on the engine and are referenced, not owned, by a LoadedModel.
    struct LoadedModel {
        Buffer   vbo;
        Buffer   ibo;
        uint32_t indexCount = 0;
        std::vector<ModelMaterialGPU> materials;
        std::vector<ModelSubmeshGPU>  submeshes;
        glm::vec3 boundsMin{0.0f}, boundsMax{0.0f};
        float    autoScale = 1.0f;   // normalises model units to a sensible size
        VkDescriptorPool descPool = VK_NULL_HANDLE;
        bool     valid = false;
    };

    // An entry in the editor asset palette: a friendly name + its GPU model.
    struct SceneAsset {
        std::string name;
        LoadedModel gpu;
    };

    // Shared model rendering resources (one pipeline for aircraft + scene objects).
    VkPipelineLayout      modelPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            modelPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout modelDescLayout = VK_NULL_HANDLE;
    VkSampler             modelSampler = VK_NULL_HANDLE;
    Image                 defaultWhiteTex;   // missing base/metal/rough fallback
    Image                 defaultNormalTex;  // missing normal fallback (flat blue)

    LoadedModel              aircraftAsset;   // the autonomous aircraft's FBX model
    std::vector<SceneAsset>  sceneAssets;     // editor palette: primitives + FBX

    // ── Scene editor state ──────────────────────────────────────
    Scene       editorScene;                  // placed objects + world params
    int         selectedObject = -1;          // index into editorScene.objects (-1 = none)
    bool        editMode = false;             // true = editor (autopilot paused, gizmos)
    int         gizmoOperation = 7;           // ImGuizmo::TRANSLATE (7) / ROTATE (120) / SCALE (896)
    int         gizmoModeLocal = 0;           // 0 = world, 1 = local
    char        scenePathBuf[256] = "scene.json";
    glm::mat4   lastProj{1.0f};               // cached for picking / gizmo
    glm::mat4   lastView{1.0f};

    // Terrain mesh (flat XZ grid, shared across all chunks)
    Buffer  vertexBuffer;
    Buffer  indexBuffer;
    uint32_t indexCount = 0;

    // LOD system (Phase 6)
    static const int MAX_LODS = 4;
    ChunkLOD lods[MAX_LODS];

    // Uniform buffers — one per frame in flight to avoid GPU read/write races.
    Buffer  uniformBuffers[MAX_FRAMES_IN_FLIGHT];
    void*   uniformMapped[MAX_FRAMES_IN_FLIGHT]{};

    // ─── Atmospheric scattering (Bruneton/Hillaire LUT pipeline) ──
    Buffer  atmosphereUniformBuffer;
    void*   atmosphereUniformMapped = nullptr;
    AtmosphereParams atmosphereParams;

    // Pre-computed LUTs (Hillaire 2020). All in R16G16B16A16_SFLOAT.
    //  transmittanceLUT   256 × 64   - sun transmittance through atmosphere
    //  multiscatterLUT     32 × 32   - precomputed multi-bounce scattering
    //  skyviewLUT         192 × 128  - per-direction sky radiance from camera
    // Bottom two are recomputed every frame; transmittance + multiscatter
    // are computed once at init (they only depend on atmosphere parameters).
    Image      transmittanceLUT;
    Image      multiscatterLUT;
    Image      skyviewLUT;
    VkSampler  lutSampler = VK_NULL_HANDLE;

    // Pipeline + descriptors for the three Bruneton compute passes.
    // We use one shared descriptor set layout (atmos UBO + 3 storage
    // images), one per pass, plus a sampler descriptor for the sky pass.
    VkDescriptorSetLayout brunetonComputeSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      brunetonDescriptorPool   = VK_NULL_HANDLE;
    VkDescriptorSet       brunetonComputeSet       = VK_NULL_HANDLE;
    VkPipelineLayout      brunetonComputeLayout    = VK_NULL_HANDLE;
    VkPipeline            transmittancePipeline    = VK_NULL_HANDLE;
    VkPipeline            multiscatterPipeline     = VK_NULL_HANDLE;
    VkPipeline            skyviewPipeline          = VK_NULL_HANDLE;

    // Sky graphics pass: full-screen triangle that samples the skyview LUT
    // and writes into the swapchain on far-depth pixels (depth==1.0).
    VkDescriptorSetLayout skyPassSetLayout    = VK_NULL_HANDLE;
    VkDescriptorSet       skyPassSet          = VK_NULL_HANDLE;
    VkPipelineLayout      skyPassPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            skyPassPipeline       = VK_NULL_HANDLE;

    // ─── Volumetric clouds (Heckel + Hillaire + Schneider) ───────
    //  Pipeline overview:
    //    1) cloud_noise_gen.comp dispatches ONCE at init, populating
    //       cloudNoise3D (128³ RGBA16F) with a 4-channel Worley-Perlin
    //       FBM. The four channels feed the shape & detail weights.
    //    2) blueNoise2D is a 64×64 R8 texture filled with a CPU-side
    //       Mitchell best-candidate sequence (low-discrepancy, perceptually
    //       blue spectrum) — used to dither the raymarch start offset
    //       so we can use ~40 samples instead of ~100 with no banding.
    //    3) cloudParamsBuffer is a per-frame UBO of phase / density
    //       / scattering knobs (mirrors atmosphere_bac_src/
    //       buffers/clouds_param_buffer.glsl).
    //    4) bruneton_clouds.{vert,frag} draws a full-screen pass that
    //       raymarches through the cloud-layer spherical shell,
    //       samples cloudNoise3D, light-marches toward the sun,
    //       multiplies by the transmittance LUT, and alpha-blends
    //       the result into the swapchain after foliage.
    Image      cloudNoise3D;
    Image      blueNoise2D;
    VkSampler  cloudNoiseSampler = VK_NULL_HANDLE;  // 3D Worley sampler (repeat)
    VkSampler  blueNoiseSampler  = VK_NULL_HANDLE;  // 2D blue-noise sampler (repeat)
    VkSampler  depthSampler      = VK_NULL_HANDLE;  // depth attachment as sampler2D
    Buffer     cloudParamsBuffer;
    void*      cloudParamsMapped = nullptr;
    uint32_t   cloudFrameIndex   = 0;
    float      cloudWindOffset   = 0.0f;
    float      cloudDensityOffsetBase = 0.55f;  // captured default for timelapse oscillation
    float      cloudTimelapsePhase = 0.0f;
    float      lastFrameDt        = 0.016f;     // most recent frame dt (for time-based anims)

    VkDescriptorSetLayout cloudNoiseGenSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      cloudDescriptorPool    = VK_NULL_HANDLE;
    VkDescriptorSet       cloudNoiseGenSet       = VK_NULL_HANDLE;
    VkPipelineLayout      cloudNoiseGenLayout    = VK_NULL_HANDLE;
    VkPipeline            cloudNoiseGenPipeline  = VK_NULL_HANDLE;

    VkDescriptorSetLayout cloudDrawSetLayout     = VK_NULL_HANDLE;
    VkDescriptorSet       cloudDrawSet           = VK_NULL_HANDLE;
    VkPipelineLayout      cloudDrawPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            cloudDrawPipeline      = VK_NULL_HANDLE;

    // ─── Temporal cloud accumulation ──────────────────────────────
    //  cloudHalfResBuffer  – half-res RGBA16F target for the raymarch.
    //  cloudAccumBuffer[2] – full-res ping-pong RGBA16F: one is written
    //                        this frame, the other is history.
    //  The temporal resolve pass (cloud_temporal.frag) bilinearly
    //  upsamples cloudHalfResBuffer, reprojects history via direction
    //  reprojection, and blends into cloudAccumBuffer[ping]. The
    //  composite pass (cloud_composite.frag) then premul-blends the
    //  accumulated result onto the swapchain.
    Image      cloudHalfResBuffer;
    Image      cloudAccumBuffer[2];
    VkSampler  cloudLinearSampler      = VK_NULL_HANDLE;  // linear/clamp for accum buffers
    Buffer     cloudTemporalUBO;                           // stores prevViewProj mat4
    void*      cloudTemporalMapped     = nullptr;
    uint32_t   cloudAccumPing          = 0;               // write target index this frame
    glm::mat4  prevCloudViewProj{1.0f};                   // last frame's ViewProj

    VkDescriptorSetLayout cloudTemporalSetLayout     = VK_NULL_HANDLE;
    VkDescriptorSet       cloudTemporalSet[2]        = {};
    VkPipelineLayout      cloudTemporalPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            cloudTemporalPipeline      = VK_NULL_HANDLE;

    VkDescriptorSetLayout cloudCompositeSetLayout    = VK_NULL_HANDLE;
    VkDescriptorSet       cloudCompositeSet[2]       = {};
    VkPipelineLayout      cloudCompositePipelineLayout = VK_NULL_HANDLE;
    VkPipeline            cloudCompositePipeline     = VK_NULL_HANDLE;

    // CloudParams std140 layout — keep in sync with bruneton_clouds.frag.
    struct CloudParams {
        glm::vec4 shapeNoiseWeights;
        glm::vec4 detailNoiseWeights;
        glm::vec4 phaseParams;          // g1, g2, blendOffset, blendFactor
        float detailNoiseMultiplier;
        float minBounds;
        float maxBounds;
        float cloudsScale;
        float detailScale;
        float densityOffset;
        float densityMultiplier;
        int   sampleCount;
        int   sampleCountToSun;
        float lightAbsTowardsSun;
        float lightAbsThroughCloud;
        float darknessThreshold;
        float windOffset;
        // ── Terrain cloud-shadow controls (read by terrain.frag) ──
        int   enableCloudShadows       = 1;
        int   cloudShadowSamples       = 8;
        float cloudShadowStrength      = 1.0f;
        float cloudShadowMinVisibility = 0.35f;
        int   cloudShadowDebug         = 0;
        int   useCloudShadowMap        = 1;
        float _pad0;
    };
    static_assert(sizeof(CloudParams) % 16 == 0, "CloudParams std140 alignment");
    CloudParams cloudParams;

    // ── Pre-baked cloud shadow transmittance map (world XZ → sun visibility) ──
    static constexpr uint32_t CLOUD_SHADOW_MAP_SIZE = 512;
    Image      cloudShadowMap;
    VkSampler  cloudShadowSampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout cloudShadowSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet       cloudShadowDescSet     = VK_NULL_HANDLE;
    VkPipelineLayout      cloudShadowLayout      = VK_NULL_HANDLE;
    VkPipeline            cloudShadowPipeline    = VK_NULL_HANDLE;
    VkDescriptorPool      cloudShadowDescPool    = VK_NULL_HANDLE;

    // Skyview LUT amortization — skip recompute when sun/camera barely moved.
    glm::vec3 lastSkyviewSunDir{0.0f, 1.0f, 0.0f};
    float     lastSkyviewCamAlt   = 0.0f;
    bool      skyviewValid        = false;
    bool      cloudShadowMapReady = false;

    // Smoothed CPU section timings (ms) for the settings overlay.
    float cpuUpdateMsSmoothed     = 0.0f;
    float cpuStreamlineMsSmoothed = 0.0f;

    // Descriptors
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool = VK_NULL_HANDLE;

    // Graphics pipeline
    VkPipelineLayout terrainPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       terrainPipeline = VK_NULL_HANDLE;

    // Ocean pass (Gerstner wave surface, shares the terrain descriptor set)
    Buffer           oceanVertexBuffer;
    Buffer           oceanIndexBuffer;
    uint32_t         oceanIndexCount     = 0;
    VkPipeline       oceanPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout oceanPipelineLayout = VK_NULL_HANDLE;
    OceanPush        oceanPush{};

    // ── Aerodynamic airflow visualization ──
    static constexpr uint32_t AERO_PARTICLE_COUNT = 8192;
    Buffer                aeroParticleSSBO;
    Buffer                aeroGlobalUBOBuf;
    Buffer                aeroSurfaceUBOBuf;
    void*                 aeroGlobalMapped  = nullptr;
    void*                 aeroSurfaceMapped = nullptr;
    VkDescriptorSetLayout aeroDescSetLayout        = VK_NULL_HANDLE;  // SSBO + global UBO
    VkDescriptorSetLayout aeroSurfaceDescSetLayout = VK_NULL_HANDLE;  // surface UBO only
    VkDescriptorPool      aeroDescPool       = VK_NULL_HANDLE;
    VkDescriptorSet       aeroDescSet        = VK_NULL_HANDLE;
    VkDescriptorSet       aeroSurfaceDescSet = VK_NULL_HANDLE;
    VkPipelineLayout      aeroAdvectLayout   = VK_NULL_HANDLE;
    VkPipeline            aeroAdvectPipeline = VK_NULL_HANDLE;
    VkPipelineLayout      aeroRenderLayout   = VK_NULL_HANDLE;
    VkPipeline            aeroRenderPipeline = VK_NULL_HANDLE;
    VkPipelineLayout      aeroSurfaceLayout  = VK_NULL_HANDLE;
    VkPipeline            aeroSurfacePipeline= VK_NULL_HANDLE;
    bool                  aeroParticlesInitialized = false;
    AeroGlobalUBO         aeroGlobalData{};

    // ── Panel Cp visualization (actual physics values, not approximated) ──
    // A separate AerodynamicBody is evaluated each frame from the aircraft's
    // current velocity so its per-panel Cp values drive the quad coloring.
    AerodynamicBody       aeroBodyViz;
    bool                  aeroBodyVizBuilt = false;
    static constexpr uint32_t AERO_MAX_PANELS = 32;
    Buffer                aeroPanelSSBO;
    void*                 aeroPanelMapped = nullptr;
    VkDescriptorSetLayout aeroPanelDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool      aeroPanelDescPool   = VK_NULL_HANDLE;
    VkDescriptorSet       aeroPanelDescSet    = VK_NULL_HANDLE;
    VkPipelineLayout      aeroPanelLayout     = VK_NULL_HANDLE;
    VkPipeline            aeroPanelPipeline   = VK_NULL_HANDLE;

    // ── CPU-integrated streamlines (MSFS-style curved paths) ──────────────
    static constexpr uint32_t STREAMLINE_UPDATE_INTERVAL = 3;
    static constexpr uint32_t STREAMLINE_MAX_VERTS    = 220000;
    uint32_t            streamlineFrameCounter = 0;
    Buffer              streamlineVBO;
    void*               streamlineMapped    = nullptr;
    uint32_t            streamlineVertCount = 0;
    VkPipelineLayout    streamlineLayout    = VK_NULL_HANDLE;
    VkPipeline          streamlinePipeline  = VK_NULL_HANDLE;

    // Transient descriptor pool for the texgen material-bake compute set.
    VkDescriptorPool      computeDescriptorPool = VK_NULL_HANDLE;

    // TexGen pipeline (procedural texture generation)
    VkPipelineLayout texgenPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       texgenPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout texgenDescriptorSetLayout = VK_NULL_HANDLE;

    // Texture array (Phase 5: triplanar mapping)
    Image      terrainTexArray;       // 12-layer array: 0-5 albedo+rough(a), 6-11 normal+ao(a)
    VkSampler  terrainTexSampler = VK_NULL_HANDLE;
    VkImageView terrainTexStorageView = VK_NULL_HANDLE;  // single-mip view for texgen writes
    uint32_t   terrainTexMipLevels = 1;
    uint32_t   terrainTexBakedRes = 0;   // resolution the array was created at

    // Command
    VkCommandPool                commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    // ── GPU per-pass profiling (timestamp queries) ──
    // 7 timestamps/frame mark the boundaries between the major GPU phases
    // (skyview, cloud-shadow bake, shadow, main, clouds, UI). Read back the
    // previous use of each frame slot after its fence is signalled, so there
    // is no stall. Smoothed and shown in the settings overlay + console.
    static constexpr uint32_t GPU_TS_PER_FRAME = 7;
    VkQueryPool gpuQueryPool        = VK_NULL_HANDLE;
    float       gpuTimestampPeriodNs = 0.0f;     // nanoseconds per tick
    bool        gpuProfilingEnabled  = false;
    bool        gpuSlotWritten[MAX_FRAMES_IN_FLIGHT] = {};
    double      gpuPassMs[6] = {};               // sky, cloudShadow, shadow, main, clouds, ui
    double      gpuTotalMs   = 0.0;
    void createGpuProfiler();
    void readGpuTimestamps(uint32_t frameSlot);

    // Sync
    std::vector<VkSemaphore> imageAvailable;       // per-swapchain-image
    std::vector<VkSemaphore> renderFinished;       // per-swapchain-image
    std::vector<VkFence>     inFlightFences;       // MAX_FRAMES_IN_FLIGHT for frame pacing
    std::vector<VkFence>     imagesInFlight;       // per-swapchain-image: which fence is in flight
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    // ─── Init helpers ─────────────────────────────────────────
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createDepthResources();
    void createHeightmapSampler();
    void initializeWorldData();       // generate the single finite world (noise + hydrology)
    void generateMasterTerrain();     // fill masterHeightData/masterFlowData at masterRes/worldSize
    void initAutonomousAircraft();
    void respawnAutonomousAircraft();
    void createMasterHeightmapTexture(); // upload world data to the single terrain GPU texture
    void createGlobalTerrainDescriptorSet(); // create globalTerrainDescSet (after all resources)
    void initVegetationSystem();     // initialize vegetation with master data
    void updateVegetation();         // per-frame vegetation growth + root density
    void createTerrainMesh();
    int  selectTerrainLOD() const;   // pick index into lods[] from camera distance
    void createOceanMesh();
    void createOceanPipeline();
    void createAeroResources();          // SSBO + UBOs + descriptor layout/pool/sets
    void createAeroPipelines();          // advect compute + streamline + surface graphics
    void createAeroPanelVizResources();  // panel SSBO + descriptor layout/pool/set
    void createAeroPanelVizPipeline();   // panel-quad pipeline
    void createStreamlinePipeline();     // CPU-integrated streamline pipeline
    void updateAeroGlobalUBO();          // CPU→GPU bridge (called each frame from draw)
    void updateAeroPanelVizData();       // evaluate panel physics + upload to SSBO
    void updateStreamlines();            // integrate paths on CPU, upload LINE_LIST VBO
    void recordAeroCompute(VkCommandBuffer cmd);                   // dispatch (outside render pass)
    void recordAeroRender(VkCommandBuffer cmd, const glm::mat4& viewProj); // inside render pass
    void destroyAeroResources();
    void createUniformBuffer();
    void createDescriptorLayouts();
    void createDescriptorPools();
    void createTextureArray();
    void dispatchTexgenAndMips();   // (re)bake material array + mip chain
    void createTexgenPipeline();
    void createShadowMap();
    void createShadowPipeline();
    void createFoliagePipeline();
    void createFoliageMesh();
    void createAircraftPipeline();
    void createAircraftMesh();

    // Model assets (textured PBR) — shared pipeline; per-asset GPU uploads.
    void createModelPipeline();
    void createModelDefaultTextures();
    bool loadAircraftModel(const std::string& fbxPath);
    void uploadModelTexture(const std::string& path, bool srgb, Image& out, bool& owned);
    void destroyModelResources();
    // Upload a CPU ModelData to GPU, fitting its longest dimension to targetSize.
    LoadedModel loadModelAsset(const ModelData& md, float targetSize);
    // Load an FBX file and upload it; returns an invalid LoadedModel on failure.
    LoadedModel loadModelAssetFromFBX(const std::string& fbxPath, float targetSize);
    void destroyLoadedModel(LoadedModel& m);
    // Populate sceneAssets with built-in primitives + every models/*.fbx found.
    void buildAssetRegistry();

    // ── Scene editor ────────────────────────────────────────────
    void drawSceneObjects(VkCommandBuffer cmd, const glm::mat4& proj, const glm::mat4& view);
    glm::mat4 sceneObjectMatrix(const SceneObject& o) const;  // full render transform
    glm::mat4 sceneObjectGizmoMatrix(const SceneObject& o) const; // T*R*S (no autofit)
    void decomposeGizmoMatrix(const glm::mat4& m, SceneObject& o) const;
    void applySceneWorldParams();             // editorScene.world -> settings (+ regen)
    void captureSceneWorldParams();           // settings -> editorScene.world
    void saveSceneToFile(const std::string& path);
    void loadSceneFromFile(const std::string& path);
    void addSceneObject(int assetIndex);
    void duplicateSceneObject(int index);
    void deleteSceneObject(int index);
    void pickSceneObject(float mouseX, float mouseY);  // ray-AABB click select
    int  assetIndexByName(const std::string& name) const;
    void drawEditorUI();   // gizmo manipulation + editor panels (ImGui frame)
    void generateGlobalFoliage();  // populate global foliage from vegetation system
    void updateGlobalFoliageBuffers(); // refresh instance buffers each frame
    void createGraphicsPipeline();

    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();
    void recreateSwapchain();
    void cleanupSwapchain();

    // ─── Atmospheric scattering ──────────────────────────────
    void createAtmosphereUBO();
    void updateAtmosphereUBO();

    // ─── Bruneton LUT pipeline ────────────────────────────────
    void createBrunetonLUTs();           // create the 3 LUT images + sampler
    void createBrunetonComputePipelines(); // create transmittance/multiscatter/skyview pipelines
    void createSkyPassPipeline();         // create the full-screen sky graphics pipeline
    void runBrunetonStaticLUTs();        // dispatch transmittance + multiscatter (once)
    void dispatchSkyviewLUT(VkCommandBuffer cmd); // per-frame skyview update
    bool needsSkyviewUpdate() const;
    void createCloudShadowResources();
    void dispatchCloudShadowMap(VkCommandBuffer cmd);

    // ─── Volumetric clouds ───────────────────────────────────
    void createCloudResources();         // 3D noise image, blue noise, samplers, UBO
    void uploadBlueNoiseTexture();       // staging-buffer upload of CPU-generated blue noise
    void createCloudNoiseGenPipeline();  // compute pipeline for Worley FBM generation
    void runCloudNoiseGeneration();      // one-shot dispatch at init
    void createCloudDrawPipeline();      // graphics pipeline for the half-res raymarch pass
    void createCloudAccumBuffers();      // create half-res + full-res accumulation images + linear sampler
    void createCloudTemporalPipeline();  // temporal resolve pipeline
    void createCloudCompositePipeline(); // composite pipeline (accum → swapchain)
    void recreateCloudAccumBuffers();    // called on swapchain resize
    void updateCloudParamsUBO();         // per-frame UBO update
    void drawCloudsPass(VkCommandBuffer cmd, uint32_t imageIndex); // record the 3-sub-pass cloud render

    // ─── ImGui / Settings UI ──────────────────────────────────
    void initImGui();
    void shutdownImGui();
    void drawSettingsUI();           // called inside recordRenderCommand
    void regenerateTerrain();        // hot-rebuild masterHeightData + hydrology
    void initPhysics();
    void rebuildPhysicsTerrain();
    void spawnPhysicsDemoBodies();
    bool uiVisible        = true;    // toggle with F1
    bool requestRegen     = false;   // set by SettingsPanel; consumed by draw()
    bool requestReseed    = false;
    bool requestResetCam  = false;
    bool requestRebuildPhysics = false;
    float fpsSmoothed     = 0.0f;
    float lastFrameTime   = 0.0f;
    bool  imguiInitialised = false;
    int   uiLandCells     = 0;
    int   uiMaxAccum      = 0;
    int   uiRiverCells    = 0;

    // ─── Per-frame ────────────────────────────────────────────
    void recordRenderCommand(VkCommandBuffer cmd, uint32_t imageIndex);
    void updateUniformBuffer(uint32_t imageIndex);

    // Single source of truth for the sun direction, in engine Y-up
    // metres space, derived from settings.sunElevation/sunAzimuth.
    // The atmosphere/cloud shaders use the swizzled Bruneton Z-up form.
    glm::vec3 currentSunDirectionYUp() const;
    void updateTimeOfDay(float dt);      // animate the sun arc when day/night is on

    // ─── Helpers ──────────────────────────────────────────────
    bool isDeviceSuitable(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    SwapchainSupport querySwapchainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& available);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& caps);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props);

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, Buffer& buf);
    void createImage(uint32_t w, uint32_t h, VkFormat fmt,
                     VkImageUsageFlags usage, VkImageAspectFlags aspect,
                     Image& img);
    void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                               VkFormat format, VkImageLayout oldLayout,
                               VkImageLayout newLayout);
    void copyBufferToImage(VkCommandBuffer cmd, VkBuffer buf, VkImage img,
                           uint32_t w, uint32_t h);

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cmd);
    VkShaderModule createShaderModule(const std::vector<char>& code);

    std::vector<char> readFile(const std::string& path);

    // ─── Validation layers ────────────────────────────────────
#ifdef _DEBUG
    const bool enableValidationLayers = true;
#else
    const bool enableValidationLayers = false;
#endif
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    bool checkValidationLayerSupport();
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void* userData);
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& info);
    VkResult createDebugUtilsMessengerEXT(VkInstance inst,
        const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkDebugUtilsMessengerEXT* pDebugMessenger);
    void destroyDebugUtilsMessengerEXT(VkInstance inst,
        VkDebugUtilsMessengerEXT debugMessenger,
        const VkAllocationCallbacks* pAllocator);
};
