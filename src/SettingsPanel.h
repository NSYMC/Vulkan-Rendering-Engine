#pragma once
// ══════════════════════════════════════════════════════════════
//  SettingsPanel — ImGui overlay for live terrain manipulation.
//
//  This wraps an older ImGui (legacy-render-pass) backend.  The
//  caller owns:
//    - The VkRenderPass created via createRenderPass()
//    - The per-image VkFramebuffer list created via createFramebuffers()
//  These are recreated when the swapchain is recreated.
//
//  Per-frame loop:
//    beginFrame();
//    draw(settings, &regen, ...);
//    renderInsideRenderPass(cmd);
//  Surrounded by vkCmdBeginRenderPass/EndRenderPass (engine does this).
// ══════════════════════════════════════════════════════════════
#include "imgui.h"
#include "ImGuizmo.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "TerrainSettings.h"
#include "PhysicsWorld.h"
#include "AutonomousAircraft.h"
#include <vector>
#include <stdexcept>

namespace SettingsPanel {

inline VkDescriptorPool gPool = VK_NULL_HANDLE;
inline VkRenderPass     gRenderPass = VK_NULL_HANDLE;
inline std::vector<VkFramebuffer> gFramebuffers;

inline void createDescriptorPool(VkDevice device) {
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER,                 256 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  256 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,           256 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           256 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          256 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          256 },
    };
    VkDescriptorPoolCreateInfo info{};
    info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    info.maxSets       = 256 * 6;
    info.poolSizeCount = (uint32_t)(sizeof(sizes)/sizeof(sizes[0]));
    info.pPoolSizes    = sizes;
    if (vkCreateDescriptorPool(device, &info, nullptr, &gPool) != VK_SUCCESS)
        throw std::runtime_error("SettingsPanel: failed to create ImGui descriptor pool");
}

// Creates a render pass that LOADs the existing swapchain image (PRESENT_SRC layout is fine
// because we transition to COLOR_ATTACHMENT_OPTIMAL inside our existing pipeline already —
// here we accept COLOR_ATTACHMENT_OPTIMAL as the initial layout).
inline void createRenderPass(VkDevice device, VkFormat colorFormat) {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format         = colorFormat;
    colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &colorRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

    VkRenderPassCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments    = &colorAttachment;
    info.subpassCount    = 1;
    info.pSubpasses      = &subpass;
    info.dependencyCount = 1;
    info.pDependencies   = &dep;

    if (vkCreateRenderPass(device, &info, nullptr, &gRenderPass) != VK_SUCCESS)
        throw std::runtime_error("SettingsPanel: failed to create ImGui render pass");
}

inline void createFramebuffers(VkDevice device,
                                const std::vector<VkImageView>& swapchainViews,
                                VkExtent2D extent)
{
    gFramebuffers.resize(swapchainViews.size());
    for (size_t i = 0; i < swapchainViews.size(); i++) {
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass      = gRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments    = &swapchainViews[i];
        fbInfo.width           = extent.width;
        fbInfo.height          = extent.height;
        fbInfo.layers          = 1;
        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &gFramebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("SettingsPanel: failed to create ImGui framebuffer");
    }
}

inline void destroyFramebuffers(VkDevice device) {
    for (auto& fb : gFramebuffers) {
        if (fb) vkDestroyFramebuffer(device, fb, nullptr);
    }
    gFramebuffers.clear();
}

inline void uploadFonts(VkDevice device, VkQueue queue, VkCommandPool pool) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(device, &ai, &cb);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    ImGui_ImplVulkan_CreateFontsTexture(cb);

    vkEndCommandBuffer(cb);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, pool, 1, &cb);

    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

inline void init(GLFWwindow* window,
                 VkInstance instance,
                 VkPhysicalDevice physical,
                 VkDevice device,
                 uint32_t queueFamily,
                 VkQueue graphicsQueue,
                 VkFormat colorFormat,
                 uint32_t minImageCount,
                 uint32_t imageCount,
                 const std::vector<VkImageView>& swapchainViews,
                 VkExtent2D extent,
                 VkCommandPool cmdPool)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding   = 6.0f;
    s.FrameRounding    = 4.0f;
    s.GrabRounding     = 4.0f;
    s.WindowBorderSize = 0.0f;
    s.Colors[ImGuiCol_WindowBg].w = 0.88f;

    createDescriptorPool(device);
    createRenderPass(device, colorFormat);
    createFramebuffers(device, swapchainViews, extent);

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance        = instance;
    initInfo.PhysicalDevice  = physical;
    initInfo.Device          = device;
    initInfo.QueueFamily     = queueFamily;
    initInfo.Queue           = graphicsQueue;
    initInfo.DescriptorPool  = gPool;
    initInfo.MinImageCount   = minImageCount;
    initInfo.ImageCount      = imageCount;
    initInfo.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;

    if (!ImGui_ImplVulkan_Init(&initInfo, gRenderPass))
        throw std::runtime_error("ImGui_ImplVulkan_Init failed");

    uploadFonts(device, graphicsQueue, cmdPool);
}

inline void beginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();
}

inline void draw(TerrainSettings& s,
                 bool* outRegenRequested,
                 bool* outReseedRequested,
                 bool* outResetCameraRequested,
                 bool* outRebuildPhysicsRequested,
                 float fps,
                 int   landCells,
                 int   maxAccum,
                 int   riverCells,
                 const PhysicsWorld* physics,
                 const AutonomousAircraft* aircraft = nullptr)
{
    ImGui::SetNextWindowPos(ImVec2(16, 16), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(380, 640), ImGuiCond_Once);
    ImGui::Begin("Terrain Settings", nullptr, ImGuiWindowFlags_NoCollapse);

    // Lightweight section divider (this ImGui build predates SeparatorText()).
    auto section = [](const char* label) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextUnformatted(label);
        ImGui::Spacing();
    };

    ImGui::Text("FPS: %.1f", fps);
    ImGui::Separator();

    if (ImGui::BeginTabBar("MainTabs", ImGuiTabBarFlags_None)) {

    // ───────────────────────── DEMO ─────────────────────────
    if (ImGui::BeginTabItem("Demo")) {
        ImGui::TextWrapped("Cinematic airport-to-airport showcase: the aircraft "
                           "taxis out, takes off, flies to the second airport and "
                           "lands, with optional day/night and cloud timelapse.");
        ImGui::Spacing();

        if (ImGui::Checkbox("Cinematic demo mode", &s.demoModeEnabled))
            *outRebuildPhysicsRequested = true;
        ImGui::Checkbox("Camera follows aircraft", &s.aircraftCameraFollowEnabled);

        ImGui::Spacing();
        section("Day / night cycle");
        ImGui::Checkbox("Animate sun (day arc)", &s.dayNightEnabled);
        ImGui::SliderFloat("Day length (s)",  &s.dayLengthSeconds, 20.0f, 600.0f, "%.0f");
        ImGui::SliderFloat("Time of day",     &s.timeOfDay01, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled("0.25 sunrise - 0.5 noon - 0.75 sunset");
        ImGui::SliderFloat("Noon sun elev (deg)", &s.sunPeakElevationDeg, 20.0f, 89.0f, "%.0f");

        ImGui::Spacing();
        section("Cloud timelapse");
        ImGui::Checkbox("Enable cloud timelapse", &s.cloudTimelapseEnabled);
        ImGui::SliderFloat("Timelapse speed x", &s.cloudTimelapseSpeed, 2.0f, 120.0f, "%.0f");

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.20f, 0.42f, 0.62f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.55f, 0.78f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.14f, 0.32f, 0.50f, 1.0f));
        if (ImGui::Button("Restart demo flight", ImVec2(ImGui::GetContentRegionAvail().x, 32.0f)))
            *outRebuildPhysicsRequested = true;
        ImGui::PopStyleColor(3);
        ImGui::EndTabItem();
    }

    // ───────────────────────── WORLD ─────────────────────────
    if (ImGui::BeginTabItem("World")) {
        section("World size");
        ImGui::TextUnformatted("Preset:");
        ImGui::SameLine();
        if (ImGui::SmallButton("1 km"))   { s.worldSize =  1024.0f; s.masterRes = 1024; *outRegenRequested = true; }
        ImGui::SameLine();
        if (ImGui::SmallButton("4 km"))   { s.worldSize =  4096.0f; s.masterRes = 2048; *outRegenRequested = true; }
        ImGui::SameLine();
        if (ImGui::SmallButton("8 km"))   { s.worldSize =  8192.0f; s.masterRes = 2048; *outRegenRequested = true; }
        ImGui::SameLine();
        if (ImGui::SmallButton("16 km"))  { s.worldSize = 16384.0f; s.masterRes = 2048; *outRegenRequested = true; }

        ImGui::DragFloat("World size (m)", &s.worldSize, 64.0f, 256.0f, 32768.0f, "%.0f");
        ImGui::TextDisabled("Seed: %u", s.worldSeed);
        ImGui::SameLine();
        if (ImGui::Button("Reseed")) *outReseedRequested = true;

        section("Terrain shape");
        ImGui::SliderFloat("Sea fraction",         &s.seaFraction,         0.10f, 0.80f);
        ImGui::SliderFloat("Total height (m)",     &s.terrainTotalHeight, 20.0f, 200.0f);
        ImGui::SliderFloat("Ridge weight",         &s.mountainRidgeWeight, 0.0f, 1.0f);
        ImGui::SliderFloat("Height scale",         &s.terrainHeightScale, 0.5f, 4.0f);
        ImGui::TextDisabled("Ridge > 0 produces sharp peaks");

        section("Rivers");
        ImGui::SliderFloat("River threshold",  &s.riverThreshold, 0.001f, 0.20f, "%.4f");
        ImGui::SliderInt  ("Min cells",        &s.riverMinCells,  1, 1000);
        ImGui::SliderFloat("Carve depth (m)",  &s.riverCarveDepth, 0.0f, 20.0f);
        ImGui::TextDisabled("Land %d | Max accum %d | Rivers %d cells", landCells, maxAccum, riverCells);

        section("Ocean");
        ImGui::SliderFloat("Wave amplitude (m)",   &s.waveAmplitude, 0.0f, 3.0f);
        ImGui::SliderFloat("Wave speed",           &s.waveSpeed,     0.1f, 3.0f);
        ImGui::SliderFloat("Shore fade depth (m)", &s.shoreDepth,    1.0f, 20.0f);
        ImGui::SliderFloat("Foam depth (m)",       &s.foamDepth,     0.5f, 5.0f);
        ImGui::TextDisabled("Ocean is applied live — no regenerate needed.");

        section("Camera");
        ImGui::SliderFloat("Move speed",        &s.cameraSpeed,      1.0f, 1000.0f);
        ImGui::SliderFloat("Sprint mult",       &s.cameraSprintMult, 1.0f, 10.0f);
        ImGui::SliderFloat("Mouse sensitivity", &s.mouseSensitivity, 0.05f, 2.0f);
        ImGui::SliderFloat("Zoom sensitivity",  &s.zoomSensitivity,  0.5f, 200.0f);
        ImGui::SliderFloat("Max zoom dist",     &s.cameraMaxDist,    50.0f, 10000.0f);
        if (ImGui::Button("Reset camera to world centre"))
            *outResetCameraRequested = true;
        ImGui::EndTabItem();
    }

    // ─────────────────────── ATMOSPHERE ───────────────────────
    if (ImGui::BeginTabItem("Atmosphere")) {
        section("Sun");
        ImGui::SliderFloat("Sun elevation (deg)", &s.sunElevation, -10.0f, 90.0f);
        ImGui::SliderFloat("Sun azimuth (deg)",   &s.sunAzimuth,    0.0f, 360.0f);
        if (s.dayNightEnabled)
            ImGui::TextDisabled("Day-arc animation (Demo tab) is driving the sun.");

        section("Atmospheric scattering");
        ImGui::Checkbox("Enable atmosphere", &s.enableAtmosphere);
        ImGui::SliderInt("View samples",  &s.atmosphereSamples, 4, 32);
        ImGui::SliderInt("Light samples", &s.atmosphereLightSamples, 2, 16);

        section("Cloud shadows");
        ImGui::Checkbox("Enable cloud shadows", &s.enableCloudShadows);
        ImGui::Checkbox("Pre-baked shadow map", &s.useCloudShadowMap);
        ImGui::SliderFloat("Shadow strength",   &s.cloudShadowStrength, 0.0f, 1.0f);
        ImGui::SliderFloat("Min visibility",    &s.cloudShadowMinVisibility, 0.0f, 1.0f);
        ImGui::EndTabItem();
    }

    // ─────────────────────── AIRCRAFT ───────────────────────
    if (ImGui::BeginTabItem("Aircraft")) {
        bool needsRegen = false;
        if (ImGui::Checkbox("Autonomous flight", &s.tempAutonomousAircraftEnabled))
            *outRebuildPhysicsRequested = true;
        ImGui::Checkbox("Camera follows plane", &s.aircraftCameraFollowEnabled);
        ImGui::SameLine();
        ImGui::Checkbox("Pause autopilot", &s.aircraftDebugPauseAutopilot);

        // ── Two-phase flight (takeoff + cruise) ─────────────────
        if (ImGui::Checkbox("Two-phase flight (takeoff+cruise)", &s.aircraftFlightEnabled))
            *outRebuildPhysicsRequested = true;
        if (s.aircraftFlightEnabled) {
            ImGui::SliderFloat("Wing area (m2)", &s.aircraftWingArea, 6.0f, 60.0f, "%.1f");
            ImGui::SliderFloat("Max thrust (N)", &s.aircraftMaxThrustN, 2000.0f, 60000.0f, "%.0f");
            ImGui::SliderFloat("Takeoff spd (m/s)", &s.aircraftTakeoffSpeedMps, 12.0f, 80.0f, "%.0f");
            ImGui::SliderFloat("Stall AoA (deg)", &s.aircraftStallAoADeg, 8.0f, 25.0f, "%.0f");
            ImGui::TextDisabled("Enter altitude/heading in the right-side panel.");

            ImGui::Checkbox("Wind + turbulence", &s.aircraftWindEnabled);
            if (s.aircraftWindEnabled) {
                ImGui::SliderFloat("Wind spd (m/s)", &s.aircraftWindSpeedMps, 0.0f, 30.0f, "%.1f");
                ImGui::SliderFloat("Wind dir (deg)", &s.aircraftWindDirDeg, 0.0f, 360.0f, "%.0f");
                ImGui::SliderFloat("Turbulence",     &s.aircraftTurbulenceIntensity, 0.0f, 1.0f, "%.2f");
            }
        }

        // ── FBX model appearance / alignment ────────────────────
        ImGui::Separator();
        ImGui::Checkbox("Use FBX model (else debug box)", &s.aircraftUseModel);
        if (s.aircraftUseModel) {
            ImGui::SliderFloat("Model scale x", &s.aircraftModelScale, 0.2f, 5.0f, "%.2f");
            ImGui::SliderFloat("Model yaw off",   &s.aircraftModelYawOffsetDeg,   -180.0f, 180.0f, "%.0f");
            ImGui::SliderFloat("Model pitch off", &s.aircraftModelPitchOffsetDeg, -180.0f, 180.0f, "%.0f");
            ImGui::SliderFloat("Model roll off",  &s.aircraftModelRollOffsetDeg,  -180.0f, 180.0f, "%.0f");
            ImGui::TextDisabled("Align nose to +X (flight forward).");
        }

        // ── Aerodynamic airflow visualization ───────────────────
        ImGui::Separator();
        ImGui::Checkbox("Airflow visualization", &s.aeroVisualizationEnabled);
        if (s.aeroVisualizationEnabled) {
            ImGui::SliderFloat("Cp overlay opacity", &s.aeroCpOverlayOpacity, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Cp scale min",        &s.aeroCpMin, -4.0f, 0.0f, "%.2f");
            ImGui::SliderFloat("Cp scale max",        &s.aeroCpMax,  0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Particle lifetime (s)", &s.aeroParticleLifetime, 2.0f, 20.0f, "%.1f");
            ImGui::SliderFloat("Spawn radius (m)",     &s.aeroSpawnRadius, 4.0f, 40.0f, "%.1f");
            ImGui::SliderFloat("Spawn upstream (m)",   &s.aeroSpawnUpstream, 5.0f, 50.0f, "%.1f");
            ImGui::SliderFloat("Streamline bend x",    &s.aeroVizStrength, 1.0f, 30.0f, "%.1f");
            ImGui::TextDisabled("Bend x exaggerates flow curvature for visibility.");
            ImGui::TextDisabled("Blue = slow/suction, green = freestream, red = fast.");
            if (physics) {
                const FlightTelemetry tel = physics->flightTelemetry();
                if (tel.active) {
                    ImGui::Separator();
                    ImGui::Text("Airspeed:   %.1f m/s", tel.airspeed);
                    ImGui::Text("AoA:        %.2f deg", tel.aoaDeg);
                    ImGui::Text("Total lift: %.0f N", tel.totalLift);
                    ImGui::Text("Total drag: %.0f N", tel.totalDrag);
                    ImGui::Text("Avg CL:     %.4f", tel.avgCl);
                }
            }
        }

        // ── Landing-gear drop test ──────────────────────────────
        ImGui::Separator();
        if (ImGui::Checkbox("Drop test (suspension)", &s.aircraftDropTestEnabled))
            *outRebuildPhysicsRequested = true;
        if (s.aircraftDropTestEnabled) {
            ImGui::SliderFloat("Drop height (m)", &s.aircraftDropHeightMeters, 1.0f, 60.0f, "%.1f");
            ImGui::SliderFloat("Mass (kg)", &s.aircraftMassKg, 200.0f, 5000.0f, "%.0f");
            ImGui::SliderFloat("Susp. rest (m)", &s.aircraftWheelRestLength, 0.1f, 2.0f, "%.2f");
            ImGui::SliderFloat("Wheel radius (m)", &s.aircraftWheelRadius, 0.05f, 1.0f, "%.2f");
            ImGui::SliderFloat("Stiffness (N/m)", &s.aircraftSuspensionStiffness, 5000.0f, 200000.0f, "%.0f");
            ImGui::SliderFloat("Damping (N.s/m)", &s.aircraftSuspensionDamping, 200.0f, 30000.0f, "%.0f");
            if (ImGui::Button("Re-drop")) *outRebuildPhysicsRequested = true;

            if (physics && physics->hasAircraftGear()) {
                ImGui::Spacing();
                ImGui::TextUnformatted("Wheels (compression / load):");
                for (int i = 0; i < physics->gearWheelCount(); ++i) {
                    const GearWheel& w = physics->gearWheel(i);
                    ImVec4 col = w.grounded ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f)
                                            : ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
                    ImGui::TextColored(col, "  #%d %s  comp=%.3f m  load=%.0f N",
                        i, w.grounded ? "GROUND" : "air   ", w.compression, w.load);
                }
            }
        }

        ImGui::SliderFloat("Spawn heading (deg)", &s.aircraftSpawnHeadingDeg, -180.0f, 180.0f, "%.0f");
        ImGui::SliderFloat("Fly-to distance (m)", &s.aircraftDestinationDistanceMeters, 700.0f, 3000.0f, "%.0f");
        s.aircraftDestinationDistanceMeters = glm::max(700.0f, s.aircraftDestinationDistanceMeters);
        ImGui::SliderFloat("Cruise alt AGL (m)", &s.aircraftCruiseAltitudeAgl, 40.0f, 400.0f);
        ImGui::SliderFloat("Cruise speed (m/s)", &s.aircraftCruiseSpeedMps, 25.0f, 90.0f);
        if (needsRegen)
            *outRegenRequested = true;

        ImGui::Separator();
        if (aircraft && aircraft->isActive()) {
            ImGui::Text("Phase: %s", aircraft->phaseName());
            ImGui::Text("Pos: (%.0f, %.0f, %.0f)", aircraft->position().x, aircraft->position().y, aircraft->position().z);
            ImGui::Text("Speed: %.1f m/s", aircraft->speedMps());
            ImGui::Text("Altitude AGL: %.0f m", aircraft->altitudeAglM());
            ImGui::Text("Dist to dest: %.0f m", aircraft->distanceToDestinationM());
            ImGui::Text("Heading err: %.1f deg", aircraft->headingErrorDeg());
            ImGui::Text("Body index: %u", aircraft->bodyIndex());
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Render visible: yes");
        } else if (s.tempAutonomousAircraftEnabled) {
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "Render visible: no");
            ImGui::TextDisabled("Restart to spawn aircraft at world centre");
        }
        ImGui::EndTabItem();
    }

    // ─────────────────────── PERFORMANCE ───────────────────────
    if (ImGui::BeginTabItem("Performance")) {
        section("Resolution & view distance");
        int mr = (int)s.masterRes;
        if (ImGui::SliderInt("Master resolution", &mr, 256, 4096))
            s.masterRes = (uint32_t)mr;
        ImGui::DragFloat("Far plane (0=auto)", &s.farPlane, 100.0f, 0.0f, 100000.0f, "%.0f");
        ImGui::TextDisabled("Effective: %.0f m", s.effectiveFarPlane());
        int tr = (int)s.texRes;
        if (ImGui::SliderInt("Material res", &tr, 256, 2048))
            s.texRes = tr;

        section("Level of detail");
        ImGui::SliderInt("Terrain view min LOD",   &s.terrainViewMinLod,   0, 3);
        ImGui::SliderInt("Terrain shadow min LOD", &s.terrainShadowMinLod, 0, 3);
        ImGui::TextDisabled("0 = full detail, higher = cheaper/coarser.");

        section("Clouds");
        ImGui::Checkbox("Adaptive cloud quality", &s.adaptiveCloudQuality);
        ImGui::SliderInt("Cloud raymarch samples", &s.cloudRaymarchSamples, 16, 128);
        ImGui::SliderInt("Streamline update interval", &s.streamlineUpdateInterval, 1, 8);

        section("Physics (Jolt)");
        ImGui::Checkbox("Enabled", &s.physicsEnabled);
        ImGui::SliderInt("Collider stride", &s.physicsMeshStride, 4, 64);
        ImGui::SliderInt("Demo spheres", &s.physicsSpawnCount, 0, 32);
        ImGui::SliderFloat("Sphere radius (m)", &s.physicsSphereRadius, 0.5f, 10.0f);
        if (physics && physics->isInitialized()) {
            ImGui::TextDisabled("Heightfield: %dx%d (%d tris)",
                physics->terrainSampleCount(), physics->terrainSampleCount(),
                physics->terrainTriangleCount());
            ImGui::TextDisabled("Dynamic bodies: %zu", physics->getDynamicBodies().size());
        } else {
            ImGui::TextDisabled("Physics not active (restart after enable)");
        }
        if (ImGui::Button("Rebuild terrain collider"))
            *outRebuildPhysicsRequested = true;
        ImGui::SameLine();
        ImGui::TextDisabled("P = spawn sphere");
        ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImVec2 sz(ImGui::GetContentRegionAvail().x, 38.0f);
    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.22f, 0.50f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.30f, 0.65f, 0.30f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.15f, 0.40f, 0.18f, 1.0f));
    if (ImGui::Button("Regenerate Terrain", sz)) *outRegenRequested = true;
    ImGui::PopStyleColor(3);

    ImGui::TextDisabled("Tip: hold LMB outside the panel to look around.");
    ImGui::TextDisabled("Use WASD/QE to fly; SHIFT to sprint.");
    ImGui::End();
}

// ── Right-side flight control panel: enter altitude (MSL) + heading, then
//    start the takeoff. Two phases: TAKEOFF (runway roll + climb out) and
//    CRUISE (hold target altitude/heading/speed). ──────────────────────────
inline void drawFlightControl(float screenWidth,
                              bool flightActive,
                              const FlightTelemetry& tel,
                              float* targetAltMSL,
                              float* targetHeadingDeg,
                              float* targetSpeedMps,
                              bool* outTakeoffRequested)
{
    const float panelW = 300.0f;
    ImGui::SetNextWindowPos(ImVec2(screenWidth - panelW - 16.0f, 16.0f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(panelW, 360.0f), ImGuiCond_Once);
    ImGui::Begin("Flight Control", nullptr, ImGuiWindowFlags_NoCollapse);

    if (!flightActive) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "Flight mode inactive");
        ImGui::TextDisabled("Enable 'aircraftFlightEnabled' and restart,");
        ImGui::TextDisabled("or Regenerate Terrain to spawn on the runway.");
        ImGui::End();
        return;
    }

    const char* ph = (tel.phase == FlightPhase::Takeoff) ? "TAKEOFF"
                   : (tel.phase == FlightPhase::Cruise)  ? "CRUISE"
                   : "IDLE (on runway)";
    ImGui::Text("Phase: %s", ph);
    ImGui::Separator();

    ImGui::TextUnformatted("Targets");
    ImGui::DragFloat("Altitude MSL (m)", targetAltMSL, 5.0f, -200.0f, 6000.0f, "%.0f");
    if (ImGui::DragFloat("Heading (deg)", targetHeadingDeg, 1.0f, 0.0f, 360.0f, "%.0f")) {
        if (*targetHeadingDeg < 0.0f)   *targetHeadingDeg += 360.0f;
        if (*targetHeadingDeg >= 360.0f) *targetHeadingDeg -= 360.0f;
    }
    ImGui::DragFloat("Speed (m/s)", targetSpeedMps, 1.0f, 15.0f, 120.0f, "%.0f");

    ImGui::Spacing();
    ImVec2 sz(ImGui::GetContentRegionAvail().x, 36.0f);
    const bool onGround = (tel.phase == FlightPhase::Idle);
    if (onGround) {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.22f, 0.50f, 0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.30f, 0.65f, 0.30f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.15f, 0.40f, 0.18f, 1.0f));
        if (ImGui::Button("Start takeoff", sz)) *outTakeoffRequested = true;
        ImGui::PopStyleColor(3);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.30f, 0.30f, 0.30f, 1.0f));
        ImGui::Button("In flight...", sz);   // inert: only meaningful on the ground
        ImGui::PopStyleColor(1);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted("Telemetry");
    ImGui::Text("Airspeed:  %.1f m/s", tel.airspeed);
    ImGui::Text("Alt MSL:   %.0f m", tel.altitudeMSL);
    ImGui::Text("Alt AGL:   %.0f m", tel.altitudeAGL);
    ImGui::Text("V/S:       %+.1f m/s", tel.verticalSpeed);
    ImGui::Text("Heading:   %.0f -> %.0f deg", tel.headingDeg, tel.targetHeadingDeg);
    ImGui::Text("Pitch/Bank:%.0f / %.0f deg", tel.pitchDeg, tel.bankDeg);
    if (tel.aoaDeg > 14.0f)
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "AoA:       %.1f deg  STALL!", tel.aoaDeg);
    else
        ImGui::Text("AoA:       %.1f deg", tel.aoaDeg);
    ImGui::Text("Throttle:  %.0f %%", tel.throttle * 100.0f);

    ImGui::End();
}

// Records a render-pass-wrapped ImGui draw onto a given swapchain image.
inline void renderToImage(VkCommandBuffer cmd, uint32_t imageIndex, VkExtent2D extent) {
    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();
    if (!drawData) return;

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass        = gRenderPass;
    rpBegin.framebuffer       = gFramebuffers[imageIndex];
    rpBegin.renderArea.offset = {0, 0};
    rpBegin.renderArea.extent = extent;
    rpBegin.clearValueCount   = 0;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(drawData, cmd);
    vkCmdEndRenderPass(cmd);
}

inline void shutdown(VkDevice device) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    destroyFramebuffers(device);
    if (gRenderPass) {
        vkDestroyRenderPass(device, gRenderPass, nullptr);
        gRenderPass = VK_NULL_HANDLE;
    }
    if (gPool) {
        vkDestroyDescriptorPool(device, gPool, nullptr);
        gPool = VK_NULL_HANDLE;
    }
}

} // namespace SettingsPanel
