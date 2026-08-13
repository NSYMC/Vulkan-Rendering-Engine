// ══════════════════════════════════════════════════════════════
//  TerrainEngine — Vulkan 1.3 procedural terrain renderer
//  Phase 10: SimpleHydrology Integration
// ══════════════════════════════════════════════════════════════

#include "VulkanEngine.h"
#include "TerrainSettings.h"
#include "imgui.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <random>
#include <fstream>
#include <cmath>

static VulkanEngine* engine = nullptr;
static bool          mouseDown = false;
static double        lastMouseX = 0, lastMouseY = 0;
static float         mouseSens  = 0.3f;
static float         zoomSens   = 5.0f;
static float         camSpeed   = 30.0f;
static float         sprintMult = 3.0f;
static float         camMinDist = 5.0f;
static float         camMaxDist = 500.0f;

// ── MSFS-24 style smooth external camera state ────────────────
// Kept as module-level so cursorPosCallback and the main loop
// share the same accumulator without passing extra pointers.
static bool      camIsFollowing   = false;
static float     camFollowYawOff  = 0.0f;
static glm::vec3 camSmoothPos     = glm::vec3(0.0f);
static float     camSmoothYaw     = -45.0f;
static float     camSmoothPitch   = -12.0f;
static float     camSmoothDist    = 45.0f;
static bool      camFollowInited  = false;

// Minimum chase distance while following (scales with aircraft size).
static float followMinDist() {
    if (!engine || !camIsFollowing) return camMinDist;
    const AircraftRenderState ars = engine->aircraftRenderState();
    if (!ars.visible) return camMinDist;
    // Never closer than ~3.5× aircraft length — prevents zoom clipping into fuselage.
    return std::max(camMinDist, ars.halfExtents.x * 3.5f);
}

// ─── Runtime sun position editor ───────────────────────────────
// Arrow keys ↑/↓/←/→ adjust sun elevation/azimuth live.
// F7 saves current settings to terrain_settings.json.
// F8 reloads settings from terrain_settings.json.

static float  sunElev  = 30.0f;
static float  sunAzim  = 45.0f;
static int    titleUpdateCounter = 0;

// Helper — true when ImGui has the mouse/keyboard focus
static inline bool uiWantsMouse() {
    if (!ImGui::GetCurrentContext()) return false;
    return ImGui::GetIO().WantCaptureMouse;
}
static inline bool uiWantsKeyboard() {
    if (!ImGui::GetCurrentContext()) return false;
    return ImGui::GetIO().WantCaptureKeyboard;
}

void cursorPosCallback(GLFWwindow* /*window*/, double xpos, double ypos) {
    if (!engine || !mouseDown || uiWantsMouse()) {
        lastMouseX = xpos; lastMouseY = ypos;
        return;
    }
    double dx = xpos - lastMouseX;
    double dy = ypos - lastMouseY;
    lastMouseX = xpos; lastMouseY = ypos;

    Camera& cam = engine->getCamera();
    if (camIsFollowing) {
        // During aircraft follow: adjust the user orbit offset and pitch freely.
        // The main loop will blend this into the smooth heading; this prevents the
        // follow logic from immediately overriding mouse input every frame.
        camFollowYawOff += (float)dx * mouseSens;
        cam.pitch       -= (float)dy * mouseSens;
        cam.pitch        = glm::clamp(cam.pitch, -89.0f, 89.0f);
    } else {
        cam.yaw   += (float)dx * mouseSens;
        cam.pitch -= (float)dy * mouseSens;
        cam.pitch  = glm::clamp(cam.pitch, -89.0f, 89.0f);
    }
}

void mouseButtonCallback(GLFWwindow* /*w*/, int button, int action, int /*mods*/) {
    if (uiWantsMouse()) { mouseDown = false; return; }
    if (button == GLFW_MOUSE_BUTTON_LEFT)
        mouseDown = (action == GLFW_PRESS);
}

void scrollCallback(GLFWwindow* /*w*/, double /*xo*/, double yo) {
    if (!engine || uiWantsMouse()) return;
    Camera& cam = engine->getCamera();
    // Logarithmic zoom — each scroll step scales distance by a small percentage
    // instead of subtracting a huge linear amount (terrain_settings had zoomSens=50!).
    const float scrollScale = 0.10f * (zoomSens / 5.0f);
    const float factor = std::exp(-(float)yo * scrollScale);
    cam.dist *= factor;
    cam.dist  = glm::clamp(cam.dist, followMinDist(), camMaxDist);
}

int main() {
    // ── Load project settings ──────────────────────────────
    TerrainSettings settings = TerrainSettings::loadFromJSON("terrain_settings.json");

    // Apply camera/input settings from config (only once)
    mouseSens  = settings.mouseSensitivity;
    zoomSens   = settings.zoomSensitivity;
    camSpeed   = settings.cameraSpeed;
    sprintMult = settings.cameraSprintMult;
    camMinDist = settings.cameraMinDist;
    camMaxDist = settings.cameraMaxDist;

    // Override with settings defaults for runtime editor
    sunElev = settings.sunElevation;
    sunAzim = settings.sunAzimuth;

    // Check for a pending regeneration seed file (from F6 restart)
    {
        std::ifstream regenFile("regen_seed.txt");
        if (regenFile.is_open()) {
            uint32_t regenSeed;
            regenFile >> regenSeed;
            settings.worldSeed = regenSeed;
            regenFile.close();
            std::remove("regen_seed.txt");
            std::cout << "[seed] loaded regeneration seed: " << regenSeed << "\n";
        }
    }

    // ── Generate seed ──────────────────────────────────
    uint32_t seed = settings.activeWorldSeed();
    settings.worldSeed = seed;

    // Display seed prominently
    std::cout << "\n";
    std::cout << "══════════════════════════════════════════════════\n";
    std::cout << "  TerrainEngine v1.0 — SimpleHydrology Integration\n";
    std::cout << "  World Seed: " << seed << "\n";
    std::cout << "  (press F6 to regenerate with random seed)\n";
    std::cout << "══════════════════════════════════════════════════\n\n";

    // Save seed to file for later reference
    {
        std::ofstream seedFile("last_seed.txt");
        if (seedFile) {
            seedFile << seed << std::endl;
            std::cout << "[seed] saved to last_seed.txt\n";
        }
    }

    try {

        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        // GLFW error callback — crucial for diagnosing window/surface issues
        glfwSetErrorCallback([](int code, const char* desc) {
            std::cerr << "[GLFW ERROR] code=" << code << " — " << desc << std::endl;
        });

        std::string title = "TerrainEngine — Seed: " + std::to_string(seed);
        GLFWwindow* window = glfwCreateWindow(1280, 720,
            title.c_str(), nullptr, nullptr);
        if (!window) throw std::runtime_error("failed to create window");

        glfwSetCursorPosCallback(window, cursorPosCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int, int) {
            if (engine) engine->onResize();
        });
        glfwSetWindowCloseCallback(window, [](GLFWwindow* w) {
            std::cout << "[GLFW] window close requested" << std::endl;
        });

        VulkanEngine eng;
        engine = &eng;
        eng.init(window, settings);

        // ── Render loop ─────────────────────────────────────
        int frameCount = 0;
        std::cout << "[main] entering render loop..." << std::endl;
        auto lastFrameTime = std::chrono::high_resolution_clock::now();
        bool pWasDown = false;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            auto frameTime = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float>(frameTime - lastFrameTime).count();
            lastFrameTime = frameTime;
            dt = glm::clamp(dt, 0.001f, 0.1f);

            // Surprise early exit check
            if (glfwWindowShouldClose(window) && frameCount == 0) {
                std::cout << "[main] WINDOW CLOSED BEFORE FIRST FRAME! frameCount=0" << std::endl;
                std::cout << "[main] window ptr: " << window << std::endl;
                break;
            }

            // Pull live values from the engine's settings (UI can change them).
            const TerrainSettings& liveSet = eng.getSettings();
            mouseSens  = liveSet.mouseSensitivity;
            zoomSens   = liveSet.zoomSensitivity;
            camSpeed   = liveSet.cameraSpeed;
            sprintMult = liveSet.cameraSprintMult;

            // Skip game input while ImGui has keyboard focus
            bool keyboardLocked = uiWantsKeyboard();

            // Jolt Physics simulation (terrain mesh + dynamic bodies)
            eng.stepPhysics(dt);

            Camera& cam = eng.getCamera();
            const AircraftRenderState ars = eng.aircraftRenderState();
            const bool followingAircraft =
                liveSet.aircraftCameraFollowEnabled && ars.visible;
            camIsFollowing = followingAircraft;   // expose to cursorPosCallback

            if (followingAircraft) {
                const glm::vec3 planePos = ars.position;

                // Heading = horizontal projection of nose direction.
                // Using the full 3D forward vector caused yaw jitter during banking.
                const glm::vec3 nose3 = ars.rotation * glm::vec3(1.0f, 0.0f, 0.0f);
                const glm::vec2 noseH(nose3.x, nose3.z);
                float headingYaw = camSmoothYaw;
                if (glm::length(noseH) > 0.05f)
                    headingYaw = glm::degrees(std::atan2(noseH.y, noseH.x));

                const float minDist = followMinDist();

                if (!camFollowInited) {
                    camSmoothDist   = glm::clamp(ars.halfExtents.x * 5.5f, minDist, 85.0f);
                    cam.dist        = camSmoothDist;
                    camSmoothPitch  = -12.0f;
                    cam.pitch       = camSmoothPitch;
                    camSmoothYaw    = headingYaw;
                    camSmoothPos    = planePos + glm::vec3(0.0f, 1.2f, 0.0f);
                    camFollowYawOff = 0.0f;
                    camFollowInited = true;
                }

                // Heavy smoothing on pivot — removes physics micro-jitter.
                const glm::vec3 targetPos = planePos + glm::vec3(0.0f, 1.2f, 0.0f);
                const float posAlpha = 1.0f - std::exp(-dt / 0.35f);
                camSmoothPos = glm::mix(camSmoothPos, targetPos, posAlpha);
                cam.pos      = camSmoothPos;

                // Smooth yaw, pitch, and zoom independently.
                float targetYaw = headingYaw + camFollowYawOff;
                float dYaw = targetYaw - camSmoothYaw;
                if (dYaw >  180.0f) dYaw -= 360.0f;
                if (dYaw < -180.0f) dYaw += 360.0f;
                const float yawAlpha   = 1.0f - std::exp(-dt / 0.90f);
                const float pitchAlpha = 1.0f - std::exp(-dt / 0.40f);
                const float distAlpha  = 1.0f - std::exp(-dt / 0.25f);

                camSmoothYaw   += dYaw * yawAlpha;
                camSmoothPitch  = glm::mix(camSmoothPitch, cam.pitch, pitchAlpha);
                camSmoothDist   = glm::mix(camSmoothDist,
                                           glm::clamp(cam.dist, minDist, camMaxDist),
                                           distAlpha);

                cam.yaw   = camSmoothYaw;
                cam.pitch = camSmoothPitch;
                cam.dist  = camSmoothDist;

            } else {
                camFollowInited  = false;
                camFollowYawOff  = 0.0f;
                camSmoothPos     = cam.pos;
                camSmoothYaw     = cam.yaw;
                camSmoothPitch   = cam.pitch;
                camSmoothDist    = cam.dist;
            }

            // WASD / QE movement (free-fly camera)
            float speed = camSpeed * dt;
            if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed *= sprintMult;
            if (keyboardLocked || followingAircraft) speed = 0.0f;

            glm::vec3 forward(cos(glm::radians(cam.yaw)) * cos(glm::radians(cam.pitch)),
                              sin(glm::radians(cam.pitch)),
                              sin(glm::radians(cam.yaw)) * cos(glm::radians(cam.pitch)));
            glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0,1,0)));
            glm::vec3 up    = glm::cross(right, forward);

            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cam.pos += forward * speed;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cam.pos -= forward * speed;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cam.pos -= right   * speed;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cam.pos += right   * speed;
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) cam.pos -= up      * speed;
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) cam.pos += up      * speed;

            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GLFW_TRUE);

            // P: spawn a physics sphere in front of the camera
            bool pDown = (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS);
            if (pDown && !pWasDown && !keyboardLocked) {
                glm::vec3 forward(
                    cos(glm::radians(cam.yaw)) * cos(glm::radians(cam.pitch)),
                    sin(glm::radians(cam.pitch)),
                    sin(glm::radians(cam.yaw)) * cos(glm::radians(cam.pitch)));
                glm::vec3 spawnPos = cam.pos + glm::normalize(forward) * 25.0f;
                const TerrainSettings& live = eng.getSettings();
                eng.getPhysics().spawnSphere(spawnPos, live.physicsSphereRadius);
                std::cout << "[physics] spawned sphere at ("
                          << spawnPos.x << ", " << spawnPos.y << ", " << spawnPos.z << ")\n";
            }
            pWasDown = pDown;

            // F1-F4: cycle debug overlay modes
            static bool f1WasDown = false, f2WasDown = false, f3WasDown = false, f4WasDown = false;
            bool f1Down = (glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS);
            bool f2Down = (glfwGetKey(window, GLFW_KEY_F2) == GLFW_PRESS);
            bool f3Down = (glfwGetKey(window, GLFW_KEY_F3) == GLFW_PRESS);
            bool f4Down = (glfwGetKey(window, GLFW_KEY_F4) == GLFW_PRESS);

            if (f1Down && !f1WasDown) {
                std::cout << "[input] F1 — debug overlay OFF\n";
                eng.setDebugOverlay(DebugAnalyzer::DebugOverlayMode::NONE);
            }
            if (f2Down && !f2WasDown) {
                std::cout << "[input] F2 — height heatmap\n";
                eng.setDebugOverlay(DebugAnalyzer::DebugOverlayMode::HEIGHT_HEATMAP);
            }
            if (f3Down && !f3WasDown) {
                std::cout << "[input] F3 — slope map\n";
                eng.setDebugOverlay(DebugAnalyzer::DebugOverlayMode::SLOPE_MAP);
            }
            if (f4Down && !f4WasDown) {
                std::cout << "[input] F4 — land/water mask\n";
                eng.setDebugOverlay(DebugAnalyzer::DebugOverlayMode::LAND_WATER_MASK);
            }
            f1WasDown = f1Down; f2WasDown = f2Down; f3WasDown = f3Down; f4WasDown = f4Down;

            // F5: re-run terrain debug analysis and save JSON report
            static bool f5WasDown = false;
            bool f5Down = (glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS);
            if (f5Down && !f5WasDown) {
                std::cout << "[input] F5 — running deep terrain analysis...\n";
                eng.runTerrainAnalysis();
            }
            f5WasDown = f5Down;

            // F9: toggle cloud shadows on terrain
            static bool f9WasDown = false;
            bool f9Down = (glfwGetKey(window, GLFW_KEY_F9) == GLFW_PRESS);
            if (f9Down && !f9WasDown) eng.toggleCloudShadows();
            f9WasDown = f9Down;

            // F10: toggle cloud-shadow transmittance debug overlay
            static bool f10WasDown = false;
            bool f10Down = (glfwGetKey(window, GLFW_KEY_F10) == GLFW_PRESS);
            if (f10Down && !f10WasDown) eng.cycleCloudShadowDebug();
            f10WasDown = f10Down;

            // ─── Runtime sun position editor ─────────────────
            TerrainSettings& set = eng.getSettingsMutable();
            bool sunChanged = false;
            const float SUN_STEP = 1.0f; // degrees per key press

            // Arrow up: raise sun (increase elevation)
            if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
                sunElev += SUN_STEP;
                if (sunElev > 90.0f) sunElev = 90.0f;
                sunChanged = true;
            }
            // Arrow down: lower sun (decrease elevation)
            if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
                sunElev -= SUN_STEP;
                if (sunElev < -10.0f) sunElev = -10.0f;
                sunChanged = true;
            }
            // Arrow left: rotate sun counter-clockwise
            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
                sunAzim -= SUN_STEP;
                if (sunAzim < 0.0f) sunAzim += 360.0f;
                sunChanged = true;
            }
            // Arrow right: rotate sun clockwise
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
                sunAzim += SUN_STEP;
                if (sunAzim >= 360.0f) sunAzim -= 360.0f;
                sunChanged = true;
            }

            if (sunChanged) {
                set.sunElevation = sunElev;
                set.sunAzimuth   = sunAzim;
            }

            // F7: save current sun position to terrain_settings.json
            static bool f7WasDown = false;
            bool f7Down = (glfwGetKey(window, GLFW_KEY_F7) == GLFW_PRESS);
            if (f7Down && !f7WasDown) {
                std::ofstream settingsFile("terrain_settings.json");
                // Write a minimal JSON with sun settings preserved
                settingsFile << "{\n"
                             << "  \"sunElevation\": " << sunElev << ",\n"
                             << "  \"sunAzimuth\": " << sunAzim << "\n"
                             << "}\n";
                settingsFile.close();
                std::cout << "[sun] saved: elev=" << sunElev << " azim=" << sunAzim << "\n";
            }
            f7WasDown = f7Down;

            // F8: reload sun position from terrain_settings.json
            static bool f8WasDown = false;
            bool f8Down = (glfwGetKey(window, GLFW_KEY_F8) == GLFW_PRESS);
            if (f8Down && !f8WasDown) {
                TerrainSettings reloaded = TerrainSettings::loadFromJSON("terrain_settings.json");
                sunElev = reloaded.sunElevation;
                sunAzim = reloaded.sunAzimuth;
                set.sunElevation = sunElev;
                set.sunAzimuth   = sunAzim;
                std::cout << "[sun] reloaded: elev=" << sunElev << " azim=" << sunAzim << "\n";
            }
            f8WasDown = f8Down;

            // Update window title every 30 frames with sun info
            titleUpdateCounter++;
            if (titleUpdateCounter >= 30) {
                titleUpdateCounter = 0;
                std::string title = "TerrainEngine — Sun: elev=" +
                    std::to_string(static_cast<int>(sunElev)) + " deg  azim=" +
                    std::to_string(static_cast<int>(sunAzim)) + " deg  (arrows=move, F7=save, F8=load)";
                glfwSetWindowTitle(window, title.c_str());
            }

            // F6: restart with new random seed
            static bool f6WasDown = false;
            bool f6Down = (glfwGetKey(window, GLFW_KEY_F6) == GLFW_PRESS);
            if (f6Down && !f6WasDown) {
                std::random_device rd;
                uint32_t newSeed = rd();
                std::cout << "[input] F6 — restarting with new seed: " << newSeed << "\n";

                // Write the new seed to a file, then restart the EXE
                std::ofstream regenFile("regen_seed.txt");
                regenFile << newSeed;
                regenFile.close();

                // Flush all console output before restarting
                std::cout << std::flush;
                std::cerr << std::flush;

                // Spawn a new instance and close this one
                system("start \"\" \"TerrainEngine.exe\"");
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            f6WasDown = f6Down;

            eng.draw();
            if (frameCount < 3) {
                std::cout << "[main] frame " << frameCount << " completed" << std::endl;
            }
            frameCount++;
        }

        std::cout << "[main] render loop exited after " << frameCount
                  << " frames, window should close: "
                  << (glfwWindowShouldClose(window) ? "yes" : "no") << std::endl;

        eng.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();

    } catch (const std::exception& e) {
        std::cerr << "\n=============================================\n";
        std::cerr << "FATAL ERROR: " << e.what() << "\n";
        std::cerr << "=============================================\n\n";
        glfwTerminate();
        system("pause");
        return 1;
    }

    std::cout << "[main] normal exit" << std::endl;
    system("pause");
    return 0;
}
