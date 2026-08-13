#include "VulkanEngine.h"
#include "DebugAnalyzer.h"
#include "FastNoiseLite.h"   // OpenSimplex2 â€” same library as SimpleHydrology
#include "SettingsPanel.h"   // ImGui runtime parameter panel
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <cstdio>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Init / Draw / Cleanup
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::init(GLFWwindow* w, const TerrainSettings& cfg) {
    window = w;
    settings = cfg;
    masterSeaLevel = settings.seaLevel;
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createCommandPool();
    createDepthResources();
    createHeightmapSampler();
    createUniformBuffer();
    createAtmosphereUBO();
    createDescriptorLayouts();
    createDescriptorPools();
    std::cout << "  [init] initializing world data (SimpleHydrology fBm noise)..." << std::endl;
    initializeWorldData();

    // Centre the orbit camera in the new world and set sensible distance.
    camera.pos  = glm::vec3(settings.worldSize * 0.5f,
                            std::max(50.0f, globalMaxHeight * 1.2f),
                            settings.worldSize * 0.5f);
    camera.dist = settings.worldSize * 0.30f;
    camera.dist = std::max(camera.dist, 50.0f);
    createTexgenPipeline();    // must be before createTextureArray (needs texgenDescriptorSetLayout)
    createTextureArray();
    createTerrainMesh();
    createOceanMesh();          // flat grid for the animated Gerstner ocean
    createFoliagePipeline();    // foliage rendering pipeline
    createFoliageMesh();        // tree geometry (cone model)
    createAircraftPipeline();   // TEMPORARY: debug aircraft pipeline
    createAircraftMesh();       // TEMPORARY: debug aircraft silhouette
    createAirportPipeline();    // flat concrete + runway/taxiway/apron markings
    buildAirportMesh();         // paved-surface geometry for both demo airports
    createGearResources();      // visible spring-damper landing-gear struts
    createModelPipeline();          // shared textured model pipeline
    createModelDefaultTextures();   // 1x1 white + flat-normal fallbacks
    if (settings.aircraftUseModel)
        loadAircraftModel("models/uploads_files_6285295_cessna172lowpoly.fbx");
    buildAssetRegistry();           // editor palette: primitives + models/*.fbx
    createShadowMap();
    createShadowPipeline();
    createGraphicsPipeline();
    createOceanPipeline();      // ocean surface (reuses terrain descriptor layout)
    createAeroResources();         // airflow particle SSBO + aero UBOs + descriptors
    createAeroPipelines();         // airflow advect/streamline + pressure-surface pipelines
    createAeroPanelVizResources(); // panel-Cp SSBO (reads from actual physics panels)
    createAeroPanelVizPipeline();  // panel quad renderer
    createStreamlinePipeline();    // CPU-integrated curved streamline renderer
    // Cloud noise volume + params buffer must exist BEFORE terrain
    // descriptor sets are written (terrain.frag now samples them for
    // cloud shadows). The noise content is generated a few lines below,
    // still before the first frame. The image's final layout
    // (SHADER_READ_ONLY_OPTIMAL) is reached in runCloudNoiseGeneration().
    createCloudResources();
    createCloudShadowResources();
    createGlobalTerrainDescriptorSet();  // must be after all textures/samplers/pipelines
    generateGlobalFoliage();    // populate tree instances from vegetation system

    // â”€â”€â”€ Bruneton / Hillaire atmospheric scattering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    //  Build the three storage-image LUTs, their compute pipelines,
    //  the sky drawing pipeline, and run the two static LUT passes
    //  (transmittance + multiscatter) once. Skyview is updated per
    //  frame in recordRenderCommand().
    std::cout << "  [init] creating Bruneton atmospheric LUTs..." << std::endl;
    createBrunetonLUTs();
    createBrunetonComputePipelines();
    createSkyPassPipeline();
    runBrunetonStaticLUTs();

    // â”€â”€â”€ Volumetric clouds (Heckel + Hillaire + Schneider) â”€â”€â”€â”€â”€
    //  Bakes the 3D Worley-Perlin noise volume once and uploads
    //  a 64Ã—64 blue-noise dither tile. The cloud raymarch pass
    //  runs every frame after foliage in recordRenderCommand().
    std::cout << "  [init] creating volumetric cloud resources..." << std::endl;
    uploadBlueNoiseTexture();
    createCloudNoiseGenPipeline();
    runCloudNoiseGeneration();
    createCloudDrawPipeline();
    createCloudAccumBuffers();
    createCloudTemporalPipeline();
    createCloudCompositePipeline();

    std::cout << "  [init] creating command buffers..." << std::endl << std::flush;
    createCommandBuffers();

    std::cout << "  [init] creating sync objects..." << std::endl << std::flush;
    createSyncObjects();
    createGpuProfiler();

    std::cout << "  [init] initialising Jolt Physics..." << std::endl << std::flush;
    initPhysics();

    std::cout << "  [init] initialising ImGui settings panel..." << std::endl << std::flush;
    initImGui();

    // Wait for all GPU work from init to complete before first frame
    std::cout << "  [init] waiting for GPU idle..." << std::endl << std::flush;
    VkResult waitResult = vkDeviceWaitIdle(device);
    if (waitResult == VK_ERROR_DEVICE_LOST) {
        throw std::runtime_error("Vulkan device lost during initialization â€” "
            "this usually indicates a GPU timeout or driver issue. "
            "Check Vulkan validation layer output for details.");
    } else if (waitResult != VK_SUCCESS) {
        std::cerr << "  [init] WARNING: vkDeviceWaitIdle returned " << waitResult << std::endl;
    }

    std::cout << "\n=============================================" << std::endl << std::flush;
    std::cout << "  [init] ENGINE INITIALIZATION COMPLETE" << std::endl << std::flush;
    std::cout << "  [init] entering render loop..." << std::endl << std::flush;
    std::cout << "=============================================\n" << std::endl << std::flush;
}

void VulkanEngine::draw() {
    static int drawCallCount = 0;
    if (drawCallCount == 0) {
        std::cout << "  [draw] ENTERING draw() for the first time..." << std::endl << std::flush;
    }
    drawCallCount++;

    // â”€â”€â”€ FPS measurement (smoothed) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    static auto sPrev = std::chrono::high_resolution_clock::now();
    auto sNow = std::chrono::high_resolution_clock::now();
    float dt  = std::chrono::duration<float>(sNow - sPrev).count();
    sPrev = sNow;
    if (dt > 0.0f) {
        float fps = 1.0f / dt;
        if (fpsSmoothed <= 0.0f) fpsSmoothed = fps;
        else                     fpsSmoothed = glm::mix(fpsSmoothed, fps, 0.1f);
    }
    lastFrameDt = glm::clamp(dt, 0.0001f, 0.1f);
    // Cinematic day/night: advance the sun arc before the UBOs are built.
    updateTimeOfDay(lastFrameDt);
    try {
        // Wait for the oldest frame-in-flight to complete (frame pacing)
        if (drawCallCount <= 2) std::cout << "  [draw] step 1: vkWaitForFences..." << std::endl << std::flush;
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // GPU profiler: this frame slot's previous submission is now complete,
        // so its timestamps are ready to read without stalling.
        readGpuTimestamps(currentFrame);
        if (gpuProfilingEnabled) {
            static double gpuPrintAccum = 0.0;
            gpuPrintAccum += dt;
            if (gpuPrintAccum >= 1.0) {
                gpuPrintAccum = 0.0;
                std::cout << "[gpu-ms] total=" << std::fixed << std::setprecision(2) << gpuTotalMs
                          << "  sky=" << gpuPassMs[0]
                          << "  cloudShadow=" << gpuPassMs[1]
                          << "  shadow=" << gpuPassMs[2]
                          << "  main=" << gpuPassMs[3]
                          << "  clouds=" << gpuPassMs[4]
                          << "  ui=" << gpuPassMs[5]
                          << "  | cpuUpd=" << cpuUpdateMsSmoothed
                          << "  fps=" << fpsSmoothed << std::endl << std::flush;
            }
        }

        // Vegetation update (periodic, not every frame)
        vegetationUpdateCounter++;
        if (vegetationUpdateCounter >= settings.vegUpdateInterval) {
            vegetationUpdateCounter = 0;
            updateVegetation();
            updateGlobalFoliageBuffers();  // sync plant positions to GPU
        }

        uint32_t imageIndex = 0;
        if (swapchain == VK_NULL_HANDLE || imageAvailable.empty()) {
            std::cerr << "  [draw] swapchain null or semaphores empty, recreating..." << std::endl;
            recreateSwapchain();
            return;
        }
        if (drawCallCount <= 2) std::cout << "  [draw] step 2: vkAcquireNextImageKHR..." << std::endl << std::flush;
        VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
            imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            std::cout << "  [draw] swapchain out of date, recreating..." << std::endl;
            recreateSwapchain();
            return;
        }
        if (result == VK_ERROR_SURFACE_LOST_KHR) {
            std::cerr << "  [draw] surface lost, recreating swapchain..." << std::endl;
            recreateSwapchain();
            return;
        }
        if (result == VK_SUBOPTIMAL_KHR) {
            std::cout << "  [draw] swapchain suboptimal, will recreate after present" << std::endl;
        } else if (result != VK_SUCCESS) {
            std::cerr << "  [draw] vkAcquireNextImageKHR failed with code " << result << std::endl;
            throw std::runtime_error("failed to acquire swapchain image (code: " + std::to_string(result) + ")");
        }

        if (drawCallCount <= 2) std::cout << "  [draw] step 3: acquired imageIndex=" << imageIndex << std::endl << std::flush;

        // If this swapchain image still has an in-flight frame, wait for it
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }
        // Mark this image as having the current frame's fence in flight
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // Update UBO BEFORE recording commands so the GPU
        // sees the current frame's camera position and MVP matrix.
        if (drawCallCount <= 2) std::cout << "  [draw] step 4: updateUniformBuffer..." << std::endl << std::flush;
        auto cpuUpdateStart = std::chrono::high_resolution_clock::now();
        updateUniformBuffer(imageIndex);
        updateAeroGlobalUBO();
        updateAeroPanelVizData();
        if (settings.aeroVisualizationEnabled) {
            const uint32_t interval = std::max(1u, (uint32_t)settings.streamlineUpdateInterval);
            if ((streamlineFrameCounter++ % interval) == 0) {
                auto slStart = std::chrono::high_resolution_clock::now();
                updateStreamlines();
                float slMs = std::chrono::duration<float, std::milli>(
                    std::chrono::high_resolution_clock::now() - slStart).count();
                cpuStreamlineMsSmoothed = (cpuStreamlineMsSmoothed <= 0.0f)
                    ? slMs : glm::mix(cpuStreamlineMsSmoothed, slMs, 0.1f);
            }
        } else {
            streamlineFrameCounter = 0;
        }
        float cpuUpdateMs = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - cpuUpdateStart).count();
        cpuUpdateMsSmoothed = (cpuUpdateMsSmoothed <= 0.0f)
            ? cpuUpdateMs : glm::mix(cpuUpdateMsSmoothed, cpuUpdateMs, 0.1f);

        // â”€â”€â”€ Build ImGui draw lists for this frame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        drawSettingsUI();

        // â”€â”€â”€ Handle hot regeneration request from the UI â”€â”€â”€â”€â”€â”€
        // (must be done after we've used the current settings to build the
        //  UBO above, so the camera state is consistent within the frame;
        //  the regen does vkDeviceWaitIdle internally.)
        if (requestReseed) {
            requestReseed = false;
            // 0 means "random next time" â€” call activeWorldSeed to bake it in
            settings.worldSeed = (uint32_t)(std::chrono::system_clock::now()
                                            .time_since_epoch().count() & 0xFFFFFFFFu);
            if (settings.worldSeed == 0) settings.worldSeed = 1;
            requestRegen = true;
        }
        if (requestRegen) {
            requestRegen = false;
            regenerateTerrain();
            // After regen, ensure the camera sits inside the new world bounds
            requestResetCam = true;
        }
        if (requestResetCam) {
            requestResetCam = false;
            camera.pos  = glm::vec3(settings.worldSize * 0.5f,
                                    std::max(50.0f, globalMaxHeight * 1.2f),
                                    settings.worldSize * 0.5f);
            camera.dist = std::max(50.0f, settings.worldSize * 0.30f);
            camera.yaw  = -45.0f;
            camera.pitch = -30.0f;
        }
        if (requestRebuildPhysics) {
            requestRebuildPhysics = false;
            rebuildPhysicsTerrain();
            respawnAutonomousAircraft();
        }

        if (drawCallCount <= 2) std::cout << "  [draw] step 5: recording commands..." << std::endl << std::flush;
        if (drawCallCount <= 2) std::cout << "  [draw] step 5a: vkResetCommandBuffer..." << std::endl << std::flush;
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        if (drawCallCount <= 2) std::cout << "  [draw] step 5b: entering recordRenderCommand..." << std::endl << std::flush;
        recordRenderCommand(commandBuffers[currentFrame], imageIndex);

        VkSemaphore waitSemaphores[] = {imageAvailable[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore signalSemaphores[] = {renderFinished[imageIndex]};

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (drawCallCount <= 2) std::cout << "  [draw] step 6: vkQueueSubmit..." << std::endl << std::flush;
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
            throw std::runtime_error("failed to submit draw command buffer");

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIndex;

        if (drawCallCount <= 2) std::cout << "  [draw] step 7: vkQueuePresentKHR..." << std::endl << std::flush;
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapchain();
        } else if (result != VK_SUCCESS) {
            std::cerr << "  [draw] vkQueuePresentKHR failed with code " << result << std::endl;
            throw std::runtime_error("failed to present (code: " + std::to_string(result) + ")");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    } catch (const std::exception& e) {
        std::cerr << "  [draw] EXCEPTION: " << e.what() << std::endl;
        throw;  // re-throw to main loop handler
    }
}

void VulkanEngine::onResize() {
    framebufferResized = true;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  ImGui integration â€” runtime settings panel
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::initImGui() {
    QueueFamilyIndices q = findQueueFamilies(physicalDevice);
    SettingsPanel::init(
        window,
        instance,
        physicalDevice,
        device,
        q.graphics.value(),
        graphicsQueue,
        swapchainFormat,
        (uint32_t)swapchainImages.size(),   // min image count
        (uint32_t)swapchainImages.size(),   // image count
        swapchainViews,
        swapchainExtent,
        commandPool
    );
    imguiInitialised = true;
    std::cout << "  [imgui] settings panel initialised" << std::endl;
}

void VulkanEngine::shutdownImGui() {
    if (!imguiInitialised) return;
    vkDeviceWaitIdle(device);
    SettingsPanel::shutdown(device);
    imguiInitialised = false;
}

void VulkanEngine::drawSettingsUI() {
    if (!imguiInitialised) return;

    SettingsPanel::beginFrame();

    if (uiVisible) {
        // Cache stats from the most-recent hydrology pass
        uiLandCells  = masterFlowData.totalLandCells;
        uiMaxAccum   = masterFlowData.maxAccumulation;
        // Count river cells once
        if (uiRiverCells == 0 && !masterFlowData.isRiver.empty()) {
            int rc = 0;
            for (bool b : masterFlowData.isRiver) if (b) rc++;
            uiRiverCells = rc;
        }

        SettingsPanel::draw(settings,
                            &requestRegen, &requestReseed, &requestResetCam,
                            &requestRebuildPhysics,
                            fpsSmoothed,
                            uiLandCells, uiMaxAccum, uiRiverCells,
                            physics.isInitialized() ? &physics : nullptr,
                            settings.tempAutonomousAircraftEnabled ? &autonomousAircraft : nullptr);

        // Right-side flight-control panel (altitude/heading + takeoff).
        bool takeoffRequested = false;
        SettingsPanel::drawFlightControl((float)swapchainExtent.width,
                                         flightActive,
                                         physics.flightTelemetry(),
                                         &flightTargetAltMSL,
                                         &flightTargetHeadingDeg,
                                         &flightTargetSpeedMps,
                                         &takeoffRequested);
        if (takeoffRequested)
            requestAircraftTakeoff();

        drawEditorUI();

        if (uiVisible) {
            ImGui::Separator();
            ImGui::Text("Perf: update %.2f ms", cpuUpdateMsSmoothed);
            if (settings.aeroVisualizationEnabled)
                ImGui::Text("Perf: streamlines %.2f ms", cpuStreamlineMsSmoothed);
            ImGui::Text("Cloud samples: %d", cloudParams.sampleCount);
            if (gpuProfilingEnabled) {
                ImGui::Separator();
                ImGui::Text("GPU total: %.2f ms (%.0f fps)", gpuTotalMs, fpsSmoothed);
                ImGui::Text("  sky %.2f | cloudShadow %.2f | shadow %.2f",
                            gpuPassMs[0], gpuPassMs[1], gpuPassMs[2]);
                ImGui::Text("  main %.2f | clouds %.2f | ui %.2f",
                            gpuPassMs[3], gpuPassMs[4], gpuPassMs[5]);
            }
            ImGui::Separator();
            ImGui::TextUnformatted("Terrain LOD floor (0=full detail, higher=faster)");
            ImGui::SliderInt("View mesh##lod", &settings.terrainViewMinLod, 0, MAX_LODS - 1);
            ImGui::SliderInt("Shadow mesh##lod", &settings.terrainShadowMinLod, 0, MAX_LODS - 1);
        }
    } else {
        // Render an empty frame so ImGui state is valid
    }
}

// â”€â”€ Hot regeneration: rebuild the world without restarting â”€â”€â”€
void VulkanEngine::regenerateTerrain() {
    std::cout << "\nâ•â•â•â•â•â• regenerating terrain at runtime â•â•â•â•â•â•" << std::endl;
    vkDeviceWaitIdle(device);

    // Destroy GPU resources that depend on master heightmap data
    if (masterHeightmapSampler) { vkDestroySampler(device, masterHeightmapSampler, nullptr); masterHeightmapSampler = VK_NULL_HANDLE; }
    if (masterHeightmapImage.view)   { vkDestroyImageView(device, masterHeightmapImage.view,   nullptr); masterHeightmapImage.view   = VK_NULL_HANDLE; }
    if (masterHeightmapImage.image)  { vkDestroyImage    (device, masterHeightmapImage.image,  nullptr); masterHeightmapImage.image  = VK_NULL_HANDLE; }
    if (masterHeightmapImage.memory) { vkFreeMemory      (device, masterHeightmapImage.memory, nullptr); masterHeightmapImage.memory = VK_NULL_HANDLE; }

    // Rebuild the terrain mesh too (masterRes / worldSize may have changed)
    if (vertexBuffer.buffer) { vkDestroyBuffer(device, vertexBuffer.buffer, nullptr); vertexBuffer.buffer = VK_NULL_HANDLE; }
    if (vertexBuffer.memory) { vkFreeMemory  (device, vertexBuffer.memory, nullptr);  vertexBuffer.memory = VK_NULL_HANDLE; }
    if (indexBuffer.buffer)  { vkDestroyBuffer(device, indexBuffer.buffer,  nullptr); indexBuffer.buffer  = VK_NULL_HANDLE; }
    if (indexBuffer.memory)  { vkFreeMemory  (device, indexBuffer.memory,  nullptr);  indexBuffer.memory  = VK_NULL_HANDLE; }

    initializeWorldData();
    createTerrainMesh();

    // ── Re-bind the new heightmap / flow / river textures into the
    //    existing global terrain descriptor set (no re-allocation,
    //    so the pool isn't exhausted by repeated regenerations).
    {
        VkDescriptorImageInfo hmInfo{};
        hmInfo.sampler     = masterHeightmapSampler;
        hmInfo.imageView   = masterHeightmapImage.view;
        hmInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet w{};
        w.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet           = globalTerrainDescSet;
        w.dstBinding       = 1;
        w.descriptorType   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount  = 1;
        w.pImageInfo       = &hmInfo;
        for (int fi = 0; fi < MAX_FRAMES_IN_FLIGHT; ++fi) {
            w.dstSet = globalTerrainDescSets[fi];
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
        }
    }

    // Re-bake the procedural material array so its seed-derived content stays
    // consistent with the regenerated world. If the requested material
    // resolution changed, fully recreate the array and rebind binding 2.
    if (terrainTexArray.image != VK_NULL_HANDLE) {
        if ((uint32_t)settings.texRes != terrainTexBakedRes) {
            if (terrainTexSampler)      { vkDestroySampler(device, terrainTexSampler, nullptr); terrainTexSampler = VK_NULL_HANDLE; }
            if (terrainTexStorageView)  { vkDestroyImageView(device, terrainTexStorageView, nullptr); terrainTexStorageView = VK_NULL_HANDLE; }
            if (terrainTexArray.view)   { vkDestroyImageView(device, terrainTexArray.view, nullptr); terrainTexArray.view = VK_NULL_HANDLE; }
            if (terrainTexArray.image)  { vkDestroyImage(device, terrainTexArray.image, nullptr); terrainTexArray.image = VK_NULL_HANDLE; }
            if (terrainTexArray.memory) { vkFreeMemory(device, terrainTexArray.memory, nullptr); terrainTexArray.memory = VK_NULL_HANDLE; }
            createTextureArray();

            VkDescriptorImageInfo texInfo{terrainTexSampler, terrainTexArray.view,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstBinding = 2;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.descriptorCount = 1;
            w.pImageInfo = &texInfo;
            for (int fi = 0; fi < MAX_FRAMES_IN_FLIGHT; ++fi) {
                w.dstSet = globalTerrainDescSets[fi];
                vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
            }
        } else {
            dispatchTexgenAndMips();
        }
    }

    rebuildPhysicsTerrain();
    buildAirportMesh();          // pads moved with the new terrain -> rebuild surfaces
    respawnAutonomousAircraft();
    std::cout << "â•â•â•â•â•â• regeneration complete â•â•â•â•â•â•\n" << std::endl;
}

void VulkanEngine::initPhysics() {
    if (!settings.physicsEnabled) {
        std::cout << "  [physics] disabled in settings\n";
        return;
    }
    if (!physics.init()) return;

    rebuildPhysicsTerrain();
    if (settings.demoModeEnabled)
        startDemoMission();
    else if (settings.aircraftFlightEnabled)
        startAircraftFlight();
    else if (settings.aircraftDropTestEnabled)
        startAircraftDropTest();
    else if (settings.tempAutonomousAircraftEnabled)
        initAutonomousAircraft();
    else
        spawnPhysicsDemoBodies();
}

void VulkanEngine::initAutonomousAircraft() {
    // Airfields were removed: spawn at world centre and fly along the configured
    // heading to a destination a fixed distance away, sampling terrain for each.
    const glm::vec2 center(settings.worldSize * 0.5f, settings.worldSize * 0.5f);
    const float heading = glm::radians(settings.aircraftSpawnHeadingDeg);
    const float dist = std::max(500.0f, settings.aircraftDestinationDistanceMeters);
    const glm::vec2 destXZ = center + dist * glm::vec2(std::cos(heading), std::sin(heading));

    FlightPoint dep;
    dep.center = center;
    dep.elevation = sampleWorldHeight(center.x, center.y) * settings.terrainHeightScale;
    dep.headingRad = heading;
    dep.enabled = true;

    FlightPoint dest;
    dest.center = destXZ;
    dest.elevation = sampleWorldHeight(destXZ.x, destXZ.y) * settings.terrainHeightScale;
    dest.headingRad = heading;
    dest.enabled = true;

    autonomousAircraft.configure(dep, dest, settings);
    if (!settings.physicsEnabled || !physics.isInitialized()) {
        // Keep the demo observable even when physics is disabled/unavailable.
        autonomousAircraft.activateWithoutPhysics();
        return;
    }
    autonomousAircraft.shutdown(physics);
    physics.clearDynamicBodies();
    if (!autonomousAircraft.spawn(physics))
        std::cout << "  [aircraft/temp] spawn failed (physics initialized)\n";
}

void VulkanEngine::respawnAutonomousAircraft() {
    stopDemoMission();
    if (settings.demoModeEnabled) {
        startDemoMission();
        return;
    }
    if (settings.aircraftFlightEnabled) {
        startAircraftFlight();
        return;
    }
    if (settings.aircraftDropTestEnabled) {
        startAircraftDropTest();
        return;
    }

    dropTestActive = false;
    dropTestBodyIndex = UINT32_MAX;
    flightActive = false;
    flightBodyIndex = UINT32_MAX;

    if (settings.tempAutonomousAircraftEnabled) {
        initAutonomousAircraft();
        return;
    }

    autonomousAircraft.shutdown(physics);
    physics.clearDynamicBodies();
    spawnPhysicsDemoBodies();
}

void VulkanEngine::startAircraftDropTest() {
    dropTestActive = false;
    dropTestBodyIndex = UINT32_MAX;
    flightActive = false;
    flightBodyIndex = UINT32_MAX;
    if (!settings.physicsEnabled || !physics.isInitialized()) {
        std::cout << "  [droptest] physics unavailable — cannot run drop test\n";
        return;
    }

    // Stop the kinematic autopilot and clear any existing bodies.
    autonomousAircraft.shutdown(physics);
    physics.clearDynamicBodies();

    // Spawn spot: world centre, on the natural terrain.
    glm::vec2 center(settings.worldSize * 0.5f, settings.worldSize * 0.5f);
    float groundElev = sampleWorldHeight(center.x, center.y) * settings.terrainHeightScale;
    float headingRad = glm::radians(settings.aircraftSpawnHeadingDeg);

    const glm::vec3 he(std::max(1.0f, settings.aircraftHalfLength),
                       std::max(0.25f, settings.aircraftHalfHeight),
                       std::max(0.25f, settings.aircraftHalfWidth));

    const float spawnY = groundElev + he.y + std::max(1.0f, settings.aircraftDropHeightMeters);
    const glm::vec3 spawnPos(center.x, spawnY, center.y);
    const glm::quat rot = glm::angleAxis(headingRad, glm::vec3(0.0f, 1.0f, 0.0f));

    // Tricycle gear: one nose wheel forward, two mains aft, all at the fuselage belly.
    std::vector<GearWheel> wheels(3);
    const float rest = std::max(0.1f, settings.aircraftWheelRestLength);
    const float rad  = std::max(0.05f, settings.aircraftWheelRadius);
    auto makeWheel = [&](float lx, float lz) {
        GearWheel w;
        w.localAttach = glm::vec3(lx, -he.y, lz);
        w.restLength = rest;
        w.radius = rad;
        return w;
    };
    wheels[0] = makeWheel( he.x * 0.55f, 0.0f);
    wheels[1] = makeWheel(-he.x * 0.25f, -he.z * 0.85f);
    wheels[2] = makeWheel(-he.x * 0.25f,  he.z * 0.85f);

    dropTestBodyIndex = physics.spawnAircraftWithGear(
        spawnPos, rot, he, settings.aircraftMassKg, wheels,
        settings.aircraftSuspensionStiffness, settings.aircraftSuspensionDamping);

    dropTestActive = (dropTestBodyIndex != UINT32_MAX);
    dropTelemetryTimer = 0.0f;
    std::cout << "  [droptest] " << (dropTestActive ? "started" : "FAILED")
              << " spawnY=" << spawnY << " groundElev=" << groundElev
              << " dropH=" << settings.aircraftDropHeightMeters << "\n";
}

AircraftFlightConfig VulkanEngine::buildFlightConfig(float runwayElevationMSL) const {
    AircraftFlightConfig cfg;
    cfg.wingArea         = std::max(1.0f, settings.aircraftWingArea);
    cfg.liftCurveSlope   = std::max(1.0f, settings.aircraftLiftCurveSlope);
    cfg.stallAoADeg      = std::clamp(settings.aircraftStallAoADeg, 6.0f, 30.0f);
    cfg.maxThrustN       = std::max(100.0f, settings.aircraftMaxThrustN);
    cfg.takeoffSpeedMps  = std::max(5.0f, settings.aircraftTakeoffSpeedMps);
    cfg.runwayElevationMSL = runwayElevationMSL;
    cfg.windEnabled         = settings.aircraftWindEnabled;
    cfg.windSpeedMps        = std::max(0.0f, settings.aircraftWindSpeedMps);
    cfg.windDirDeg          = settings.aircraftWindDirDeg;
    cfg.turbulenceIntensity = std::clamp(settings.aircraftTurbulenceIntensity, 0.0f, 1.0f);
    return cfg;
}

void VulkanEngine::startAircraftFlight() {
    flightActive = false;
    flightBodyIndex = UINT32_MAX;
    dropTestActive = false;
    dropTestBodyIndex = UINT32_MAX;
    if (!settings.physicsEnabled || !physics.isInitialized()) {
        std::cout << "  [flight] physics unavailable — cannot start flight\n";
        return;
    }

    // Stop any kinematic autopilot and clear existing bodies.
    autonomousAircraft.shutdown(physics);
    physics.clearDynamicBodies();

    // Spawn at world centre, resting on the natural terrain.
    glm::vec2 center(settings.worldSize * 0.5f, settings.worldSize * 0.5f);
    float groundElev = sampleWorldHeight(center.x, center.y) * settings.terrainHeightScale;
    float headingRad = glm::radians(settings.aircraftSpawnHeadingDeg);

    const glm::vec3 he(std::max(1.0f, settings.aircraftHalfLength),
                       std::max(0.25f, settings.aircraftHalfHeight),
                       std::max(0.25f, settings.aircraftHalfWidth));

    const float rest = std::max(0.1f, settings.aircraftWheelRestLength);
    const float rad  = std::max(0.05f, settings.aircraftWheelRadius);

    // Spawn resting on the gear (wheels just touching) so it sits on the runway.
    const float spawnY = groundElev + he.y + rest + rad;
    const glm::vec3 spawnPos(center.x, spawnY, center.y);
    const glm::quat rot = glm::angleAxis(headingRad, glm::vec3(0.0f, 1.0f, 0.0f));

    std::vector<GearWheel> wheels(3);
    auto makeWheel = [&](float lx, float lz) {
        GearWheel w;
        w.localAttach = glm::vec3(lx, -he.y, lz);
        w.restLength = rest;
        w.radius = rad;
        return w;
    };
    wheels[0] = makeWheel( he.x * 0.55f, 0.0f);
    wheels[1] = makeWheel(-he.x * 0.25f, -he.z * 0.85f);
    wheels[2] = makeWheel(-he.x * 0.25f,  he.z * 0.85f);

    flightBodyIndex = physics.spawnAircraftWithGear(
        spawnPos, rot, he, settings.aircraftMassKg, wheels,
        settings.aircraftSuspensionStiffness, settings.aircraftSuspensionDamping);
    flightActive = (flightBodyIndex != UINT32_MAX);
    if (!flightActive) {
        std::cout << "  [flight] spawn FAILED\n";
        return;
    }

    // Seed GUI targets from the runway on first spawn; keep user edits afterward.
    const float runwayHeadingDeg = glm::degrees(headingRad);
    if (!flightTargetsInit) {
        flightTargetAltMSL = groundElev + std::max(30.0f, settings.aircraftCruiseAltitudeAgl);
        flightTargetHeadingDeg = runwayHeadingDeg;
        flightTargetSpeedMps = std::max(20.0f, settings.aircraftCruiseSpeedMps);
        flightTargetsInit = true;
    }

    flightRunwayElevMSL = groundElev;
    physics.setFlightEnabled(true, buildFlightConfig(groundElev));
    {
        // Shape-based aerodynamics: derive a realistic wing half-span from the
        // reference area at AR≈7 (keeps total area correct while giving the
        // panel layout / airflow vortices a believable spanwise extent).
        const float wingArea = std::max(1.0f, settings.aircraftWingArea);
        const float halfSpan = 0.5f * std::sqrt(7.0f * wingArea);
        AerodynamicBody ab;
        ab.buildFromHalfExtents(settings.aircraftHalfLength, halfSpan,
                                settings.aircraftHalfHeight, settings.aircraftMassKg,
                                wingArea, settings.aircraftStallAoADeg,
                                settings.aircraftLiftCurveSlope);
        physics.setAerodynamicBody(ab);
    }
    physics.setFlightTargets(flightTargetAltMSL,
                             glm::radians(flightTargetHeadingDeg),
                             flightTargetSpeedMps);
    flightTelemetryTimer = 0.0f;

    std::cout << "  [flight] ready on runway: groundElev=" << groundElev
              << " headingDeg=" << runwayHeadingDeg
              << " targetAltMSL=" << flightTargetAltMSL
              << " (press Start takeoff)\n";
}

void VulkanEngine::requestAircraftTakeoff() {
    if (flightActive && !missionActive) physics.requestTakeoff();
}

// ══════════════════════════════════════════════════════════════
//  Cinematic demo mission — taxi → takeoff → fly → land → taxi → park
// ══════════════════════════════════════════════════════════════

const char* VulkanEngine::missionPhaseName() const {
    switch (missionPhase) {
        case MissionPhase::Idle:        return "Idle";
        case MissionPhase::TaxiOut:     return "Taxi to runway";
        case MissionPhase::Lineup:      return "Line up";
        case MissionPhase::TakeoffRoll: return "Takeoff roll";
        case MissionPhase::Climb:       return "Climb";
        case MissionPhase::Cruise:      return "Cruise";
        case MissionPhase::Descent:     return "Descent";
        case MissionPhase::Approach:    return "Final approach";
        case MissionPhase::Flare:       return "Flare";
        case MissionPhase::Rollout:     return "Landing rollout";
        case MissionPhase::TaxiIn:      return "Taxi to gate";
        case MissionPhase::Park:        return "Parked";
    }
    return "?";
}

namespace {

inline float missionWrapToPi(float a) {
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kTwoPi = 2.0f * kPi;
    a = std::fmod(a + kPi, kTwoPi);
    if (a < 0.0f) a += kTwoPi;
    return a - kPi;
}

// Walk along the polyline from the current segment and return a point `dist`
// metres ahead — pure-pursuit steering avoids snapping the nose at each corner.
inline glm::vec2 pathLookahead(const std::vector<glm::vec2>& path, size_t wp,
                               glm::vec2 pos, float dist) {
    if (path.empty()) return pos;
    wp = std::min(wp, path.size() - 1);
    glm::vec2 segStart = pos;
    float remaining = dist;
    for (size_t i = wp; i < path.size(); ++i) {
        const glm::vec2 segEnd = path[i];
        glm::vec2 seg = segEnd - segStart;
        const float segLen = glm::length(seg);
        if (segLen < 1e-3f) {
            segStart = segEnd;
            continue;
        }
        if (remaining <= segLen)
            return segStart + seg * (remaining / segLen);
        remaining -= segLen;
        segStart = segEnd;
    }
    return path.back();
}

inline float polylineLength(const std::vector<glm::vec2>& path) {
    float len = 0.0f;
    for (size_t i = 1; i < path.size(); ++i)
        len += glm::length(path[i] - path[i - 1]);
    return len;
}

} // namespace

// Departure: gate → apron exit → parallel taxiway → runway hold/line-up point.
void VulkanEngine::buildDepartureTaxi() {
    missionPath.clear();
    missionWaypoint = 0;
    const Airport& A = airports[missionFrom];
    const float travelSign = missionTowardPosEnd ? 1.0f : -1.0f;
    const float halfR = A.halfRunway();
    const float startEndU = -travelSign * (halfR - A.taxiwayWidth * 0.6f);

    missionPath.push_back(A.local(A.apronCenterU(), A.apronCenterV() - A.apronHalfWid * 0.45f));
    missionPath.push_back(A.local(A.apronCenterU(), A.taxiwayOffset * 0.85f + A.apronCenterV() * 0.15f));
    missionPath.push_back(A.local(A.apronCenterU(), A.taxiwayOffset));
    missionPath.push_back(A.local(startEndU * 0.55f + A.apronCenterU() * 0.45f, A.taxiwayOffset));
    missionPath.push_back(A.local(startEndU, A.taxiwayOffset));
    missionPath.push_back(A.local(startEndU, A.taxiwayOffset * 0.72f));
    missionPath.push_back(A.local(startEndU, A.taxiwayOffset * 0.42f));
    missionPath.push_back(A.local(startEndU, 0.0f));
}

// Arrival: current runway position → taxiway exit → parallel taxiway → gate.
void VulkanEngine::buildArrivalTaxi() {
    missionPath.clear();
    missionWaypoint = 0;
    const Airport& B = airports[missionTo];
    const float travelSign = missionTowardPosEnd ? 1.0f : -1.0f;
    const float halfR = B.halfRunway();
    const float exitEndU = travelSign * (halfR - B.taxiwayWidth * 0.6f);

    missionPath.push_back(B.local(exitEndU, 0.0f));
    missionPath.push_back(B.local(exitEndU, B.taxiwayOffset * 0.35f));
    missionPath.push_back(B.local(exitEndU, B.taxiwayOffset));
    missionPath.push_back(B.local(B.apronCenterU() * 0.6f + exitEndU * 0.4f, B.taxiwayOffset));
    missionPath.push_back(B.local(B.apronCenterU(), B.taxiwayOffset));
    const AirportGate g = B.gates.empty() ? AirportGate{B.center, B.headingRad} : B.gates[0];
    missionPath.push_back(g.position);
}

// Scenic cruise: bulge the route around the terrain centre so the demo flies
// a wide arc and has room to climb into the cloud layer before descending.
void VulkanEngine::buildScenicFlightPath() {
    missionFlightPath.clear();
    missionFlightWaypoint = 0;
    const Airport& A = airports[missionFrom];
    const Airport& B = airports[missionTo];
    const glm::vec2 wc(settings.worldSize * 0.5f);

    // Bulge the route around the terrain mass (world centre), not the direct chord.
    const glm::vec2 arcCenter = wc;
    const float maxRadius = std::min(std::min(arcCenter.x, arcCenter.y),
                                     std::min(settings.worldSize - arcCenter.x,
                                              settings.worldSize - arcCenter.y)) - 180.0f;
    const float arcRadius = std::min(
        std::max(glm::length(A.center - arcCenter), glm::length(B.center - arcCenter)) * 1.18f,
        maxRadius);

    float angA = std::atan2(A.center.y - arcCenter.y, A.center.x - arcCenter.x);
    float angB = std::atan2(B.center.y - arcCenter.y, B.center.x - arcCenter.x);
    float delta = angB - angA;
    // Always take the longer scenic arc around the terrain mass.
    constexpr float kTwoPi = 6.28318530718f;
    if (std::abs(delta) < 3.14159265f)
        delta += (delta >= 0.0f) ? -kTwoPi : kTwoPi;

    const int n = 20;
    for (int i = 1; i <= n; ++i) {
        const float t = (float)i / (float)(n + 1);
        const float ang = angA + delta * t;
        missionFlightPath.push_back(arcCenter + glm::vec2(std::cos(ang), std::sin(ang)) * arcRadius);
    }

    const float travelSign = missionTowardPosEnd ? 1.0f : -1.0f;
    const float halfR = B.halfRunway();
    const float entryU = -travelSign * (halfR - 50.0f);
    const glm::vec2 threshold = B.local(entryU, 0.0f);
    const glm::vec2 rwyDir(std::cos(B.headingRad), std::sin(B.headingRad));
    const float appDist = glm::min(3200.0f, glm::length(threshold - A.center) * 0.42f);
    glm::vec2 approachFix = threshold - rwyDir * appDist;
    const float margin = 220.0f;
    approachFix = glm::clamp(approachFix, glm::vec2(margin), glm::vec2(settings.worldSize - margin));
    missionFlightPath.push_back(approachFix);

    missionFlightPathLen = polylineLength(missionFlightPath)
                         + glm::length(A.center - missionFlightPath.front())
                         + glm::length(B.center - missionFlightPath.back());
}

void VulkanEngine::startDemoMission() {
    flightActive = false; flightBodyIndex = UINT32_MAX;
    dropTestActive = false; dropTestBodyIndex = UINT32_MAX;
    missionActive = false;
    if (!settings.physicsEnabled || !physics.isInitialized()) {
        std::cout << "  [demo] physics unavailable — cannot start mission\n";
        return;
    }
    if (!airportsValid) layoutAirports();

    autonomousAircraft.shutdown(physics);
    physics.clearDynamicBodies();

    missionFrom = 0; missionTo = 1;
    const Airport& A = airports[missionFrom];
    const Airport& B = airports[missionTo];
    missionTowardPosEnd = glm::dot(B.center - A.center, A.dir()) > 0.0f;

    const glm::vec3 he(std::max(1.0f, settings.aircraftHalfLength),
                       std::max(0.25f, settings.aircraftHalfHeight),
                       std::max(0.25f, settings.aircraftHalfWidth));
    missionAircraftHalf = he;
    const float rest = std::max(0.1f, settings.aircraftWheelRestLength);
    const float rad  = std::max(0.05f, settings.aircraftWheelRadius);

    const AirportGate gate = A.gates.empty() ? AirportGate{A.center, A.headingRad} : A.gates[0];
    const float spawnY = A.elevation + he.y + rest + rad;
    const glm::vec3 spawnPos(gate.position.x, spawnY, gate.position.y);
    const glm::quat rot = glm::angleAxis(gate.headingRad, glm::vec3(0.0f, 1.0f, 0.0f));

    std::vector<GearWheel> wheels(3);
    auto makeWheel = [&](float lx, float lz) {
        GearWheel w; w.localAttach = glm::vec3(lx, -he.y, lz); w.restLength = rest; w.radius = rad; return w;
    };
    wheels[0] = makeWheel( he.x * 0.55f, 0.0f);
    wheels[1] = makeWheel(-he.x * 0.25f, -he.z * 0.85f);
    wheels[2] = makeWheel(-he.x * 0.25f,  he.z * 0.85f);

    flightBodyIndex = physics.spawnAircraftWithGear(
        spawnPos, rot, he, settings.aircraftMassKg, wheels,
        settings.aircraftSuspensionStiffness, settings.aircraftSuspensionDamping);
    if (flightBodyIndex == UINT32_MAX) { std::cout << "  [demo] spawn FAILED\n"; return; }
    flightActive = true;
    flightRunwayElevMSL = A.elevation;

    physics.setFlightEnabled(true, buildFlightConfig(A.elevation));
    {
        const float wingArea = std::max(1.0f, settings.aircraftWingArea);
        const float halfSpan = 0.5f * std::sqrt(7.0f * wingArea);
        AerodynamicBody ab;
        ab.buildFromHalfExtents(settings.aircraftHalfLength, halfSpan,
                                settings.aircraftHalfHeight, settings.aircraftMassKg,
                                wingArea, settings.aircraftStallAoADeg,
                                settings.aircraftLiftCurveSlope);
        physics.setAerodynamicBody(ab);
    }
    physics.setMissionMode(true);
    missionActive = true;
    missionPhase = MissionPhase::TaxiOut;
    missionPhaseTimer = 0.0f;
    missionHoldTimer = 0.0f;
    buildDepartureTaxi();
    buildScenicFlightPath();
    std::cout << "  [demo] mission started: gate A -> takeoff -> fly -> land B\n";
}

void VulkanEngine::updateDemoMission(float dt) {
    if (!missionActive || flightBodyIndex == UINT32_MAX) return;
    const auto& bodies = physics.getDynamicBodies();
    if (flightBodyIndex >= bodies.size()) return;
    const PhysicsBodyState& b = bodies[flightBodyIndex];
    missionPhaseTimer += dt;

    const Airport& A = airports[missionFrom];
    const Airport& B = airports[missionTo];
    const glm::vec2 posXZ(b.position.x, b.position.z);
    const glm::vec3 vel = b.velocity;
    const float groundSpeed = glm::length(glm::vec2(vel.x, vel.z));
    const glm::vec3 fwd = b.rotation * glm::vec3(1.0f, 0.0f, 0.0f);
    const float curHdg = std::atan2(fwd.z, fwd.x);

    // Operational direction of travel along the shared runway axis.
    const glm::vec2 travelVec = B.center - A.center;
    const float opHeading = std::atan2(travelVec.y, travelVec.x);
    const float travelSign = missionTowardPosEnd ? 1.0f : -1.0f;
    const float halfR = B.halfRunway();

    // Localizer: steer to capture and hold the extended runway centerline
    // (through B along the runway axis) by biasing heading from cross-track error.
    const glm::vec2 rwyAxis(std::cos(opHeading), std::sin(opHeading));
    const glm::vec2 rwyPerp(-rwyAxis.y, rwyAxis.x);
    const float landHdg = missionTowardPosEnd ? B.headingRad : B.headingRad + 3.14159265f;
    const glm::vec2 landDir(std::cos(landHdg), std::sin(landHdg));

    const float gtanDeg = glm::clamp(settings.aircraftGlideSlopeDeg, 2.5f, 5.0f);
    const float gtan = std::tan(glm::radians(gtanDeg));
    const float Vr   = std::max(12.0f, settings.aircraftTakeoffSpeedMps);
    const float Vcr  = std::max(Vr + 12.0f, settings.aircraftCruiseSpeedMps);
    const float Vapp = Vr * 1.3f;
    // Cruise altitude: sized to the scenic route length. Target the cloud layer
    // when the distance allows it; never force a floor above what the leg fits.
    const float climbTan   = std::tan(glm::radians(9.0f));
    const float cloudTargetAGL = 2200.0f;
    const float routeLen = std::max(missionFlightPathLen, glm::length(travelVec));
    const float fitAGL     = (routeLen - 1200.0f) / (1.0f / gtan + 1.0f / climbTan);
    const float cruiseAGL  = glm::clamp(std::min(cloudTargetAGL, fitAGL), 250.0f, 2800.0f);
    const float cruiseMSL  = std::max(A.elevation, B.elevation) + cruiseAGL;

    // Touchdown aim point on the destination runway (just past the threshold).
    const float entryU_B = -travelSign * (halfR - 50.0f);
    const glm::vec2 touchdownXZ = B.local(entryU_B + travelSign * 160.0f, 0.0f);
    const float distTD = glm::length(touchdownXZ - posXZ);
    const float altAGL_B = b.position.y - B.elevation;

    bool anyGrounded = false;
    for (int i = 0; i < physics.gearWheelCount(); ++i)
        if (physics.gearWheel(i).grounded) anyGrounded = true;

    FlightControl fc;
    auto bearingTo = [&](glm::vec2 t) { glm::vec2 d = t - posXZ; return std::atan2(d.y, d.x); };

    auto centerlineHeading = [&]() {
        const float xtrack = glm::dot(posXZ - B.center, rwyPerp);
        return landHdg + glm::clamp(-xtrack * 0.02f, -0.6f, 0.6f);
    };
    auto interceptHeading = [&]() {
        return (distTD < 1800.0f) ? centerlineHeading() : bearingTo(touchdownXZ);
    };
    auto checkRunwayOvershoot = [&]() -> bool {
        const float beyond = glm::dot(posXZ - touchdownXZ, landDir);
        if (beyond > 180.0f && altAGL_B > 12.0f) {
            missionPhase = MissionPhase::Climb;
            missionPhaseTimer = 0.0f;
            missionFlightWaypoint = missionFlightPath.empty() ? 0
                : std::min(missionFlightWaypoint, missionFlightPath.size() - 1);
            return true;
        }
        return false;
    };

    auto steerAlongPath = [&](const std::vector<glm::vec2>& path, size_t& wp,
                              float lookahead, float straightSpeed, float turnSpeed) {
        if (wp >= path.size()) return;
        const glm::vec2 steerPt = pathLookahead(path, wp, posXZ, lookahead);
        fc.targetHeadingRad = bearingTo(steerPt);
        const float wpHdg = bearingTo(path[wp]);
        const float turnErr = std::abs(missionWrapToPi(wpHdg - curHdg));
        fc.targetSpeedMps = turnErr > 0.45f ? turnSpeed
                          : (turnErr > 0.22f ? straightSpeed * 0.65f : straightSpeed);
        const float dist = glm::length(path[wp] - posXZ);
        const bool last = (wp + 1 >= path.size());
        if (dist < (last ? 10.0f : 18.0f)) wp++;
    };

    auto headingToFlight = [&]() {
        if (missionFlightPath.empty()) return centerlineHeading();
        if (missionFlightWaypoint >= missionFlightPath.size())
            return centerlineHeading();
        const float la = 900.0f + groundSpeed * 12.0f;
        const glm::vec2 steerPt = pathLookahead(missionFlightPath, missionFlightWaypoint,
                                                posXZ, la);
        return bearingTo(steerPt);
    };

    switch (missionPhase) {
    case MissionPhase::TaxiOut: {
        fc.mode = FlightControl::Taxi; fc.displayPhase = FlightPhase::Taxi;
        if (missionWaypoint < missionPath.size()) {
            const float la = 8.0f + groundSpeed * 2.0f;
            steerAlongPath(missionPath, missionWaypoint, la, 9.0f, 4.5f);
        } else {
            missionPhase = MissionPhase::Lineup; missionPhaseTimer = 0.0f;
        }
        break;
    }
    case MissionPhase::Lineup: {
        fc.mode = FlightControl::Taxi; fc.displayPhase = FlightPhase::Taxi;
        fc.targetHeadingRad = opHeading; fc.targetSpeedMps = 3.0f;
        float hdgErr = std::abs(std::atan2(std::sin(opHeading - curHdg), std::cos(opHeading - curHdg)));
        if ((hdgErr < 0.10f && groundSpeed < 3.5f) || missionPhaseTimer > 6.0f) {
            missionPhase = MissionPhase::TakeoffRoll; missionPhaseTimer = 0.0f;
        }
        break;
    }
    case MissionPhase::TakeoffRoll: {
        fc.mode = FlightControl::TakeoffRoll; fc.displayPhase = FlightPhase::Takeoff;
        fc.targetHeadingRad = opHeading;
        float altAGL_A = b.position.y - A.elevation;
        if (altAGL_A > 12.0f && vel.y > 0.5f) {
            missionPhase = MissionPhase::Climb; missionPhaseTimer = 0.0f;
        }
        break;
    }
    case MissionPhase::Climb: {
        fc.mode = FlightControl::Airborne; fc.displayPhase = FlightPhase::Climb;
        fc.targetHeadingRad = headingToFlight();
        fc.targetAltMSL = cruiseMSL; fc.targetSpeedMps = Vcr;
        if (!missionFlightPath.empty() && missionFlightWaypoint < missionFlightPath.size()) {
            if (glm::length(missionFlightPath[missionFlightWaypoint] - posXZ) < 650.0f)
                missionFlightWaypoint++;
        }
        if (b.position.y >= cruiseMSL - 40.0f) { missionPhase = MissionPhase::Cruise; }
        break;
    }
    case MissionPhase::Cruise: {
        fc.mode = FlightControl::Airborne; fc.displayPhase = FlightPhase::Cruise;
        fc.targetHeadingRad = headingToFlight();
        fc.targetAltMSL = cruiseMSL; fc.targetSpeedMps = Vcr;
        if (!missionFlightPath.empty() && missionFlightWaypoint < missionFlightPath.size()) {
            if (glm::length(missionFlightPath[missionFlightWaypoint] - posXZ) < 550.0f)
                missionFlightWaypoint++;
        }
        const bool pastScenic = missionFlightPath.empty()
                             || missionFlightWaypoint + 2 >= missionFlightPath.size();
        const float descentRange = cruiseAGL / gtan + 350.0f;
        if (pastScenic && distTD < descentRange) { missionPhase = MissionPhase::Descent; }
        break;
    }
    case MissionPhase::Descent: {
        fc.mode = FlightControl::Airborne; fc.displayPhase = FlightPhase::Descent;
        if (checkRunwayOvershoot()) {
            fc.displayPhase = FlightPhase::Climb;
            fc.targetHeadingRad = bearingTo(touchdownXZ);
            fc.targetAltMSL = cruiseMSL; fc.targetSpeedMps = Vcr;
            break;
        }
        fc.targetHeadingRad = interceptHeading();
        const float glideAlt = B.elevation + distTD * gtan;
        fc.targetAltMSL = glm::clamp(glideAlt, B.elevation + 3.0f, cruiseMSL);
        fc.targetSpeedMps = glm::mix(Vcr, Vapp, 0.5f);
        fc.climbRateFF = (glideAlt < cruiseMSL) ? -gtan * groundSpeed : 0.0f;
        if (distTD < 1500.0f) { missionPhase = MissionPhase::Approach; }
        break;
    }
    case MissionPhase::Approach: {
        fc.mode = FlightControl::Airborne; fc.displayPhase = FlightPhase::Approach;
        if (checkRunwayOvershoot()) {
            fc.displayPhase = FlightPhase::Climb;
            fc.targetHeadingRad = bearingTo(touchdownXZ);
            fc.targetAltMSL = cruiseMSL; fc.targetSpeedMps = Vcr;
            break;
        }
        fc.targetHeadingRad = interceptHeading();
        const float glideAlt = B.elevation + distTD * gtan;
        fc.targetAltMSL = glm::clamp(glideAlt, B.elevation + 2.0f, cruiseMSL);
        fc.targetSpeedMps = Vapp;
        fc.climbRateFF = (glideAlt < cruiseMSL) ? -gtan * groundSpeed : 0.0f;
        if (altAGL_B < 12.0f) { missionPhase = MissionPhase::Flare; missionPhaseTimer = 0.0f; }
        break;
    }
    case MissionPhase::Flare: {
        fc.mode = FlightControl::Airborne; fc.displayPhase = FlightPhase::Flare;
        fc.targetHeadingRad = centerlineHeading();
        fc.targetAltMSL = B.elevation + 1.0f;       // settle gently onto the wheels
        fc.targetSpeedMps = Vapp * 0.9f;
        if (anyGrounded && altAGL_B < (missionAircraftHalf.y + 1.2f)) {
            missionPhase = MissionPhase::Rollout; missionPhaseTimer = 0.0f;
        } else if (missionPhaseTimer > 12.0f) {     // safety: never get stuck flaring
            missionPhase = MissionPhase::Rollout; missionPhaseTimer = 0.0f;
        }
        break;
    }
    case MissionPhase::Rollout: {
        fc.mode = FlightControl::Rollout; fc.displayPhase = FlightPhase::Rollout;
        fc.targetHeadingRad = centerlineHeading(); fc.targetSpeedMps = 0.0f; fc.brake = 1.0f;
        if (groundSpeed < 4.0f) {
            buildArrivalTaxi();
            missionPhase = MissionPhase::TaxiIn; missionPhaseTimer = 0.0f;
        }
        break;
    }
    case MissionPhase::TaxiIn: {
        fc.mode = FlightControl::Taxi; fc.displayPhase = FlightPhase::Taxi;
        if (missionWaypoint < missionPath.size()) {
            const float la = 8.0f + groundSpeed * 2.0f;
            steerAlongPath(missionPath, missionWaypoint, la, 8.0f, 4.0f);
        } else {
            missionPhase = MissionPhase::Park; missionPhaseTimer = 0.0f; missionHoldTimer = 0.0f;
        }
        break;
    }
    case MissionPhase::Park: {
        fc.mode = FlightControl::Hold; fc.displayPhase = FlightPhase::Parked;
        fc.targetHeadingRad = curHdg; fc.brake = 1.0f;
        missionHoldTimer += dt;
        if (missionHoldTimer > 6.0f) {
            // Fly the return leg: swap departure/destination and start over.
            std::swap(missionFrom, missionTo);
            const Airport& nA = airports[missionFrom];
            const Airport& nB = airports[missionTo];
            missionTowardPosEnd = glm::dot(nB.center - nA.center, nA.dir()) > 0.0f;
            const float rest = std::max(0.1f, settings.aircraftWheelRestLength);
            const float rad  = std::max(0.05f, settings.aircraftWheelRadius);
            const AirportGate g = nA.gates.empty() ? AirportGate{nA.center, nA.headingRad} : nA.gates[0];
            const float y = nA.elevation + missionAircraftHalf.y + rest + rad;
            physics.placeAircraft(glm::vec3(g.position.x, y, g.position.y),
                                  glm::angleAxis(g.headingRad, glm::vec3(0.0f, 1.0f, 0.0f)));
            flightRunwayElevMSL = nA.elevation;
            physics.setFlightConfig(buildFlightConfig(nA.elevation));
            buildDepartureTaxi();
            buildScenicFlightPath();
            missionPhase = MissionPhase::TaxiOut; missionHoldTimer = 0.0f; missionPhaseTimer = 0.0f;
        }
        break;
    }
    default: break;
    }

    physics.setFlightControl(fc);
}

void VulkanEngine::stopDemoMission() {
    missionActive = false;
    physics.setMissionMode(false);
    physics.resetFlightToIdle();
}

// ── Visible landing gear ──────────────────────────────────────────────
// A small host-visible mesh (3 struts + 3 wheels as boxes) rebuilt every
// frame from live suspension telemetry, so the oleo struts visibly compress
// and rebound on touchdown. Drawn with the aircraft pipeline in world space.
namespace { struct GVert { glm::vec3 pos, color, normal; }; }

void VulkanEngine::createGearResources() {
    // 6 boxes (3 struts + 3 wheels) * 24 verts/box.
    const uint32_t maxBoxes = 6;
    gearVertexCapacity = maxBoxes * 24;
    const VkDeviceSize vbSize = gearVertexCapacity * sizeof(GVert);

    createBuffer(vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        gearVertexBuffer);
    vkMapMemory(device, gearVertexBuffer.memory, 0, vbSize, 0, &gearVertexMapped);

    // Static index buffer: 6 boxes * 36 indices, box winding matches createAircraftMesh.
    std::vector<uint32_t> indices;
    for (uint32_t bx = 0; bx < maxBoxes; ++bx) {
        uint32_t base = bx * 24;
        for (uint32_t f = 0; f < 6; ++f) {
            uint32_t b = base + f * 4;
            indices.push_back(b + 0); indices.push_back(b + 1); indices.push_back(b + 2);
            indices.push_back(b + 0); indices.push_back(b + 2); indices.push_back(b + 3);
        }
    }
    const VkDeviceSize ibSize = indices.size() * sizeof(uint32_t);
    Buffer staging;
    createBuffer(ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
    void* data; vkMapMemory(device, staging.memory, 0, ibSize, 0, &data);
    memcpy(data, indices.data(), ibSize); vkUnmapMemory(device, staging.memory);
    createBuffer(ibSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gearIndexBuffer);
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy copy{}; copy.size = ibSize;
    vkCmdCopyBuffer(cmd, staging.buffer, gearIndexBuffer.buffer, 1, &copy);
    endSingleTimeCommands(cmd);
    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);
    gearResourcesReady = true;
}

void VulkanEngine::updateGearMesh() {
    gearIndexCount = 0;
    if (!gearResourcesReady || !gearVertexMapped) return;
    if (!flightActive || flightBodyIndex == UINT32_MAX) return;
    if (!physics.hasAircraftGear()) return;
    const auto& bodies = physics.getDynamicBodies();
    if (flightBodyIndex >= bodies.size()) return;
    const PhysicsBodyState& body = bodies[flightBodyIndex];

    const glm::mat4 M = glm::translate(glm::mat4(1.0f), body.position)
                      * glm::mat4_cast(body.rotation);

    std::vector<GVert> verts;
    verts.reserve(gearVertexCapacity);

    auto addBox = [&](glm::vec3 mn, glm::vec3 mx, glm::vec3 color) {
        const glm::vec3 c[8] = {
            {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mx.x, mx.y, mn.z}, {mn.x, mx.y, mn.z},
            {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z}, {mx.x, mx.y, mx.z}, {mn.x, mx.y, mx.z},
        };
        struct Face { int a, b, c2, d; glm::vec3 n; };
        const Face faces[6] = {
            {1, 0, 3, 2, { 0, 0,-1}}, {4, 5, 6, 7, { 0, 0, 1}},
            {0, 4, 7, 3, {-1, 0, 0}}, {5, 1, 2, 6, { 1, 0, 0}},
            {3, 7, 6, 2, { 0, 1, 0}}, {0, 1, 5, 4, { 0,-1, 0}},
        };
        for (const Face& f : faces) {
            glm::vec3 wn = glm::normalize(glm::vec3(M * glm::vec4(f.n, 0.0f)));
            auto wp = [&](int i){ return glm::vec3(M * glm::vec4(c[i], 1.0f)); };
            verts.push_back({wp(f.a), color, wn});
            verts.push_back({wp(f.b), color, wn});
            verts.push_back({wp(f.c2), color, wn});
            verts.push_back({wp(f.d), color, wn});
        }
    };

    const glm::vec3 kStrut(0.85f, 0.86f, 0.90f);   // bright so compression reads
    const glm::vec3 kTire (0.06f, 0.06f, 0.07f);
    const int n = physics.gearWheelCount();
    int drawn = 0;
    for (int i = 0; i < n && drawn < 3; ++i, ++drawn) {
        const GearWheel& w = physics.gearWheel(i);
        const float strutLen = std::max(0.05f, w.restLength - w.compression);   // visible travel
        const float r = std::max(0.08f, w.radius);
        const float t = r * 0.45f;          // strut half-thickness
        const glm::vec3 a = w.localAttach;  // body-local attach (top of strut)
        const float wheelY = a.y - strutLen;       // wheel centre (body-local)
        // Strut: thin box from attach down to wheel centre.
        addBox(glm::vec3(a.x - t, wheelY, a.z - t),
               glm::vec3(a.x + t, a.y,    a.z + t), kStrut);
        // Wheel: flattened box (axle across the body's Z).
        addBox(glm::vec3(a.x - r,        wheelY - r, a.z - r * 0.5f),
               glm::vec3(a.x + r,        wheelY + r, a.z + r * 0.5f), kTire);
    }

    if (verts.empty()) return;
    if (verts.size() > gearVertexCapacity) verts.resize(gearVertexCapacity);
    memcpy(gearVertexMapped, verts.data(), verts.size() * sizeof(GVert));
    gearIndexCount = (uint32_t)((verts.size() / 24) * 36);
}

AircraftRenderState VulkanEngine::aircraftRenderState() const {
    const uint32_t idx = flightActive ? flightBodyIndex
                       : (dropTestActive ? dropTestBodyIndex : UINT32_MAX);
    if (idx != UINT32_MAX) {
        const auto& bodies = physics.getDynamicBodies();
        if (idx < bodies.size()) {
            const PhysicsBodyState& b = bodies[idx];
            AircraftRenderState rs;
            rs.position = b.position;
            rs.rotation = b.rotation;
            rs.halfExtents = b.halfExtents;
            rs.visible = true;
            return rs;
        }
    }
    return autonomousAircraft.renderState();
}

glm::vec3 VulkanEngine::aircraftLinearVelocity() const {
    const uint32_t idx = flightActive ? flightBodyIndex
                       : (dropTestActive ? dropTestBodyIndex : UINT32_MAX);
    if (idx != UINT32_MAX) {
        const auto& bodies = physics.getDynamicBodies();
        if (idx < bodies.size())
            return bodies[idx].velocity;
    }
    return autonomousAircraft.velocity();
}

void VulkanEngine::stepDropTestTelemetry(float deltaTime) {
    if (!dropTestActive || dropTestBodyIndex == UINT32_MAX) return;
    const auto& bodies = physics.getDynamicBodies();
    if (dropTestBodyIndex >= bodies.size()) return;
    const PhysicsBodyState& b = bodies[dropTestBodyIndex];

    dropTelemetryTimer += deltaTime;
    if (dropTelemetryTimer < 0.5f) return;
    dropTelemetryTimer = 0.0f;

    // Fuselage-to-ground clearance: min over the 8 box corners (world up only).
    float minClearance = 1e30f;
    const glm::quat q = b.rotation;
    for (int sx = -1; sx <= 1; sx += 2)
    for (int sy = -1; sy <= 1; sy += 2)
    for (int sz = -1; sz <= 1; sz += 2) {
        glm::vec3 local(sx * b.halfExtents.x, sy * b.halfExtents.y, sz * b.halfExtents.z);
        glm::vec3 world = b.position + q * local;
        float ground = sampleWorldHeight(world.x, world.z) * settings.terrainHeightScale;
        minClearance = std::min(minClearance, world.y - ground);
    }

    std::cout << "[droptest] y=" << b.position.y
              << " vY=" << b.velocity.y
              << " fuselageClearance=" << minClearance
              << (minClearance <= 0.0f ? " <-- FUSELAGE CONTACT!" : "")
              << " | wheels:";
    for (int i = 0; i < physics.gearWheelCount(); ++i) {
        const GearWheel& w = physics.gearWheel(i);
        std::cout << " [" << i << (w.grounded ? "G" : ".")
                  << " comp=" << w.compression
                  << " load=" << w.load << "N]";
    }
    std::cout << "\n";
}

void VulkanEngine::spawnPhysicsDemoBodies() {
    if (!settings.physicsEnabled || !physics.isInitialized()) return;

    const float cx = settings.worldSize * 0.5f;
    const float cz = settings.worldSize * 0.5f;
    const int n = std::max(0, settings.physicsSpawnCount);
    int spawned = 0;
    for (int i = 0; i < n; ++i) {
        const float ox = ((i % 4) - 1.5f) * 12.0f;
        const float oz = ((i / 4) - 1.5f) * 12.0f;
        float x = cx + ox;
        float z = cz + oz;
        float groundY = sampleWorldHeight(x, z) * settings.terrainHeightScale;

        // If spawn point is underwater, search nearby for dry land.
        if (groundY < settings.seaLevel) {
            bool found = false;
            for (float r = 20.0f; r <= 200.0f && !found; r += 20.0f) {
                for (int a = 0; a < 8; ++a) {
                    const float ang = (float)a * (3.14159265f / 4.0f);
                    const float tx = cx + cosf(ang) * r;
                    const float tz = cz + sinf(ang) * r;
                    const float th = sampleWorldHeight(tx, tz) * settings.terrainHeightScale;
                    if (th >= settings.seaLevel) {
                        x = tx; z = tz; groundY = th;
                        found = true;
                        break;
                    }
                }
            }
            if (!found) continue;
        }

        const float y = groundY + 40.0f + (float)(i % 3) * 8.0f;
        physics.spawnSphere(glm::vec3(x, y, z), settings.physicsSphereRadius);
        ++spawned;
    }
    if (spawned > 0)
        std::cout << "  [physics] spawned " << spawned << " demo spheres on land (P = spawn more)\n";
}

void VulkanEngine::rebuildPhysicsTerrain() {
    if (!settings.physicsEnabled || !physics.isInitialized()) return;

    if (masterHeightData.empty()) return;
    physics.rebuildTerrain(
        masterHeightData,
        settings.masterRes,
        settings.worldSize,
        settings.terrainHeightScale,
        settings.seaLevel,
        settings.physicsMeshStride);
}

void VulkanEngine::stepPhysics(float deltaTime) {
    // Cinematic demo: the mission controller computes the per-frame flight
    // command (taxi/takeoff/airborne/rollout) before the physics step executes
    // it with real forces.
    if (missionActive && !editMode) {
        updateDemoMission(deltaTime);
    }
    // Push the latest GUI targets + aero config to the flight model each frame
    // (so panel sliders are live without a respawn). Skipped in mission mode.
    else if (flightActive) {
        physics.setFlightConfig(buildFlightConfig(flightRunwayElevMSL));
        physics.setFlightTargets(flightTargetAltMSL,
                                 glm::radians(flightTargetHeadingDeg),
                                 flightTargetSpeedMps);
    }

    if (!dropTestActive && !flightActive &&
        settings.tempAutonomousAircraftEnabled && autonomousAircraft.isActive()) {
        // Edit mode freezes the flight state machine so objects can be placed.
        autonomousAircraft.setAutopilotPaused(settings.aircraftDebugPauseAutopilot || editMode);
        autonomousAircraft.update(deltaTime, physics);
    }
    if (settings.physicsEnabled && physics.isInitialized())
        physics.step(deltaTime);

    // Rebuild the visible landing-gear struts from the live suspension state.
    updateGearMesh();

    stepDropTestTelemetry(deltaTime);

    // 1 Hz flight telemetry to the console.
    if (flightActive) {
        flightTelemetryTimer += deltaTime;
        if (flightTelemetryTimer >= 1.0f) {
            flightTelemetryTimer = 0.0f;
            const FlightTelemetry t = physics.flightTelemetry();
            if (missionActive) {
                const auto& bs = physics.getDynamicBodies();
                glm::vec3 p = (flightBodyIndex < bs.size()) ? bs[flightBodyIndex].position : glm::vec3(0);
                float gnd = sampleWorldHeight(p.x, p.z) * settings.terrainHeightScale;
                std::cout << "[demo] " << missionPhaseName()
                          << " pos=(" << p.x << "," << p.y << "," << p.z << ")"
                          << " gnd=" << gnd
                          << " spd=" << t.airspeed << " gs=" << glm::length(glm::vec2(bs[flightBodyIndex].velocity.x, bs[flightBodyIndex].velocity.z))
                          << " altMSL=" << t.altitudeMSL << " thr=" << t.throttle
                          << " pitch=" << t.pitchDeg << " cmdP=" << t.cmdPitchDeg
                          << " tgtAlt=" << t.targetAltitudeMSL << " vs=" << t.verticalSpeed << " aoa=" << t.aoaDeg
                          << " wp=" << missionWaypoint << "/" << missionPath.size() << "\n";
            }
            const char* ph = (t.phase == FlightPhase::Takeoff) ? "TAKEOFF"
                           : (t.phase == FlightPhase::Cruise)  ? "CRUISE" : "IDLE";
            std::cout << "[flight] " << ph
                      << " spd=" << t.airspeed << "m/s altMSL=" << t.altitudeMSL
                      << " agl=" << t.altitudeAGL << " vs=" << t.verticalSpeed
                      << " aoa=" << t.aoaDeg << "deg hdg=" << t.headingDeg
                      << "->" << t.targetHeadingDeg << " thr=" << t.throttle
                      << " pitch=" << t.pitchDeg << " bank=" << t.bankDeg << "\n";
        }
    }
}

void VulkanEngine::cleanup() {
    vkDeviceWaitIdle(device);

    autonomousAircraft.shutdown(physics);
    physics.shutdown();

    // ImGui must be shut down before destroying its descriptor pool / device
    shutdownImGui();

    // Sync objects
    for (auto& sem : renderFinished) vkDestroySemaphore(device, sem, nullptr);
    for (auto& sem : imageAvailable) vkDestroySemaphore(device, sem, nullptr);
    for (auto& fence : inFlightFences) vkDestroyFence(device, fence, nullptr);
    imagesInFlight.clear();

    vkDestroyCommandPool(device, commandPool, nullptr);

    // Pipeline
    vkDestroyPipeline(device, terrainPipeline, nullptr);
    vkDestroyPipelineLayout(device, terrainPipelineLayout, nullptr);

    // Ocean pass
    if (oceanPipeline)        vkDestroyPipeline(device, oceanPipeline, nullptr);
    if (oceanPipelineLayout)  vkDestroyPipelineLayout(device, oceanPipelineLayout, nullptr);
    if (oceanVertexBuffer.buffer) vkDestroyBuffer(device, oceanVertexBuffer.buffer, nullptr);
    if (oceanVertexBuffer.memory) vkFreeMemory(device, oceanVertexBuffer.memory, nullptr);
    if (oceanIndexBuffer.buffer)  vkDestroyBuffer(device, oceanIndexBuffer.buffer, nullptr);
    if (oceanIndexBuffer.memory)  vkFreeMemory(device, oceanIndexBuffer.memory, nullptr);

    // Aerodynamic airflow visualization
    destroyAeroResources();

    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorPool(device, computeDescriptorPool, nullptr);

    // Shadow map
    vkDestroyPipeline(device, shadowPipeline, nullptr);
    vkDestroyPipelineLayout(device, shadowPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, shadowDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(device, shadowDescriptorPool, nullptr);
    vkDestroySampler(device, shadowSampler, nullptr);
    vkDestroyImageView(device, shadowMap.view, nullptr);
    vkDestroyImage(device, shadowMap.image, nullptr);
    vkFreeMemory(device, shadowMap.memory, nullptr);
    vkDestroyBuffer(device, shadowUniformBuffer.buffer, nullptr);
    vkFreeMemory(device, shadowUniformBuffer.memory, nullptr);

    // Foliage
    vkDestroyPipeline(device, foliagePipeline, nullptr);
    vkDestroyPipelineLayout(device, foliagePipelineLayout, nullptr);
    vkDestroyBuffer(device, foliageVertexBuffer.buffer, nullptr);
    vkFreeMemory(device, foliageVertexBuffer.memory, nullptr);
    vkDestroyBuffer(device, foliageIndexBuffer.buffer, nullptr);
    vkFreeMemory(device, foliageIndexBuffer.memory, nullptr);
    vkDestroyBuffer(device, globalFoliageInstanceBuffer.buffer, nullptr);
    vkFreeMemory(device, globalFoliageInstanceBuffer.memory, nullptr);
    vkDestroyBuffer(device, globalFoliageColorBuffer.buffer, nullptr);
    vkFreeMemory(device, globalFoliageColorBuffer.memory, nullptr);

    // TEMPORARY: aircraft
    vkDestroyPipeline(device, aircraftPipeline, nullptr);
    vkDestroyPipelineLayout(device, aircraftPipelineLayout, nullptr);
    vkDestroyBuffer(device, aircraftVertexBuffer.buffer, nullptr);
    vkFreeMemory(device, aircraftVertexBuffer.memory, nullptr);
    vkDestroyBuffer(device, aircraftIndexBuffer.buffer, nullptr);
    vkFreeMemory(device, aircraftIndexBuffer.memory, nullptr);

    // Landing gear
    if (gearVertexMapped)      { vkUnmapMemory(device, gearVertexBuffer.memory); gearVertexMapped = nullptr; }
    if (gearVertexBuffer.buffer) vkDestroyBuffer(device, gearVertexBuffer.buffer, nullptr);
    if (gearVertexBuffer.memory) vkFreeMemory(device, gearVertexBuffer.memory, nullptr);
    if (gearIndexBuffer.buffer)  vkDestroyBuffer(device, gearIndexBuffer.buffer, nullptr);
    if (gearIndexBuffer.memory)  vkFreeMemory(device, gearIndexBuffer.memory, nullptr);

    // Airports
    if (airportPipeline)       vkDestroyPipeline(device, airportPipeline, nullptr);
    if (airportPipelineLayout) vkDestroyPipelineLayout(device, airportPipelineLayout, nullptr);
    if (airportVertexBuffer.buffer) vkDestroyBuffer(device, airportVertexBuffer.buffer, nullptr);
    if (airportVertexBuffer.memory) vkFreeMemory(device, airportVertexBuffer.memory, nullptr);
    if (airportIndexBuffer.buffer)  vkDestroyBuffer(device, airportIndexBuffer.buffer, nullptr);
    if (airportIndexBuffer.memory)  vkFreeMemory(device, airportIndexBuffer.memory, nullptr);

    destroyModelResources();

    // TexGen + texture array
    vkDestroyPipeline(device, texgenPipeline, nullptr);
    vkDestroyPipelineLayout(device, texgenPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, texgenDescriptorSetLayout, nullptr);
    vkDestroySampler(device, terrainTexSampler, nullptr);
    if (terrainTexStorageView) vkDestroyImageView(device, terrainTexStorageView, nullptr);
    vkDestroyImageView(device, terrainTexArray.view, nullptr);
    vkDestroyImage(device, terrainTexArray.image, nullptr);
    vkFreeMemory(device, terrainTexArray.memory, nullptr);

    // Heightmap sampler
    vkDestroySampler(device, heightmapSampler, nullptr);

    // Master heightmap (CPU-only, no GPU resources to free)

    // Debug overlay resources
    if (debugOverlaySampler) vkDestroySampler(device, debugOverlaySampler, nullptr);
    if (debugOverlayImage.view) vkDestroyImageView(device, debugOverlayImage.view, nullptr);
    if (debugOverlayImage.image) vkDestroyImage(device, debugOverlayImage.image, nullptr);
    if (debugOverlayImage.memory) vkFreeMemory(device, debugOverlayImage.memory, nullptr);

    // Mesh
    vkDestroyBuffer(device, vertexBuffer.buffer, nullptr);
    vkFreeMemory(device, vertexBuffer.memory, nullptr);
    vkDestroyBuffer(device, indexBuffer.buffer, nullptr);
    vkFreeMemory(device, indexBuffer.memory, nullptr);

    // UBOs (one per frame in flight)
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (uniformBuffers[i].buffer) vkDestroyBuffer(device, uniformBuffers[i].buffer, nullptr);
        if (uniformBuffers[i].memory) vkFreeMemory(device, uniformBuffers[i].memory, nullptr);
    }

    if (gpuQueryPool) vkDestroyQueryPool(device, gpuQueryPool, nullptr);

    // Cloud shadow pre-bake
    if (cloudShadowPipeline)    vkDestroyPipeline(device, cloudShadowPipeline, nullptr);
    if (cloudShadowLayout)      vkDestroyPipelineLayout(device, cloudShadowLayout, nullptr);
    if (cloudShadowSetLayout)   vkDestroyDescriptorSetLayout(device, cloudShadowSetLayout, nullptr);
    if (cloudShadowDescPool)    vkDestroyDescriptorPool(device, cloudShadowDescPool, nullptr);
    if (cloudShadowSampler)     vkDestroySampler(device, cloudShadowSampler, nullptr);
    if (cloudShadowMap.view)    vkDestroyImageView(device, cloudShadowMap.view, nullptr);
    if (cloudShadowMap.image)   vkDestroyImage(device, cloudShadowMap.image, nullptr);
    if (cloudShadowMap.memory)  vkFreeMemory(device, cloudShadowMap.memory, nullptr);

    // Bruneton atmosphere LUTs + pipelines
    if (skyPassPipeline)         vkDestroyPipeline(device, skyPassPipeline, nullptr);
    if (skyPassPipelineLayout)   vkDestroyPipelineLayout(device, skyPassPipelineLayout, nullptr);
    if (skyPassSetLayout)        vkDestroyDescriptorSetLayout(device, skyPassSetLayout, nullptr);
    if (transmittancePipeline)   vkDestroyPipeline(device, transmittancePipeline, nullptr);
    if (multiscatterPipeline)    vkDestroyPipeline(device, multiscatterPipeline, nullptr);
    if (skyviewPipeline)         vkDestroyPipeline(device, skyviewPipeline, nullptr);
    if (brunetonComputeLayout)   vkDestroyPipelineLayout(device, brunetonComputeLayout, nullptr);
    if (brunetonComputeSetLayout) vkDestroyDescriptorSetLayout(device, brunetonComputeSetLayout, nullptr);
    if (brunetonDescriptorPool)  vkDestroyDescriptorPool(device, brunetonDescriptorPool, nullptr);
    if (lutSampler)              vkDestroySampler(device, lutSampler, nullptr);
    auto destroyLUT = [&](Image& im) {
        if (im.view)   vkDestroyImageView(device, im.view, nullptr);
        if (im.image)  vkDestroyImage(device, im.image, nullptr);
        if (im.memory) vkFreeMemory(device, im.memory, nullptr);
    };
    destroyLUT(transmittanceLUT);
    destroyLUT(multiscatterLUT);
    destroyLUT(skyviewLUT);

    // Volumetric clouds
    if (cloudDrawPipeline)        vkDestroyPipeline(device, cloudDrawPipeline, nullptr);
    if (cloudDrawPipelineLayout)  vkDestroyPipelineLayout(device, cloudDrawPipelineLayout, nullptr);
    if (cloudDrawSetLayout)       vkDestroyDescriptorSetLayout(device, cloudDrawSetLayout, nullptr);
    // Temporal cloud accumulation
    if (cloudTemporalPipeline)        vkDestroyPipeline(device, cloudTemporalPipeline, nullptr);
    if (cloudTemporalPipelineLayout)  vkDestroyPipelineLayout(device, cloudTemporalPipelineLayout, nullptr);
    if (cloudTemporalSetLayout)       vkDestroyDescriptorSetLayout(device, cloudTemporalSetLayout, nullptr);
    if (cloudCompositePipeline)       vkDestroyPipeline(device, cloudCompositePipeline, nullptr);
    if (cloudCompositePipelineLayout) vkDestroyPipelineLayout(device, cloudCompositePipelineLayout, nullptr);
    if (cloudCompositeSetLayout)      vkDestroyDescriptorSetLayout(device, cloudCompositeSetLayout, nullptr);
    if (cloudLinearSampler)           vkDestroySampler(device, cloudLinearSampler, nullptr);
    if (cloudNoiseGenPipeline)    vkDestroyPipeline(device, cloudNoiseGenPipeline, nullptr);
    if (cloudNoiseGenLayout)      vkDestroyPipelineLayout(device, cloudNoiseGenLayout, nullptr);
    if (cloudNoiseGenSetLayout)   vkDestroyDescriptorSetLayout(device, cloudNoiseGenSetLayout, nullptr);
    if (cloudDescriptorPool)      vkDestroyDescriptorPool(device, cloudDescriptorPool, nullptr);
    if (cloudNoiseSampler)        vkDestroySampler(device, cloudNoiseSampler, nullptr);
    if (blueNoiseSampler)         vkDestroySampler(device, blueNoiseSampler, nullptr);
    if (depthSampler)             vkDestroySampler(device, depthSampler, nullptr);
    destroyLUT(cloudNoise3D);
    destroyLUT(blueNoise2D);
    destroyLUT(cloudHalfResBuffer);
    destroyLUT(cloudAccumBuffer[0]);
    destroyLUT(cloudAccumBuffer[1]);
    if (cloudParamsMapped) {
        vkUnmapMemory(device, cloudParamsBuffer.memory);
        cloudParamsMapped = nullptr;
    }
    if (cloudParamsBuffer.buffer) {
        vkDestroyBuffer(device, cloudParamsBuffer.buffer, nullptr);
        vkFreeMemory(device, cloudParamsBuffer.memory, nullptr);
    }
    if (cloudTemporalMapped) {
        vkUnmapMemory(device, cloudTemporalUBO.memory);
        cloudTemporalMapped = nullptr;
    }
    if (cloudTemporalUBO.buffer) {
        vkDestroyBuffer(device, cloudTemporalUBO.buffer, nullptr);
        vkFreeMemory(device, cloudTemporalUBO.memory, nullptr);
    }

    // Atmosphere UBO
    if (atmosphereUniformMapped) {
        vkUnmapMemory(device, atmosphereUniformBuffer.memory);
        atmosphereUniformMapped = nullptr;
    }
    if (atmosphereUniformBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, atmosphereUniformBuffer.buffer, nullptr);
        vkFreeMemory(device, atmosphereUniformBuffer.memory, nullptr);
    }

    // Depth
    vkDestroyImageView(device, depth.view, nullptr);
    vkDestroyImage(device, depth.image, nullptr);
    vkFreeMemory(device, depth.memory, nullptr);

    cleanupSwapchain();

    vkDestroyDevice(device, nullptr);

    if (enableValidationLayers)
        destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Instance
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

bool VulkanEngine::checkValidationLayerSupport() {
    uint32_t count;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> available(count);
    vkEnumerateInstanceLayerProperties(&count, available.data());
    for (const char* name : validationLayers) {
        bool found = false;
        for (const auto& layer : available) {
            if (strcmp(name, layer.layerName) == 0) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

void VulkanEngine::createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport())
        throw std::runtime_error("validation layers requested but not available");

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "TerrainEngine";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "TerrainEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);
    if (enableValidationLayers)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
        populateDebugMessengerCreateInfo(debugInfo);
        createInfo.pNext = &debugInfo;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("failed to create instance");
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Debug messenger
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanEngine::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* userData)
{
    std::cerr << "[Vulkan] " << data->pMessage << "\n";
    return VK_FALSE;
}

void VulkanEngine::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& info) {
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                         | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                     | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                     | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugCallback;
}

VkResult VulkanEngine::createDebugUtilsMessengerEXT(VkInstance inst,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(inst, "vkCreateDebugUtilsMessengerEXT");
    if (func) return func(inst, pCreateInfo, pAllocator, pDebugMessenger);
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void VulkanEngine::destroyDebugUtilsMessengerEXT(VkInstance inst,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(inst, "vkDestroyDebugUtilsMessengerEXT");
    if (func) func(inst, debugMessenger, pAllocator);
}

void VulkanEngine::setupDebugMessenger() {
    if (!enableValidationLayers) return;
    VkDebugUtilsMessengerCreateInfoEXT info{};
    populateDebugMessengerCreateInfo(info);
    if (createDebugUtilsMessengerEXT(instance, &info, nullptr, &debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("failed to set up debug messenger");
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Surface
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("failed to create window surface");
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Physical device
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

QueueFamilyIndices VulkanEngine::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; i++) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            indices.graphics = i;
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport) indices.present = i;
        if (indices.isComplete()) break;
    }
    return indices;
}

bool VulkanEngine::isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    if (!indices.isComplete()) return false;

    // Check required extensions
    uint32_t extCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, available.data());

    std::set<std::string> required = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
        // dynamicRendering and synchronization2 are core Vulkan 1.3 features
    };
    for (const auto& ext : available)
        required.erase(ext.extensionName);

    if (!required.empty()) return false;

    // Check swapchain support
    SwapchainSupport swap = querySwapchainSupport(device);
    return !swap.formats.empty() && !swap.presentModes.empty();
}

SwapchainSupport VulkanEngine::querySwapchainSupport(VkPhysicalDevice device) {
    SwapchainSupport support;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &support.capabilities);

    uint32_t count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr);
    if (count) {
        support.formats.resize(count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, support.formats.data());
    }
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, nullptr);
    if (count) {
        support.presentModes.resize(count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, support.presentModes.data());
    }
    return support;
}

void VulkanEngine::pickPhysicalDevice() {
    uint32_t count;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) throw std::runtime_error("no Vulkan-capable GPU found");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    for (auto& d : devices) {
        if (isDeviceSuitable(d)) { physicalDevice = d; return; }
    }
    throw std::runtime_error("no suitable GPU found");
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Logical device
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    graphicsFamily = indices.graphics.value();
    presentFamily = indices.present.value();

    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    std::set<uint32_t> unique = {graphicsFamily, presentFamily};
    float priority = 1.0f;
    for (uint32_t family : unique) {
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = family;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        queueInfos.push_back(qi);
    }

    // Vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.features.samplerAnisotropy = VK_TRUE;
    features2.pNext = &features13;

    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
        // dynamicRendering and synchronization2 are core in Vulkan 1.3
    };

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &features2;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos = queueInfos.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("failed to create logical device");

    vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, presentFamily, 0, &presentQueue);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Swapchain
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

VkSurfaceFormatKHR VulkanEngine::chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& available)
{
    for (const auto& f : available)
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    return available[0];
}

VkPresentModeKHR VulkanEngine::chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR>& available)
{
    for (const auto& m : available)
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanEngine::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& caps) {
    if (caps.currentExtent.width != UINT32_MAX) return caps.currentExtent;
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    return {
        std::clamp((uint32_t)w, caps.minImageExtent.width, caps.maxImageExtent.width),
        std::clamp((uint32_t)h, caps.minImageExtent.height, caps.maxImageExtent.height)
    };
}

void VulkanEngine::createSwapchain() {
    SwapchainSupport support = querySwapchainSupport(physicalDevice);
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(support.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(support.presentModes);
    VkExtent2D extent = chooseSwapExtent(support.capabilities);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
        imageCount = support.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t families[] = {indices.graphics.value(), indices.present.value()};
    if (indices.graphics != indices.present) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = families;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS)
        throw std::runtime_error("failed to create swapchain");

    swapchainFormat = surfaceFormat.format;
    swapchainExtent = extent;

    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

    swapchainViews.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; i++) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = swapchainImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapchainFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &viewInfo, nullptr, &swapchainViews[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create swapchain image view");
    }
}

void VulkanEngine::cleanupSwapchain() {
    vkDestroyImageView(device, depth.view, nullptr);
    vkDestroyImage(device, depth.image, nullptr);
    vkFreeMemory(device, depth.memory, nullptr);
    for (auto& v : swapchainViews) vkDestroyImageView(device, v, nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

void VulkanEngine::recreateSwapchain() {
    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);
    while (w == 0 || h == 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &w, &h);
    }
    vkDeviceWaitIdle(device);

    // Destroy old sync objects before recreating swapchain
    for (auto& sem : renderFinished) vkDestroySemaphore(device, sem, nullptr);
    for (auto& sem : imageAvailable) vkDestroySemaphore(device, sem, nullptr);
    for (auto& fence : inFlightFences) vkDestroyFence(device, fence, nullptr);
    imagesInFlight.clear();

    cleanupSwapchain();
    createSwapchain();
    createDepthResources();
    recreateCloudAccumBuffers();  // recreate size-dependent cloud temporal buffers

    // Cloud raymarch samples the depth image. The view was recreated by
    // createDepthResources() so the existing descriptor entry now points
    // at a destroyed view â€” rewrite it.
    if (cloudDrawSet != VK_NULL_HANDLE) {
        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler     = depthSampler;
        depthInfo.imageView   = depth.view;
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = cloudDrawSet;
        w.dstBinding = 2;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo = &depthInfo;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }

    createSyncObjects();

    // Rebuild ImGui framebuffers if the panel is active
    if (imguiInitialised) {
        SettingsPanel::destroyFramebuffers(device);
        SettingsPanel::createFramebuffers(device, swapchainViews, swapchainExtent);
    }

    currentFrame = 0;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Depth
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createDepthResources() {
    // SAMPLED usage is added so the cloud raymarch pass can read the
    // depth buffer as a sampler2D (terrain occlusion) after the main
    // pass ends. The image is reused as a depth attachment elsewhere
    // so this single image carries both usages.
    createImage(swapchainExtent.width, swapchainExtent.height,
        VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
        | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT, depth);

    // Transition to DEPTH_ATTACHMENT_OPTIMAL (stay there forever)
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_NONE;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    barrier.image = depth.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
    endSingleTimeCommands(cmd);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Heightmap sampler (shared)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createHeightmapSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16.0f;
    samplerInfo.maxLod = 0.0f;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &heightmapSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create heightmap sampler");
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  World data initialization â€” matches SimpleHydrology's terrain approach:
//    OpenSimplex2 + 8-octave FBm (freq*2, scale*0.6 per octave) then
//    normalised to [0,1], edge-falloff applied, scaled to metres.
//    A second ridged-FBm mountain layer is blended onto high ground.
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::initializeWorldData() {
    worldSeed = settings.activeWorldSeed();
    masterSeaLevel = settings.seaLevel;

    generateMasterTerrain();

    createMasterHeightmapTexture();
    std::cout << "  [debug] running terrain topology analysis..." << std::endl;
    runTerrainAnalysis();
    initVegetationSystem();
}

void VulkanEngine::generateMasterTerrain() {
    // Single finite world: one heightmap at masterRes over worldSize metres,
    // origin (0,0). Produces masterHeightData + masterFlowData (D-infinity
    // hydrology, rivers, and flat lakes) in one pass — no regions, no chunks.
    const uint32_t R = std::max<uint32_t>(64, settings.masterRes);
    const float worldSize = settings.worldSize;

    // Lay out the demo airports up front so their pads can be flattened into
    // the heightmap below (elevations are resolved from the terrain itself).
    layoutAirports();

    std::vector<float> heights((size_t)R * R, 0.0f);

    const uint32_t seed = settings.worldSeed == 0 ? settings.activeWorldSeed() : settings.worldSeed;
    FastNoiseLite baseNoise;
    baseNoise.SetSeed((int)seed);
    baseNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);

    FastNoiseLite ridgeNoise;
    ridgeNoise.SetSeed((int)(seed ^ 0xDEADBEEFu));
    ridgeNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2S);
    ridgeNoise.SetFractalType(FastNoiseLite::FractalType_Ridged);
    ridgeNoise.SetFractalOctaves(6);
    ridgeNoise.SetFractalLacunarity(2.1f);
    ridgeNoise.SetFractalGain(0.55f);
    ridgeNoise.SetFrequency(2.5f);

    const float cell = worldSize / (float)(R - 1);
    const float totalScale = 0.6f * (1.0f - std::pow(0.6f, 8.0f)) / (1.0f - 0.6f);
    const float invNoiseRange = 1.0f / (2.0f * totalScale);

    for (uint32_t y = 0; y < R; ++y) {
        for (uint32_t x = 0; x < R; ++x) {
            float wx = (float)x * cell;
            float wz = (float)y * cell;
            float nx = wx / worldSize;
            float nz = wz / worldSize;

            float h = 0.0f, freq = 1.0f, scale = 0.6f;
            for (int o = 0; o < 8; ++o) {
                baseNoise.SetFrequency(freq);
                h += scale * baseNoise.GetNoise(nx, nz);
                freq *= 2.0f;
                scale *= 0.6f;
            }
            h = glm::clamp(0.5f + h * invNoiseRange, 0.0f, 1.0f);

            if (settings.mountainRidgeWeight > 0.001f) {
                float ridge = ridgeNoise.GetNoise(nx, nz) * 0.5f + 0.5f;
                float landFrac = glm::smoothstep(settings.seaFraction,
                                                 settings.seaFraction + 0.25f,
                                                 h);
                h = glm::clamp(h + ridge * landFrac * settings.mountainRidgeWeight, 0.0f, 1.0f);
            }

            heights[(size_t)y * R + x] =
                (h - settings.seaFraction) * settings.terrainTotalHeight;
        }
    }

    MasterHydrology::computeMasterHydrology(
        heights, R, settings.seaLevel,
        settings.riverThreshold, masterFlowData, settings.riverMinCells,
        settings.lakeMinDepth);

    std::vector<float> carveAmount((size_t)R * R, 0.0f);
    std::vector<float> influenceDepth((size_t)R * R, 0.0f);
    std::vector<float> influenceRadius((size_t)R * R, 0.0f);
    for (uint32_t i = 0; i < R * R; ++i) {
        if (!masterFlowData.isRiver[i]) continue;
        float norm = (float)masterFlowData.flowAccumulation[i]
                   / (float)std::max(1, masterFlowData.maxAccumulation);
        influenceDepth[i] = settings.riverCarveDepth
                          * (std::log2(1.0f + norm * 99.0f) / std::log2(100.0f));
        influenceRadius[i] = 1.0f + 3.0f * std::sqrt(norm);
    }
    const int MAX_RING = 4;
    for (int y = 0; y < (int)R; ++y) {
        for (int x = 0; x < (int)R; ++x) {
            uint32_t idx = (uint32_t)y * R + (uint32_t)x;
            float bestCarve = 0.0f;
            for (int dy = -MAX_RING; dy <= MAX_RING; ++dy) {
                int ny = y + dy;
                if (ny < 0 || ny >= (int)R) continue;
                for (int dx = -MAX_RING; dx <= MAX_RING; ++dx) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= (int)R) continue;
                    uint32_t nidx = (uint32_t)ny * R + (uint32_t)nx;
                    if (influenceDepth[nidx] <= 0.0f) continue;
                    float dist = std::sqrt((float)(dx * dx + dy * dy));
                    if (dist > influenceRadius[nidx]) continue;
                    float t = dist / influenceRadius[nidx];
                    float w = 0.5f + 0.5f * std::cos(t * 3.14159265f);
                    bestCarve = std::max(bestCarve, influenceDepth[nidx] * w);
                }
            }
            carveAmount[idx] = bestCarve;
        }
    }

    // Soften the carved valleys: a few separable box-blur passes turn the
    // cosine-ring channels into gently graded riverbeds instead of sharp
    // V-notches. Lake basins are left untouched here (flattened below).
    if (settings.carveSmoothPasses > 0) {
        std::vector<float> tmp(carveAmount.size(), 0.0f);
        for (int pass = 0; pass < settings.carveSmoothPasses; ++pass) {
            for (int y = 0; y < (int)R; ++y) {
                for (int x = 0; x < (int)R; ++x) {
                    float sum = 0.0f; int cnt = 0;
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = x + dx;
                        if (nx < 0 || nx >= (int)R) continue;
                        sum += carveAmount[(size_t)y * R + nx]; ++cnt;
                    }
                    tmp[(size_t)y * R + x] = sum / (float)cnt;
                }
            }
            for (int y = 0; y < (int)R; ++y) {
                for (int x = 0; x < (int)R; ++x) {
                    float sum = 0.0f; int cnt = 0;
                    for (int dy = -1; dy <= 1; ++dy) {
                        int ny = y + dy;
                        if (ny < 0 || ny >= (int)R) continue;
                        sum += tmp[(size_t)ny * R + x]; ++cnt;
                    }
                    carveAmount[(size_t)y * R + x] = sum / (float)cnt;
                }
            }
        }
    }

    for (uint32_t i = 0; i < R * R; ++i) {
        if (masterFlowData.isLake[i]) continue;   // don't carve through standing water
        heights[i] -= carveAmount[i];
    }
    // Flatten lake basins to their spill (water-table) elevation.
    for (uint32_t i = 0; i < R * R; ++i) {
        if (masterFlowData.isLake[i])
            heights[i] = masterFlowData.waterLevel[i];
    }

    // Carve flat concrete airport pads (with a smooth skirt) into the terrain.
    flattenAirportPads(heights, R);

    masterHeightData = std::move(heights);

    globalMinHeight = 1e30f;
    globalMaxHeight = -1e30f;
    int riverCells = 0;
    for (uint32_t i = 0; i < R * R; ++i) {
        globalMinHeight = std::min(globalMinHeight, masterHeightData[i]);
        globalMaxHeight = std::max(globalMaxHeight, masterHeightData[i]);
        if (masterFlowData.isRiver[i] || masterFlowData.isLake[i]) ++riverCells;
    }

    uiLandCells  = masterFlowData.totalLandCells;
    uiMaxAccum   = masterFlowData.maxAccumulation;
    uiRiverCells = riverCells;

    std::cout << "  [terrain] generated master " << R << "x" << R
              << " over " << worldSize << "m"
              << " height=[" << globalMinHeight << "," << globalMaxHeight << "]"
              << " water cells=" << riverCells << "\n";
}

// ══════════════════════════════════════════════════════════════
//  Airports — layout, terrain flattening, paved-surface mesh
// ══════════════════════════════════════════════════════════════

void VulkanEngine::layoutAirports() {
    // Scale runway dimensions sensibly with world size so a small world
    // still gets a usable strip and a large one isn't dominated by it.
    for (Airport& ap : airports) {
        ap.runwayLength  = glm::clamp(settings.worldSize * 0.22f, 600.0f, 1400.0f);
        ap.runwayWidth   = 45.0f;
        ap.taxiwayWidth  = 23.0f;
        ap.taxiwayOffset = 95.0f;
        ap.apronHalfLen  = 80.0f;
        ap.apronHalfWid  = 55.0f;
        ap.gateCount     = 4;
    }
    layoutDemoAirports(settings.worldSize, airports[0], airports[1]);
    airportsValid = true;
}

// Flatten an oriented concrete pad around each airport into `heights` (metres),
// with a smooth falloff skirt so the pad blends into the surrounding terrain.
// Also fills each airport's pad elevation. Must run on the local heights array
// inside generateMasterTerrain(), before it is moved into masterHeightData.
void VulkanEngine::flattenAirportPads(std::vector<float>& heights, uint32_t res) const {
    if (!airportsValid || heights.size() != (size_t)res * res) return;
    const uint32_t R = res;
    const float cell = settings.worldSize / (float)(R - 1);
    const float falloff = 55.0f;   // skirt width (m)

    for (const Airport& ap : airports) {
        const glm::vec2 dir = ap.dir();
        const glm::vec2 side = ap.side();

        // Footprint half-extents in the local (u along runway, v across) frame.
        const float hu = ap.halfRunway() + 30.0f;
        const float vMin = -(ap.runwayWidth * 0.5f + 18.0f);
        const float vMax = ap.apronCenterV() + ap.apronHalfWid + 18.0f;
        const float vc = 0.5f * (vMin + vMax);
        const float hv = 0.5f * (vMax - vMin);

        // World-space AABB bounding the oriented footprint (+ skirt) → cell range.
        const float reachU = hu + falloff;
        const float reachV = hv + falloff;
        glm::vec2 ext = glm::abs(dir) * reachU + glm::abs(side) * reachV;
        glm::vec2 cMin = ap.center - ext;
        glm::vec2 cMax = ap.center + ext;
        int x0 = std::max(0, (int)std::floor(cMin.x / cell));
        int x1 = std::min((int)R - 1, (int)std::ceil(cMax.x / cell));
        int y0 = std::max(0, (int)std::floor(cMin.y / cell));
        int y1 = std::min((int)R - 1, (int)std::ceil(cMax.y / cell));

        // Pass 1: average terrain height inside the core footprint → pad level.
        double sum = 0.0; long long cnt = 0;
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                glm::vec2 w(x * cell, y * cell);
                glm::vec2 rel = w - ap.center;
                float u = glm::dot(rel, dir), v = glm::dot(rel, side);
                if (std::abs(u) <= hu && std::abs(v - vc) <= hv) {
                    sum += heights[(size_t)y * R + x];
                    ++cnt;
                }
            }
        }
        float padElev = (cnt > 0) ? (float)(sum / (double)cnt) : 0.0f;
        padElev = std::max(padElev, settings.seaLevel + 4.0f);   // keep clear of water
        const_cast<Airport&>(ap).elevation = padElev;

        // Pass 2: flatten core to pad, blend skirt out to original terrain.
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                glm::vec2 w(x * cell, y * cell);
                glm::vec2 rel = w - ap.center;
                float u = glm::dot(rel, dir), v = glm::dot(rel, side);
                float du = std::max(0.0f, std::abs(u) - hu);
                float dv = std::max(0.0f, std::abs(v - vc) - hv);
                float dist = std::sqrt(du * du + dv * dv);
                if (dist >= falloff) continue;
                float t = dist / falloff;
                float s = t * t * (3.0f - 2.0f * t);   // smoothstep
                size_t idx = (size_t)y * R + x;
                heights[idx] = glm::mix(padElev, heights[idx], s);
            }
        }

        std::cout << "  [airport] pad at (" << ap.center.x << "," << ap.center.y
                  << ") elev=" << padElev << " heading="
                  << glm::degrees(ap.headingRad) << "deg\n";
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Master hydrology textures â€” continuous cross-chunk rivers
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Master heightmap texture upload â€” feeds SimpleHydrology
//  eroded terrain to GPU chunk rendering
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

namespace {
// World-space climate humidity in [0,1]: domain-warped value-noise made drier
// with altitude, then boosted near water. Feeds terrain.frag's dry/lush grass,
// desert push, and snow-line shift.
float humHash(glm::vec2 p, uint32_t seed) {
    float h = std::sin(p.x * 127.1f + p.y * 311.7f + (float)seed * 0.000137f) * 43758.5453f;
    return h - std::floor(h);
}
float humValueNoise(glm::vec2 p, uint32_t seed) {
    glm::vec2 i = glm::floor(p), f = p - i;
    f = f * f * (3.0f - 2.0f * f);
    float a = humHash(i, seed),                    b = humHash(i + glm::vec2(1, 0), seed);
    float c = humHash(i + glm::vec2(0, 1), seed),  d = humHash(i + glm::vec2(1, 1), seed);
    return glm::mix(glm::mix(a, b, f.x), glm::mix(c, d, f.x), f.y);
}
float humFbm(glm::vec2 p, uint32_t seed) {
    float v = 0.0f, amp = 0.5f;
    for (int o = 0; o < 4; ++o) { v += amp * humValueNoise(p, seed + (uint32_t)o * 17u); p *= 2.1f; amp *= 0.5f; }
    return v;
}
float computeHumidity(float wx, float wz, float h, float seaLevel, float peak, uint32_t seed) {
    glm::vec2 p(wx, wz);
    glm::vec2 warp(humFbm(p * 0.0009f + glm::vec2(11.3f), seed),
                   humFbm(p * 0.0009f + glm::vec2(57.1f), seed + 91u));
    float climate = glm::clamp(humFbm(p * 0.0009f + warp * 1.5f, seed) * 1.15f, 0.0f, 1.0f);
    float altDry  = glm::clamp((h - seaLevel) / std::max(1.0f, peak), 0.0f, 1.0f);
    return glm::clamp(climate * (1.0f - 0.45f * altDry), 0.0f, 1.0f);
}
} // namespace

void VulkanEngine::createMasterHeightmapTexture() {
    if (masterHeightData.empty()) {
        std::cerr << "[master heightmap] no data to upload\n";
        return;
    }

    std::cout << "  [master heightmap] uploading "
              << settings.masterRes << "x" << settings.masterRes
              << " terrain to GPU..." << std::endl;

    // Create image â€” RGBA32F to match per-chunk heightmap channel layout:
    // .r=height, .g=humidity, .b=riverMask, .a=height
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    imgInfo.extent = {settings.masterRes, settings.masterRes, 1};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &imgInfo, nullptr, &masterHeightmapImage.image) != VK_SUCCESS)
        throw std::runtime_error("failed to create master heightmap image");

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, masterHeightmapImage.image, &memReqs);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &masterHeightmapImage.memory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate master heightmap memory");
    vkBindImageMemory(device, masterHeightmapImage.image, masterHeightmapImage.memory, 0);

    // Image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = masterHeightmapImage.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &viewInfo, nullptr, &masterHeightmapImage.view) != VK_SUCCESS)
        throw std::runtime_error("failed to create master heightmap image view");

    // Sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &masterHeightmapSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create master heightmap sampler");

    // Upload data via staging buffer â€” convert R32F â†’ RGBA32F
    // Per-pixel layout: .r=height, .g=humidity, .b=riverMask, .a=height
    const uint32_t R = settings.masterRes;
    uint32_t pixelCount = R * R;
    const float cell = settings.worldSize / (float)std::max(1u, R - 1u);
    const uint32_t humSeed = settings.activeWorldSeed();
    const float humPeak = std::max(1.0f, settings.terrainTotalHeight * 0.6f);
    VkDeviceSize dataSize = sizeof(float) * 4 * pixelCount;
    std::vector<float> rgbaData(pixelCount * 4, 0.0f);
    for (uint32_t i = 0; i < pixelCount; i++) {
        float h = masterHeightData[i];
        rgbaData[i * 4 + 0] = h;  // .r = height

        // .b = water mask: 0=none, (0,0.5)=riparian bank, [0.5,1]=open water
        // (river channel gradient by discharge, or a flat lake at ~1.0).
        float riverMask = 0.0f;
        bool isLake  = (i < masterFlowData.isLake.size()  && masterFlowData.isLake[i]);
        bool isRiver = (i < masterFlowData.isRiver.size() && masterFlowData.isRiver[i]);
        if (isLake) {
            riverMask = 1.0f;
        } else if (isRiver) {
            float norm = (float)masterFlowData.flowAccumulation[i]
                       / (float)std::max(1, masterFlowData.maxAccumulation);
            riverMask = 0.5f + 0.5f * std::sqrt(norm);
        } else {
            float bank = 0.0f;
            if (i < masterFlowData.riverStrength.size())
                bank = masterFlowData.riverStrength[i] * 0.49f;
            uint32_t x = i % R;
            uint32_t y = i / R;
            if (x > 0 && x < R - 1 && y > 0 && y < R - 1) {
                float maxNeighNorm = 0.0f;
                for (int n = 0; n < 8; n++) {
                    uint32_t ni = (y + MasterHydrology::DY8[n]) * R
                                + (x + MasterHydrology::DX8[n]);
                    bool nWater = (ni < masterFlowData.isRiver.size() && masterFlowData.isRiver[ni])
                               || (ni < masterFlowData.isLake.size()  && masterFlowData.isLake[ni]);
                    if (nWater) {
                        float nn = (float)masterFlowData.flowAccumulation[ni]
                                 / (float)std::max(1, masterFlowData.maxAccumulation);
                        maxNeighNorm = std::max(maxNeighNorm, std::max(nn, 0.25f));
                    }
                }
                if (maxNeighNorm > 0.0f)
                    bank = std::max(bank, 0.15f + 0.25f * std::sqrt(maxNeighNorm));
            }
            riverMask = std::min(bank, 0.49f);
        }

        // .g = humidity: climate field, wetter near water.
        const float wx = (float)(i % R) * cell;
        const float wz = (float)(i / R) * cell;
        float humidity = computeHumidity(wx, wz, h, settings.seaLevel, humPeak, humSeed);
        if (riverMask >= 0.5f)      humidity = std::max(humidity, 0.90f);
        else if (riverMask > 0.05f) humidity = std::max(humidity, 0.60f + 0.30f * (riverMask / 0.5f));
        rgbaData[i * 4 + 1] = glm::clamp(humidity, 0.0f, 1.0f);

        rgbaData[i * 4 + 2] = riverMask;
        rgbaData[i * 4 + 3] = h;  // .a = height (vertex shader reads this)
    }
    
    Buffer staging;
    createBuffer(dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging);

    void* mapped;
    vkMapMemory(device, staging.memory, 0, dataSize, 0, &mapped);
    memcpy(mapped, rgbaData.data(), dataSize);
    vkUnmapMemory(device, staging.memory);

    VkCommandBuffer cmd = beginSingleTimeCommands();

    // Transition: undefined â†’ transfer dst
    VkImageMemoryBarrier2 preBarrier{};
    preBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    preBarrier.srcAccessMask = 0;
    preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    preBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    preBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    preBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    preBarrier.image = masterHeightmapImage.image;
    preBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    preBarrier.subresourceRange.levelCount = 1;
    preBarrier.subresourceRange.layerCount = 1;
    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &preBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = {settings.masterRes, settings.masterRes, 1};
    vkCmdCopyBufferToImage(cmd, staging.buffer, masterHeightmapImage.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    // Transition: transfer dst â†’ shader read only
    VkImageMemoryBarrier2 postBarrier{};
    postBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    postBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    postBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    postBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    postBarrier.image = masterHeightmapImage.image;
    postBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    postBarrier.subresourceRange.levelCount = 1;
    postBarrier.subresourceRange.layerCount = 1;
    VkDependencyInfo postDep{};
    postDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    postDep.imageMemoryBarrierCount = 1;
    postDep.pImageMemoryBarriers = &postBarrier;
    vkCmdPipelineBarrier2(cmd, &postDep);

    endSingleTimeCommands(cmd);

    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);

    std::cout << "  [master heightmap] uploaded to GPU" << std::endl;
}

void VulkanEngine::createGlobalTerrainDescriptorSet() {
    VkDescriptorSetLayout layouts[MAX_FRAMES_IN_FLIGHT];
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        layouts[i] = descriptorSetLayout;

    VkDescriptorSetAllocateInfo descAlloc{};
    descAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAlloc.descriptorPool = descriptorPool;
    descAlloc.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    descAlloc.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(device, &descAlloc, globalTerrainDescSets) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate global terrain descriptor sets");
    globalTerrainDescSet = globalTerrainDescSets[0];

    VkDescriptorImageInfo hmInfo{};
    hmInfo.sampler = masterHeightmapSampler;
    hmInfo.imageView = masterHeightmapImage.view;
    hmInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo texInfo{};
    texInfo.sampler = terrainTexSampler;
    texInfo.imageView = terrainTexArray.view;
    texInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo shadowInfo{};
    shadowInfo.sampler = shadowSampler;
    shadowInfo.imageView = shadowMap.view;
    shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo atmoGlobInfo{};
    atmoGlobInfo.buffer = atmosphereUniformBuffer.buffer;
    atmoGlobInfo.offset = 0;
    atmoGlobInfo.range = sizeof(AtmosphereParams);

    VkDescriptorBufferInfo cloudGlobInfo{};
    cloudGlobInfo.buffer = cloudParamsBuffer.buffer;
    cloudGlobInfo.offset = 0;
    cloudGlobInfo.range = sizeof(CloudParams);

    VkDescriptorImageInfo cloudNoiseGlobInfo{};
    cloudNoiseGlobInfo.sampler = cloudNoiseSampler;
    cloudNoiseGlobInfo.imageView = cloudNoise3D.view;
    cloudNoiseGlobInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo cloudShadowInfo{};
    cloudShadowInfo.sampler = cloudShadowSampler;
    cloudShadowInfo.imageView = cloudShadowMap.view;
    cloudShadowInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    for (int fi = 0; fi < MAX_FRAMES_IN_FLIGHT; ++fi) {
        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = uniformBuffers[fi].buffer;
        uboInfo.offset = 0;
        uboInfo.range = sizeof(FrameData);

        VkWriteDescriptorSet writes[8]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = globalTerrainDescSets[fi];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &uboInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = globalTerrainDescSets[fi];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &hmInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = globalTerrainDescSets[fi];
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &texInfo;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = globalTerrainDescSets[fi];
        writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].descriptorCount = 1;
        writes[3].pImageInfo = &shadowInfo;

        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = globalTerrainDescSets[fi];
        writes[4].dstBinding = 4;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[4].descriptorCount = 1;
        writes[4].pBufferInfo = &atmoGlobInfo;

        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet = globalTerrainDescSets[fi];
        writes[5].dstBinding = 5;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[5].descriptorCount = 1;
        writes[5].pBufferInfo = &cloudGlobInfo;

        writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[6].dstSet = globalTerrainDescSets[fi];
        writes[6].dstBinding = 6;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[6].descriptorCount = 1;
        writes[6].pImageInfo = &cloudNoiseGlobInfo;

        writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[7].dstSet = globalTerrainDescSets[fi];
        writes[7].dstBinding = 7;
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[7].descriptorCount = 1;
        writes[7].pImageInfo = &cloudShadowInfo;

        vkUpdateDescriptorSets(device, 8, writes, 0, nullptr);
    }

    // Complete the global shadow descriptor set (binding 1 = master heightmap)
    // so the non-infinite/legacy shadow path binds a fully-written, layout-
    // compatible set instead of the 5-binding graphics set.
    if (shadowDescriptorSet != VK_NULL_HANDLE) {
        VkWriteDescriptorSet shadowHm{};
        shadowHm.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        shadowHm.dstSet = shadowDescriptorSet;
        shadowHm.dstBinding = 1;
        shadowHm.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        shadowHm.descriptorCount = 1;
        shadowHm.pImageInfo = &hmInfo;
        vkUpdateDescriptorSets(device, 1, &shadowHm, 0, nullptr);
    }

    std::cout << "  [global terrain] descriptor set created" << std::endl;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Debug terrain analysis â€” public entry point for keybinding
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::runTerrainAnalysis() {
    if (masterHeightData.empty()) {
        std::cerr << "[DebugAnalyzer] no world data to analyze\n";
        return;
    }

    // SimpleHydrology integration: plate maps removed â€” pass nullptr for
    // optional plate/crust analysis (geomorph classification still works
    // from height data alone).
    DebugAnalyzer::AnalysisResult result;
    DebugAnalyzer::analyze(masterHeightData, settings.masterRes,
                           settings.worldSize, result, nullptr, nullptr);
    DebugAnalyzer::saveJsonReport(result, "terrain_analysis.json");

    // Cache global min/max for debug overlay shader
    globalMinHeight = result.minHeight;
    globalMaxHeight = result.maxHeight;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Debug overlay system â€” real-time terrain visualization toggle
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::uploadDebugOverlayTexture(DebugAnalyzer::DebugOverlayMode mode) {
    if (masterHeightData.empty()) return;

    // Generate debug overlay RGBA8 data on CPU
    // SimpleHydrology: no plate/crust maps â€” pass nullptr
    std::vector<uint8_t> rgbaData = DebugAnalyzer::generateDebugOverlay(
        masterHeightData, settings.masterRes, mode, nullptr, nullptr);

    // Lazily create the debug overlay GPU image + sampler on first use
    if (debugOverlayImage.image == VK_NULL_HANDLE) {
        // Sampler
        VkSamplerCreateInfo sampInfo{};
        sampInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampInfo.magFilter = VK_FILTER_LINEAR;
        sampInfo.minFilter = VK_FILTER_LINEAR;
        sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device, &sampInfo, nullptr, &debugOverlaySampler) != VK_SUCCESS)
            throw std::runtime_error("failed to create debug overlay sampler");

        // Image
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imgInfo.extent = {settings.masterRes, settings.masterRes, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(device, &imgInfo, nullptr, &debugOverlayImage.image) != VK_SUCCESS)
            throw std::runtime_error("failed to create debug overlay image");

        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, debugOverlayImage.image, &memReqs);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &debugOverlayImage.memory) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate debug overlay memory");
        vkBindImageMemory(device, debugOverlayImage.image, debugOverlayImage.memory, 0);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = debugOverlayImage.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &viewInfo, nullptr, &debugOverlayImage.view) != VK_SUCCESS)
            throw std::runtime_error("failed to create debug overlay image view");
    }

    // Upload via staging buffer
    VkDeviceSize dataSize = rgbaData.size();
    Buffer staging;
    createBuffer(dataSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging);

    void* mapped;
    vkMapMemory(device, staging.memory, 0, dataSize, 0, &mapped);
    memcpy(mapped, rgbaData.data(), dataSize);
    vkUnmapMemory(device, staging.memory);

    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkImageMemoryBarrier2 preBarrier{};
    preBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    preBarrier.srcAccessMask = VK_ACCESS_2_NONE;
    preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    preBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    preBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    preBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    preBarrier.image = debugOverlayImage.image;
    preBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    preBarrier.subresourceRange.levelCount = 1;
    preBarrier.subresourceRange.layerCount = 1;

    VkDependencyInfo preDep{};
    preDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    preDep.imageMemoryBarrierCount = 1;
    preDep.pImageMemoryBarriers = &preBarrier;
    vkCmdPipelineBarrier2(cmd, &preDep);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = {settings.masterRes, settings.masterRes, 1};
    vkCmdCopyBufferToImage(cmd, staging.buffer, debugOverlayImage.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    VkImageMemoryBarrier2 postBarrier{};
    postBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    postBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    postBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    postBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    postBarrier.image = debugOverlayImage.image;
    postBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    postBarrier.subresourceRange.levelCount = 1;
    postBarrier.subresourceRange.layerCount = 1;

    VkDependencyInfo postDep{};
    postDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    postDep.imageMemoryBarrierCount = 1;
    postDep.pImageMemoryBarriers = &postBarrier;
    vkCmdPipelineBarrier2(cmd, &postDep);

    endSingleTimeCommands(cmd);

    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);
}

void VulkanEngine::setDebugOverlay(DebugAnalyzer::DebugOverlayMode mode) {
    if (mode == debugOverlayMode) return;
    debugOverlayMode = mode;

    const char* modeNames[] = {
        "OFF (normal)", "Height Heatmap", "Slope Map", "Plate Boundaries",
        "Land/Water Mask", "Mountain Peaks", "Ridge Lines", "Geomorphology", "Flow Accumulation"
    };
    int m = (int)mode;
    std::cout << "[debug] overlay mode: " << (m <= 8 ? modeNames[m] : "?")
              << " (" << m << ")" << std::endl;

    if (mode == DebugAnalyzer::DebugOverlayMode::NONE)
        return;

    // Update cached global min/max in case terrain was regenerated
    if (!masterHeightData.empty()) {
        float hMin = 1e30f, hMax = -1e30f;
        for (float h : masterHeightData) {
            hMin = std::min(hMin, h);
            hMax = std::max(hMax, h);
        }
        globalMinHeight = hMin;
        globalMaxHeight = hMax;
    }

    uploadDebugOverlayTexture(mode);
}

void VulkanEngine::cycleDebugOverlay() {
    int next = ((int)debugOverlayMode + 1) % 9; // 0-8
    setDebugOverlay((DebugAnalyzer::DebugOverlayMode)next);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  CPU-side terrain height query â€” bilinear sample from the
//  master heightmap (guaranteed to match the GPU heightmap).
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

float VulkanEngine::sampleWorldHeight(float worldX, float worldZ) {
    if (masterHeightData.empty()) return 0.0f;

    const float R = (float)(settings.masterRes - 1);
    float u = worldX / settings.worldSize;
    float v = worldZ / settings.worldSize;
    u = std::max(0.0f, std::min(1.0f, u));
    v = std::max(0.0f, std::min(1.0f, v));

    float fx = u * R;
    float fz = v * R;
    int x0 = (int)fx, z0 = (int)fz;
    int x1 = std::min((int)(settings.masterRes - 1), x0 + 1);
    int z1 = std::min((int)(settings.masterRes - 1), z0 + 1);
    float tx = fx - (float)x0;
    float tz = fz - (float)z0;

    const uint32_t W = settings.masterRes;
    float h00 = masterHeightData[(uint32_t)z0 * W + (uint32_t)x0];
    float h10 = masterHeightData[(uint32_t)z0 * W + (uint32_t)x1];
    float h01 = masterHeightData[(uint32_t)z1 * W + (uint32_t)x0];
    float h11 = masterHeightData[(uint32_t)z1 * W + (uint32_t)x1];

    return glm::mix(glm::mix(h00, h10, tx), glm::mix(h01, h11, tx), tz);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Terrain mesh
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createTerrainMesh() {
    // Single terrain mesh covering the full master world area
    const int RES = (int)settings.masterRes;
    const float WORLD_SIZE = settings.worldSize;

    // Flat XZ grid, Y=0 (height comes from texture)
    std::vector<glm::vec3> vertices(RES * RES);
    for (int z = 0; z < RES; z++) {
        for (int x = 0; x < RES; x++) {
            vertices[z * RES + x] = glm::vec3(
                (float)x / (RES - 1) * WORLD_SIZE,
                0.0f,
                (float)z / (RES - 1) * WORLD_SIZE
            );
        }
    }

    // Generate multi-LOD indices (shared vertex buffer, segmented index buffer)
    // LOD distances auto-scale with world size so an 8 km world still LODs sensibly.
    std::vector<uint32_t> indices;
    const float worldScale = WORLD_SIZE / 512.0f;     // 1.0 at 512m world
    const float lodDistances[] = {
        0.0f,
         80.0f  * worldScale,
        200.0f  * worldScale,
        500.0f  * worldScale
    };

    for (int lod = 0; lod < MAX_LODS; lod++) {
        int step = 1 << lod;                    // vertex stride: 1, 2, 4, 8
        int lodRes = RES / step;                // vertices per side for this LOD

        lods[lod].firstIndex = static_cast<uint32_t>(indices.size());
        lods[lod].minDistance = lodDistances[lod];

        for (int z = 0; z < lodRes - 1; z++) {
            for (int x = 0; x < lodRes - 1; x++) {
                uint32_t tl = (z * step) * RES + (x * step);
                uint32_t tr = tl + step;
                uint32_t bl = ((z + 1) * step) * RES + (x * step);
                uint32_t br = bl + step;
                indices.push_back(tl); indices.push_back(bl); indices.push_back(tr);
                indices.push_back(tr); indices.push_back(bl); indices.push_back(br);
            }
        }
        lods[lod].indexCount = static_cast<uint32_t>(indices.size()) - lods[lod].firstIndex;
    }
    indexCount = lods[0].indexCount;  // keep for compatibility

    // Create vertex buffer
    VkDeviceSize vbSize = sizeof(glm::vec3) * vertices.size();
    Buffer staging;
    createBuffer(vbSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
    void* data;
    vkMapMemory(device, staging.memory, 0, vbSize, 0, &data);
    memcpy(data, vertices.data(), vbSize);
    vkUnmapMemory(device, staging.memory);

    createBuffer(vbSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy copy{};
    copy.size = vbSize;
    vkCmdCopyBuffer(cmd, staging.buffer, vertexBuffer.buffer, 1, &copy);
    endSingleTimeCommands(cmd);
    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);

    // Create index buffer (all LODs in one buffer)
    VkDeviceSize ibSize = sizeof(uint32_t) * indices.size();
    createBuffer(ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
    vkMapMemory(device, staging.memory, 0, ibSize, 0, &data);
    memcpy(data, indices.data(), ibSize);
    vkUnmapMemory(device, staging.memory);

    createBuffer(ibSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer);

    cmd = beginSingleTimeCommands();
    copy.size = ibSize;
    vkCmdCopyBuffer(cmd, staging.buffer, indexBuffer.buffer, 1, &copy);
    endSingleTimeCommands(cmd);
    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);
}

int VulkanEngine::selectTerrainLOD() const {
    const glm::vec3 pos = camera.eyePos();
    const float worldHalf = settings.worldSize * 0.5f;
    const float horizToCenter = glm::length(glm::vec2(pos.x, pos.z) - glm::vec2(worldHalf));
    const float metric = glm::max(pos.y * 0.35f, horizToCenter - worldHalf * 0.25f);

    int lod = 0;
    for (int i = 1; i < MAX_LODS; ++i) {
        if (metric >= lods[i].minDistance)
            lod = i;
    }
    return lod;
}

// ══════════════════════════════════════════════════════════════
//  Ocean mesh — flat XZ grid; the Gerstner vertex shader lifts it
//  to sea level and animates the surface. Host-visible buffers.
// ══════════════════════════════════════════════════════════════
void VulkanEngine::createOceanMesh() {
    // 257x257 vertices (256 quads) gives ~16 m/vertex over a 4 km world.
    const int N = 257;
    const float step = settings.worldSize / float(N - 1);

    std::vector<glm::vec3> verts;
    verts.reserve((size_t)N * N);
    for (int z = 0; z < N; z++)
        for (int x = 0; x < N; x++)
            verts.push_back({ x * step, 0.0f, z * step });

    std::vector<uint32_t> idx;
    idx.reserve((size_t)(N - 1) * (N - 1) * 6);
    for (int z = 0; z < N - 1; z++)
        for (int x = 0; x < N - 1; x++) {
            uint32_t base = (uint32_t)(z * N + x);
            idx.push_back(base);
            idx.push_back(base + 1);
            idx.push_back(base + N + 1);
            idx.push_back(base);
            idx.push_back(base + N + 1);
            idx.push_back(base + N);
        }

    oceanIndexCount = static_cast<uint32_t>(idx.size());

    VkDeviceSize vbSize = sizeof(glm::vec3) * verts.size();
    createBuffer(vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 oceanVertexBuffer);
    void* data;
    vkMapMemory(device, oceanVertexBuffer.memory, 0, vbSize, 0, &data);
    memcpy(data, verts.data(), (size_t)vbSize);
    vkUnmapMemory(device, oceanVertexBuffer.memory);

    VkDeviceSize ibSize = sizeof(uint32_t) * idx.size();
    createBuffer(ibSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 oceanIndexBuffer);
    vkMapMemory(device, oceanIndexBuffer.memory, 0, ibSize, 0, &data);
    memcpy(data, idx.data(), (size_t)ibSize);
    vkUnmapMemory(device, oceanIndexBuffer.memory);
}

// ══════════════════════════════════════════════════════════════
//  Ocean pipeline — reuses the terrain descriptor set layout (set 0)
//  so no new descriptor pool/set is needed. Depth test LESS_OR_EQUAL
//  so the surface naturally meets the terrain at the coastline.
// ══════════════════════════════════════════════════════════════
void VulkanEngine::createOceanPipeline() {
    auto vertCode = readFile("shaders/ocean.vert.spv");
    auto fragCode = readFile("shaders/ocean.frag.spv");
    VkShaderModule vertMod = createShaderModule(vertCode);
    VkShaderModule fragMod = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertMod;
    vertStage.pName = "main";
    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragMod;
    fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

    VkVertexInputBindingDescription binding{};
    binding.binding   = 0;
    binding.stride    = sizeof(glm::vec3);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attr{};
    attr.binding  = 0;
    attr.location = 0;
    attr.format   = VK_FORMAT_R32G32B32_SFLOAT;
    attr.offset   = 0;

    VkPipelineVertexInputStateCreateInfo vtxInfo{};
    vtxInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vtxInfo.vertexBindingDescriptionCount   = 1;
    vtxInfo.pVertexBindingDescriptions      = &binding;
    vtxInfo.vertexAttributeDescriptionCount = 1;
    vtxInfo.pVertexAttributeDescriptions    = &attr;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode    = VK_CULL_MODE_NONE;   // waves can flip facing near crests
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.blendEnable    = VK_FALSE;
    cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                         | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &cbAtt;

    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(OceanPush);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &descriptorSetLayout;  // same as terrain
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &oceanPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create ocean pipeline layout");

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &swapchainFormat;
    renderingInfo.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.pNext = &renderingInfo;
    pipeInfo.stageCount          = 2;
    pipeInfo.pStages             = stages;
    pipeInfo.pVertexInputState   = &vtxInfo;
    pipeInfo.pInputAssemblyState = &ia;
    pipeInfo.pViewportState      = &vp;
    pipeInfo.pRasterizationState = &rast;
    pipeInfo.pMultisampleState   = &ms;
    pipeInfo.pDepthStencilState  = &ds;
    pipeInfo.pColorBlendState    = &cb;
    pipeInfo.pDynamicState       = &dyn;
    pipeInfo.layout              = oceanPipelineLayout;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &oceanPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create ocean pipeline");
    vkDestroyShaderModule(device, vertMod, nullptr);
    vkDestroyShaderModule(device, fragMod, nullptr);

    // Push constant defaults. seaLevel is the WORLD-space waterline, which the
    // terrain shader treats as y=0 (stored height 0 × heightScale), so the ocean
    // plane sits at 0 regardless of the hydrology drain level (settings.seaLevel).
    oceanPush.worldOrigin    = glm::vec2(0.0f);
    oceanPush.worldSize      = settings.worldSize;
    oceanPush.heightScale    = settings.terrainHeightScale;
    oceanPush.heightmapRes   = (int)settings.masterRes;
    oceanPush.seaLevel       = 0.0f;
    oceanPush.waveAmplitude  = settings.waveAmplitude;
    oceanPush.waveSpeed      = settings.waveSpeed;
    oceanPush.shoreDepth     = settings.shoreDepth;
    oceanPush.foamDepth      = settings.foamDepth;
    oceanPush._pad0          = 0.0f;
}

// ══════════════════════════════════════════════════════════════
//  Aerodynamic airflow visualization
//   · Particle SSBO advected through an analytical flow field
//     (horseshoe vortices + fuselage source/sink) by a compute pass.
//   · CPU-integrated streamlines: exact curved paths resolved at 0.2 m step
//     size so leading-edge and tip-vortex curvature is fully visible.
//   · Pressure (Cp) overlay re-draws the aircraft model semi-transparent.
//  Driven entirely from the aircraft render state + velocity, so it
//  tracks whichever aircraft is currently visible (autonomous or geared).
// ══════════════════════════════════════════════════════════════

// ── CPU-side flow-field helpers (mirror of GPU airflow_advect.comp) ─────
// Used by updateStreamlines() to integrate streamline paths with fine steps.
namespace {

static constexpr float AERO_PI = 3.14159265358979f;

glm::vec3 cpuBiotSavartSeg(const glm::vec3& P,
                            const glm::vec3& A, const glm::vec3& B,
                            float Gamma, float coreR) {
    glm::vec3 r1 = P-A, r2 = P-B, r0 = B-A;
    glm::vec3 cx = glm::cross(r1, r2);
    float cl2 = glm::dot(cx, cx);
    if (cl2 < coreR*coreR) return glm::vec3(0);
    float r1l = glm::length(r1), r2l = glm::length(r2);
    if (r1l < 1e-5f || r2l < 1e-5f) return glm::vec3(0);
    return (Gamma / (4.f*AERO_PI*cl2)) * cx
           * (glm::dot(r0,r1)/r1l - glm::dot(r0,r2)/r2l);
}

glm::vec3 cpuBiotSavartSemiInf(const glm::vec3& P,
                                const glm::vec3& A, const glm::vec3& d,
                                float Gamma, float coreR) {
    glm::vec3 rA = P-A, cxDR = glm::cross(d, rA);
    float hSq = glm::dot(cxDR, cxDR);
    if (hSq < coreR*coreR) return glm::vec3(0);
    float h = std::sqrt(hSq), rAl = glm::length(rA);
    if (rAl < 1e-5f) return glm::vec3(0);
    return (Gamma/(4.f*AERO_PI*h)) * (1.f + glm::dot(rA,d)/rAl) * (cxDR/h);
}

glm::vec3 cpuSourceVel(const glm::vec3& P, const glm::vec3& O, float Q) {
    glm::vec3 r = P - O;
    float l = glm::length(r);
    if (l < 0.08f) return glm::vec3(0);
    return (Q / (4.f*AERO_PI*l*l*l)) * r;
}

// Evaluates the "wind-tunnel" flow velocity at Pworld:
//   V_body(P_body) = V∞_body + vizStrength * (vortex + source)
//   rotated back to world (no translation) → same shape as GPU shader.
//
//  vizStrength exaggerates the perturbation only (NOT the freestream) so
//  the streamline deflection around the wing is clearly visible. The true
//  induced velocity for a light aircraft is only ~1-2% of freestream, which
//  would render as dead-straight lines; this factor makes the physically-
//  shaped deflection readable, exactly like wind-tunnel smoke at high AoA.
glm::vec3 cpuFlowVelocityWorld(const glm::vec3& Pworld,
                                const AeroGlobalUBO& u,
                                float vizStrength) {
    glm::vec3 P = glm::vec3(u.worldToBody * glm::vec4(Pworld, 1.f));

    float Gw  = u.vortexParams.x, hs  = u.vortexParams.y, cr  = u.vortexParams.z;
    float Gs  = u.stabParams.x,   hss = u.stabParams.y,   crs = u.stabParams.z;
    float Q   = u.sourceParams.x, hf  = u.sourceParams.y;
    glm::vec3 fsb(u.freestreamBody);

    // Trailing vortex direction = downstream in body frame
    glm::vec3 trail = glm::normalize(fsb - glm::vec3(1e-4f, 0, 0));

    // Accumulate the perturbation (vortex + source) separately, then scale.
    glm::vec3 pert(0.0f);
    glm::vec3 tL(0, 0, -hs), tR(0, 0, hs);
    pert += cpuBiotSavartSeg    (P, tL, tR,  Gw, cr);
    pert += cpuBiotSavartSemiInf(P, tL, trail, -Gw, cr);
    pert += cpuBiotSavartSemiInf(P, tR, trail,  Gw, cr);

    float ta = -hf * 1.15f;
    glm::vec3 sL(ta, 0, -hss), sR(ta, 0, hss);
    pert += cpuBiotSavartSeg    (P, sL, sR,  Gs, crs);
    pert += cpuBiotSavartSemiInf(P, sL, trail, -Gs, crs);
    pert += cpuBiotSavartSemiInf(P, sR, trail,  Gs, crs);

    pert += cpuSourceVel(P, glm::vec3( hf, 0, 0),  Q);
    pert += cpuSourceVel(P, glm::vec3(-hf, 0, 0), -Q);

    glm::vec3 V = fsb + pert * vizStrength;
    return glm::mat3(u.bodyToWorld) * V;   // rotate-only → world frame
}

} // anonymous namespace

void VulkanEngine::createAeroResources() {
    // ── Particle SSBO (host-visible; advected in place by the compute pass) ──
    VkDeviceSize ssboSize = sizeof(AeroParticle) * AERO_PARTICLE_COUNT;
    createBuffer(ssboSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 aeroParticleSSBO);
    {
        void* d = nullptr;
        vkMapMemory(device, aeroParticleSSBO.memory, 0, ssboSize, 0, &d);
        std::memset(d, 0, (size_t)ssboSize);
        vkUnmapMemory(device, aeroParticleSSBO.memory);
    }

    // ── Global UBO (persistently mapped) ──
    createBuffer(sizeof(AeroGlobalUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 aeroGlobalUBOBuf);
    vkMapMemory(device, aeroGlobalUBOBuf.memory, 0, sizeof(AeroGlobalUBO), 0, &aeroGlobalMapped);

    // ── Surface overlay UBO (persistently mapped) ──
    createBuffer(sizeof(AeroSurfaceUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 aeroSurfaceUBOBuf);
    vkMapMemory(device, aeroSurfaceUBOBuf.memory, 0, sizeof(AeroSurfaceUBO), 0, &aeroSurfaceMapped);

    // ── Descriptor set layout: SSBO (b0) + global UBO (b1) ──
    {
        VkDescriptorSetLayoutBinding b[2]{};
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;
        b[1].binding = 1;
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT
                        | VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 2;
        li.pBindings = b;
        if (vkCreateDescriptorSetLayout(device, &li, nullptr, &aeroDescSetLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create aero descriptor set layout");
    }
    // ── Surface UBO layout (b0) ──
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1;
        li.pBindings = &b;
        if (vkCreateDescriptorSetLayout(device, &li, nullptr, &aeroSurfaceDescSetLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create aero surface descriptor set layout");
    }

    // ── Pool + sets ──
    VkDescriptorPoolSize ps[2]{};
    ps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[0].descriptorCount = 1;
    ps[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ps[1].descriptorCount = 2;
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 2;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(device, &pci, nullptr, &aeroDescPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create aero descriptor pool");

    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = aeroDescPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &aeroDescSetLayout;
        if (vkAllocateDescriptorSets(device, &ai, &aeroDescSet) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate aero descriptor set");

        ai.pSetLayouts = &aeroSurfaceDescSetLayout;
        if (vkAllocateDescriptorSets(device, &ai, &aeroSurfaceDescSet) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate aero surface descriptor set");
    }

    VkDescriptorBufferInfo ssboInfo{ aeroParticleSSBO.buffer, 0, ssboSize };
    VkDescriptorBufferInfo gubInfo { aeroGlobalUBOBuf.buffer, 0, sizeof(AeroGlobalUBO) };
    VkDescriptorBufferInfo subInfo { aeroSurfaceUBOBuf.buffer, 0, sizeof(AeroSurfaceUBO) };

    VkWriteDescriptorSet w[3]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = aeroDescSet; w[0].dstBinding = 0;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[0].descriptorCount = 1; w[0].pBufferInfo = &ssboInfo;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = aeroDescSet; w[1].dstBinding = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[1].descriptorCount = 1; w[1].pBufferInfo = &gubInfo;
    w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[2].dstSet = aeroSurfaceDescSet; w[2].dstBinding = 0;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[2].descriptorCount = 1; w[2].pBufferInfo = &subInfo;
    vkUpdateDescriptorSets(device, 3, w, 0, nullptr);
}

void VulkanEngine::createAeroPipelines() {
    // ── 1. Advection compute pipeline ──
    {
        struct AdvectPC { float dt; uint32_t N; uint32_t frame; float time; };
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = sizeof(AdvectPC);
        VkPipelineLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        li.setLayoutCount = 1;
        li.pSetLayouts = &aeroDescSetLayout;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pc;
        if (vkCreatePipelineLayout(device, &li, nullptr, &aeroAdvectLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create aero advect layout");

        auto code = readFile("shaders/airflow_advect.comp.spv");
        VkShaderModule mod = createShaderModule(code);
        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = mod;
        cpi.stage.pName = "main";
        cpi.layout = aeroAdvectLayout;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &aeroAdvectPipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create aero advect pipeline");
        vkDestroyShaderModule(device, mod, nullptr);
    }

    VkPipelineRenderingCreateInfo dynRender{};
    dynRender.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    dynRender.colorAttachmentCount    = 1;
    dynRender.pColorAttachmentFormats = &swapchainFormat;
    dynRender.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynStates;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // ── 2. Streamline render pipeline (LINE_LIST, additive, no depth write) ──
    {
        auto vsC = readFile("shaders/airflow_render.vert.spv");
        auto fsC = readFile("shaders/airflow_render.frag.spv");
        VkShaderModule vs = createShaderModule(vsC);
        VkShaderModule fs = createShaderModule(fsC);
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vs; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vtx{};
        vtx.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;  // no vertex buffer
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        VkPipelineRasterizationStateCreateInfo rast{};
        rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rast.polygonMode = VK_POLYGON_MODE_FILL;
        rast.cullMode = VK_CULL_MODE_NONE;
        rast.lineWidth = 1.0f;

        VkPipelineColorBlendAttachmentState cbAtt{};
        cbAtt.blendEnable = VK_TRUE;
        cbAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cbAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cbAtt.colorBlendOp = VK_BLEND_OP_ADD;
        cbAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cbAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cbAtt.alphaBlendOp = VK_BLEND_OP_ADD;
        cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                             | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1; cb.pAttachments = &cbAtt;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_FALSE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;

        struct RenderPC { glm::mat4 mvp; float lw, p0, p1, p2; };
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pc.offset = 0; pc.size = sizeof(RenderPC);
        VkPipelineLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        li.setLayoutCount = 1; li.pSetLayouts = &aeroDescSetLayout;
        li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pc;
        if (vkCreatePipelineLayout(device, &li, nullptr, &aeroRenderLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create aero render layout");

        VkGraphicsPipelineCreateInfo gpi{};
        gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpi.pNext = &dynRender;
        gpi.stageCount = 2; gpi.pStages = stages;
        gpi.pVertexInputState = &vtx; gpi.pInputAssemblyState = &ia;
        gpi.pViewportState = &vp; gpi.pRasterizationState = &rast;
        gpi.pMultisampleState = &ms; gpi.pDepthStencilState = &ds;
        gpi.pColorBlendState = &cb; gpi.pDynamicState = &dyn;
        gpi.layout = aeroRenderLayout;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, nullptr, &aeroRenderPipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create aero render pipeline");
        vkDestroyShaderModule(device, vs, nullptr);
        vkDestroyShaderModule(device, fs, nullptr);
    }

    // ── 3. Pressure-surface overlay pipeline (model geometry, alpha blend) ──
    {
        auto vsC = readFile("shaders/aero_surface.vert.spv");
        auto fsC = readFile("shaders/aero_surface.frag.spv");
        VkShaderModule vs = createShaderModule(vsC);
        VkShaderModule fs = createShaderModule(fsC);
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vs; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0; binding.stride = sizeof(ModelVertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        VkVertexInputAttributeDescription attrs[4]{};
        attrs[0].location = 0; attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;    attrs[0].offset = offsetof(ModelVertex, pos);
        attrs[1].location = 1; attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;    attrs[1].offset = offsetof(ModelVertex, normal);
        attrs[2].location = 2; attrs[2].format = VK_FORMAT_R32G32_SFLOAT;       attrs[2].offset = offsetof(ModelVertex, uv);
        attrs[3].location = 3; attrs[3].format = VK_FORMAT_R32G32B32A32_SFLOAT; attrs[3].offset = offsetof(ModelVertex, tangent);
        VkPipelineVertexInputStateCreateInfo vtx{};
        vtx.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vtx.vertexBindingDescriptionCount = 1; vtx.pVertexBindingDescriptions = &binding;
        vtx.vertexAttributeDescriptionCount = 4; vtx.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineRasterizationStateCreateInfo rast{};
        rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rast.polygonMode = VK_POLYGON_MODE_FILL;
        rast.cullMode = VK_CULL_MODE_NONE;
        rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rast.lineWidth = 1.0f;

        VkPipelineColorBlendAttachmentState cbAtt{};
        cbAtt.blendEnable = VK_TRUE;
        cbAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cbAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cbAtt.colorBlendOp = VK_BLEND_OP_ADD;
        cbAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cbAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cbAtt.alphaBlendOp = VK_BLEND_OP_ADD;
        cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                             | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1; cb.pAttachments = &cbAtt;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_FALSE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        struct SurfPC { glm::mat4 mvp; glm::mat4 model; };
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pc.offset = 0; pc.size = sizeof(SurfPC);
        VkPipelineLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        li.setLayoutCount = 1; li.pSetLayouts = &aeroSurfaceDescSetLayout;
        li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pc;
        if (vkCreatePipelineLayout(device, &li, nullptr, &aeroSurfaceLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create aero surface layout");

        VkGraphicsPipelineCreateInfo gpi{};
        gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpi.pNext = &dynRender;
        gpi.stageCount = 2; gpi.pStages = stages;
        gpi.pVertexInputState = &vtx; gpi.pInputAssemblyState = &ia;
        gpi.pViewportState = &vp; gpi.pRasterizationState = &rast;
        gpi.pMultisampleState = &ms; gpi.pDepthStencilState = &ds;
        gpi.pColorBlendState = &cb; gpi.pDynamicState = &dyn;
        gpi.layout = aeroSurfaceLayout;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, nullptr, &aeroSurfacePipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create aero surface pipeline");
        vkDestroyShaderModule(device, vs, nullptr);
        vkDestroyShaderModule(device, fs, nullptr);
    }
}

void VulkanEngine::updateAeroGlobalUBO() {
    if (!settings.aeroVisualizationEnabled) return;
    AircraftRenderState rs = aircraftRenderState();
    if (!rs.visible) return;

    const glm::mat4 bodyToWorld = glm::translate(glm::mat4(1.0f), rs.position)
                                * glm::mat4_cast(rs.rotation);
    const glm::mat4 worldToBody = glm::inverse(bodyToWorld);
    const glm::mat3 worldToBodyRot = glm::mat3(worldToBody);

    glm::vec3 vAircraft = aircraftLinearVelocity();
    glm::vec3 vInfWorld = -vAircraft;               // air flows toward the aircraft
    float airspeed = glm::length(vInfWorld);
    if (airspeed < 1e-3f) { vInfWorld = glm::vec3(0.0f); }

    glm::vec3 vInfBody = worldToBodyRot * vInfWorld;

    // AoA: angle between the chord direction (+X nose) and the apparent wind.
    // vInfBody ≈ (−V, 0, 0) for level flight (wind blows from nose toward tail).
    // We want AoA = 0 for level flight → use atan2(−vInfBody.y, −vInfBody.x).
    float aoaRad = (airspeed > 1.0f) ? std::atan2(-vInfBody.y, -vInfBody.x) : 0.0f;
    const float slope = settings.aircraftLiftCurveSlope;
    const float stall = glm::radians(settings.aircraftStallAoADeg);
    float cl;
    {
        float aoaAbs = std::abs(aoaRad);
        float clPeak = slope * stall;
        if (aoaAbs <= stall) cl = slope * aoaRad;
        else {
            float t = glm::clamp((aoaAbs - stall) / stall, 0.0f, 1.0f);
            float sgn = (aoaRad >= 0.0f) ? 1.0f : -1.0f;
            cl = sgn * clPeak * (1.0f - 0.6f * t);
        }
    }

    const float wingArea = std::max(1.0f, settings.aircraftWingArea);
    const float span     = 0.5f * std::sqrt(7.0f * wingArea) * 2.0f;  // AR≈7 wingspan
    const float fuseR    = std::max(0.2f, settings.aircraftHalfHeight);
    const float Gamma0   = (span > 0.01f && airspeed > 0.5f)
                         ? airspeed * wingArea * cl / (glm::pi<float>() * span) : 0.0f;
    const float coreR    = span * 0.025f;
    const float Q        = 2.0f * glm::pi<float>() * fuseR * fuseR * airspeed;

    AeroGlobalUBO& g = aeroGlobalData;
    g.bodyToWorld     = bodyToWorld;
    g.worldToBody     = worldToBody;
    g.freestreamWorld = glm::vec4(vInfWorld, airspeed);
    g.freestreamBody  = glm::vec4(vInfBody, glm::degrees(aoaRad));
    g.aircraftDims    = glm::vec4(span * 0.5f, settings.aircraftHalfLength, fuseR,
                                  (float)glfwGetTime());
    g.vortexParams    = glm::vec4(Gamma0, span * 0.5f, coreR, 0.0f);
    g.sourceParams    = glm::vec4(Q, settings.aircraftHalfLength, fuseR, 0.0f);
    g.stabParams      = glm::vec4(Gamma0 * -0.10f, span * 0.5f * 0.35f, coreR * 0.8f, 0.0f);
    g.particleConfig  = glm::vec4(std::max(1.0f, settings.aeroParticleLifetime),
                                  std::max(1.0f, settings.aeroSpawnRadius),
                                  std::max(2.0f, settings.aeroSpawnUpstream), // boxHalfLen
                                  (float)AERO_PARTICLE_COUNT);
    g.flags           = glm::ivec4((int)AERO_PARTICLE_COUNT, 0,
                                   aeroParticlesInitialized ? 0 : 1, (int)currentFrame);
    // Simulation centre = camera orbit target so the particle volume tracks the viewer,
    // not the aircraft.  When the camera follows the aircraft the two coincide; when the
    // user orbits to a different angle the streamlines are always visible around the
    // subject they're looking at.
    g.simVolume       = glm::vec4(camera.pos, 0.0f);
    aeroParticlesInitialized = true;
    std::memcpy(aeroGlobalMapped, &g, sizeof(AeroGlobalUBO));

    AeroSurfaceUBO s{};
    float qDyn = 0.5f * 1.225f * airspeed * airspeed;
    s.freestreamBody = glm::vec4(vInfBody, qDyn);
    s.visualConfig   = glm::vec4(settings.aeroCpOverlayOpacity, settings.aeroCpMin,
                                 settings.aeroCpMax, 0.8f);
    s.surfParams     = glm::vec4(settings.aircraftHalfLength, glm::degrees(aoaRad), 0.0f, 0.0f);
    std::memcpy(aeroSurfaceMapped, &s, sizeof(AeroSurfaceUBO));
}

void VulkanEngine::recordAeroCompute(VkCommandBuffer cmd) {
    if (!settings.aeroVisualizationEnabled) return;
    if (!aircraftRenderState().visible) return;

    static double prevT = glfwGetTime();
    double now = glfwGetTime();
    float dt = (float)glm::clamp(now - prevT, 0.0, 0.05);
    prevT = now;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, aeroAdvectPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            aeroAdvectLayout, 0, 1, &aeroDescSet, 0, nullptr);
    struct AdvectPC { float dt; uint32_t N; uint32_t frame; float time; } apc{
        dt, AERO_PARTICLE_COUNT, currentFrame, (float)glfwGetTime() };
    vkCmdPushConstants(cmd, aeroAdvectLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(apc), &apc);
    uint32_t groups = (AERO_PARTICLE_COUNT + 63) / 64;
    vkCmdDispatch(cmd, groups, 1, 1);

    // Compute writes → vertex reads barrier
    VkBufferMemoryBarrier bmb{};
    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = aeroParticleSSBO.buffer;
    bmb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                         0, 0, nullptr, 1, &bmb, 0, nullptr);
}

void VulkanEngine::recordAeroRender(VkCommandBuffer cmd, const glm::mat4& viewProj) {
    if (!settings.aeroVisualizationEnabled) return;
    AircraftRenderState rs = aircraftRenderState();
    if (!rs.visible) return;

    // ── Pressure-surface overlay (only when the FBX model is being shown) ──
    const bool useModel = aircraftAsset.valid && settings.aircraftUseModel
                       && aircraftAsset.indexCount > 0;
    if (useModel) {
        const glm::vec3 center = 0.5f * (aircraftAsset.boundsMin + aircraftAsset.boundsMax);
        const float scl = aircraftAsset.autoScale * std::max(0.01f, settings.aircraftModelScale);
        glm::mat4 offset =
            glm::rotate(glm::mat4(1.0f), glm::radians(settings.aircraftModelYawOffsetDeg),   glm::vec3(0,1,0)) *
            glm::rotate(glm::mat4(1.0f), glm::radians(settings.aircraftModelPitchOffsetDeg), glm::vec3(0,0,1)) *
            glm::rotate(glm::mat4(1.0f), glm::radians(settings.aircraftModelRollOffsetDeg),  glm::vec3(1,0,0));
        glm::mat4 model = glm::translate(glm::mat4(1.0f), rs.position)
                        * glm::mat4_cast(rs.rotation)
                        * offset
                        * glm::scale(glm::mat4(1.0f), glm::vec3(scl))
                        * glm::translate(glm::mat4(1.0f), -center);
        struct SurfPC { glm::mat4 mvp; glm::mat4 model; } spc;
        spc.mvp = viewProj * model;
        spc.model = model;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, aeroSurfacePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                aeroSurfaceLayout, 0, 1, &aeroSurfaceDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, aeroSurfaceLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(spc), &spc);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &aircraftAsset.vbo.buffer, &off);
        vkCmdBindIndexBuffer(cmd, aircraftAsset.ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, aircraftAsset.indexCount, 1, 0, 0, 0);
    }

    // ── CPU-integrated curved streamlines (MSFS-style) ─────────────────
    // Draw first (additive, behind panels) so they show through the aircraft.
    if (streamlinePipeline != VK_NULL_HANDLE && streamlineVertCount >= 2) {
        struct SLPush { glm::mat4 mvp; float vMin, vMax, time, opacity; } slp{};
        slp.mvp     = viewProj;
        slp.vMin    = aeroGlobalData.freestreamWorld.w * 0.30f;
        slp.vMax    = aeroGlobalData.freestreamWorld.w * 1.75f;
        slp.time    = static_cast<float>(glfwGetTime());
        slp.opacity = 0.82f;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, streamlinePipeline);
        VkBuffer     slBuf[]    = { streamlineVBO.buffer };
        VkDeviceSize slOffsets[]= { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, slBuf, slOffsets);
        vkCmdPushConstants(cmd, streamlineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(SLPush), &slp);
        vkCmdDraw(cmd, streamlineVertCount, 1, 0, 0);
    }

    // (The old GPU particle "streamlines" were removed — the CPU-integrated
    //  curved streamlines above replace them with physically-shaped, fully
    //  visible flow lines instead of sparse moving dots.)

    // ── Panel Cp quads — actual physics-computed pressure distribution ──
    // Each of the 24 strip-theory panels is drawn as two triangles
    // colored by its computed Cp (blue=suction, green=ambient, red=stagnation).
    if (aeroPanelPipeline != VK_NULL_HANDLE && !aeroBodyViz.empty()) {
        const uint32_t numP = std::min((uint32_t)aeroBodyViz.panels().size(), AERO_MAX_PANELS);
        struct PanelPC { glm::mat4 mvp; uint32_t numPanels; float p0, p1, p2; } ppc{};
        ppc.mvp = viewProj;
        ppc.numPanels = numP;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, aeroPanelPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                aeroPanelLayout, 0, 1, &aeroPanelDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, aeroPanelLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(ppc), &ppc);
        // 6 vertices per panel (2 triangles), no index buffer
        vkCmdDraw(cmd, numP * 6, 1, 0, 0);
    }
}

void VulkanEngine::destroyAeroResources() {
    if (aeroAdvectPipeline) { vkDestroyPipeline(device, aeroAdvectPipeline, nullptr); aeroAdvectPipeline = VK_NULL_HANDLE; }
    if (aeroAdvectLayout)   { vkDestroyPipelineLayout(device, aeroAdvectLayout, nullptr); aeroAdvectLayout = VK_NULL_HANDLE; }
    if (aeroRenderPipeline) { vkDestroyPipeline(device, aeroRenderPipeline, nullptr); aeroRenderPipeline = VK_NULL_HANDLE; }
    if (aeroRenderLayout)   { vkDestroyPipelineLayout(device, aeroRenderLayout, nullptr); aeroRenderLayout = VK_NULL_HANDLE; }
    if (aeroSurfacePipeline){ vkDestroyPipeline(device, aeroSurfacePipeline, nullptr); aeroSurfacePipeline = VK_NULL_HANDLE; }
    if (aeroSurfaceLayout)  { vkDestroyPipelineLayout(device, aeroSurfaceLayout, nullptr); aeroSurfaceLayout = VK_NULL_HANDLE; }
    if (aeroDescPool)       { vkDestroyDescriptorPool(device, aeroDescPool, nullptr); aeroDescPool = VK_NULL_HANDLE; }
    if (aeroDescSetLayout)  { vkDestroyDescriptorSetLayout(device, aeroDescSetLayout, nullptr); aeroDescSetLayout = VK_NULL_HANDLE; }
    if (aeroSurfaceDescSetLayout) { vkDestroyDescriptorSetLayout(device, aeroSurfaceDescSetLayout, nullptr); aeroSurfaceDescSetLayout = VK_NULL_HANDLE; }
    if (aeroGlobalMapped)   { vkUnmapMemory(device, aeroGlobalUBOBuf.memory); aeroGlobalMapped = nullptr; }
    if (aeroSurfaceMapped)  { vkUnmapMemory(device, aeroSurfaceUBOBuf.memory); aeroSurfaceMapped = nullptr; }
    if (aeroGlobalUBOBuf.buffer)  { vkDestroyBuffer(device, aeroGlobalUBOBuf.buffer, nullptr); vkFreeMemory(device, aeroGlobalUBOBuf.memory, nullptr); aeroGlobalUBOBuf = {}; }
    if (aeroSurfaceUBOBuf.buffer) { vkDestroyBuffer(device, aeroSurfaceUBOBuf.buffer, nullptr); vkFreeMemory(device, aeroSurfaceUBOBuf.memory, nullptr); aeroSurfaceUBOBuf = {}; }
    if (aeroParticleSSBO.buffer)  { vkDestroyBuffer(device, aeroParticleSSBO.buffer, nullptr); vkFreeMemory(device, aeroParticleSSBO.memory, nullptr); aeroParticleSSBO = {}; }
    // Panel visualization
    if (aeroPanelPipeline)   { vkDestroyPipeline(device, aeroPanelPipeline, nullptr); aeroPanelPipeline = VK_NULL_HANDLE; }
    if (aeroPanelLayout)     { vkDestroyPipelineLayout(device, aeroPanelLayout, nullptr); aeroPanelLayout = VK_NULL_HANDLE; }
    if (aeroPanelDescPool)   { vkDestroyDescriptorPool(device, aeroPanelDescPool, nullptr); aeroPanelDescPool = VK_NULL_HANDLE; }
    if (aeroPanelDescLayout) { vkDestroyDescriptorSetLayout(device, aeroPanelDescLayout, nullptr); aeroPanelDescLayout = VK_NULL_HANDLE; }
    if (aeroPanelSSBO.buffer){ vkDestroyBuffer(device, aeroPanelSSBO.buffer, nullptr); vkFreeMemory(device, aeroPanelSSBO.memory, nullptr); aeroPanelSSBO = {}; }
    // Streamlines
    if (streamlinePipeline)  { vkDestroyPipeline(device, streamlinePipeline, nullptr); streamlinePipeline = VK_NULL_HANDLE; }
    if (streamlineLayout)    { vkDestroyPipelineLayout(device, streamlineLayout, nullptr); streamlineLayout = VK_NULL_HANDLE; }
    if (streamlineMapped)    { vkUnmapMemory(device, streamlineVBO.memory); streamlineMapped = nullptr; }
    if (streamlineVBO.buffer){ vkDestroyBuffer(device, streamlineVBO.buffer, nullptr); vkFreeMemory(device, streamlineVBO.memory, nullptr); streamlineVBO = {}; }
}

// ══════════════════════════════════════════════════════════════
//  Panel Cp visualization — directly shows physics-computed
//  pressure distribution on the 24 strip-theory panels.
// ══════════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════════
//  CPU-integrated streamlines
// ══════════════════════════════════════════════════════════════
void VulkanEngine::createStreamlinePipeline() {
    // ── Vertex buffer (host-visible, persistently mapped) ──
    VkDeviceSize vboSize = sizeof(StreamlineVertex) * STREAMLINE_MAX_VERTS;
    createBuffer(vboSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 streamlineVBO);
    vkMapMemory(device, streamlineVBO.memory, 0, vboSize, 0, &streamlineMapped);
    std::memset(streamlineMapped, 0, (size_t)vboSize);

    // ── Pipeline ──
    auto vsC = readFile("shaders/streamline.vert.spv");
    auto fsC = readFile("shaders/streamline.frag.spv");
    VkShaderModule vs = createShaderModule(vsC);
    VkShaderModule fs = createShaderModule(fsC);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs; stages[1].pName = "main";

    // Vertex input: one binding, three attributes (pos, speed, t)
    VkVertexInputBindingDescription bind{};
    bind.binding   = 0;
    bind.stride    = sizeof(StreamlineVertex);
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[3]{};
    attrs[0].location = 0; attrs[0].binding = 0;
    attrs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset   = offsetof(StreamlineVertex, x);
    attrs[1].location = 1; attrs[1].binding = 0;
    attrs[1].format   = VK_FORMAT_R32_SFLOAT;
    attrs[1].offset   = offsetof(StreamlineVertex, speed);
    attrs[2].location = 2; attrs[2].binding = 0;
    attrs[2].format   = VK_FORMAT_R32_SFLOAT;
    attrs[2].offset   = offsetof(StreamlineVertex, t);

    VkPipelineVertexInputStateCreateInfo vtx{};
    vtx.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vtx.vertexBindingDescriptionCount   = 1;    vtx.pVertexBindingDescriptions   = &bind;
    vtx.vertexAttributeDescriptionCount = 3;    vtx.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType     = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode  = VK_CULL_MODE_NONE;
    rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth = 1.0f;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynStates;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Standard alpha blend — additive was nearly invisible on bright sky/terrain
    // and made lines appear to flicker in/out as the background changed.
    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.blendEnable         = VK_TRUE;
    cbAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cbAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cbAtt.colorBlendOp        = VK_BLEND_OP_ADD;
    cbAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cbAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cbAtt.alphaBlendOp        = VK_BLEND_OP_ADD;
    cbAtt.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                              | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &cbAtt;

    // No depth test — streamlines are a view overlay and must stay visible
    // over terrain/sky/aircraft regardless of occlusion (MSFS-style).
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    // Push constant: MVP (64) + vMin,vMax,time,opacity (16) = 80 bytes
    struct SLPush { glm::mat4 mvp; float vMin, vMax, time, opacity; };
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pc.offset     = 0;
    pc.size       = sizeof(SLPush);

    VkPipelineLayoutCreateInfo li{};
    li.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges    = &pc;
    if (vkCreatePipelineLayout(device, &li, nullptr, &streamlineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create streamline pipeline layout");

    VkPipelineRenderingCreateInfo dynRender{};
    dynRender.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    dynRender.colorAttachmentCount    = 1;
    dynRender.pColorAttachmentFormats = &swapchainFormat;
    dynRender.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo gpi{};
    gpi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpi.pNext               = &dynRender;
    gpi.stageCount          = 2; gpi.pStages = stages;
    gpi.pVertexInputState   = &vtx;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState      = &vp;
    gpi.pRasterizationState = &rast;
    gpi.pMultisampleState   = &ms;
    gpi.pDepthStencilState  = &ds;
    gpi.pColorBlendState    = &cb;
    gpi.pDynamicState       = &dyn;
    gpi.layout              = streamlineLayout;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, nullptr, &streamlinePipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create streamline pipeline");

    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);
}

// Integrate streamlines on the CPU and upload to the VBO.
// Seeds on upstream cross-sections perpendicular to the freestream (classic
// wind-tunnel inlet).  Uses aircraft body axes so lines fill a 3-D volume
// and curve over/under the wing — not a flat camera-facing sheet.
void VulkanEngine::updateStreamlines() {
    if (!settings.aeroVisualizationEnabled) return;
    AircraftRenderState rs = aircraftRenderState();
    if (!rs.visible) return;

    const AeroGlobalUBO& u = aeroGlobalData;
    const float airspeed = u.freestreamWorld.w;
    if (airspeed < 0.5f) return;

    // Air travels in freestreamWorld direction (toward the aircraft).
    glm::vec3 flowDir(u.freestreamWorld);
    if (glm::length(flowDir) < 0.5f) {
        flowDir = glm::normalize(glm::mat3(u.bodyToWorld) * glm::vec3(-1.0f, 0.0f, 0.0f));
    } else {
        flowDir = glm::normalize(flowDir);
    }

    // Cross-section axes from aircraft body frame (span = wing, vert = up).
    const glm::mat3 bodyRot(u.bodyToWorld);
    glm::vec3 spanAxis = glm::normalize(bodyRot * glm::vec3(0.0f, 0.0f, 1.0f));
    glm::vec3 vertAxis = glm::normalize(bodyRot * glm::vec3(0.0f, 1.0f, 0.0f));

    const glm::vec3 acPos      = rs.position;
    const float   halfSpan   = u.vortexParams.y * 1.35f;
    const float   halfVert   = std::max(halfSpan * 0.65f, u.sourceParams.z * 2.5f);
    const float   upstream   = std::max(6.0f, settings.aeroSpawnUpstream);
    const float   downstream = upstream * 1.15f;
    const float   vizK       = std::max(1.0f, settings.aeroVizStrength);

    const int   GRID_N   = 14;
    const int   LAYERS   = 3;       // stacked upstream slices → 3-D volume, not one sheet
    const int   MAX_STEPS = 90;
    const float STEP     = 0.18f;
    const float maxArc   = STEP * float(MAX_STEPS);

    auto* verts = reinterpret_cast<StreamlineVertex*>(streamlineMapped);
    uint32_t vcount = 0;

    auto emitStreamline = [&](glm::vec3 pos) {
        glm::vec3 prevPos = pos;
        float     prevT   = 0.0f;
        float     prevSpd = airspeed;
        bool      hasPrev = false;
        float     arcLen  = 0.0f;

        for (int s = 0; s < MAX_STEPS; ++s) {
            glm::vec3 k1  = cpuFlowVelocityWorld(pos, u, vizK);
            float     spd = glm::length(k1);
            if (spd < 0.3f) break;
            glm::vec3 dir1 = k1 / spd;
            glm::vec3 mid  = pos + dir1 * STEP;
            glm::vec3 k2   = cpuFlowVelocityWorld(mid, u, vizK);
            float     spd2 = glm::length(k2);
            glm::vec3 dir2 = (spd2 > 0.3f) ? (k2 / spd2) : dir1;
            pos = pos + 0.5f * (dir1 + dir2) * STEP;

            arcLen += STEP;
            float curT = arcLen / maxArc;

            if (hasPrev && vcount + 2 <= STREAMLINE_MAX_VERTS) {
                verts[vcount++] = { prevPos.x, prevPos.y, prevPos.z, prevSpd, prevT };
                verts[vcount++] = { pos.x,     pos.y,     pos.z,     spd,     curT  };
            }

            prevPos = pos;
            prevSpd = spd;
            prevT   = curT;
            hasPrev = true;

            if (arcLen >= maxArc) break;
            // Stop once well past the aircraft tail in flow direction.
            const float along = glm::dot(pos - acPos, flowDir);
            if (along > downstream) break;
        }
    };

    for (int layer = 0; layer < LAYERS; ++layer) {
        const float layerBack = upstream + layer * (upstream * 0.22f);
        const glm::vec3 planeCenter = acPos - flowDir * layerBack;

        for (int iy = 0; iy < GRID_N && vcount + MAX_STEPS * 2 < STREAMLINE_MAX_VERTS; ++iy) {
            for (int ix = 0; ix < GRID_N && vcount + MAX_STEPS * 2 < STREAMLINE_MAX_VERTS; ++ix) {
                const float fu = (ix / float(GRID_N - 1) - 0.5f) * 2.0f;
                const float fv = (iy / float(GRID_N - 1) - 0.5f) * 2.0f;
                const glm::vec3 seed = planeCenter
                                     + fu * halfSpan * spanAxis
                                     + fv * halfVert * vertAxis;
                emitStreamline(seed);
            }
        }
    }

    streamlineVertCount = vcount;
}

void VulkanEngine::createAeroPanelVizResources() {
    VkDeviceSize ssboSize = sizeof(PanelVizData) * AERO_MAX_PANELS;
    createBuffer(ssboSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 aeroPanelSSBO);
    vkMapMemory(device, aeroPanelSSBO.memory, 0, ssboSize, 0, &aeroPanelMapped);
    std::memset(aeroPanelMapped, 0, (size_t)ssboSize);

    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 1;
    li.pBindings = &b;
    if (vkCreateDescriptorSetLayout(device, &li, nullptr, &aeroPanelDescLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create aero panel descriptor set layout");

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 1;
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(device, &pci, nullptr, &aeroPanelDescPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create aero panel descriptor pool");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = aeroPanelDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &aeroPanelDescLayout;
    if (vkAllocateDescriptorSets(device, &ai, &aeroPanelDescSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate aero panel descriptor set");

    VkDescriptorBufferInfo bi2{ aeroPanelSSBO.buffer, 0, ssboSize };
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = aeroPanelDescSet;
    w.dstBinding = 0;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.descriptorCount = 1;
    w.pBufferInfo = &bi2;
    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
}

void VulkanEngine::createAeroPanelVizPipeline() {
    auto vsC = readFile("shaders/aero_panel_viz.vert.spv");
    auto fsC = readFile("shaders/aero_panel_viz.frag.spv");
    VkShaderModule vs = createShaderModule(vsC);
    VkShaderModule fs = createShaderModule(fsC);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = vs; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vtx{};
    vtx.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;  // no vertex buffer
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode    = VK_CULL_MODE_NONE;   // panels must be visible from both sides
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth   = 1.0f;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynStates;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Alpha blend so panels are semi-transparent over the aircraft model
    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.blendEnable = VK_TRUE;
    cbAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cbAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cbAtt.colorBlendOp = VK_BLEND_OP_ADD;
    cbAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cbAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cbAtt.alphaBlendOp = VK_BLEND_OP_ADD;
    cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                         | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &cbAtt;

    // Depth test but no depth write (overlay)
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp  = VK_COMPARE_OP_LESS_OR_EQUAL;

    // Push constant: MVP + numPanels + 3 padding floats
    struct PanelPC { glm::mat4 mvp; uint32_t numPanels; float p0, p1, p2; };
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pc.offset = 0;
    pc.size = sizeof(PanelPC);
    VkPipelineLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    li.setLayoutCount = 1; li.pSetLayouts = &aeroPanelDescLayout;
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(device, &li, nullptr, &aeroPanelLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create aero panel pipeline layout");

    VkPipelineRenderingCreateInfo dynRender{};
    dynRender.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    dynRender.colorAttachmentCount    = 1;
    dynRender.pColorAttachmentFormats = &swapchainFormat;
    dynRender.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo gpi{};
    gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpi.pNext = &dynRender;
    gpi.stageCount = 2; gpi.pStages = stages;
    gpi.pVertexInputState = &vtx; gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vp; gpi.pRasterizationState = &rast;
    gpi.pMultisampleState = &ms; gpi.pDepthStencilState = &ds;
    gpi.pColorBlendState = &cb; gpi.pDynamicState = &dyn;
    gpi.layout = aeroPanelLayout;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, nullptr, &aeroPanelPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create aero panel pipeline");
    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);
}

void VulkanEngine::updateAeroPanelVizData() {
    if (!settings.aeroVisualizationEnabled) return;
    AircraftRenderState rs = aircraftRenderState();
    if (!rs.visible) return;

    // Build the panel model once (or if settings change); geometry is fixed.
    if (!aeroBodyVizBuilt) {
        const float wingArea = std::max(1.0f, settings.aircraftWingArea);
        const float halfSpan = 0.5f * std::sqrt(7.0f * wingArea);
        aeroBodyViz.buildFromHalfExtents(
            settings.aircraftHalfLength, halfSpan, settings.aircraftHalfHeight,
            settings.aircraftMassKg, wingArea,
            settings.aircraftStallAoADeg, settings.aircraftLiftCurveSlope);
        aeroBodyVizBuilt = true;
    }

    // Transform: aircraft world position + orientation
    const glm::mat4 bodyToWorld = glm::translate(glm::mat4(1.0f), rs.position)
                                * glm::mat4_cast(rs.rotation);
    const glm::mat4 worldToBody = glm::inverse(bodyToWorld);
    const glm::mat3 bodyRot     = glm::mat3(bodyToWorld);

    // Evaluate aerodynamics for the current velocity (body-frame freestream)
    const glm::vec3 vAircraft   = aircraftLinearVelocity();
    const glm::vec3 vInfBody    = glm::mat3(worldToBody) * (-vAircraft);
    const float     airspeed    = glm::length(vAircraft);
    const float     qDyn        = 0.5f * 1.225f * airspeed * airspeed;
    aeroBodyViz.compute(vInfBody, 1.225f);  // updates Cl/Cd/Cp on each panel

    // Pack world-space panel data for the GPU
    const auto& panelList = aeroBodyViz.panels();
    const uint32_t N = std::min((uint32_t)panelList.size(), AERO_MAX_PANELS);
    std::vector<PanelVizData> vizData(N);
    for (uint32_t i = 0; i < N; ++i) {
        const AeroPanel& p = panelList[i];
        // World-space center
        glm::vec3 wc = glm::vec3(bodyToWorld * glm::vec4(p.center, 1.0f));
        // World-space half-vectors
        glm::vec3 wCH = bodyRot * (p.chordDir * (p.chord * 0.5f));
        glm::vec3 wSH = bodyRot * (p.spanDir  * (p.span  * 0.5f));
        float liftN = qDyn * p.area * p.Cl;

        vizData[i].center    = glm::vec4(wc, p.Cp);
        vizData[i].chordHalf = glm::vec4(wCH, 0.0f);
        vizData[i].spanHalf  = glm::vec4(wSH, 0.0f);
        vizData[i].meta      = glm::vec4(liftN, float(p.groupID), p.localAoADeg, p.Cl);
    }

    VkDeviceSize uploadSize = sizeof(PanelVizData) * N;
    std::memcpy(aeroPanelMapped, vizData.data(), (size_t)uploadSize);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Uniform buffer
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createUniformBuffer() {
    VkDeviceSize size = sizeof(FrameData);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uniformBuffers[i]);
        vkMapMemory(device, uniformBuffers[i].memory, 0, size, 0, &uniformMapped[i]);
    }
}

// â”€â”€â”€ Earth atmospheric parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//  Values mirror atmosphere_bac_src/source/model/sky_model.cpp's
//  SetupEarthAtmosphere(): kilometers, Bruneton density-profile
//  layer convention. Layers are laid out as flat float[12] then
//  packed into 3 vec4s matching the GLSL std140 array stride.
static void fillEarthAtmosphere(AtmosphereParams& p) {
    const float EarthBottomRadius        = 6360.0f;        // km
    const float EarthTopRadius           = 6460.0f;        // km
    const float EarthRayleighScaleHeight = 8.0f;           // km
    const float EarthMieScaleHeight      = 1.2f;           // km

    p.solar_irradiance       = glm::vec3(1.474f, 1.8504f, 1.91198f);
    p.sun_angular_radius     = 0.004675f;
    p.bottom_radius          = EarthBottomRadius;
    p.top_radius             = EarthTopRadius;
    p.ground_albedo          = glm::vec3(0.0f);
    p._pad0 = p._pad1 = p._pad2 = p._pad4 = 0.0f;
    p._pad3 = glm::vec2(0.0f);

    // â”€â”€ Rayleigh â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // density profile: layer 0 = (0,0,0,0,0), layer 1 = (0, 1, -1/H, 0, 0)
    // Packed as float[12], then 3 vec4s.
    {
        float r[12] = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,                 // layer 0 first 5
            0.0f, 1.0f, -1.0f / EarthRayleighScaleHeight, // layer 1 (exp_term, exp_scale)
            0.0f, 0.0f,                                    // layer 1 (linear, constant)
            0.0f, 0.0f                                     // padding
        };
        memcpy(&p.rayleigh_density[0], r, sizeof(r));
    }
    p.rayleigh_scattering = glm::vec3(0.005802f, 0.013558f, 0.033100f);

    // â”€â”€ Mie â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    {
        float m[12] = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, -1.0f / EarthMieScaleHeight,
            0.0f, 0.0f,
            0.0f, 0.0f
        };
        memcpy(&p.mie_density[0], m, sizeof(m));
    }
    p.mie_scattering         = glm::vec3(0.003996f);
    p.mie_extinction         = glm::vec3(0.004440f);
    p.mie_phase_function_g   = 0.8f;
    {
        glm::vec3 ext = p.mie_extinction;
        glm::vec3 sca = p.mie_scattering;
        p.mie_absorption = glm::max(ext - sca, glm::vec3(0.0f));
    }

    // â”€â”€ Ozone absorption (two altitude layers) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // layer 0: (width=25, exp_term=0, exp_scale=0, linear_term=1/15, constant_term=-2/3)
    // layer 1: (width=0,  exp_term=0, exp_scale=0, linear_term=-1/15, constant_term=8/3)
    {
        float a[12] = {
            25.0f, 0.0f, 0.0f, 1.0f / 15.0f, -2.0f / 3.0f,   // layer 0 (5 floats)
            0.0f, 0.0f, 0.0f,                                  // layer 1 first 3
            -1.0f / 15.0f, 8.0f / 3.0f,                        // layer 1 last 2
            0.0f, 0.0f                                          // padding
        };
        memcpy(&p.absorption_density[0], a, sizeof(a));
    }
    p.absorption_extinction = glm::vec3(0.000650f, 0.001881f, 0.000085f);

    p.TransmittanceTexDimensions    = glm::vec2(256.0f, 64.0f);
    p.MultiscatteringTexDimensions  = glm::vec2(32.0f, 32.0f);
    p.SkyViewTexDimensions          = glm::vec2(192.0f, 128.0f);
    p.AEPerspectiveTexDimensions    = glm::vec4(32.0f, 32.0f, 32.0f, 0.0f);

    p.cameraScale                   = 0.001f;  // engine meters â†’ km
    p.camera_position               = glm::vec3(0.0f);
    p.sun_direction                 = glm::vec3(0.0f, 0.0f, 1.0f);
}

glm::vec3 VulkanEngine::currentSunDirectionYUp() const {
    // Engine Y-up: x=east, y=up, z=north. Elevation above horizon,
    // azimuth measured from +X around the up axis.
    return glm::normalize(glm::vec3(
        cos(glm::radians(settings.sunAzimuth)) * cos(glm::radians(settings.sunElevation)),
        sin(glm::radians(settings.sunElevation)),
        sin(glm::radians(settings.sunAzimuth)) * cos(glm::radians(settings.sunElevation))
    ));
}

void VulkanEngine::createAtmosphereUBO() {
    fillEarthAtmosphere(atmosphereParams);
    // Initialize sun direction from settings (Bruneton Z-up convention:
    // we lay Y on the engine-up axis into Bruneton's Z slot).
    glm::vec3 sunYup = currentSunDirectionYUp();
    atmosphereParams.sun_direction = glm::vec3(sunYup.x, sunYup.z, sunYup.y);

    VkDeviceSize size = sizeof(AtmosphereParams);
    createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        atmosphereUniformBuffer);
    vkMapMemory(device, atmosphereUniformBuffer.memory, 0, size, 0, &atmosphereUniformMapped);
    memcpy(atmosphereUniformMapped, &atmosphereParams, size);
}

void VulkanEngine::updateAtmosphereUBO() {
    glm::vec3 sunYup = currentSunDirectionYUp();
    atmosphereParams.sun_direction = glm::vec3(sunYup.x, sunYup.z, sunYup.y);
    atmosphereParams.camera_position = camera.eyePos();
    memcpy(atmosphereUniformMapped, &atmosphereParams, sizeof(AtmosphereParams));
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Bruneton/Hillaire LUT pipeline
//
//  Three compute passes that build precomputed scattering LUTs in
//  R16G16B16A16_SFLOAT storage images, plus one full-screen
//  graphics pass that samples the skyview LUT for far-depth
//  pixels and adds a sun disc.
//
//  Sized per atmosphere_bac_src defaults:
//    transmittance: 256 Ã— 64    (computed once)
//    multiscatter:   32 Ã— 32    (computed once)
//    skyview:       192 Ã— 128   (recomputed every frame â€” sun moves)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createBrunetonLUTs() {
    // â”€â”€ Three storage images (rgba16f) used as both compute targets
    // and sampled by the sky pass. Storage + Sampled usage so we can
    // imageStore from compute and texture() from fragment shaders.
    auto makeLUT = [&](Image& out, uint32_t w, uint32_t h) {
        VkImageCreateInfo ii{};
        ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ii.extent = {w, h, 1};
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_STORAGE_BIT
                 | VK_IMAGE_USAGE_SAMPLED_BIT;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &ii, nullptr, &out.image) != VK_SUCCESS)
            throw std::runtime_error("failed to create Bruneton LUT image");

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device, out.image, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &ai, nullptr, &out.memory) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate Bruneton LUT memory");
        vkBindImageMemory(device, out.image, out.memory, 0);

        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = out.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &vi, nullptr, &out.view) != VK_SUCCESS)
            throw std::runtime_error("failed to create Bruneton LUT view");
    };

    makeLUT(transmittanceLUT, 256, 64);
    makeLUT(multiscatterLUT,   32, 32);
    makeLUT(skyviewLUT,       192, 128);

    // Transition all three to GENERAL so compute can imageStore them.
    {
        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkImageMemoryBarrier2 b[3]{};
        Image* imgs[3] = {&transmittanceLUT, &multiscatterLUT, &skyviewLUT};
        for (int i = 0; i < 3; ++i) {
            b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b[i].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b[i].srcAccessMask = VK_ACCESS_2_NONE;
            b[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b[i].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            b[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b[i].image = imgs[i]->image;
            b[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b[i].subresourceRange.levelCount = 1;
            b[i].subresourceRange.layerCount = 1;
        }
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 3;
        dep.pImageMemoryBarriers = b;
        vkCmdPipelineBarrier2(cmd, &dep);
        endSingleTimeCommands(cmd);
    }

    // Linear-clamp sampler used by the sky pass to read the LUTs.
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxAnisotropy = 1.0f;
    si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    si.unnormalizedCoordinates = VK_FALSE;
    si.compareEnable = VK_FALSE;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    if (vkCreateSampler(device, &si, nullptr, &lutSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create Bruneton LUT sampler");
}

void VulkanEngine::createBrunetonComputePipelines() {
    // â”€â”€ Descriptor set layout: binding 0 = atmosphere UBO,
    //    bindings 1..3 = three LUT storage images.
    VkDescriptorSetLayoutBinding bs[4]{};
    bs[0].binding = 0;
    bs[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bs[0].descriptorCount = 1;
    bs[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (int i = 1; i <= 3; ++i) {
        bs[i].binding = i;
        bs[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bs[i].descriptorCount = 1;
        bs[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 4;
    li.pBindings = bs;
    if (vkCreateDescriptorSetLayout(device, &li, nullptr,
                                     &brunetonComputeSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create bruneton compute DS layout");

    // â”€â”€ Descriptor pool with capacity for compute set + sky pass set.
    VkDescriptorPoolSize ps[3]{};
    ps[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ps[0].descriptorCount = 4;  // compute + sky pass + headroom
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps[1].descriptorCount = 6;
    ps[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[2].descriptorCount = 4;
    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.maxSets = 4;
    pi.poolSizeCount = 3;
    pi.pPoolSizes = ps;
    if (vkCreateDescriptorPool(device, &pi, nullptr,
                                &brunetonDescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create bruneton descriptor pool");

    // â”€â”€ Allocate & populate the compute descriptor set â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = brunetonDescriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &brunetonComputeSetLayout;
    if (vkAllocateDescriptorSets(device, &ai, &brunetonComputeSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate bruneton compute DS");

    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = atmosphereUniformBuffer.buffer;
    bufInfo.offset = 0;
    bufInfo.range = sizeof(AtmosphereParams);

    VkDescriptorImageInfo imgInfo[3]{};
    Image* imgs[3] = {&transmittanceLUT, &multiscatterLUT, &skyviewLUT};
    for (int i = 0; i < 3; ++i) {
        imgInfo[i].imageView = imgs[i]->view;
        imgInfo[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    VkWriteDescriptorSet ws[4]{};
    ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[0].dstSet = brunetonComputeSet;
    ws[0].dstBinding = 0;
    ws[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ws[0].descriptorCount = 1;
    ws[0].pBufferInfo = &bufInfo;
    for (int i = 1; i <= 3; ++i) {
        ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet = brunetonComputeSet;
        ws[i].dstBinding = i;
        ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ws[i].descriptorCount = 1;
        ws[i].pImageInfo = &imgInfo[i - 1];
    }
    vkUpdateDescriptorSets(device, 4, ws, 0, nullptr);

    // â”€â”€ Pipeline layout shared by all three compute pipelines â”€â”€â”€â”€â”€
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &brunetonComputeSetLayout;
    if (vkCreatePipelineLayout(device, &pli, nullptr,
                                &brunetonComputeLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create bruneton compute layout");

    auto makeCompute = [&](const char* spv, VkPipeline& outPipe) {
        auto code = readFile(spv);
        VkShaderModule mod = createShaderModule(code);
        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = mod;
        cpi.stage.pName = "main";
        cpi.layout = brunetonComputeLayout;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi,
                                      nullptr, &outPipe) != VK_SUCCESS)
            throw std::runtime_error(std::string("failed to create bruneton pipeline: ") + spv);
        vkDestroyShaderModule(device, mod, nullptr);
    };
    makeCompute("shaders/bruneton_transmittance.comp.spv", transmittancePipeline);
    makeCompute("shaders/bruneton_multiscatter.comp.spv",  multiscatterPipeline);
    makeCompute("shaders/bruneton_skyview.comp.spv",       skyviewPipeline);
}

void VulkanEngine::createSkyPassPipeline() {
    // â”€â”€ Descriptor: binding 0 = atmosphere UBO, binding 1 = skyview sampler
    VkDescriptorSetLayoutBinding bs[2]{};
    bs[0].binding = 0;
    bs[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bs[0].descriptorCount = 1;
    bs[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bs[1].binding = 1;
    bs[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bs[1].descriptorCount = 1;
    bs[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 2;
    li.pBindings = bs;
    if (vkCreateDescriptorSetLayout(device, &li, nullptr,
                                     &skyPassSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create sky pass DS layout");

    // Allocate and populate the sky descriptor set from the same pool.
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = brunetonDescriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &skyPassSetLayout;
    if (vkAllocateDescriptorSets(device, &ai, &skyPassSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate sky pass DS");

    // Sky pass reads skyview as a sampled image in SHADER_READ_ONLY layout.
    // The pre-frame compute -> sky-pass barrier handles the GENERAL ->
    // SHADER_READ_ONLY transition; the descriptor itself is written with
    // SHADER_READ_ONLY_OPTIMAL to match Vulkan's expectations.
    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = atmosphereUniformBuffer.buffer;
    bufInfo.offset = 0;
    bufInfo.range = sizeof(AtmosphereParams);

    VkDescriptorImageInfo imgInfo{};
    imgInfo.sampler = lutSampler;
    imgInfo.imageView = skyviewLUT.view;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet ws[2]{};
    ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[0].dstSet = skyPassSet;
    ws[0].dstBinding = 0;
    ws[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ws[0].descriptorCount = 1;
    ws[0].pBufferInfo = &bufInfo;

    ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[1].dstSet = skyPassSet;
    ws[1].dstBinding = 1;
    ws[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ws[1].descriptorCount = 1;
    ws[1].pImageInfo = &imgInfo;

    vkUpdateDescriptorSets(device, 2, ws, 0, nullptr);

    // â”€â”€ Pipeline layout: one set + a 16-aligned push block (invViewProj + camPos)
    struct SkyPush {
        glm::mat4 invViewProj;
        glm::vec4 cameraPos;
    };
    VkPushConstantRange pr{};
    pr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pr.offset = 0;
    pr.size = sizeof(SkyPush);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &skyPassSetLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(device, &pli, nullptr,
                                &skyPassPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create sky pass pipeline layout");

    // â”€â”€ Build the graphics pipeline (screen triangle, depth test only) â”€â”€
    auto vertCode = readFile("shaders/bruneton_sky.vert.spv");
    auto fragCode = readFile("shaders/bruneton_sky.frag.spv");
    VkShaderModule vs = createShaderModule(vertCode);
    VkShaderModule fs = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo viState{};
    viState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo iaState{};
    iaState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vpState{};
    vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpState.viewportCount = 1;
    vpState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Depth test EQUAL_OR_GREATER so sky writes only on far pixels
    // (terrain has cleared depth to 1.0 and writes ~near values).
    // We do NOT write depth.
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;  // sky has depth=1.0

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState dyns[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{};
    dy.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dy.dynamicStateCount = 2;
    dy.pDynamicStates = dyns;

    VkPipelineRenderingCreateInfo rendInfo{};
    rendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendInfo.colorAttachmentCount = 1;
    rendInfo.pColorAttachmentFormats = &swapchainFormat;
    rendInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext = &rendInfo;
    pi.stageCount = 2;
    pi.pStages = stages;
    pi.pVertexInputState = &viState;
    pi.pInputAssemblyState = &iaState;
    pi.pViewportState = &vpState;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &dy;
    pi.layout = skyPassPipelineLayout;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr,
                                   &skyPassPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create sky pass pipeline");

    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);
}

void VulkanEngine::runBrunetonStaticLUTs() {
    // Push fresh atmosphere parameters before the LUTs read them.
    updateAtmosphereUBO();

    VkCommandBuffer cmd = beginSingleTimeCommands();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        brunetonComputeLayout, 0, 1, &brunetonComputeSet, 0, nullptr);

    // â”€â”€ Transmittance: 256x64, local 8x4 â†’ 32x16 workgroups
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, transmittancePipeline);
    vkCmdDispatch(cmd, 32, 16, 1);

    // Barrier so multiscatter pass sees transmittance writes.
    VkMemoryBarrier2 mem{};
    mem.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mem.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mem.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mem.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mem.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mem;
    vkCmdPipelineBarrier2(cmd, &dep);

    // â”€â”€ Multiscatter: 32x32, local 1x1x64 â†’ 32x32 workgroups (each
    // workgroup writes one texel; 64 z-threads cooperate via shared mem)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, multiscatterPipeline);
    vkCmdDispatch(cmd, 32, 32, 1);

    endSingleTimeCommands(cmd);
}

void VulkanEngine::dispatchSkyviewLUT(VkCommandBuffer cmd) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, skyviewPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        brunetonComputeLayout, 0, 1, &brunetonComputeSet, 0, nullptr);
    vkCmdDispatch(cmd, 12, 8, 1);
}

bool VulkanEngine::needsSkyviewUpdate() const {
    if (!skyviewValid) return true;
    if (glm::dot(atmosphereParams.sun_direction, lastSkyviewSunDir)
            < 1.0f - settings.skyviewUpdateSunThreshold)
        return true;
    const float camAlt = camera.eyePos().y;
    if (std::abs(camAlt - lastSkyviewCamAlt) > settings.skyviewUpdateAltThreshold)
        return true;
    return false;
}

void VulkanEngine::createCloudShadowResources() {
    const uint32_t mapSize = CLOUD_SHADOW_MAP_SIZE;
    createImage(mapSize, mapSize, VK_FORMAT_R16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, cloudShadowMap);

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(device, &si, nullptr, &cloudShadowSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud shadow sampler");

    VkDescriptorSetLayoutBinding bindings[5]{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 5;
    li.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &li, nullptr, &cloudShadowSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud shadow DS layout");

    VkDescriptorPoolSize ps[3]{};
    ps[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2};
    ps[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2};
    ps[2] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1;
    pci.poolSizeCount = 3;
    pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(device, &pci, nullptr, &cloudShadowDescPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud shadow descriptor pool");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = cloudShadowDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &cloudShadowSetLayout;
    if (vkAllocateDescriptorSets(device, &ai, &cloudShadowDescSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate cloud shadow DS");

    VkDescriptorBufferInfo atmoInfo{atmosphereUniformBuffer.buffer, 0, sizeof(AtmosphereParams)};
    VkDescriptorBufferInfo cloudInfo{cloudParamsBuffer.buffer, 0, sizeof(CloudParams)};
    VkDescriptorImageInfo noiseInfo{cloudNoiseSampler, cloudNoise3D.view,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo hmInfo{masterHeightmapSampler, masterHeightmapImage.view,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo outInfo{};
    outInfo.imageView = cloudShadowMap.view;
    outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[5]{};
    for (int wi = 0; wi < 5; ++wi) {
        writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[wi].dstSet = cloudShadowDescSet;
        writes[wi].descriptorCount = 1;
    }
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &atmoInfo;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo = &cloudInfo;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &noiseInfo;
    writes[3].dstBinding = 3;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &hmInfo;
    writes[4].dstBinding = 4;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[4].pImageInfo = &outInfo;
    vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);

    auto code = readFile("shaders/cloud_shadow.comp.spv");
    VkShaderModule mod = createShaderModule(code);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &cloudShadowSetLayout;
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(float) * 8;  // CloudShadowPush
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &cloudShadowLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud shadow pipeline layout");

    VkComputePipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pi.stage.module = mod;
    pi.stage.pName = "main";
    pi.layout = cloudShadowLayout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &cloudShadowPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud shadow pipeline");
    vkDestroyShaderModule(device, mod, nullptr);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    {
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.image = cloudShadowMap.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    dispatchCloudShadowMap(cmd);
    endSingleTimeCommands(cmd);

    std::cout << "  [cloud-shadow] " << mapSize << "x" << mapSize
              << " transmittance map ready" << std::endl;
}

void VulkanEngine::dispatchCloudShadowMap(VkCommandBuffer cmd) {
    if (!settings.enableCloudShadows || !settings.useCloudShadowMap
        || cloudShadowPipeline == VK_NULL_HANDLE)
        return;

    if (cloudShadowMapReady) {
        VkImageMemoryBarrier2 pre{};
        pre.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        pre.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        pre.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        pre.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        pre.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        pre.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        pre.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        pre.image = cloudShadowMap.image;
        pre.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &pre;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    struct CloudShadowPush {
        glm::vec3 cameraPos;
        float worldExtent;
        float heightScale;
        int heightmapRes;
        int shadowSamples;
        float shadowStrength;
        float minVisibility;
    } push{};
    push.cameraPos      = camera.eyePos();
    push.worldExtent    = settings.worldSize;
    push.heightScale    = settings.terrainHeightScale;
    push.heightmapRes   = (int)settings.masterRes;
    push.shadowSamples  = settings.cloudShadowSamples;
    push.shadowStrength = settings.cloudShadowStrength;
    push.minVisibility  = settings.cloudShadowMinVisibility;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cloudShadowPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        cloudShadowLayout, 0, 1, &cloudShadowDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, cloudShadowLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(push), &push);
    vkCmdDispatch(cmd, (CLOUD_SHADOW_MAP_SIZE + 7) / 8, (CLOUD_SHADOW_MAP_SIZE + 7) / 8, 1);

    VkImageMemoryBarrier2 post{};
    post.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    post.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    post.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    post.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    post.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    post.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    post.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    post.image = cloudShadowMap.image;
    post.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &post;
    vkCmdPipelineBarrier2(cmd, &dep);
    cloudShadowMapReady = true;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Volumetric clouds (Heckel + Hillaire + Schneider pipeline)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createCloudResources() {
    // â”€â”€ 3D Worley-Perlin noise volume (128Â³, RGBA16F) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    //  Compute writes via STORAGE; graphics samples via SAMPLED.
    {
        VkImageCreateInfo ii{};
        ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType = VK_IMAGE_TYPE_3D;
        ii.format    = VK_FORMAT_R16G16B16A16_SFLOAT;
        ii.extent    = {128, 128, 128};
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling  = VK_IMAGE_TILING_OPTIMAL;
        ii.usage   = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &ii, nullptr, &cloudNoise3D.image) != VK_SUCCESS)
            throw std::runtime_error("failed to create cloud noise 3D image");

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device, cloudNoise3D.image, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &ai, nullptr, &cloudNoise3D.memory) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate cloud noise 3D memory");
        vkBindImageMemory(device, cloudNoise3D.image, cloudNoise3D.memory, 0);

        VkImageViewCreateInfo vi{};
        vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image    = cloudNoise3D.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vi.format   = VK_FORMAT_R16G16B16A16_SFLOAT;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &vi, nullptr, &cloudNoise3D.view) != VK_SUCCESS)
            throw std::runtime_error("failed to create cloud noise 3D view");
    }

    // â”€â”€ Blue noise tile (64Ã—64 R8) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    //  CPU-side Mitchell best-candidate sequence (see uploadBlueNoiseTexture).
    {
        VkImageCreateInfo ii{};
        ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format    = VK_FORMAT_R8_UNORM;
        ii.extent    = {64, 64, 1};
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling  = VK_IMAGE_TILING_OPTIMAL;
        ii.usage   = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &ii, nullptr, &blueNoise2D.image) != VK_SUCCESS)
            throw std::runtime_error("failed to create blue noise image");

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device, blueNoise2D.image, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &ai, nullptr, &blueNoise2D.memory) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate blue noise memory");
        vkBindImageMemory(device, blueNoise2D.image, blueNoise2D.memory, 0);

        VkImageViewCreateInfo vi{};
        vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image    = blueNoise2D.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format   = VK_FORMAT_R8_UNORM;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &vi, nullptr, &blueNoise2D.view) != VK_SUCCESS)
            throw std::runtime_error("failed to create blue noise view");
    }

    // â”€â”€ Samplers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    //  cloudNoiseSampler: REPEAT so the 3D Worley tile across world space.
    //  blueNoiseSampler:  REPEAT so the 64Ã—64 tile covers the framebuffer.
    //  depthSampler:      NEAREST so the cloud raymarch sees the exact
    //                     depth-buffer reading without bilinear blur.
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.maxAnisotropy = 1.0f;
        si.unnormalizedCoordinates = VK_FALSE;
        si.compareEnable = VK_FALSE;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        if (vkCreateSampler(device, &si, nullptr, &cloudNoiseSampler) != VK_SUCCESS)
            throw std::runtime_error("failed to create cloud noise sampler");

        si.magFilter = VK_FILTER_NEAREST;
        si.minFilter = VK_FILTER_NEAREST;
        if (vkCreateSampler(device, &si, nullptr, &blueNoiseSampler) != VK_SUCCESS)
            throw std::runtime_error("failed to create blue noise sampler");

        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device, &si, nullptr, &depthSampler) != VK_SUCCESS)
            throw std::runtime_error("failed to create depth sampler");
    }

    // â”€â”€ Cloud params UBO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    createBuffer(sizeof(CloudParams),
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 cloudParamsBuffer);
    vkMapMemory(device, cloudParamsBuffer.memory, 0,
                sizeof(CloudParams), 0, &cloudParamsMapped);

    // Temporal UBO — holds the previous frame's ViewProj matrix for
    // direction reprojection in cloud_temporal.frag.
    createBuffer(sizeof(glm::mat4),
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 cloudTemporalUBO);
    vkMapMemory(device, cloudTemporalUBO.memory, 0,
                sizeof(glm::mat4), 0, &cloudTemporalMapped);
    glm::mat4 identity(1.0f);
    std::memcpy(cloudTemporalMapped, &identity, sizeof(glm::mat4));

    // â”€â”€ Default parameters (mirror atmosphere_bac_src/main_application.cpp) â”€
    cloudParams = {};
    cloudParams.shapeNoiseWeights      = glm::vec4(0.625f, 0.250f, 0.125f, 0.0f);
    cloudParams.detailNoiseWeights     = glm::vec4(0.625f, 0.250f, 0.125f, 0.0f);
    cloudParams.phaseParams            = glm::vec4(0.83f, 0.30f, 0.80f, 0.15f);
    cloudParams.detailNoiseMultiplier  = 0.40f;
    cloudParams.minBounds              = 1.5f;     // km above surface
    cloudParams.maxBounds              = 4.0f;     // 2.5 km thick layer
    cloudParams.cloudsScale            = 0.30f;
    cloudParams.detailScale            = 6.0f;
    cloudParams.densityOffset          = 0.55f;
    cloudDensityOffsetBase             = 0.55f;   // remembered for timelapse oscillation
    cloudParams.densityMultiplier      = 2.2f;
    cloudParams.sampleCount            = 64;
    cloudParams.sampleCountToSun       = 5;
    cloudParams.lightAbsTowardsSun     = 1.2f;
    cloudParams.lightAbsThroughCloud   = 0.65f;
    cloudParams.darknessThreshold      = 0.20f;
    cloudParams.windOffset             = 0.0f;
    // Terrain cloud-shadow controls (driven by TerrainSettings / JSON).
    cloudParams.enableCloudShadows       = settings.enableCloudShadows ? 1 : 0;
    cloudParams.cloudShadowSamples       = settings.cloudShadowSamples;
    cloudParams.cloudShadowStrength      = settings.cloudShadowStrength;
    cloudParams.cloudShadowMinVisibility = settings.cloudShadowMinVisibility;
    cloudParams.cloudShadowDebug         = 0;
    cloudParams.useCloudShadowMap        = settings.useCloudShadowMap ? 1 : 0;
    std::memcpy(cloudParamsMapped, &cloudParams, sizeof(CloudParams));
}

void VulkanEngine::uploadBlueNoiseTexture() {
    // â”€â”€ CPU-side Mitchell best-candidate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    //  Produces a perceptually 'blue' 64Ã—64 R8 dithering tile.
    //  We place 64*64 = 4096 candidate points by, for each new point,
    //  picking the candidate (of 8) that maximises the minimum
    //  distance to all previously-placed points. This is the same
    //  technique used in Heitz-Belcour and other Hillaire 2020-era
    //  dithering literature; the result is indistinguishable from a
    //  proper void-and-cluster pattern at this resolution.
    constexpr int   W = 64;
    constexpr int   H = 64;
    constexpr int   N = W * H;
    std::vector<uint8_t> tile(N);

    std::mt19937 rng(0xB1C10D5Au);  // arbitrary fixed seed for reproducible blue noise
    std::uniform_int_distribution<int> randCoord(0, W - 1);
    auto torusDist2 = [](int dx, int dy) {
        if (dx >  W/2) dx -= W;
        if (dx < -W/2) dx += W;
        if (dy >  H/2) dy -= H;
        if (dy < -H/2) dy += H;
        return dx*dx + dy*dy;
    };

    std::vector<std::pair<int,int>> placed;
    placed.reserve(N);
    std::vector<int> nearestDist(W * H, std::numeric_limits<int>::max());

    for (int i = 0; i < N; ++i) {
        constexpr int CANDS = 8;
        int   bestX = 0, bestY = 0;
        int   bestScore = -1;
        for (int c = 0; c < CANDS; ++c) {
            int x = randCoord(rng);
            int y = randCoord(rng);
            // Score is distance to nearest already-placed sample (toroidal).
            int idx = y * W + x;
            int s   = nearestDist[idx];
            if (s > bestScore) { bestScore = s; bestX = x; bestY = y; }
        }

        // Encode rank as the 8-bit intensity: earlier picks (low rank)
        // map to low values, later picks to high values. The resulting
        // ramp has a blue-noise spatial spectrum.
        tile[bestY * W + bestX] = static_cast<uint8_t>((i * 255) / (N - 1));
        placed.emplace_back(bestX, bestY);

        // Update nearestDist field with new point.
        for (int yy = 0; yy < H; ++yy)
        for (int xx = 0; xx < W; ++xx) {
            int d = torusDist2(xx - bestX, yy - bestY);
            int& nd = nearestDist[yy * W + xx];
            if (d < nd) nd = d;
        }
    }

    // â”€â”€ Stage to GPU â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Buffer staging{};
    createBuffer(N, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
    void* mapped = nullptr;
    vkMapMemory(device, staging.memory, 0, N, 0, &mapped);
    std::memcpy(mapped, tile.data(), N);
    vkUnmapMemory(device, staging.memory);

    VkCommandBuffer cmd = beginSingleTimeCommands();

    // Transition UNDEFINED â†’ TRANSFER_DST.
    VkImageMemoryBarrier2 pre{};
    pre.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    pre.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    pre.srcAccessMask = VK_ACCESS_2_NONE;
    pre.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    pre.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    pre.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    pre.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    pre.image = blueNoise2D.image;
    pre.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    pre.subresourceRange.levelCount = 1;
    pre.subresourceRange.layerCount = 1;
    VkDependencyInfo preDep{};
    preDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    preDep.imageMemoryBarrierCount = 1;
    preDep.pImageMemoryBarriers = &pre;
    vkCmdPipelineBarrier2(cmd, &preDep);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {(uint32_t)W, (uint32_t)H, 1};
    vkCmdCopyBufferToImage(cmd, staging.buffer, blueNoise2D.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition TRANSFER_DST â†’ SHADER_READ_ONLY_OPTIMAL.
    VkImageMemoryBarrier2 post{};
    post.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    post.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    post.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    post.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    post.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    post.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    post.image = blueNoise2D.image;
    post.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    post.subresourceRange.levelCount = 1;
    post.subresourceRange.layerCount = 1;
    VkDependencyInfo postDep{};
    postDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    postDep.imageMemoryBarrierCount = 1;
    postDep.pImageMemoryBarriers = &post;
    vkCmdPipelineBarrier2(cmd, &postDep);

    endSingleTimeCommands(cmd);

    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);
}

void VulkanEngine::createCloudNoiseGenPipeline() {
    // â”€â”€ Descriptor pool & layouts (pool also serves the draw pass) â”€
    VkDescriptorPoolSize ps[3]{};
    ps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps[0].descriptorCount = 2;        // 1 for gen, slack for future
    ps[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ps[1].descriptorCount = 8;        // atmos+cloud(draw) + temporal(×2) + slack
    ps[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[2].descriptorCount = 20;       // draw(4)+temporal(2×2)+composite(2)+slack
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 10;
    pci.poolSizeCount = 3;
    pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(device, &pci, nullptr, &cloudDescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud descriptor pool");

    // â”€â”€ Noise-gen set layout: binding 0 = storage 3D â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 1;
    li.pBindings = &b;
    if (vkCreateDescriptorSetLayout(device, &li, nullptr, &cloudNoiseGenSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud noise gen DS layout");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = cloudDescriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &cloudNoiseGenSetLayout;
    if (vkAllocateDescriptorSets(device, &ai, &cloudNoiseGenSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate cloud noise gen DS");

    VkDescriptorImageInfo storageInfo{};
    storageInfo.imageView   = cloudNoise3D.view;
    storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = cloudNoiseGenSet;
    w.dstBinding = 0;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w.descriptorCount = 1;
    w.pImageInfo = &storageInfo;
    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);

    struct CloudNoisePush {
        glm::ivec4 texDims;
        glm::ivec4 numDivisions;
        glm::vec4  persistence;
    };
    VkPushConstantRange pr{};
    pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pr.size = sizeof(CloudNoisePush);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &cloudNoiseGenSetLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &cloudNoiseGenLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud noise gen layout");

    auto code = readFile("shaders/cloud_noise_gen.comp.spv");
    VkShaderModule mod = createShaderModule(code);
    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = cloudNoiseGenLayout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr,
                                  &cloudNoiseGenPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud noise gen pipeline");
    vkDestroyShaderModule(device, mod, nullptr);
}

void VulkanEngine::runCloudNoiseGeneration() {
    // Transition cloudNoise3D UNDEFINED â†’ GENERAL for storage writes.
    {
        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.srcAccessMask = VK_ACCESS_2_NONE;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.image = cloudNoise3D.image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
        endSingleTimeCommands(cmd);
    }

    // Single dispatch covering the full 128Â³ volume (4Ã—4Ã—4 local).
    VkCommandBuffer cmd = beginSingleTimeCommands();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cloudNoiseGenPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        cloudNoiseGenLayout, 0, 1, &cloudNoiseGenSet, 0, nullptr);

    struct CloudNoisePush {
        glm::ivec4 texDims;
        glm::ivec4 numDivisions;
        glm::vec4  persistence;
    } push;
    push.texDims      = glm::ivec4(128, 128, 128, 0);
    // R: 4 cells base; G: 8; B: 16; A: 32 â€” matches reference channel layout.
    push.numDivisions = glm::ivec4(4, 8, 16, 32);
    push.persistence  = glm::vec4(0.7f, 0.6f, 0.5f, 0.5f);
    vkCmdPushConstants(cmd, cloudNoiseGenLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(cmd, 128 / 4, 128 / 4, 128 / 4);

    // Transition GENERAL â†’ SHADER_READ_ONLY for sampling.
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.image = cloudNoise3D.image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep);
    endSingleTimeCommands(cmd);
}

void VulkanEngine::createCloudDrawPipeline() {
    // â”€â”€ Descriptor set: atmosUBO, cloudUBO, depth, worley, transm, blue â”€
    VkDescriptorSetLayoutBinding bs[6]{};
    bs[0].binding = 0; bs[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bs[1].binding = 1; bs[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bs[2].binding = 2; bs[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bs[3].binding = 3; bs[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bs[4].binding = 4; bs[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bs[5].binding = 5; bs[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    for (int i = 0; i < 6; ++i) {
        bs[i].descriptorCount = 1;
        bs[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 6;
    li.pBindings = bs;
    if (vkCreateDescriptorSetLayout(device, &li, nullptr, &cloudDrawSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud draw DS layout");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = cloudDescriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &cloudDrawSetLayout;
    if (vkAllocateDescriptorSets(device, &ai, &cloudDrawSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate cloud draw DS");

    VkDescriptorBufferInfo atmosBuf{};
    atmosBuf.buffer = atmosphereUniformBuffer.buffer;
    atmosBuf.range  = sizeof(AtmosphereParams);

    VkDescriptorBufferInfo cloudBuf{};
    cloudBuf.buffer = cloudParamsBuffer.buffer;
    cloudBuf.range  = sizeof(CloudParams);

    VkDescriptorImageInfo depthInfo{};
    depthInfo.sampler     = depthSampler;
    depthInfo.imageView   = depth.view;
    depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo worleyInfo{};
    worleyInfo.sampler     = cloudNoiseSampler;
    worleyInfo.imageView   = cloudNoise3D.view;
    worleyInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo transmInfo{};
    transmInfo.sampler     = lutSampler;
    transmInfo.imageView   = transmittanceLUT.view;
    // transmittanceLUT lives in GENERAL after init (storage layout). Vulkan
    // permits sampling from GENERAL when the image was created with SAMPLED
    // usage, which it was.
    transmInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo blueInfo{};
    blueInfo.sampler     = blueNoiseSampler;
    blueInfo.imageView   = blueNoise2D.view;
    blueInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet ws[6]{};
    for (int i = 0; i < 6; ++i) {
        ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet = cloudDrawSet;
        ws[i].dstBinding = i;
        ws[i].descriptorCount = 1;
    }
    ws[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ws[0].pBufferInfo = &atmosBuf;
    ws[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ws[1].pBufferInfo = &cloudBuf;
    ws[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ws[2].pImageInfo = &depthInfo;
    ws[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ws[3].pImageInfo = &worleyInfo;
    ws[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ws[4].pImageInfo = &transmInfo;
    ws[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ws[5].pImageInfo = &blueInfo;
    vkUpdateDescriptorSets(device, 6, ws, 0, nullptr);

    struct CloudsPush {
        glm::mat4 invViewProj;
        glm::vec4 cameraPos;
        glm::vec4 viewport;  // .xy fbSize, .z frameIndex, .w nearPlane
    };
    VkPushConstantRange pr{};
    pr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pr.size = sizeof(CloudsPush);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &cloudDrawSetLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &cloudDrawPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud draw layout");

    auto vertCode = readFile("shaders/bruneton_clouds.vert.spv");
    auto fragCode = readFile("shaders/bruneton_clouds.frag.spv");
    VkShaderModule vs = createShaderModule(vertCode);
    VkShaderModule fs = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs; stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo viState{};
    viState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo iaState{};
    iaState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vpState{};
    vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpState.viewportCount = 1;
    vpState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    // Pre-multiplied alpha blending was here; now this pipeline renders to
    // cloudHalfResBuffer (RGBA16F) and the composite pass handles blending.
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable    = VK_FALSE;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState dyns[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{};
    dy.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dy.dynamicStateCount = 2;
    dy.pDynamicStates = dyns;

    VkPipelineRenderingCreateInfo rendInfo{};
    rendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendInfo.colorAttachmentCount = 1;
    // Renders to half-res RGBA16F intermediate, not the swapchain.
    VkFormat halfResCloudFmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    rendInfo.pColorAttachmentFormats = &halfResCloudFmt;
    // No depth attachment in the cloud pass -- we sample depth instead.

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext = &rendInfo;
    pi.stageCount = 2;
    pi.pStages = stages;
    pi.pVertexInputState = &viState;
    pi.pInputAssemblyState = &iaState;
    pi.pViewportState = &vpState;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &dy;
    pi.layout = cloudDrawPipelineLayout;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr,
                                   &cloudDrawPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud draw pipeline");

    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);
}

// Animate the sun through a full day/night arc when enabled. Advances the
// normalized time-of-day, derives elevation (sine arc peaking at noon) and a
// continuously sweeping azimuth, and forces the skyview LUT to refresh so the
// sky colour tracks the moving sun smoothly.
void VulkanEngine::updateTimeOfDay(float dt) {
    if (!settings.dayNightEnabled) return;
    const float len = std::max(5.0f, settings.dayLengthSeconds);
    settings.timeOfDay01 += dt / len;
    settings.timeOfDay01 -= std::floor(settings.timeOfDay01);   // wrap to [0,1)

    const float p = settings.timeOfDay01;
    const float twoPi = 6.28318530718f;
    float elev = settings.sunPeakElevationDeg * std::sin((p - 0.25f) * twoPi);
    settings.sunElevation = glm::clamp(elev, -9.0f, 89.0f);
    settings.sunAzimuth   = std::fmod(p * 360.0f + 90.0f, 360.0f);

    skyviewValid = false;   // sun moved → rebuild the skyview LUT this frame
}

void VulkanEngine::updateCloudParamsUBO() {
    cloudFrameIndex++;

    // Cloud drift. Normal mode keeps the original gentle fixed step; timelapse
    // mode advances much faster (dt-scaled) and oscillates coverage so clouds
    // visibly build up and dissipate like a real sped-up sky.
    if (settings.cloudTimelapseEnabled) {
        const float spd = std::max(1.0f, settings.cloudTimelapseSpeed);
        cloudWindOffset   += lastFrameDt * 0.036f * spd;
        cloudTimelapsePhase += lastFrameDt * 0.05f * spd;
        const float osc = 0.5f + 0.5f * std::sin(cloudTimelapsePhase);
        cloudParams.densityOffset = glm::mix(0.40f, 0.74f, osc);  // form ↔ dissipate
    } else {
        cloudWindOffset += 0.0006f;
        cloudParams.densityOffset = cloudDensityOffsetBase;
    }
    cloudParams.windOffset = cloudWindOffset;

    cloudParams.enableCloudShadows       = settings.enableCloudShadows ? 1 : 0;
    cloudParams.useCloudShadowMap        = settings.useCloudShadowMap ? 1 : 0;
    cloudParams.cloudShadowSamples       = settings.cloudShadowSamples;
    cloudParams.cloudShadowStrength      = settings.cloudShadowStrength;
    cloudParams.cloudShadowMinVisibility = settings.cloudShadowMinVisibility;

    // Demo: lower the cloud-base slightly so the achievable cruise altitude on a
    // 4 km world actually penetrates the volumetric layer during the scenic leg.
    if (settings.demoModeEnabled) {
        cloudParams.minBounds = 0.55f;
        cloudParams.maxBounds = 3.2f;
    } else {
        cloudParams.minBounds = 1.5f;
        cloudParams.maxBounds = 4.0f;
    }

    if (settings.adaptiveCloudQuality && fpsSmoothed > 0.0f) {
        int samples = settings.cloudRaymarchSamples;
        if (fpsSmoothed < 45.0f) samples = std::min(samples, 48);
        if (fpsSmoothed < 35.0f) samples = std::min(samples, 40);
        if (fpsSmoothed < 25.0f) samples = std::min(samples, 32);
        cloudParams.sampleCount = samples;
    } else {
        cloudParams.sampleCount = settings.cloudRaymarchSamples;
    }

    std::memcpy(cloudParamsMapped, &cloudParams, sizeof(CloudParams));
}

void VulkanEngine::toggleCloudShadows() {
    cloudParams.enableCloudShadows = cloudParams.enableCloudShadows ? 0 : 1;
    std::cout << "[cloud-shadows] " << (cloudParams.enableCloudShadows ? "ON" : "OFF") << std::endl;
}

void VulkanEngine::cycleCloudShadowDebug() {
    cloudParams.cloudShadowDebug = cloudParams.cloudShadowDebug ? 0 : 1;
    std::cout << "[cloud-shadows] debug overlay "
              << (cloudParams.cloudShadowDebug ? "ON (terrain shows cloud transmittance)" : "OFF")
              << std::endl;
}

void VulkanEngine::createCloudAccumBuffers() {
    const uint32_t hw = (swapchainExtent.width  + 1) / 2;
    const uint32_t hh = (swapchainExtent.height + 1) / 2;

    createImage(hw, hh, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, cloudHalfResBuffer);

    for (int i = 0; i < 2; ++i)
        createImage(swapchainExtent.width, swapchainExtent.height,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, cloudAccumBuffer[i]);

    // Linear/clamp sampler for bilinear upscale and temporal history reads.
    VkSamplerCreateInfo si{};
    si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter    = VK_FILTER_LINEAR;
    si.minFilter    = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxAnisotropy = 1.0f;
    si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if (vkCreateSampler(device, &si, nullptr, &cloudLinearSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud linear sampler");

    // Transition all three images to SHADER_READ_ONLY_OPTIMAL so the first
    // temporal pass can sample history without a layout validation error.
    VkCommandBuffer cmd = beginSingleTimeCommands();
    Image* imgs[3] = { &cloudHalfResBuffer, &cloudAccumBuffer[0], &cloudAccumBuffer[1] };
    for (auto* img : imgs) {
        VkImageMemoryBarrier2 b{};
        b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.srcAccessMask = VK_ACCESS_2_NONE;
        b.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.image         = img->image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    endSingleTimeCommands(cmd);
}

void VulkanEngine::createCloudTemporalPipeline() {
    // Descriptor layout: binding 0 = new half-res samples,
    //                    binding 1 = accumulated history,
    //                    binding 2 = temporal UBO (prevViewProj).
    VkDescriptorSetLayoutBinding bs[3]{};
    bs[0].binding        = 0;
    bs[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bs[1].binding        = 1;
    bs[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bs[2].binding        = 2;
    bs[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    for (int i = 0; i < 3; ++i) {
        bs[i].descriptorCount = 1;
        bs[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 3;
    li.pBindings    = bs;
    if (vkCreateDescriptorSetLayout(device, &li, nullptr, &cloudTemporalSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud temporal DS layout");

    // Two descriptor sets (indexed by cloudAccumPing so history = 1-ping).
    //  cloudTemporalSet[0]: binding1 = cloudAccumBuffer[1] (history when ping=0)
    //  cloudTemporalSet[1]: binding1 = cloudAccumBuffer[0] (history when ping=1)
    VkDescriptorSetLayout layouts2[2] = { cloudTemporalSetLayout, cloudTemporalSetLayout };
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = cloudDescriptorPool;
    ai.descriptorSetCount = 2;
    ai.pSetLayouts        = layouts2;
    if (vkAllocateDescriptorSets(device, &ai, cloudTemporalSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate cloud temporal DS");

    VkDescriptorImageInfo halfResInfo{};
    halfResInfo.sampler     = cloudLinearSampler;
    halfResInfo.imageView   = cloudHalfResBuffer.view;
    halfResInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo temporalBuf{};
    temporalBuf.buffer = cloudTemporalUBO.buffer;
    temporalBuf.offset = 0;
    temporalBuf.range  = sizeof(glm::mat4);

    for (int s = 0; s < 2; ++s) {
        VkDescriptorImageInfo histInfo{};
        histInfo.sampler     = cloudLinearSampler;
        histInfo.imageView   = cloudAccumBuffer[1 - s].view;  // history = other buffer
        histInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet ws[3]{};
        ws[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[0].dstSet          = cloudTemporalSet[s];
        ws[0].dstBinding      = 0;
        ws[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ws[0].descriptorCount = 1;
        ws[0].pImageInfo      = &halfResInfo;

        ws[1]            = ws[0];
        ws[1].dstBinding = 1;
        ws[1].pImageInfo = &histInfo;

        ws[2].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[2].dstSet           = cloudTemporalSet[s];
        ws[2].dstBinding       = 2;
        ws[2].descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ws[2].descriptorCount  = 1;
        ws[2].pBufferInfo      = &temporalBuf;

        vkUpdateDescriptorSets(device, 3, ws, 0, nullptr);
    }

    // Push constants: same layout as the raymarch pass (mat4+vec4+vec4 = 96 B).
    VkPushConstantRange pr{};
    pr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pr.size       = sizeof(glm::mat4) + 2 * sizeof(glm::vec4);  // 96

    VkPipelineLayoutCreateInfo pli{};
    pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount         = 1;
    pli.pSetLayouts            = &cloudTemporalSetLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges    = &pr;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &cloudTemporalPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud temporal pipeline layout");

    auto vertCode = readFile("shaders/bruneton_clouds.vert.spv");
    auto fragCode = readFile("shaders/cloud_temporal.frag.spv");
    VkShaderModule vs = createShaderModule(vertCode);
    VkShaderModule fs = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs; stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo viState{};
    viState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo iaState{};
    iaState.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vpState{};
    vpState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpState.viewportCount = 1; vpState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType     = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode  = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depthState{};
    depthState.sType           = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthState.depthTestEnable = VK_FALSE;
    depthState.depthWriteEnable = VK_FALSE;

    // No blending: writes resolved RGBA16F directly to accumulation buffer.
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable    = VK_FALSE;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    VkDynamicState dyns[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy{};
    dy.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dy.dynamicStateCount = 2; dy.pDynamicStates = dyns;

    VkFormat accumFmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkPipelineRenderingCreateInfo rendInfo{};
    rendInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendInfo.colorAttachmentCount    = 1;
    rendInfo.pColorAttachmentFormats = &accumFmt;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext               = &rendInfo;
    pi.stageCount          = 2; pi.pStages = stages;
    pi.pVertexInputState   = &viState;
    pi.pInputAssemblyState = &iaState;
    pi.pViewportState      = &vpState;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState   = &ms;
    pi.pDepthStencilState  = &depthState;
    pi.pColorBlendState    = &cb;
    pi.pDynamicState       = &dy;
    pi.layout              = cloudTemporalPipelineLayout;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr,
                                   &cloudTemporalPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud temporal pipeline");

    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);
}

void VulkanEngine::createCloudCompositePipeline() {
    // Single binding: the temporally accumulated full-res cloud buffer.
    VkDescriptorSetLayoutBinding b{};
    b.binding        = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 1; li.pBindings = &b;
    if (vkCreateDescriptorSetLayout(device, &li, nullptr, &cloudCompositeSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud composite DS layout");

    // cloudCompositeSet[s] reads from cloudAccumBuffer[s].
    VkDescriptorSetLayout layouts2[2] = { cloudCompositeSetLayout, cloudCompositeSetLayout };
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = cloudDescriptorPool;
    ai.descriptorSetCount = 2;
    ai.pSetLayouts        = layouts2;
    if (vkAllocateDescriptorSets(device, &ai, cloudCompositeSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate cloud composite DS");

    for (int s = 0; s < 2; ++s) {
        VkDescriptorImageInfo info{};
        info.sampler     = cloudLinearSampler;
        info.imageView   = cloudAccumBuffer[s].view;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = cloudCompositeSet[s];
        w.dstBinding      = 0;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo      = &info;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }

    // No push constants needed for the trivial composite pass.
    VkPipelineLayoutCreateInfo pli{};
    pli.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &cloudCompositeSetLayout;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &cloudCompositePipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud composite pipeline layout");

    auto vertCode = readFile("shaders/bruneton_clouds.vert.spv");
    auto fragCode = readFile("shaders/cloud_composite.frag.spv");
    VkShaderModule vs = createShaderModule(vertCode);
    VkShaderModule fs = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs; stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo viState{};
    viState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo iaState{};
    iaState.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vpState{};
    vpState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpState.viewportCount = 1; vpState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType     = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode  = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depthState{};
    depthState.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthState.depthTestEnable  = VK_FALSE;
    depthState.depthWriteEnable = VK_FALSE;

    // Premultiplied alpha: src.rgb + dst.rgb * (1 - src.a).
    // The accumulated buffer stores (premultiplied cloud colour, 1 - transmittance)
    // so this blend correctly composites clouds over terrain + sky.
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable         = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp        = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.alphaBlendOp        = VK_BLEND_OP_ADD;
    cba.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    VkDynamicState dyns[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy{};
    dy.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dy.dynamicStateCount = 2; dy.pDynamicStates = dyns;

    VkPipelineRenderingCreateInfo rendInfo{};
    rendInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendInfo.colorAttachmentCount    = 1;
    rendInfo.pColorAttachmentFormats = &swapchainFormat;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext               = &rendInfo;
    pi.stageCount          = 2; pi.pStages = stages;
    pi.pVertexInputState   = &viState;
    pi.pInputAssemblyState = &iaState;
    pi.pViewportState      = &vpState;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState   = &ms;
    pi.pDepthStencilState  = &depthState;
    pi.pColorBlendState    = &cb;
    pi.pDynamicState       = &dy;
    pi.layout              = cloudCompositePipelineLayout;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr,
                                   &cloudCompositePipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create cloud composite pipeline");

    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);
}

void VulkanEngine::recreateCloudAccumBuffers() {
    vkDeviceWaitIdle(device);

    auto destroyImg = [&](Image& im) {
        if (im.view)   { vkDestroyImageView(device, im.view, nullptr);   im.view   = VK_NULL_HANDLE; }
        if (im.image)  { vkDestroyImage(device, im.image, nullptr);      im.image  = VK_NULL_HANDLE; }
        if (im.memory) { vkFreeMemory(device, im.memory, nullptr);       im.memory = VK_NULL_HANDLE; }
    };
    destroyImg(cloudHalfResBuffer);
    destroyImg(cloudAccumBuffer[0]);
    destroyImg(cloudAccumBuffer[1]);

    const uint32_t hw = (swapchainExtent.width  + 1) / 2;
    const uint32_t hh = (swapchainExtent.height + 1) / 2;

    createImage(hw, hh, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, cloudHalfResBuffer);
    for (int i = 0; i < 2; ++i)
        createImage(swapchainExtent.width, swapchainExtent.height,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, cloudAccumBuffer[i]);

    // Transition to SHADER_READ_ONLY_OPTIMAL.
    VkCommandBuffer cmd = beginSingleTimeCommands();
    Image* imgs[3] = { &cloudHalfResBuffer, &cloudAccumBuffer[0], &cloudAccumBuffer[1] };
    for (auto* img : imgs) {
        VkImageMemoryBarrier2 b2{};
        b2.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b2.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b2.srcAccessMask = VK_ACCESS_2_NONE;
        b2.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b2.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b2.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        b2.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b2.image         = img->image;
        b2.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b2.subresourceRange.levelCount = 1;
        b2.subresourceRange.layerCount = 1;
        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b2;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    endSingleTimeCommands(cmd);

    // Update temporal and composite descriptor sets with new image views.
    VkDescriptorImageInfo halfResInfo{};
    halfResInfo.sampler     = cloudLinearSampler;
    halfResInfo.imageView   = cloudHalfResBuffer.view;
    halfResInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    for (int s = 0; s < 2; ++s) {
        VkDescriptorImageInfo histInfo{};
        histInfo.sampler     = cloudLinearSampler;
        histInfo.imageView   = cloudAccumBuffer[1 - s].view;
        histInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet tw[2]{};
        tw[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        tw[0].dstSet          = cloudTemporalSet[s];
        tw[0].dstBinding      = 0;
        tw[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        tw[0].descriptorCount = 1;
        tw[0].pImageInfo      = &halfResInfo;
        tw[1]            = tw[0];
        tw[1].dstBinding = 1;
        tw[1].pImageInfo = &histInfo;
        vkUpdateDescriptorSets(device, 2, tw, 0, nullptr);

        VkDescriptorImageInfo accumInfo{};
        accumInfo.sampler     = cloudLinearSampler;
        accumInfo.imageView   = cloudAccumBuffer[s].view;
        accumInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet cw{};
        cw.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        cw.dstSet          = cloudCompositeSet[s];
        cw.dstBinding      = 0;
        cw.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        cw.descriptorCount = 1;
        cw.pImageInfo      = &accumInfo;
        vkUpdateDescriptorSets(device, 1, &cw, 0, nullptr);
    }

    // Reset temporal state so the new buffers warm up cleanly.
    cloudAccumPing    = 0;
    prevCloudViewProj = glm::mat4(1.0f);
}

void VulkanEngine::drawCloudsPass(VkCommandBuffer cmd, uint32_t imageIndex) {
    const uint32_t ping = cloudAccumPing;
    const uint32_t pong = 1u - ping;

    // Update temporal UBO with the PREVIOUS frame's ViewProj so the temporal
    // resolve shader can reproject history to the current screen position.
    std::memcpy(cloudTemporalMapped, &prevCloudViewProj, sizeof(glm::mat4));

    // Helper lambda: single-image colour-layout transition via pipeline barrier.
    auto transitionColor = [&](VkImage img,
                                VkImageLayout from, VkImageLayout to,
                                VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                                VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
        VkImageMemoryBarrier2 b{};
        b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask  = srcStage;  b.srcAccessMask = srcAccess;
        b.dstStageMask  = dstStage;  b.dstAccessMask = dstAccess;
        b.oldLayout     = from;      b.newLayout     = to;
        b.image         = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    };

    // Current ViewProj + inverse (shared across all three sub-passes).
    glm::mat4 proj = glm::perspective(glm::radians(60.0f),
        swapchainExtent.width / (float)swapchainExtent.height,
        0.1f, settings.effectiveFarPlane());
    proj[1][1] *= -1;
    const glm::mat4 viewProj    = proj * camera.view();
    const glm::mat4 invViewProj = glm::inverse(viewProj);
    const VkExtent2D halfExt    = { (swapchainExtent.width  + 1) / 2,
                                    (swapchainExtent.height + 1) / 2 };

    // ══ Pass 1: Cloud Raymarch (half-resolution) ══════════════════════
    // Renders to cloudHalfResBuffer at half the swapchain resolution.
    // The cloud shader is unchanged: inUV covers [0,1]×[0,1] regardless
    // of the render target size, so ray reconstruction and depth sampling
    // are correct. The blue-noise dither still tiles the 64×64 texture.
    transitionColor(cloudHalfResBuffer.image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    {
        VkRenderingAttachmentInfo ca{};
        ca.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        ca.imageView   = cloudHalfResBuffer.view;
        ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ca.loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ca.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo ri{};
        ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent    = halfExt;
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments    = &ca;
        vkCmdBeginRendering(cmd, &ri);

        VkViewport vp{};
        vp.width = (float)halfExt.width; vp.height = (float)halfExt.height;
        vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{}; sc.extent = halfExt;
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cloudDrawPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            cloudDrawPipelineLayout, 0, 1, &cloudDrawSet, 0, nullptr);

        struct CloudsPush {
            glm::mat4 invViewProj;
            glm::vec4 cameraPos;
            glm::vec4 viewport;
        } push;
        push.invViewProj = invViewProj;
        push.cameraPos   = glm::vec4(camera.eyePos(), 1.0f);
        push.viewport    = glm::vec4(
            (float)swapchainExtent.width, (float)swapchainExtent.height,
            (float)cloudFrameIndex, 0.1f);
        vkCmdPushConstants(cmd, cloudDrawPipelineLayout,
            VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    }

    // ══ Pass 2: Temporal Resolve (full-resolution) ════════════════════
    // Bilinearly upsamples the half-res new samples, reprojects and blends
    // with history (cloudAccumBuffer[pong]) into cloudAccumBuffer[ping].
    transitionColor(cloudHalfResBuffer.image,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    transitionColor(cloudAccumBuffer[ping].image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    {
        VkRenderingAttachmentInfo ca{};
        ca.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        ca.imageView   = cloudAccumBuffer[ping].view;
        ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ca.loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ca.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo ri{};
        ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent    = swapchainExtent;
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments    = &ca;
        vkCmdBeginRendering(cmd, &ri);

        VkViewport vp{};
        vp.width = (float)swapchainExtent.width; vp.height = (float)swapchainExtent.height;
        vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{}; sc.extent = swapchainExtent;
        vkCmdSetScissor(cmd, 0, 1, &sc);

        // cloudTemporalSet[ping] has binding 1 = cloudAccumBuffer[1-ping] (history = pong).
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cloudTemporalPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            cloudTemporalPipelineLayout, 0, 1, &cloudTemporalSet[ping], 0, nullptr);

        struct TemporalPush {
            glm::mat4 invViewProj;
            glm::vec4 cameraPos;
            glm::vec4 viewport;
        } tpush;
        tpush.invViewProj = invViewProj;
        tpush.cameraPos   = glm::vec4(camera.eyePos(), 1.0f);
        tpush.viewport    = glm::vec4(0.0f, 0.0f, (float)cloudFrameIndex, 0.0f);
        vkCmdPushConstants(cmd, cloudTemporalPipelineLayout,
            VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(tpush), &tpush);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    }

    // ══ Pass 3: Composite — blend accumulated clouds onto swapchain ═══
    transitionColor(cloudAccumBuffer[ping].image,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    {
        VkRenderingAttachmentInfo ca{};
        ca.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        ca.imageView   = swapchainViews[imageIndex];
        ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ca.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;   // preserve terrain / sky
        ca.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo ri{};
        ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent    = swapchainExtent;
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments    = &ca;
        vkCmdBeginRendering(cmd, &ri);

        VkViewport vp{};
        vp.width = (float)swapchainExtent.width; vp.height = (float)swapchainExtent.height;
        vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{}; sc.extent = swapchainExtent;
        vkCmdSetScissor(cmd, 0, 1, &sc);

        // cloudCompositeSet[ping] reads from cloudAccumBuffer[ping] (just written).
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cloudCompositePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            cloudCompositePipelineLayout, 0, 1, &cloudCompositeSet[ping], 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    }

    // Advance ping-pong: current output becomes next frame's history.
    cloudAccumPing = pong;

    // Store this frame's ViewProj for reprojection in the next frame.
    prevCloudViewProj = viewProj;
}

void VulkanEngine::updateUniformBuffer(uint32_t /*imageIndex*/) {
    FrameData frame{};
    glm::mat4 proj = glm::perspective(glm::radians(60.0f),
        swapchainExtent.width / (float)swapchainExtent.height, 0.1f, settings.effectiveFarPlane());
    proj[1][1] *= -1;  // Vulkan Y-flip
    frame.mvp = proj * camera.view();
    frame.cameraPos = camera.eyePos();
    // Unified sun: terrain lighting follows the same vector that drives
    // the visible sky and the shadow frustum below.
    frame.lightDir = currentSunDirectionYUp();
    frame.time = static_cast<float>(glfwGetTime());

    // Compute light-space view-projection matrix covering the finite world.
    float shadowSpan = settings.worldSize;
    float worldHalf = shadowSpan * 0.5f;
    float maxHeight = globalMaxHeight * settings.terrainHeightScale + 50.0f;
    glm::vec3 worldCenter(worldHalf, 0.0f, worldHalf);
    glm::vec3 lightPos = worldCenter - frame.lightDir * (maxHeight + 200.0f);
    glm::vec3 lightTarget = worldCenter;
    glm::vec3 lightUp = glm::vec3(0.0f, 1.0f, 0.0f);
    // If light direction is nearly parallel to up, pick a different up vector
    if (std::abs(glm::dot(frame.lightDir, lightUp)) > 0.99f)
        lightUp = glm::vec3(1.0f, 0.0f, 0.0f);

    glm::mat4 lightView = glm::lookAt(lightPos, lightTarget, lightUp);
    float orthoSize = worldHalf * 1.1f;  // full world + 10% margin
    glm::mat4 lightProj = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize,
                                     0.5f, maxHeight * 1.5f);
    lightProj[1][1] *= -1;  // Vulkan Y-flip
    frame.lightMVP = lightProj * lightView;
    currentLightMVP = frame.lightMVP;   // shared with the shadow pass

    memcpy(uniformMapped[currentFrame], &frame, sizeof(FrameData));

    // Also update the shadow UBO with the same lightMVP
    if (shadowUniformMapped)
        memcpy(shadowUniformMapped, &frame.lightMVP, sizeof(glm::mat4));

    updateAtmosphereUBO();

    // Advance cloud wind/frame state BEFORE terrain is recorded so the
    // terrain cloud-shadow pass and the visible cloud pass sample the
    // exact same cloud field this frame (no one-frame lag).
    updateCloudParamsUBO();
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Descriptors
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createDescriptorLayouts() {
    // Graphics: binding 0 = UBO, binding 1 = sampler
    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding = 0;
    uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding samplerBinding{};
    samplerBinding.binding = 1;
    samplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerBinding.descriptorCount = 1;
    samplerBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding texArrayBinding{};
    texArrayBinding.binding = 2;
    texArrayBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    texArrayBinding.descriptorCount = 1;
    texArrayBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding shadowBinding{};
    shadowBinding.binding = 3;
    shadowBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowBinding.descriptorCount = 1;
    shadowBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding atmosphereBinding{};
    atmosphereBinding.binding = 4;
    atmosphereBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    atmosphereBinding.descriptorCount = 1;
    atmosphereBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 5 = CloudParams UBO, binding 6 = 3D Worley noise — used by
    // terrain.frag to project cloud shadows onto the ground.
    VkDescriptorSetLayoutBinding cloudParamsBinding{};
    cloudParamsBinding.binding = 5;
    cloudParamsBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    cloudParamsBinding.descriptorCount = 1;
    cloudParamsBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding cloudNoiseBinding{};
    cloudNoiseBinding.binding = 6;
    cloudNoiseBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    cloudNoiseBinding.descriptorCount = 1;
    cloudNoiseBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding cloudShadowMapBinding{};
    cloudShadowMapBinding.binding = 7;
    cloudShadowMapBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    cloudShadowMapBinding.descriptorCount = 1;
    cloudShadowMapBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding bindings[] = {uboBinding, samplerBinding, texArrayBinding,
                                               shadowBinding, atmosphereBinding,
                                               cloudParamsBinding, cloudNoiseBinding,
                                               cloudShadowMapBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 8;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor set layout");
}

void VulkanEngine::createDescriptorPools() {
    // Single finite terrain: a small fixed pool covers the global terrain set,
    // the shadow set, and incidental UI/model sets.
    const int poolCapacity = 128;

    // Graphics pool: poolCapacity sets, each with 3 UBOs + 5 COMBINED_IMAGE_SAMPLERs.
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          static_cast<uint32_t>(poolCapacity * 3)},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  static_cast<uint32_t>(poolCapacity * 5)}
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = static_cast<uint32_t>(poolCapacity);
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics descriptor pool");

    // Small pool for the texgen material-bake compute set (one STORAGE_IMAGE
    // set, allocated and freed transiently during dispatchTexgenAndMips()).
    VkDescriptorPoolSize compPoolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8}
    };
    VkDescriptorPoolCreateInfo compPoolInfo{};
    compPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    compPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    compPoolInfo.maxSets = 8;
    compPoolInfo.poolSizeCount = 1;
    compPoolInfo.pPoolSizes = compPoolSizes;
    if (vkCreateDescriptorPool(device, &compPoolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create texgen compute descriptor pool");
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Pipelines
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createGraphicsPipeline() {
    auto vertCode = readFile("shaders/terrain.vert.spv");
    auto fragCode = readFile("shaders/terrain.frag.spv");
    VkShaderModule vertShader = createShaderModule(vertCode);
    VkShaderModule fragShader = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShader;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShader;
    fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    // Vertex input
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(glm::vec3);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDesc{};
    attrDesc.binding = 0;
    attrDesc.location = 0;
    attrDesc.format = VK_FORMAT_R32G32B32_SFLOAT;
    attrDesc.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = 1;
    vertexInput.pVertexAttributeDescriptions = &attrDesc;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlend{};
    colorBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                              | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlend;

    // Dynamic state (viewport + scissor)
    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynStates;

    // Push constant range for per-chunk data (vertex + fragment for debug overlay)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(ChunkPush);

    // Pipeline layout
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &terrainPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout");

    // Dynamic rendering
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &swapchainFormat;
    renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.pNext = &renderingInfo;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pDepthStencilState = &depthStencil;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.pDynamicState = &dynamicState;
    pipeInfo.layout = terrainPipelineLayout;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &terrainPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics pipeline");

    vkDestroyShaderModule(device, vertShader, nullptr);
    vkDestroyShaderModule(device, fragShader, nullptr);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Erosion pipelines (Phase 4)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Texture array (Phase 5: Triplanar)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createTextureArray() {
    const uint32_t TEX_RES = (uint32_t)settings.texRes;
    const int TEX_LAYERS = 12;   // 0-5 albedo+roughness(a), 6-11 normal+ao(a)
    VkFormat texFormat = VK_FORMAT_R8G8B8A8_UNORM;

    terrainTexMipLevels = (uint32_t)std::floor(std::log2((float)TEX_RES)) + 1u;
    const uint32_t MIPS = terrainTexMipLevels;

    // Sampler: trilinear + anisotropic across the full mip chain.
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = (float)MIPS;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &terrainTexSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create terrain texture sampler");

    // Storage + sampled + transfer (blit) usage; full mip chain.
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = texFormat;
    imgInfo.extent = {TEX_RES, TEX_RES, 1};
    imgInfo.mipLevels = MIPS;
    imgInfo.arrayLayers = TEX_LAYERS;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                  | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(device, &imgInfo, nullptr, &terrainTexArray.image) != VK_SUCCESS)
        throw std::runtime_error("failed to create texture array image");

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, terrainTexArray.image, &memReqs);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &terrainTexArray.memory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate texture array memory");
    vkBindImageMemory(device, terrainTexArray.image, terrainTexArray.memory, 0);

    // Sampled view: full mip chain, all layers (bound to terrain.frag).
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = terrainTexArray.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = texFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = MIPS;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = TEX_LAYERS;
    if (vkCreateImageView(device, &viewInfo, nullptr, &terrainTexArray.view) != VK_SUCCESS)
        throw std::runtime_error("failed to create texture array view");

    // Storage view: single mip (level 0), all layers — texgen writes here.
    viewInfo.subresourceRange.levelCount = 1;
    if (vkCreateImageView(device, &viewInfo, nullptr, &terrainTexStorageView) != VK_SUCCESS)
        throw std::runtime_error("failed to create texture array storage view");

    terrainTexBakedRes = TEX_RES;
    dispatchTexgenAndMips();
}

// (Re)generate the material array contents (albedo+roughness, normals) and the
// full mip chain. Safe to call again on seed/regen — operates on the existing
// image + views.
void VulkanEngine::dispatchTexgenAndMips() {
    const uint32_t TEX_RES = (uint32_t)settings.texRes;
    const int TEX_LAYERS = 12;
    const uint32_t MIPS = terrainTexMipLevels;

    VkCommandBuffer cmd = beginSingleTimeCommands();

    auto layerRange = [&](uint32_t baseMip, uint32_t mipCount) {
        VkImageSubresourceRange r{};
        r.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        r.baseMipLevel = baseMip; r.levelCount = mipCount;
        r.baseArrayLayer = 0; r.layerCount = TEX_LAYERS;
        return r;
    };
    auto barrier = [&](VkImageLayout oldL, VkImageLayout newL,
                       VkAccessFlags2 srcA, VkAccessFlags2 dstA,
                       VkPipelineStageFlags2 srcS, VkPipelineStageFlags2 dstS,
                       VkImageSubresourceRange range) {
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = srcS; b.srcAccessMask = srcA;
        b.dstStageMask = dstS; b.dstAccessMask = dstA;
        b.oldLayout = oldL; b.newLayout = newL;
        b.image = terrainTexArray.image; b.subresourceRange = range;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    };

    // Level 0 -> GENERAL for the compute write.
    barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            layerRange(0, 1));

    VkDescriptorSet texgenSet = VK_NULL_HANDLE;
    {
        VkDescriptorSetAllocateInfo setAlloc{};
        setAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setAlloc.descriptorPool = computeDescriptorPool;
        setAlloc.descriptorSetCount = 1;
        setAlloc.pSetLayouts = &texgenDescriptorSetLayout;
        if (vkAllocateDescriptorSets(device, &setAlloc, &texgenSet) != VK_SUCCESS) {
            std::cerr << "texgen descriptor set allocation failed\n";
            vkEndCommandBuffer(cmd);
            vkFreeCommandBuffers(device, commandPool, 1, &cmd);
            return;
        }
        VkDescriptorImageInfo imgDesc{};
        imgDesc.imageView = terrainTexStorageView;
        imgDesc.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = texgenSet; write.dstBinding = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.descriptorCount = 1; write.pImageInfo = &imgDesc;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, texgenPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        texgenPipelineLayout, 0, 1, &texgenSet, 0, nullptr);
    struct { int resolution; float seed; } texgenPC =
        {(int)TEX_RES, (float)settings.activeWorldSeed() * 0.001f + 1.0f};
    vkCmdPushConstants(cmd, texgenPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(texgenPC), &texgenPC);
    vkCmdDispatch(cmd, (TEX_RES + 7) / 8, (TEX_RES + 7) / 8, TEX_LAYERS);

    // ── Mip chain via successive blits (all 12 layers at once) ──
    // Level 0: GENERAL (compute write) -> TRANSFER_SRC.
    barrier(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            layerRange(0, 1));

    int32_t mw = (int32_t)TEX_RES, mh = (int32_t)TEX_RES;
    for (uint32_t i = 1; i < MIPS; ++i) {
        const int32_t nw = std::max(1, mw / 2), nh = std::max(1, mh / 2);
        // Dst mip -> TRANSFER_DST.
        barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_ACCESS_2_NONE, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                layerRange(i, 1));
        VkImageBlit blit{};
        blit.srcOffsets[1] = {mw, mh, 1};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, (uint32_t)TEX_LAYERS};
        blit.dstOffsets[1] = {nw, nh, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, (uint32_t)TEX_LAYERS};
        vkCmdBlitImage(cmd,
            terrainTexArray.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            terrainTexArray.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);
        // Previous mip is done being read -> SHADER_READ_ONLY.
        barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                layerRange(i - 1, 1));
        // This mip becomes the source for the next iteration.
        barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                layerRange(i, 1));
        mw = nw; mh = nh;
    }
    // Last mip: TRANSFER_SRC -> SHADER_READ_ONLY.
    barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            layerRange(MIPS - 1, 1));

    endSingleTimeCommands(cmd);
    vkFreeDescriptorSets(device, computeDescriptorPool, 1, &texgenSet);
    std::cout << "  [texgen] baked " << TEX_LAYERS << " layers @" << TEX_RES
              << " with " << MIPS << " mips\n";
}

void VulkanEngine::createTexgenPipeline() {
    // Descriptor set layout: binding 0 = storage image2DArray
    VkDescriptorSetLayoutBinding imgBinding{};
    imgBinding.binding = 0;
    imgBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    imgBinding.descriptorCount = 1;
    imgBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &imgBinding;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &texgenDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create texgen descriptor set layout");

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 8; // int resolution + float seed

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount = 1;
    pipeLayoutInfo.pSetLayouts = &texgenDescriptorSetLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &texgenPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create texgen pipeline layout");

    auto code = readFile("shaders/texgen.comp.spv");
    VkShaderModule module = createShaderModule(code);

    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = module;
    pipeInfo.stage.pName = "main";
    pipeInfo.layout = texgenPipelineLayout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &texgenPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create texgen pipeline");

    vkDestroyShaderModule(device, module, nullptr);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Shadow map (Phase 7)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createShadowMap() {
    // Create depth image for shadow map
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = VK_FORMAT_D32_SFLOAT;
    imgInfo.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device, &imgInfo, nullptr, &shadowMap.image) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadow map image");

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, shadowMap.image, &memReqs);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &shadowMap.memory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate shadow map memory");
    vkBindImageMemory(device, shadowMap.image, shadowMap.memory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = shadowMap.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_D32_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &viewInfo, nullptr, &shadowMap.view) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadow map view");

    // â”€â”€ Initialize shadow map to max depth (1.0) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Until the shadow pass is fully integrated, clear to 1.0 so
    // the depth comparison always passes (no shadows cast).
    // This prevents the fragment shader from reading garbage depth
    // values that would darken the entire scene.
    {
        VkCommandBuffer cmd = beginSingleTimeCommands();

        // UNDEFINED â†’ DEPTH_ATTACHMENT_OPTIMAL for clear
        VkImageMemoryBarrier2 preBarrier{};
        preBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        preBarrier.srcAccessMask = VK_ACCESS_2_NONE;
        preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        preBarrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        preBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        preBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        preBarrier.image = shadowMap.image;
        preBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        preBarrier.subresourceRange.levelCount = 1;
        preBarrier.subresourceRange.layerCount = 1;

        VkDependencyInfo preDep{};
        preDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        preDep.imageMemoryBarrierCount = 1;
        preDep.pImageMemoryBarriers = &preBarrier;
        vkCmdPipelineBarrier2(cmd, &preDep);

        // Dynamic rendering clear to 1.0 (max depth in Vulkan)
        VkRenderingAttachmentInfo depthAttach{};
        depthAttach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttach.imageView = shadowMap.view;
        depthAttach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttach.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo renderInfo{};
        renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderInfo.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
        renderInfo.layerCount = 1;
        renderInfo.pDepthAttachment = &depthAttach;
        vkCmdBeginRendering(cmd, &renderInfo);
        vkCmdEndRendering(cmd);

        // DEPTH_ATTACHMENT_OPTIMAL â†’ DEPTH_READ_ONLY_OPTIMAL for sampling
        VkImageMemoryBarrier2 postBarrier{};
        postBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        postBarrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        postBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        postBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        postBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        postBarrier.image = shadowMap.image;
        postBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        postBarrier.subresourceRange.levelCount = 1;
        postBarrier.subresourceRange.layerCount = 1;

        VkDependencyInfo postDep{};
        postDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        postDep.imageMemoryBarrierCount = 1;
        postDep.pImageMemoryBarriers = &postBarrier;
        vkCmdPipelineBarrier2(cmd, &postDep);

        endSingleTimeCommands(cmd);
    }

    // Shadow sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.compareEnable = VK_TRUE;
    samplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &shadowSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadow sampler");

    // Shadow UBO
    VkDeviceSize uboSize = sizeof(glm::mat4);  // light MVP
    createBuffer(uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        shadowUniformBuffer);
    vkMapMemory(device, shadowUniformBuffer.memory, 0, uboSize, 0, &shadowUniformMapped);
}

void VulkanEngine::createShadowPipeline() {
    // Descriptor layout: binding 0 = UBO (lightMVP), binding 1 = heightmap sampler
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &shadowDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadow descriptor set layout");

    // Pool — single finite terrain shadow set plus headroom.
    int poolCapacity = 16;
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(poolCapacity)},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(poolCapacity)}
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = static_cast<uint32_t>(poolCapacity);
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &shadowDescriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadow descriptor pool");

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo setAlloc{};
    setAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAlloc.descriptorPool = shadowDescriptorPool;
    setAlloc.descriptorSetCount = 1;
    setAlloc.pSetLayouts = &shadowDescriptorSetLayout;
    if (vkAllocateDescriptorSets(device, &setAlloc, &shadowDescriptorSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate shadow descriptor set");

    // Write UBO
    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = shadowUniformBuffer.buffer;
    uboInfo.offset = 0;
    uboInfo.range = sizeof(glm::mat4);

    VkWriteDescriptorSet writeUBO{};
    writeUBO.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeUBO.dstSet = shadowDescriptorSet;
    writeUBO.dstBinding = 0;
    writeUBO.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeUBO.descriptorCount = 1;
    writeUBO.pBufferInfo = &uboInfo;

    vkUpdateDescriptorSets(device, 1, &writeUBO, 0, nullptr);
    // Note: binding 1 (heightmap sampler) is written per-chunk in recordShadowPass

    // Push constants for per-chunk data
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(ChunkPush);

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount = 1;
    pipeLayoutInfo.pSetLayouts = &shadowDescriptorSetLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &shadowPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadow pipeline layout");

    // Create depth-only graphics pipeline
    auto vertCode = readFile("shaders/shadow.vert.spv");
    VkShaderModule vertShader = createShaderModule(vertCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShader;
    vertStage.pName = "main";

    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(glm::vec3);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDesc{};
    attrDesc.binding = 0;
    attrDesc.location = 0;
    attrDesc.format = VK_FORMAT_R32G32B32_SFLOAT;
    attrDesc.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = 1;
    vertexInput.pVertexAttributeDescriptions = &attrDesc;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 1.25f;
    rasterizer.depthBiasSlopeFactor = 1.75f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynStates;

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.pNext = &renderingInfo;
    pipeInfo.stageCount = 1;
    pipeInfo.pStages = &vertStage;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pDepthStencilState = &depthStencil;
    pipeInfo.pDynamicState = &dynamicState;
    pipeInfo.layout = shadowPipelineLayout;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &shadowPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create shadow pipeline");

    vkDestroyShaderModule(device, vertShader, nullptr);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Foliage (Phase 7)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createFoliageMesh() {
    // Simple tree: cone (canopy) + cylinder (trunk)
    // Using 8 vertical segments for cone, 8 for cylinder
    std::vector<glm::vec3> verts;
    std::vector<uint32_t> indices;

    // Trunk: cylinder from y=0 to y=1.5, radius=0.15
    const int SEGS = 8;
    const float TRUNK_H = 1.5f;
    const float TRUNK_R = 0.15f;

    for (int i = 0; i <= SEGS; i++) {
        float a = (float(i) / SEGS) * 6.28318f;
        float cx = cos(a) * TRUNK_R;
        float cz = sin(a) * TRUNK_R;
        verts.push_back({cx, 0.0f, cz});      // bottom ring
        verts.push_back({cx, TRUNK_H, cz});    // top ring
    }

    // Trunk indices
    uint32_t baseV = 0;
    for (int i = 0; i < SEGS; i++) {
        uint32_t b0 = baseV + i * 2;
        uint32_t b1 = b0 + 1;
        uint32_t t0 = b0 + 2;
        uint32_t t1 = b0 + 3;
        indices.push_back(b0); indices.push_back(b1); indices.push_back(t0);
        indices.push_back(t0); indices.push_back(b1); indices.push_back(t1);
    }

    // Canopy: cone from y=1.0 to y=3.5, base radius=1.2
    baseV = uint32_t(verts.size());
    const float CANOPY_BASE = 1.0f;
    const float CANOPY_TOP = 3.5f;
    const float CANOPY_R = 1.2f;
    for (int i = 0; i <= SEGS; i++) {
        float a = (float(i) / SEGS) * 6.28318f;
        float cx = cos(a) * CANOPY_R;
        float cz = sin(a) * CANOPY_R;
        verts.push_back({cx, CANOPY_BASE, cz});
    }
    verts.push_back({0.0f, CANOPY_TOP, 0.0f});  // apex

    uint32_t apex = baseV + SEGS + 1;
    for (int i = 0; i < SEGS; i++) {
        // Canopy base ring cap (triangles between adjacent base vertices)
        uint32_t v0 = baseV + i;
        uint32_t v1 = baseV + i + 1;
        indices.push_back(v0); indices.push_back(v1); indices.push_back(apex);
        // Base cap (center fill)
        indices.push_back(v0); indices.push_back(apex); indices.push_back(v1);
    }

    foliageIndexCount = uint32_t(indices.size());

    // Upload vertex buffer
    VkDeviceSize vbSize = verts.size() * sizeof(glm::vec3);
    Buffer staging;
    createBuffer(vbSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
    void* data;
    vkMapMemory(device, staging.memory, 0, vbSize, 0, &data);
    memcpy(data, verts.data(), vbSize);
    vkUnmapMemory(device, staging.memory);

    createBuffer(vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, foliageVertexBuffer);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy copy{};
    copy.size = vbSize;
    vkCmdCopyBuffer(cmd, staging.buffer, foliageVertexBuffer.buffer, 1, &copy);
    endSingleTimeCommands(cmd);
    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);

    // Upload index buffer
    VkDeviceSize ibSize = indices.size() * sizeof(uint32_t);
    createBuffer(ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
    vkMapMemory(device, staging.memory, 0, ibSize, 0, &data);
    memcpy(data, indices.data(), ibSize);
    vkUnmapMemory(device, staging.memory);

    createBuffer(ibSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, foliageIndexBuffer);

    cmd = beginSingleTimeCommands();
    copy.size = ibSize;
    vkCmdCopyBuffer(cmd, staging.buffer, foliageIndexBuffer.buffer, 1, &copy);
    endSingleTimeCommands(cmd);
    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);
}

void VulkanEngine::createFoliagePipeline() {
    auto vertCode = readFile("shaders/foliage.vert.spv");
    auto fragCode = readFile("shaders/foliage.frag.spv");
    VkShaderModule vertShader = createShaderModule(vertCode);
    VkShaderModule fragShader = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShader;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShader;
    fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    // Vertex input: binding 0 = model vertices, binding 1 = instance data (posScale), binding 2 = instance data (color)
    VkVertexInputBindingDescription bindings[3]{};
    bindings[0].binding = 0;
    bindings[0].stride = sizeof(glm::vec3);
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    bindings[1].binding = 1;
    bindings[1].stride = sizeof(glm::vec4);  // posScale
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    bindings[2].binding = 2;
    bindings[2].stride = sizeof(glm::vec4);  // color
    bindings[2].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription attrs[3]{};
    attrs[0].binding = 0;
    attrs[0].location = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset = 0;

    attrs[1].binding = 1;
    attrs[1].location = 1;
    attrs[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[1].offset = 0;

    attrs[2].binding = 2;
    attrs[2].location = 2;
    attrs[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[2].offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 3;
    vertexInput.pVertexBindingDescriptions = bindings;
    vertexInput.vertexAttributeDescriptionCount = 3;
    vertexInput.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlend{};
    colorBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                              | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlend;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynStates;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(glm::mat4);  // MVP

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 0;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &foliagePipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create foliage pipeline layout");

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &swapchainFormat;
    renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.pNext = &renderingInfo;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pDepthStencilState = &depthStencil;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.pDynamicState = &dynamicState;
    pipeInfo.layout = foliagePipelineLayout;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &foliagePipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create foliage pipeline");

    vkDestroyShaderModule(device, vertShader, nullptr);
    vkDestroyShaderModule(device, fragShader, nullptr);
}

// ══════════════════════════════════════════════════════════════
//  TEMPORARY debug aircraft — fuselage + wings + tail, bright color.
//  Built in local model space where 1.0 unit = aircraft half-length
//  (the renderer scales by halfExtents.x). +X = nose/forward,
//  +Y = up, +Z = right wing.
// ══════════════════════════════════════════════════════════════
void VulkanEngine::createAircraftMesh() {
    struct AVert { glm::vec3 pos; glm::vec3 color; glm::vec3 normal; };
    std::vector<AVert> verts;
    std::vector<uint32_t> indices;

    auto addBox = [&](glm::vec3 mn, glm::vec3 mx, glm::vec3 color) {
        const glm::vec3 c[8] = {
            {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mx.x, mx.y, mn.z}, {mn.x, mx.y, mn.z},
            {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z}, {mx.x, mx.y, mx.z}, {mn.x, mx.y, mx.z},
        };
        struct Face { int a, b, c2, d; glm::vec3 n; };
        const Face faces[6] = {
            {1, 0, 3, 2, { 0,  0, -1}},  // -Z
            {4, 5, 6, 7, { 0,  0,  1}},  // +Z
            {0, 4, 7, 3, {-1,  0,  0}},  // -X
            {5, 1, 2, 6, { 1,  0,  0}},  // +X
            {3, 7, 6, 2, { 0,  1,  0}},  // +Y
            {0, 1, 5, 4, { 0, -1,  0}},  // -Y
        };
        for (const Face& f : faces) {
            uint32_t base = (uint32_t)verts.size();
            verts.push_back({c[f.a], color, f.n});
            verts.push_back({c[f.b], color, f.n});
            verts.push_back({c[f.c2], color, f.n});
            verts.push_back({c[f.d], color, f.n});
            indices.push_back(base + 0); indices.push_back(base + 1); indices.push_back(base + 2);
            indices.push_back(base + 0); indices.push_back(base + 2); indices.push_back(base + 3);
        }
    };

    const glm::vec3 kOrange(1.00f, 0.45f, 0.05f);
    const glm::vec3 kYellow(0.97f, 0.85f, 0.10f);
    const glm::vec3 kRed   (0.92f, 0.13f, 0.08f);
    const glm::vec3 kWhite (0.95f, 0.95f, 0.95f);

    // Fuselage
    addBox({-1.00f, -0.16f, -0.14f}, {1.00f, 0.16f, 0.14f}, kOrange);
    // Nose marker (forward direction, bright red)
    addBox({1.00f, -0.10f, -0.10f}, {1.32f, 0.10f, 0.10f}, kRed);
    // Cockpit cap (top, white) — helps read roll/pitch
    addBox({0.35f, 0.16f, -0.10f}, {0.75f, 0.30f, 0.10f}, kWhite);
    // Main wings (span along Z)
    addBox({-0.18f, -0.02f, -1.40f}, {0.40f, 0.05f, 1.40f}, kYellow);
    // Horizontal stabilizer
    addBox({-1.02f, 0.00f, -0.58f}, {-0.78f, 0.05f, 0.58f}, kYellow);
    // Vertical stabilizer (fin)
    addBox({-1.02f, 0.00f, -0.05f}, {-0.74f, 0.55f, 0.05f}, kRed);

    // Landing gear wheels (dark) — canonical positions mirror the tricycle gear
    // (scaled by halfLength at draw time). Cosmetic: drawn at rest extension.
    const glm::vec3 kTire(0.07f, 0.07f, 0.08f);
    const float wy = -0.30f;   // belly + a little suspension drop
    const float wr = 0.07f;    // wheel half-size in canonical units
    auto addWheel = [&](float cx, float cz) {
        addBox({cx - wr, wy - wr, cz - wr * 0.6f}, {cx + wr, wy + wr, cz + wr * 0.6f}, kTire);
    };
    addWheel( 0.55f, 0.0f);    // nose
    addWheel(-0.25f, -0.22f);  // left main
    addWheel(-0.25f,  0.22f);  // right main

    aircraftIndexCount = (uint32_t)indices.size();

    VkDeviceSize vbSize = verts.size() * sizeof(AVert);
    Buffer staging;
    createBuffer(vbSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
    void* data;
    vkMapMemory(device, staging.memory, 0, vbSize, 0, &data);
    memcpy(data, verts.data(), vbSize);
    vkUnmapMemory(device, staging.memory);
    createBuffer(vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, aircraftVertexBuffer);
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy copy{};
    copy.size = vbSize;
    vkCmdCopyBuffer(cmd, staging.buffer, aircraftVertexBuffer.buffer, 1, &copy);
    endSingleTimeCommands(cmd);
    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);

    VkDeviceSize ibSize = indices.size() * sizeof(uint32_t);
    createBuffer(ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
    vkMapMemory(device, staging.memory, 0, ibSize, 0, &data);
    memcpy(data, indices.data(), ibSize);
    vkUnmapMemory(device, staging.memory);
    createBuffer(ibSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, aircraftIndexBuffer);
    cmd = beginSingleTimeCommands();
    copy.size = ibSize;
    vkCmdCopyBuffer(cmd, staging.buffer, aircraftIndexBuffer.buffer, 1, &copy);
    endSingleTimeCommands(cmd);
    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);

    std::cout << "  [aircraft/temp] debug mesh built: " << verts.size()
              << " verts, " << aircraftIndexCount << " indices\n";
}

void VulkanEngine::createAircraftPipeline() {
    auto vertCode = readFile("shaders/aircraft.vert.spv");
    auto fragCode = readFile("shaders/aircraft.frag.spv");
    VkShaderModule vertShader = createShaderModule(vertCode);
    VkShaderModule fragShader = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShader;
    vertStage.pName = "main";
    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShader;
    fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    // Interleaved vertex: pos(3) + color(3) + normal(3) = 9 floats.
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(float) * 9;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[3]{};
    attrs[0].binding = 0; attrs[0].location = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = 0;
    attrs[1].binding = 0; attrs[1].location = 1;
    attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = sizeof(float) * 3;
    attrs[2].binding = 0; attrs[2].location = 2;
    attrs[2].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[2].offset = sizeof(float) * 6;

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 3;
    vertexInput.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlend{};
    colorBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                              | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlend;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynStates;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(glm::mat4) * 2;  // mvp + model

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 0;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &aircraftPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create aircraft pipeline layout");

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &swapchainFormat;
    renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.pNext = &renderingInfo;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pDepthStencilState = &depthStencil;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.pDynamicState = &dynamicState;
    pipeInfo.layout = aircraftPipelineLayout;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &aircraftPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create aircraft pipeline");

    vkDestroyShaderModule(device, vertShader, nullptr);
    vkDestroyShaderModule(device, fragShader, nullptr);
}

// Airport paved-surface pipeline: same vertex format / push constants as the
// aircraft pipeline (pos+color+normal, mvp+model) but with depth bias (so the
// near-ground concrete wins over the terrain) and no face culling.
void VulkanEngine::createAirportPipeline() {
    auto vertCode = readFile("shaders/aircraft.vert.spv");
    auto fragCode = readFile("shaders/aircraft.frag.spv");
    VkShaderModule vertShader = createShaderModule(vertCode);
    VkShaderModule fragShader = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShader; vertStage.pName = "main";
    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShader; fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    VkVertexInputBindingDescription binding{};
    binding.binding = 0; binding.stride = sizeof(float) * 9;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attrs[3]{};
    attrs[0].binding = 0; attrs[0].location = 0; attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = 0;
    attrs[1].binding = 0; attrs[1].location = 1; attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = sizeof(float) * 3;
    attrs[2].binding = 0; attrs[2].location = 2; attrs[2].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[2].offset = sizeof(float) * 6;
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 3;
    vertexInput.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1; viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;
    rasterizer.depthBiasEnable = VK_TRUE;            // pull concrete toward camera
    rasterizer.depthBiasConstantFactor = -2.0f;
    rasterizer.depthBiasSlopeFactor = -2.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlend{};
    colorBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                              | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlend;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynStates;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0; pushRange.size = sizeof(glm::mat4) * 2;
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 0;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &airportPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create airport pipeline layout");

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &swapchainFormat;
    renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.pNext = &renderingInfo;
    pipeInfo.stageCount = 2; pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pDepthStencilState = &depthStencil;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.pDynamicState = &dynamicState;
    pipeInfo.layout = airportPipelineLayout;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &airportPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create airport pipeline");

    vkDestroyShaderModule(device, vertShader, nullptr);
    vkDestroyShaderModule(device, fragShader, nullptr);
}

// (Re)build the combined paved-surface mesh for both airports.
void VulkanEngine::buildAirportMesh() {
    if (!airportsValid) return;

    struct AVert { glm::vec3 pos; glm::vec3 color; glm::vec3 normal; };
    std::vector<AVert> verts;
    std::vector<uint32_t> indices;
    const glm::vec3 N(0.0f, 1.0f, 0.0f);

    for (const Airport& ap : airports) {
        const glm::vec2 dir = ap.dir();
        const glm::vec2 side = ap.side();
        for (const AirportQuad& q : ap.buildSurfaceQuads()) {
            const float y = ap.elevation + q.yOffset;
            // q.center is already world XZ; q.headingRad == ap.headingRad.
            glm::vec2 d(std::cos(q.headingRad), std::sin(q.headingRad));
            glm::vec2 s(-d.y, d.x);
            uint32_t base = (uint32_t)verts.size();
            for (int sl = -1; sl <= 1; sl += 2) {
                for (int sw = -1; sw <= 1; sw += 2) {
                    glm::vec2 c = q.center + d * (sl * q.halfLen) + s * (sw * q.halfWid);
                    verts.push_back({ glm::vec3(c.x, y, c.y), q.color, N });
                }
            }
            // corner order: (sl,sw) = (-,-),(-,+),(+,-),(+,+) → 0,1,2,3
            indices.push_back(base + 0); indices.push_back(base + 2); indices.push_back(base + 3);
            indices.push_back(base + 0); indices.push_back(base + 3); indices.push_back(base + 1);
        }
    }

    airportIndexCount = (uint32_t)indices.size();
    if (verts.empty() || indices.empty()) return;

    // Destroy any previous buffers (rebuild path on regen).
    if (airportVertexBuffer.buffer) { vkDestroyBuffer(device, airportVertexBuffer.buffer, nullptr); airportVertexBuffer.buffer = VK_NULL_HANDLE; }
    if (airportVertexBuffer.memory) { vkFreeMemory(device, airportVertexBuffer.memory, nullptr); airportVertexBuffer.memory = VK_NULL_HANDLE; }
    if (airportIndexBuffer.buffer)  { vkDestroyBuffer(device, airportIndexBuffer.buffer, nullptr);  airportIndexBuffer.buffer = VK_NULL_HANDLE; }
    if (airportIndexBuffer.memory)  { vkFreeMemory(device, airportIndexBuffer.memory, nullptr);     airportIndexBuffer.memory = VK_NULL_HANDLE; }

    auto uploadBuffer = [&](const void* src, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
        Buffer staging;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
        void* data; vkMapMemory(device, staging.memory, 0, size, 0, &data);
        memcpy(data, src, size); vkUnmapMemory(device, staging.memory);
        createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out);
        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkBufferCopy copy{}; copy.size = size;
        vkCmdCopyBuffer(cmd, staging.buffer, out.buffer, 1, &copy);
        endSingleTimeCommands(cmd);
        vkDestroyBuffer(device, staging.buffer, nullptr);
        vkFreeMemory(device, staging.memory, nullptr);
    };

    uploadBuffer(verts.data(), verts.size() * sizeof(AVert), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, airportVertexBuffer);
    uploadBuffer(indices.data(), indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, airportIndexBuffer);

    std::cout << "  [airport] paved-surface mesh: " << verts.size()
              << " verts, " << airportIndexCount << " indices\n";
}

// ── Textured FBX model: pipeline, default textures, loading, rendering ──

void VulkanEngine::createModelPipeline() {
    // Descriptor set 0: base, metallic, roughness, normal samplers + material UBO.
    VkDescriptorSetLayoutBinding binds[5]{};
    for (int i = 0; i < 4; ++i) {
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    binds[4].binding = 4;
    binds[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binds[4].descriptorCount = 1;
    binds[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslInfo{};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = 5;
    dslInfo.pBindings = binds;
    if (vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &modelDescLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create model descriptor set layout");

    // Shared sampler (repeat, trilinear).
    VkSamplerCreateInfo sampInfo{};
    sampInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampInfo.magFilter = VK_FILTER_LINEAR;
    sampInfo.minFilter = VK_FILTER_LINEAR;
    sampInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampInfo.maxLod = 0.0f;
    if (vkCreateSampler(device, &sampInfo, nullptr, &modelSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create model sampler");

    auto vertCode = readFile("shaders/model.vert.spv");
    auto fragCode = readFile("shaders/model.frag.spv");
    VkShaderModule vertShader = createShaderModule(vertCode);
    VkShaderModule fragShader = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShader; vertStage.pName = "main";
    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShader; fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    // ModelVertex: pos(3) normal(3) uv(2) tangent(4) = 48 bytes.
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(ModelVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[4]{};
    attrs[0].location = 0; attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;    attrs[0].offset = offsetof(ModelVertex, pos);
    attrs[1].location = 1; attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;    attrs[1].offset = offsetof(ModelVertex, normal);
    attrs[2].location = 2; attrs[2].format = VK_FORMAT_R32G32_SFLOAT;       attrs[2].offset = offsetof(ModelVertex, uv);
    attrs[3].location = 3; attrs[3].format = VK_FORMAT_R32G32B32A32_SFLOAT; attrs[3].offset = offsetof(ModelVertex, tangent);

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 4;
    vertexInput.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1; viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;   // FBX winding varies; draw both sides
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlend{};
    colorBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                              | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlend;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2; dynamicState.pDynamicStates = dynStates;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(glm::mat4) * 2;   // mvp + model

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &modelDescLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &modelPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create model pipeline layout");

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &swapchainFormat;
    renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.pNext = &renderingInfo;
    pipeInfo.stageCount = 2; pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pDepthStencilState = &depthStencil;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.pDynamicState = &dynamicState;
    pipeInfo.layout = modelPipelineLayout;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &modelPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create model pipeline");

    vkDestroyShaderModule(device, vertShader, nullptr);
    vkDestroyShaderModule(device, fragShader, nullptr);
    std::cout << "  [model] pipeline created\n";
}

void VulkanEngine::createModelDefaultTextures() {
    auto upload1x1 = [&](uint8_t r, uint8_t g, uint8_t b, uint8_t a, Image& out) {
        const uint8_t px[4] = {r, g, b, a};
        Buffer staging;
        createBuffer(4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
        void* data; vkMapMemory(device, staging.memory, 0, 4, 0, &data);
        std::memcpy(data, px, 4); vkUnmapMemory(device, staging.memory);
        createImage(1, 1, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, out);

        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkImageMemoryBarrier toDst{};
        toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image = out.image;
        toDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        toDst.srcAccessMask = 0; toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toDst);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {1, 1, 1};
        vkCmdCopyBufferToImage(cmd, staging.buffer, out.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VkImageMemoryBarrier toRead = toDst;
        toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toRead);
        endSingleTimeCommands(cmd);
        vkDestroyBuffer(device, staging.buffer, nullptr);
        vkFreeMemory(device, staging.memory, nullptr);
    };
    upload1x1(255, 255, 255, 255, defaultWhiteTex);   // base/metal/rough fallback
    upload1x1(128, 128, 255, 255, defaultNormalTex);  // flat tangent-space normal
}

void VulkanEngine::uploadModelTexture(const std::string& path, bool srgb, Image& out, bool& owned) {
    owned = false;
    if (path.empty()) return;
    ImageRGBA8 img = ModelLoader::loadImageRGBA8(path, 2048);
    if (!img.valid()) return;

    const VkFormat fmt = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    const VkDeviceSize sz = (VkDeviceSize)img.pixels.size();

    Buffer staging;
    createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
    void* data; vkMapMemory(device, staging.memory, 0, sz, 0, &data);
    std::memcpy(data, img.pixels.data(), (size_t)sz); vkUnmapMemory(device, staging.memory);

    createImage((uint32_t)img.width, (uint32_t)img.height, fmt,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, out);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkImageMemoryBarrier toDst{};
    toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.image = out.image;
    toDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    toDst.srcAccessMask = 0; toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toDst);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {(uint32_t)img.width, (uint32_t)img.height, 1};
    vkCmdCopyBufferToImage(cmd, staging.buffer, out.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VkImageMemoryBarrier toRead = toDst;
    toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toRead);
    endSingleTimeCommands(cmd);
    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);
    owned = true;
}

namespace {

// ── Built-in editor primitives (CPU ModelData) ──────────────────
// A unit cube centred on the origin (−0.5..0.5), 24 verts so each face has
// its own flat normal/UV. One default material (tinted, no textures).
ModelData makeBoxModel() {
    ModelData md;
    const glm::vec3 n[6] = {
        { 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0}, { 0,-1, 0}, { 0, 0, 1}, { 0, 0,-1}
    };
    const glm::vec4 t[6] = {
        {0,0,-1,1}, {0,0,1,1}, {1,0,0,1}, {1,0,0,1}, {1,0,0,1}, {-1,0,0,1}
    };
    // Four corners per face (CCW seen from outside).
    const glm::vec3 faceVerts[6][4] = {
        {{ 0.5f,-0.5f, 0.5f},{ 0.5f,-0.5f,-0.5f},{ 0.5f, 0.5f,-0.5f},{ 0.5f, 0.5f, 0.5f}}, // +X
        {{-0.5f,-0.5f,-0.5f},{-0.5f,-0.5f, 0.5f},{-0.5f, 0.5f, 0.5f},{-0.5f, 0.5f,-0.5f}}, // -X
        {{-0.5f, 0.5f, 0.5f},{ 0.5f, 0.5f, 0.5f},{ 0.5f, 0.5f,-0.5f},{-0.5f, 0.5f,-0.5f}}, // +Y
        {{-0.5f,-0.5f,-0.5f},{ 0.5f,-0.5f,-0.5f},{ 0.5f,-0.5f, 0.5f},{-0.5f,-0.5f, 0.5f}}, // -Y
        {{-0.5f,-0.5f, 0.5f},{ 0.5f,-0.5f, 0.5f},{ 0.5f, 0.5f, 0.5f},{-0.5f, 0.5f, 0.5f}}, // +Z
        {{ 0.5f,-0.5f,-0.5f},{-0.5f,-0.5f,-0.5f},{-0.5f, 0.5f,-0.5f},{ 0.5f, 0.5f,-0.5f}}, // -Z
    };
    const glm::vec2 uvs[4] = {{0,1},{1,1},{1,0},{0,0}};
    for (int f = 0; f < 6; ++f) {
        uint32_t base = (uint32_t)md.vertices.size();
        for (int c = 0; c < 4; ++c) {
            ModelVertex v;
            v.pos = faceVerts[f][c];
            v.normal = n[f];
            v.uv = uvs[c];
            v.tangent = t[f];
            md.vertices.push_back(v);
        }
        md.indices.push_back(base + 0); md.indices.push_back(base + 1); md.indices.push_back(base + 2);
        md.indices.push_back(base + 0); md.indices.push_back(base + 2); md.indices.push_back(base + 3);
    }
    md.boundsMin = glm::vec3(-0.5f);
    md.boundsMax = glm::vec3(0.5f);
    ModelMaterial mat;
    mat.name = "Box";
    mat.baseColorFactor = glm::vec4(0.75f, 0.76f, 0.78f, 1.0f);
    mat.metallicFactor = 0.0f;
    mat.roughnessFactor = 0.85f;
    md.materials.push_back(mat);
    md.submeshes.push_back({0, (uint32_t)md.indices.size(), 0});
    md.valid = true;
    return md;
}

// A UV sphere of radius 0.5 centred on the origin.
ModelData makeSphereModel(int stacks = 24, int slices = 32) {
    ModelData md;
    const float R = 0.5f;
    for (int i = 0; i <= stacks; ++i) {
        float v = (float)i / (float)stacks;
        float phi = v * 3.14159265358979f;          // 0..pi
        float y = std::cos(phi);
        float r = std::sin(phi);
        for (int j = 0; j <= slices; ++j) {
            float u = (float)j / (float)slices;
            float theta = u * 2.0f * 3.14159265358979f;
            glm::vec3 nrm(r * std::cos(theta), y, r * std::sin(theta));
            ModelVertex vert;
            vert.pos = nrm * R;
            vert.normal = nrm;
            vert.uv = glm::vec2(u, v);
            vert.tangent = glm::vec4(-std::sin(theta), 0.0f, std::cos(theta), 1.0f);
            md.vertices.push_back(vert);
        }
    }
    const int cols = slices + 1;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = (uint32_t)(i * cols + j);
            uint32_t b = (uint32_t)((i + 1) * cols + j);
            md.indices.push_back(a);     md.indices.push_back(b);     md.indices.push_back(a + 1);
            md.indices.push_back(a + 1); md.indices.push_back(b);     md.indices.push_back(b + 1);
        }
    }
    md.boundsMin = glm::vec3(-R);
    md.boundsMax = glm::vec3(R);
    ModelMaterial mat;
    mat.name = "Sphere";
    mat.baseColorFactor = glm::vec4(0.80f, 0.55f, 0.35f, 1.0f);
    mat.metallicFactor = 0.0f;
    mat.roughnessFactor = 0.6f;
    md.materials.push_back(mat);
    md.submeshes.push_back({0, (uint32_t)md.indices.size(), 0});
    md.valid = true;
    return md;
}

} // namespace

VulkanEngine::LoadedModel VulkanEngine::loadModelAsset(const ModelData& md, float targetSize) {
    LoadedModel lm;
    if (!md.valid || md.vertices.empty() || md.indices.empty())
        return lm;

    // Vertex + index buffers (device-local via staging).
    auto uploadBuffer = [&](const void* src, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& dst) {
        Buffer staging;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
        void* data; vkMapMemory(device, staging.memory, 0, size, 0, &data);
        std::memcpy(data, src, (size_t)size); vkUnmapMemory(device, staging.memory);
        createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dst);
        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkBufferCopy copy{}; copy.size = size;
        vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &copy);
        endSingleTimeCommands(cmd);
        vkDestroyBuffer(device, staging.buffer, nullptr);
        vkFreeMemory(device, staging.memory, nullptr);
    };
    uploadBuffer(md.vertices.data(), md.vertices.size() * sizeof(ModelVertex),
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, lm.vbo);
    uploadBuffer(md.indices.data(), md.indices.size() * sizeof(uint32_t),
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT, lm.ibo);
    lm.indexCount = (uint32_t)md.indices.size();

    lm.boundsMin = md.boundsMin;
    lm.boundsMax = md.boundsMax;
    const glm::vec3 ext = md.boundsMax - md.boundsMin;
    const float longest = std::max(ext.x, std::max(ext.y, ext.z));
    lm.autoScale = (longest > 1e-4f) ? (std::max(0.01f, targetSize) / longest) : 1.0f;

    // Descriptor pool sized to material count.
    const uint32_t nmat = (uint32_t)std::max<size_t>(1, md.materials.size());
    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; poolSizes[0].descriptorCount = nmat * 4;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         poolSizes[1].descriptorCount = nmat;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = nmat;
    poolInfo.poolSizeCount = 2; poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &lm.descPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create model descriptor pool");

    lm.materials.resize(md.materials.size());
    for (size_t i = 0; i < md.materials.size(); ++i) {
        const ModelMaterial& m = md.materials[i];
        ModelMaterialGPU& g = lm.materials[i];

        uploadModelTexture(m.baseColorPath, true,  g.base,      g.ownBase);
        uploadModelTexture(m.metallicPath,  false, g.metal,     g.ownMetal);
        uploadModelTexture(m.roughnessPath, false, g.rough,     g.ownRough);
        uploadModelTexture(m.normalPath,    false, g.normalMap, g.ownNormal);

        // Material UBO: baseColorFactor + (metallic, roughness, hasNormal, hasMetalRough).
        struct MatUBO { glm::vec4 baseColorFactor; glm::vec4 params; } ubo;
        ubo.baseColorFactor = m.baseColorFactor;
        ubo.params = glm::vec4(m.metallicFactor, m.roughnessFactor,
                               g.ownNormal ? 1.0f : 0.0f,
                               (g.ownMetal || g.ownRough) ? 1.0f : 0.0f);
        createBuffer(sizeof(MatUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, g.ubo);
        void* data; vkMapMemory(device, g.ubo.memory, 0, sizeof(MatUBO), 0, &data);
        std::memcpy(data, &ubo, sizeof(MatUBO)); vkUnmapMemory(device, g.ubo.memory);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = lm.descPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &modelDescLayout;
        if (vkAllocateDescriptorSets(device, &allocInfo, &g.descriptor) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate model descriptor set");

        VkImageView views[4] = {
            g.ownBase   ? g.base.view      : defaultWhiteTex.view,
            g.ownMetal  ? g.metal.view     : defaultWhiteTex.view,
            g.ownRough  ? g.rough.view     : defaultWhiteTex.view,
            g.ownNormal ? g.normalMap.view : defaultNormalTex.view,
        };
        VkDescriptorImageInfo imgInfos[4]{};
        VkWriteDescriptorSet writes[5]{};
        for (int b = 0; b < 4; ++b) {
            imgInfos[b].sampler = modelSampler;
            imgInfos[b].imageView = views[b];
            imgInfos[b].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = g.descriptor;
            writes[b].dstBinding = (uint32_t)b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[b].pImageInfo = &imgInfos[b];
        }
        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = g.ubo.buffer; bufInfo.offset = 0; bufInfo.range = sizeof(MatUBO);
        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = g.descriptor;
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[4].pBufferInfo = &bufInfo;
        vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
    }

    lm.submeshes.clear();
    for (const ModelSubmesh& s : md.submeshes)
        lm.submeshes.push_back({s.indexOffset, s.indexCount, s.materialIndex});

    lm.valid = true;
    return lm;
}

VulkanEngine::LoadedModel VulkanEngine::loadModelAssetFromFBX(const std::string& fbxPath, float targetSize) {
    // Resolve path: relative to CWD first, then the source tree.
    std::string path = fbxPath;
    { std::ifstream f(path, std::ios::binary);
      if (!f.good()) {
          std::string alt = std::string(TE_SOURCE_DIR) + "/" + fbxPath;
          std::ifstream f2(alt, std::ios::binary);
          if (f2.good()) path = alt;
      }
    }
    ModelData md = ModelLoader::loadFBX(path);
    if (!md.valid) {
        std::cout << "  [model] FBX NOT loaded: " << fbxPath << "\n";
        return LoadedModel{};
    }
    return loadModelAsset(md, targetSize);
}

bool VulkanEngine::loadAircraftModel(const std::string& fbxPath) {
    destroyLoadedModel(aircraftAsset);
    const float fuselageLen = 2.0f * std::max(0.5f, settings.aircraftHalfLength);
    aircraftAsset = loadModelAssetFromFBX(fbxPath, fuselageLen);
    if (!aircraftAsset.valid) {
        std::cout << "  [model] aircraft model NOT loaded; using debug box\n";
        return false;
    }
    std::cout << "  [model] aircraft model READY: autoScale=" << aircraftAsset.autoScale
              << " submeshes=" << aircraftAsset.submeshes.size() << "\n";
    return true;
}

void VulkanEngine::buildAssetRegistry() {
    for (SceneAsset& a : sceneAssets)
        destroyLoadedModel(a.gpu);
    sceneAssets.clear();

    // Built-in primitives first so the editor is usable with no FBX present.
    sceneAssets.push_back({"Box",    loadModelAsset(makeBoxModel(),    2.0f)});
    sceneAssets.push_back({"Sphere", loadModelAsset(makeSphereModel(), 2.0f)});

    // Discover every *.fbx under models/ (CWD first, then the source tree).
    namespace fs = std::filesystem;
    std::vector<fs::path> dirs;
    dirs.emplace_back("models");
    dirs.emplace_back(fs::path(TE_SOURCE_DIR) / "models");
    std::error_code ec;
    for (const fs::path& dir : dirs) {
        if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) continue;
        for (const fs::directory_entry& e : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            if (!e.is_regular_file()) continue;
            std::string ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c){ return (char)std::tolower(c); });
            if (ext != ".fbx") continue;
            LoadedModel lm = loadModelAssetFromFBX(e.path().string(), 8.0f);
            if (lm.valid)
                sceneAssets.push_back({e.path().stem().string(), std::move(lm)});
        }
        break;  // only the first existing models dir
    }
    std::cout << "  [editor] asset registry: " << sceneAssets.size() << " assets\n";
}

void VulkanEngine::destroyLoadedModel(LoadedModel& m) {
    auto destroyImage = [&](Image& im) {
        if (im.view)   vkDestroyImageView(device, im.view, nullptr);
        if (im.image)  vkDestroyImage(device, im.image, nullptr);
        if (im.memory) vkFreeMemory(device, im.memory, nullptr);
        im = Image{};
    };
    for (ModelMaterialGPU& g : m.materials) {
        if (g.ownBase)   destroyImage(g.base);
        if (g.ownMetal)  destroyImage(g.metal);
        if (g.ownRough)  destroyImage(g.rough);
        if (g.ownNormal) destroyImage(g.normalMap);
        if (g.ubo.buffer) vkDestroyBuffer(device, g.ubo.buffer, nullptr);
        if (g.ubo.memory) vkFreeMemory(device, g.ubo.memory, nullptr);
    }
    m.materials.clear();
    m.submeshes.clear();
    if (m.descPool) { vkDestroyDescriptorPool(device, m.descPool, nullptr); m.descPool = VK_NULL_HANDLE; }
    if (m.vbo.buffer) { vkDestroyBuffer(device, m.vbo.buffer, nullptr); vkFreeMemory(device, m.vbo.memory, nullptr); m.vbo = Buffer{}; }
    if (m.ibo.buffer) { vkDestroyBuffer(device, m.ibo.buffer, nullptr); vkFreeMemory(device, m.ibo.memory, nullptr); m.ibo = Buffer{}; }
    m.indexCount = 0;
    m.valid = false;
}

void VulkanEngine::destroyModelResources() {
    auto destroyImage = [&](Image& im) {
        if (im.view)   vkDestroyImageView(device, im.view, nullptr);
        if (im.image)  vkDestroyImage(device, im.image, nullptr);
        if (im.memory) vkFreeMemory(device, im.memory, nullptr);
        im = Image{};
    };
    destroyLoadedModel(aircraftAsset);
    for (SceneAsset& a : sceneAssets)
        destroyLoadedModel(a.gpu);
    sceneAssets.clear();

    destroyImage(defaultWhiteTex);
    destroyImage(defaultNormalTex);
    if (modelSampler)         { vkDestroySampler(device, modelSampler, nullptr); modelSampler = VK_NULL_HANDLE; }
    if (modelPipeline)        { vkDestroyPipeline(device, modelPipeline, nullptr); modelPipeline = VK_NULL_HANDLE; }
    if (modelPipelineLayout)  { vkDestroyPipelineLayout(device, modelPipelineLayout, nullptr); modelPipelineLayout = VK_NULL_HANDLE; }
    if (modelDescLayout)      { vkDestroyDescriptorSetLayout(device, modelDescLayout, nullptr); modelDescLayout = VK_NULL_HANDLE; }
}

// ══════════════════════════════════════════════════════════════
//  Scene editor — placement, transforms, picking, persistence
// ══════════════════════════════════════════════════════════════

int VulkanEngine::assetIndexByName(const std::string& name) const {
    for (size_t i = 0; i < sceneAssets.size(); ++i)
        if (sceneAssets[i].name == name) return (int)i;
    return -1;
}

glm::mat4 VulkanEngine::sceneObjectGizmoMatrix(const SceneObject& o) const {
    glm::mat4 T = glm::translate(glm::mat4(1.0f), o.position);
    glm::mat4 R = glm::mat4_cast(glm::quat(glm::radians(o.eulerDeg)));
    glm::mat4 S = glm::scale(glm::mat4(1.0f), o.scale);
    return T * R * S;
}

glm::mat4 VulkanEngine::sceneObjectMatrix(const SceneObject& o) const {
    glm::vec3 center(0.0f);
    float autoS = 1.0f;
    if (o.assetIndex >= 0 && o.assetIndex < (int)sceneAssets.size()) {
        const LoadedModel& m = sceneAssets[o.assetIndex].gpu;
        center = 0.5f * (m.boundsMin + m.boundsMax);
        autoS = m.autoScale;
    }
    return sceneObjectGizmoMatrix(o)
         * glm::scale(glm::mat4(1.0f), glm::vec3(autoS))
         * glm::translate(glm::mat4(1.0f), -center);
}

void VulkanEngine::decomposeGizmoMatrix(const glm::mat4& m, SceneObject& o) const {
    o.position = glm::vec3(m[3]);
    glm::vec3 c0(m[0]), c1(m[1]), c2(m[2]);
    glm::vec3 s(glm::length(c0), glm::length(c1), glm::length(c2));
    o.scale = s;
    glm::mat3 rot(
        s.x > 1e-6f ? c0 / s.x : glm::vec3(1, 0, 0),
        s.y > 1e-6f ? c1 / s.y : glm::vec3(0, 1, 0),
        s.z > 1e-6f ? c2 / s.z : glm::vec3(0, 0, 1));
    glm::quat q = glm::quat_cast(rot);
    o.eulerDeg = glm::degrees(glm::eulerAngles(q));
}

void VulkanEngine::drawSceneObjects(VkCommandBuffer cmd, const glm::mat4& proj, const glm::mat4& view) {
    if (editorScene.objects.empty() || modelPipeline == VK_NULL_HANDLE) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, modelPipeline);
    struct Push { glm::mat4 mvp; glm::mat4 model; } push;
    for (const SceneObject& o : editorScene.objects) {
        if (o.assetIndex < 0 || o.assetIndex >= (int)sceneAssets.size()) continue;
        const LoadedModel& lm = sceneAssets[o.assetIndex].gpu;
        if (!lm.valid || lm.indexCount == 0) continue;
        glm::mat4 model = sceneObjectMatrix(o);
        push.mvp = proj * view * model;
        push.model = model;
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &lm.vbo.buffer, &off);
        vkCmdBindIndexBuffer(cmd, lm.ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdPushConstants(cmd, modelPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
            0, sizeof(push), &push);
        for (const ModelSubmeshGPU& sm : lm.submeshes) {
            if (sm.materialIndex < 0 || sm.materialIndex >= (int)lm.materials.size()) continue;
            VkDescriptorSet ds = lm.materials[sm.materialIndex].descriptor;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                modelPipelineLayout, 0, 1, &ds, 0, nullptr);
            vkCmdDrawIndexed(cmd, sm.indexCount, 1, sm.indexOffset, 0, 0);
        }
    }
}

void VulkanEngine::captureSceneWorldParams() {
    editorScene.world.captureFrom(settings);
}

void VulkanEngine::applySceneWorldParams() {
    editorScene.world.applyTo(settings);
    regenerateTerrain();
}

void VulkanEngine::addSceneObject(int assetIndex) {
    if (assetIndex < 0 || assetIndex >= (int)sceneAssets.size()) return;
    SceneObject o;
    o.name = sceneAssets[assetIndex].name;
    o.assetIndex = assetIndex;
    o.assetName = sceneAssets[assetIndex].name;
    // Place at the camera's orbit focus, lifted onto the terrain.
    const glm::vec2 c(camera.pos.x, camera.pos.z);
    o.position = glm::vec3(c.x,
        sampleWorldHeight(c.x, c.y) * settings.terrainHeightScale + 1.0f, c.y);
    editorScene.objects.push_back(o);
    selectedObject = (int)editorScene.objects.size() - 1;
}

void VulkanEngine::duplicateSceneObject(int index) {
    if (index < 0 || index >= (int)editorScene.objects.size()) return;
    SceneObject o = editorScene.objects[index];
    o.position += glm::vec3(2.0f, 0.0f, 2.0f);
    o.name += " copy";
    editorScene.objects.push_back(o);
    selectedObject = (int)editorScene.objects.size() - 1;
}

void VulkanEngine::deleteSceneObject(int index) {
    if (index < 0 || index >= (int)editorScene.objects.size()) return;
    editorScene.objects.erase(editorScene.objects.begin() + index);
    if (selectedObject == index) selectedObject = -1;
    else if (selectedObject > index) --selectedObject;
}

void VulkanEngine::saveSceneToFile(const std::string& path) {
    captureSceneWorldParams();
    for (SceneObject& o : editorScene.objects)
        if (o.assetIndex >= 0 && o.assetIndex < (int)sceneAssets.size())
            o.assetName = sceneAssets[o.assetIndex].name;
    editorScene.saveToJSON(path);
}

void VulkanEngine::loadSceneFromFile(const std::string& path) {
    Scene s;
    if (!s.loadFromJSON(path)) return;
    editorScene = s;
    // Re-resolve asset references by name (ordering may differ between runs).
    for (SceneObject& o : editorScene.objects) {
        int idx = assetIndexByName(o.assetName);
        if (idx >= 0) o.assetIndex = idx;
        else o.assetIndex = glm::clamp(o.assetIndex, 0,
                 std::max(0, (int)sceneAssets.size() - 1));
    }
    selectedObject = -1;
    applySceneWorldParams();   // rebuild terrain to match the saved world
}

void VulkanEngine::pickSceneObject(float mouseX, float mouseY) {
    if (swapchainExtent.width == 0 || swapchainExtent.height == 0) return;
    const float w = (float)swapchainExtent.width;
    const float h = (float)swapchainExtent.height;
    // GL-style NDC (lastProj/lastView are the editor's non-Y-flipped matrices).
    const float ndcX = 2.0f * mouseX / w - 1.0f;
    const float ndcY = 1.0f - 2.0f * mouseY / h;

    glm::mat4 invVP = glm::inverse(lastProj * lastView);
    glm::vec4 pNear = invVP * glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 pFar  = invVP * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
    if (std::abs(pNear.w) < 1e-6f || std::abs(pFar.w) < 1e-6f) return;
    pNear /= pNear.w; pFar /= pFar.w;
    const glm::vec3 ro = camera.eyePos();
    const glm::vec3 rd = glm::normalize(glm::vec3(pFar) - glm::vec3(pNear));

    int best = -1;
    float bestT = 1e30f;
    for (size_t i = 0; i < editorScene.objects.size(); ++i) {
        const SceneObject& o = editorScene.objects[i];
        if (o.assetIndex < 0 || o.assetIndex >= (int)sceneAssets.size()) continue;
        const LoadedModel& lm = sceneAssets[o.assetIndex].gpu;
        const glm::mat4 model = sceneObjectMatrix(o);

        // World-space AABB of the asset's local bounds under this transform.
        glm::vec3 mn(1e30f), mx(-1e30f);
        for (int c = 0; c < 8; ++c) {
            glm::vec3 corner(
                (c & 1) ? lm.boundsMax.x : lm.boundsMin.x,
                (c & 2) ? lm.boundsMax.y : lm.boundsMin.y,
                (c & 4) ? lm.boundsMax.z : lm.boundsMin.z);
            glm::vec3 wp = glm::vec3(model * glm::vec4(corner, 1.0f));
            mn = glm::min(mn, wp);
            mx = glm::max(mx, wp);
        }

        // Slab ray/AABB intersection.
        float t0 = 0.0f, t1 = 1e30f;
        bool hit = true;
        for (int a = 0; a < 3; ++a) {
            float inv = 1.0f / (std::abs(rd[a]) > 1e-9f ? rd[a] : 1e-9f);
            float ta = (mn[a] - ro[a]) * inv;
            float tb = (mx[a] - ro[a]) * inv;
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta);
            t1 = std::min(t1, tb);
            if (t0 > t1) { hit = false; break; }
        }
        if (hit && t0 < bestT) { bestT = t0; best = (int)i; }
    }
    selectedObject = best;
}

void VulkanEngine::drawEditorUI() {
    ImGuiIO& io = ImGui::GetIO();
    const float w = io.DisplaySize.x;
    const float h = io.DisplaySize.y;

    // The editor uses a standard (non-Y-flipped) projection so ImGuizmo's screen
    // mapping and our picking line up with the rendered image (the Vulkan flip
    // cancels with the framebuffer orientation on screen).
    glm::mat4 proj = glm::perspective(glm::radians(60.0f),
        (h > 0.0f ? w / h : 1.0f), 0.1f, settings.effectiveFarPlane());
    glm::mat4 view = camera.view();
    lastProj = proj;
    lastView = view;

    // ── Toolbar ──
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Editor")) {
        if (ImGui::Button("New")) { editorScene.objects.clear(); selectedObject = -1; }
        ImGui::SameLine();
        if (ImGui::Button("Save")) saveSceneToFile(scenePathBuf);
        ImGui::SameLine();
        if (ImGui::Button("Load")) loadSceneFromFile(scenePathBuf);
        ImGui::InputText("Scene file", scenePathBuf, sizeof(scenePathBuf));

        ImGui::Separator();
        bool em = editMode;
        if (ImGui::Checkbox("Edit mode (pause flight)", &em)) {
            editMode = em;
            if (editMode) captureSceneWorldParams();
        }

        ImGui::Text("Gizmo:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Move", gizmoOperation == ImGuizmo::TRANSLATE)) gizmoOperation = ImGuizmo::TRANSLATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate", gizmoOperation == ImGuizmo::ROTATE)) gizmoOperation = ImGuizmo::ROTATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Scale", gizmoOperation == ImGuizmo::SCALE)) gizmoOperation = ImGuizmo::SCALE;
        if (ImGui::RadioButton("World", gizmoModeLocal == ImGuizmo::WORLD)) gizmoModeLocal = ImGuizmo::WORLD;
        ImGui::SameLine();
        if (ImGui::RadioButton("Local", gizmoModeLocal == ImGuizmo::LOCAL)) gizmoModeLocal = ImGuizmo::LOCAL;

        ImGui::Separator();
        static int addAsset = 0;
        if (!sceneAssets.empty()) {
            addAsset = glm::clamp(addAsset, 0, (int)sceneAssets.size() - 1);
            if (ImGui::BeginCombo("Asset", sceneAssets[addAsset].name.c_str())) {
                for (int i = 0; i < (int)sceneAssets.size(); ++i)
                    if (ImGui::Selectable(sceneAssets[i].name.c_str(), i == addAsset)) addAsset = i;
                ImGui::EndCombo();
            }
            ImGui::SameLine();
            if (ImGui::Button("Add")) addSceneObject(addAsset);
        } else {
            ImGui::TextDisabled("No assets available");
        }
        ImGui::Text("%d object(s)", (int)editorScene.objects.size());
    }
    ImGui::End();

    // ── Hierarchy + Inspector ──
    ImGui::SetNextWindowPos(ImVec2(10, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330, 360), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Hierarchy")) {
        for (int i = 0; i < (int)editorScene.objects.size(); ++i) {
            ImGui::PushID(i);
            std::string label = editorScene.objects[i].name + "##" + std::to_string(i);
            if (ImGui::Selectable(label.c_str(), i == selectedObject)) selectedObject = i;
            ImGui::PopID();
        }
        ImGui::Separator();
        if (selectedObject >= 0 && selectedObject < (int)editorScene.objects.size()) {
            SceneObject& o = editorScene.objects[selectedObject];
            char nameBuf[128];
            std::snprintf(nameBuf, sizeof(nameBuf), "%s", o.name.c_str());
            if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf))) o.name = nameBuf;
            if (!sceneAssets.empty()) {
                int ai = glm::clamp(o.assetIndex, 0, (int)sceneAssets.size() - 1);
                if (ImGui::BeginCombo("Mesh", sceneAssets[ai].name.c_str())) {
                    for (int i = 0; i < (int)sceneAssets.size(); ++i)
                        if (ImGui::Selectable(sceneAssets[i].name.c_str(), i == ai)) {
                            o.assetIndex = i; o.assetName = sceneAssets[i].name;
                        }
                    ImGui::EndCombo();
                }
            }
            ImGui::DragFloat3("Position", &o.position.x, 0.1f);
            ImGui::DragFloat3("Rotation", &o.eulerDeg.x, 0.5f);
            ImGui::DragFloat3("Scale", &o.scale.x, 0.01f, 0.001f, 1000.0f);
            if (ImGui::Button("Duplicate")) duplicateSceneObject(selectedObject);
            ImGui::SameLine();
            if (ImGui::Button("Delete")) deleteSceneObject(selectedObject);
        } else {
            ImGui::TextDisabled("No object selected");
        }
    }
    ImGui::End();

    // ── World params ──
    ImGui::SetNextWindowPos(ImVec2(w - 340, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("World")) {
        WorldParams& wp = editorScene.world;
        int seed = (int)wp.worldSeed;
        if (ImGui::InputInt("Seed", &seed)) wp.worldSeed = (uint32_t)std::max(0, seed);
        ImGui::DragFloat("World size", &wp.worldSize, 10.0f, 500.0f, 20000.0f, "%.0f");
        int mres = (int)wp.masterRes;
        if (ImGui::InputInt("Master res", &mres)) wp.masterRes = (uint32_t)glm::clamp(mres, 64, 4096);
        ImGui::SliderFloat("Sea fraction", &wp.seaFraction, 0.0f, 1.0f);
        ImGui::DragFloat("Total height", &wp.terrainTotalHeight, 1.0f, 1.0f, 1000.0f);
        ImGui::SliderFloat("Ridge weight", &wp.mountainRidgeWeight, 0.0f, 1.0f);
        ImGui::SliderFloat("Sun elevation", &wp.sunElevation, 0.0f, 90.0f);
        ImGui::SliderFloat("Sun azimuth", &wp.sunAzimuth, 0.0f, 360.0f);
        if (ImGui::Button("Capture current")) captureSceneWorldParams();
        ImGui::SameLine();
        if (ImGui::Button("Apply (regen)")) applySceneWorldParams();
    }
    ImGui::End();

    // ── Gizmo manipulation + click-select (edit mode only) ──
    if (editMode) {
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetRect(0, 0, w, h);
        if (selectedObject >= 0 && selectedObject < (int)editorScene.objects.size()) {
            SceneObject& o = editorScene.objects[selectedObject];
            glm::mat4 m = sceneObjectGizmoMatrix(o);
            ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj),
                (ImGuizmo::OPERATION)gizmoOperation, (ImGuizmo::MODE)gizmoModeLocal,
                glm::value_ptr(m));
            if (ImGuizmo::IsUsing())
                decomposeGizmoMatrix(m, o);
        }
        if (ImGui::IsMouseClicked(0) && !io.WantCaptureMouse &&
            !ImGuizmo::IsOver() && !ImGuizmo::IsUsing())
            pickSceneObject(io.MousePos.x, io.MousePos.y);
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Command pool / buffers
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::createCommandPool() {
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = graphicsFamily;
    if (vkCreateCommandPool(device, &info, nullptr, &commandPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create command pool");
}

void VulkanEngine::createCommandBuffers() {
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate command buffers");
}

void VulkanEngine::recordRenderCommand(VkCommandBuffer cmd, uint32_t imageIndex) {
    static int recordCallCount = 0;
    recordCallCount++;
    if (recordCallCount <= 2) {
        std::cout << "    [record] ENTER recordRenderCommand (#" << recordCallCount << ")" << std::endl << std::flush;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin command buffer");

    if (recordCallCount <= 2) std::cout << "    [record] step A: beginCommandBuffer OK" << std::endl << std::flush;

    // GPU profiler: reset this frame slot's timestamps before any writes.
    const uint32_t tsBase = currentFrame * GPU_TS_PER_FRAME;
    auto writeTs = [&](uint32_t localIdx) {
        if (gpuProfilingEnabled)
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 gpuQueryPool, tsBase + localIdx);
    };
    if (gpuProfilingEnabled) {
        vkCmdResetQueryPool(cmd, gpuQueryPool, tsBase, GPU_TS_PER_FRAME);
        gpuSlotWritten[currentFrame] = true;
    }

    // Transition swapchain image for color attachment
    VkImageMemoryBarrier2 colorBarrier{};
    colorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    colorBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    colorBarrier.srcAccessMask = VK_ACCESS_2_NONE;
    colorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    colorBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    colorBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorBarrier.image = swapchainImages[imageIndex];
    colorBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorBarrier.subresourceRange.levelCount = 1;
    colorBarrier.subresourceRange.layerCount = 1;

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &colorBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    writeTs(0);  // frame start

    if (recordCallCount <= 2) std::cout << "    [record] step B: color barrier OK, dispatching skyview LUT..." << std::endl << std::flush;

    const bool updateSkyview = needsSkyviewUpdate();
    if (updateSkyview) {
        dispatchSkyviewLUT(cmd);
        lastSkyviewSunDir = atmosphereParams.sun_direction;
        lastSkyviewCamAlt = camera.eyePos().y;
        skyviewValid = true;
    }

    // Transition skyview LUT for fragment shader sampling.
    {
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = updateSkyview ? VK_ACCESS_2_SHADER_WRITE_BIT : VK_ACCESS_2_NONE;
        b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.image = skyviewLUT.image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        VkDependencyInfo skyDep{};
        skyDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        skyDep.imageMemoryBarrierCount = 1;
        skyDep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &skyDep);
    }

    writeTs(1);  // after skyview LUT

    dispatchCloudShadowMap(cmd);

    writeTs(2);  // after cloud-shadow bake

    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    //  Shadow pass: render terrain to shadow map from light POV
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    {
        // Transition shadow map: DEPTH_READ_ONLY â†’ DEPTH_ATTACHMENT
        VkImageMemoryBarrier2 shadowPreBarrier{};
        shadowPreBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        shadowPreBarrier.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        shadowPreBarrier.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        shadowPreBarrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        shadowPreBarrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        shadowPreBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        shadowPreBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        shadowPreBarrier.image = shadowMap.image;
        shadowPreBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        shadowPreBarrier.subresourceRange.levelCount = 1;
        shadowPreBarrier.subresourceRange.layerCount = 1;

        VkDependencyInfo shadowPreDep{};
        shadowPreDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        shadowPreDep.imageMemoryBarrierCount = 1;
        shadowPreDep.pImageMemoryBarriers = &shadowPreBarrier;
        vkCmdPipelineBarrier2(cmd, &shadowPreDep);

        if (recordCallCount <= 2) std::cout << "    [record] step C: shadow barrier OK, begin shadow rendering..." << std::endl << std::flush;

        // Dynamic rendering â€” depth only
        VkRenderingAttachmentInfo shadowDepthAttach{};
        shadowDepthAttach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        shadowDepthAttach.imageView = shadowMap.view;
        shadowDepthAttach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        shadowDepthAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        shadowDepthAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        shadowDepthAttach.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo shadowRenderInfo{};
        shadowRenderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        shadowRenderInfo.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
        shadowRenderInfo.layerCount = 1;
        shadowRenderInfo.pDepthAttachment = &shadowDepthAttach;

        vkCmdBeginRendering(cmd, &shadowRenderInfo);

        if (recordCallCount <= 2) std::cout << "    [record] step D: shadow beginRendering OK, drawing shadow terrain..." << std::endl << std::flush;

        VkViewport shadowViewport{};
        shadowViewport.x = 0; shadowViewport.y = 0;
        shadowViewport.width = (float)SHADOW_MAP_SIZE;
        shadowViewport.height = (float)SHADOW_MAP_SIZE;
        shadowViewport.minDepth = 0.0f; shadowViewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &shadowViewport);

        VkRect2D shadowScissor{};
        shadowScissor.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
        vkCmdSetScissor(cmd, 0, 1, &shadowScissor);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);

        VkDeviceSize vOffset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer.buffer, &vOffset);
        vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        // Reuse the exact light matrix computed in updateUniformBuffer()
        // (and uploaded to the shadow UBO) so the depth render, terrain
        // shadow lookup, and culling frustum can never diverge.
        Frustum lightFrustum = extractFrustum(currentLightMVP);
        (void)lightFrustum;  // regions are drawn unconditionally for now

        const float sMinY = globalMinHeight * settings.terrainHeightScale - 10.0f;
        const float sMaxY = globalMaxHeight * settings.terrainHeightScale + 10.0f;

        if (shadowDescriptorSet != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                shadowPipelineLayout, 0, 1, &shadowDescriptorSet, 0, nullptr);

            ChunkPush push{};
            push.worldOrigin = glm::vec2(0.0f);
            push.worldSize = settings.worldSize;
            push.heightScale = settings.terrainHeightScale;
            push.heightmapRes = (int)settings.masterRes;
            push.debugOverlayMode = 0;
            push.globalMinHeight = 0.0f;
            push.globalMaxHeight = 1.0f;
            vkCmdPushConstants(cmd, shadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                0, sizeof(ChunkPush), &push);
            // Shadow casters are low-frequency: a coarse mesh produces an
            // essentially identical 2048² depth map with far fewer triangles.
            // Never render the shadow pass finer than the configured floor
            // (and never finer than the view LOD when the camera is far).
            const int shadowFloor = std::clamp(settings.terrainShadowMinLod, 0, MAX_LODS - 1);
            const int terrainLod = std::max(selectTerrainLOD(), shadowFloor);
            vkCmdDrawIndexed(cmd, lods[terrainLod].indexCount, 1,
                lods[terrainLod].firstIndex, 0, 0);
        }

        if (recordCallCount <= 2) std::cout << "    [record] step E: shadow draw OK, ending shadow rendering..." << std::endl << std::flush;

        vkCmdEndRendering(cmd);

        // Transition shadow map back: DEPTH_ATTACHMENT â†’ DEPTH_READ_ONLY
        VkImageMemoryBarrier2 shadowPostBarrier{};
        shadowPostBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        shadowPostBarrier.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        shadowPostBarrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        shadowPostBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        shadowPostBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        shadowPostBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        shadowPostBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        shadowPostBarrier.image = shadowMap.image;
        shadowPostBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        shadowPostBarrier.subresourceRange.levelCount = 1;
        shadowPostBarrier.subresourceRange.layerCount = 1;

        VkDependencyInfo shadowPostDep{};
        shadowPostDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        shadowPostDep.imageMemoryBarrierCount = 1;
        shadowPostDep.pImageMemoryBarriers = &shadowPostBarrier;
        vkCmdPipelineBarrier2(cmd, &shadowPostDep);
    }

    writeTs(3);  // after shadow pass

    if (recordCallCount <= 2) std::cout << "    [record] step F: shadow pass complete, starting main rendering..." << std::endl << std::flush;

    // Dynamic rendering
    VkRenderingAttachmentInfo colorAttach{};
    colorAttach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttach.imageView = swapchainViews[imageIndex];
    colorAttach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttach.clearValue.color = {{0.1f, 0.1f, 0.15f, 1.0f}};  // dark blue-gray for troubleshooting

    VkRenderingAttachmentInfo depthAttach{};
    depthAttach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttach.imageView = depth.view;
    depthAttach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttach.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.extent = swapchainExtent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttach;
    renderingInfo.pDepthAttachment = &depthAttach;

    // Airflow particle advection runs as a compute pass; it must be issued
    // OUTSIDE the dynamic-rendering scope (Vulkan forbids dispatch inside
    // vkCmdBeginRendering…EndRendering).
    recordAeroCompute(cmd);

    vkCmdBeginRendering(cmd, &renderingInfo);

    if (recordCallCount <= 2) std::cout << "    [record] step G: main beginRendering OK" << std::endl << std::flush;

    // Viewport + scissor
    VkViewport viewport{};
    viewport.x = 0; viewport.y = 0;
    viewport.width = (float)swapchainExtent.width;
    viewport.height = (float)swapchainExtent.height;
    viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = swapchainExtent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind pipeline + shared mesh once
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, terrainPipeline);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    // Build frustum for culling
    glm::mat4 proj = glm::perspective(glm::radians(60.0f),
        swapchainExtent.width / (float)swapchainExtent.height, 0.1f, settings.effectiveFarPlane());
    proj[1][1] *= -1;  // Vulkan Y-flip
    Frustum frustum = extractFrustum(proj * camera.view());

    // Height bounds from actual terrain data Ã— heightScale
    const float estimatedMinY = globalMinHeight * settings.terrainHeightScale - 10.0f;
    const float estimatedMaxY = globalMaxHeight * settings.terrainHeightScale + 10.0f;

    if (globalTerrainDescSets[currentFrame] != VK_NULL_HANDLE) {
        const VkDescriptorSet terrainSet = globalTerrainDescSets[currentFrame];
        glm::vec3 minB(0.0f, estimatedMinY, 0.0f);
        glm::vec3 maxB(settings.worldSize, estimatedMaxY, settings.worldSize);
        if (isAabbInFrustum(frustum, minB, maxB)) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                terrainPipelineLayout, 0, 1, &terrainSet, 0, nullptr);

            ChunkPush push{};
            push.worldOrigin = glm::vec2(0.0f);
            push.worldSize = settings.worldSize;
            push.heightScale = settings.terrainHeightScale;
            push.heightmapRes = (int)settings.masterRes;
            push.debugOverlayMode = (int)debugOverlayMode;
            push.globalMinHeight = globalMinHeight;
            push.globalMaxHeight = globalMaxHeight;
            vkCmdPushConstants(cmd, terrainPipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(ChunkPush), &push);

            // View mesh LOD floor: one vertex per heightmap texel (LOD 0) is
            // far finer than the heightmap or screen can resolve and is the
            // frame's dominant (vertex-bound) cost. The floor trades that
            // invisible geometry for a large GPU saving.
            const int viewFloor = std::clamp(settings.terrainViewMinLod, 0, MAX_LODS - 1);
            const int terrainLod = std::max(selectTerrainLOD(), viewFloor);
            vkCmdDrawIndexed(cmd, lods[terrainLod].indexCount, 1,
                lods[terrainLod].firstIndex, 0, 0);
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  Ocean pass — animated Gerstner surface at the waterline.
    //  Shares the terrain set-0 descriptor; depth LESS_OR_EQUAL so it
    //  meets the terrain flush at the coast and is occluded by land.
    // ═══════════════════════════════════════════════════════════
    if (oceanPipeline != VK_NULL_HANDLE && globalTerrainDescSets[currentFrame] != VK_NULL_HANDLE) {
        const VkDescriptorSet terrainSet = globalTerrainDescSets[currentFrame];
        // Pull live wave tuning from settings (ImGui can change these).
        oceanPush.worldSize    = settings.worldSize;
        oceanPush.heightScale  = settings.terrainHeightScale;
        oceanPush.heightmapRes = (int)settings.masterRes;
        oceanPush.waveAmplitude = settings.waveAmplitude;
        oceanPush.waveSpeed     = settings.waveSpeed;
        oceanPush.shoreDepth    = settings.shoreDepth;
        oceanPush.foamDepth     = settings.foamDepth;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, oceanPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            oceanPipelineLayout, 0, 1, &terrainSet, 0, nullptr);
        vkCmdPushConstants(cmd, oceanPipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(OceanPush), &oceanPush);
        VkBuffer oceanVBufs[] = { oceanVertexBuffer.buffer };
        VkDeviceSize oceanOffsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, oceanVBufs, oceanOffsets);
        vkCmdBindIndexBuffer(cmd, oceanIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, oceanIndexCount, 1, 0, 0, 0);
    }

    if (recordCallCount <= 2) std::cout << "    [record] step H: terrain draw OK, drawing sky..." << std::endl << std::flush;

    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    //  Sky pass â€” full-screen triangle that fills depth==1.0 pixels
    //  with the skyview LUT colour (sampled per view direction).
    //  Depth test is LESS_OR_EQUAL with the vertex shader emitting
    //  depth=1.0, so terrain fragments (depth<1.0) always win.
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPassPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            skyPassPipelineLayout, 0, 1, &skyPassSet, 0, nullptr);

        struct SkyPush {
            glm::mat4 invViewProj;
            glm::vec4 cameraPos;
        } skyPush;
        glm::mat4 projSky = glm::perspective(glm::radians(60.0f),
            swapchainExtent.width / (float)swapchainExtent.height, 0.1f, settings.effectiveFarPlane());
        projSky[1][1] *= -1;
        skyPush.invViewProj = glm::inverse(projSky * camera.view());
        skyPush.cameraPos = glm::vec4(camera.eyePos(), 1.0f);

        vkCmdPushConstants(cmd, skyPassPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(skyPush), &skyPush);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    // ── Airport paved surfaces (concrete + runway/taxiway/apron markings) ──
    if (airportPipeline != VK_NULL_HANDLE && airportIndexCount > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, airportPipeline);
        struct ApPush { glm::mat4 mvp; glm::mat4 model; } app;
        app.model = glm::mat4(1.0f);   // geometry already in world space
        app.mvp = proj * camera.view();
        vkCmdPushConstants(cmd, airportPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
            0, sizeof(app), &app);
        VkDeviceSize aoff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &airportVertexBuffer.buffer, &aoff);
        vkCmdBindIndexBuffer(cmd, airportIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, airportIndexCount, 1, 0, 0, 0);
    }

    // â”€â”€ Foliage pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if (globalFoliageCount > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, foliagePipeline);
        // NOTE: foliage pipeline layout has no descriptor sets (setLayoutCount=0),
        // only push constants for MVP. Binding an incompatible pipeline above
        // invalidates push constants, so the view-proj must be pushed here.
        glm::mat4 foliageMvp = proj * camera.view();
        vkCmdPushConstants(cmd, foliagePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
            0, sizeof(glm::mat4), &foliageMvp);

        VkDeviceSize fOff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &foliageVertexBuffer.buffer, &fOff);
        vkCmdBindIndexBuffer(cmd, foliageIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindVertexBuffers(cmd, 1, 1, &globalFoliageInstanceBuffer.buffer, &fOff);
        vkCmdBindVertexBuffers(cmd, 2, 1, &globalFoliageColorBuffer.buffer, &fOff);

        vkCmdDrawIndexed(cmd, foliageIndexCount, globalFoliageCount, 0, 0, 0);
    }

    // ── Aircraft pass: textured FBX model if loaded, else debug box ──
    {
        AircraftRenderState rs = aircraftRenderState();
        struct AircraftPush { glm::mat4 mvp; glm::mat4 model; } apush;

        const bool useModel = aircraftAsset.valid && settings.aircraftUseModel && aircraftAsset.indexCount > 0;
        if (rs.visible && useModel) {
            // Orientation offset (align model nose to +X) + auto-fit scale,
            // centering the model's bounding box on the physics body.
            const glm::vec3 center = 0.5f * (aircraftAsset.boundsMin + aircraftAsset.boundsMax);
            const float scl = aircraftAsset.autoScale * std::max(0.01f, settings.aircraftModelScale);
            glm::mat4 offset =
                glm::rotate(glm::mat4(1.0f), glm::radians(settings.aircraftModelYawOffsetDeg),   glm::vec3(0,1,0)) *
                glm::rotate(glm::mat4(1.0f), glm::radians(settings.aircraftModelPitchOffsetDeg), glm::vec3(0,0,1)) *
                glm::rotate(glm::mat4(1.0f), glm::radians(settings.aircraftModelRollOffsetDeg),  glm::vec3(1,0,0));
            glm::mat4 model = glm::translate(glm::mat4(1.0f), rs.position)
                            * glm::mat4_cast(rs.rotation)
                            * offset
                            * glm::scale(glm::mat4(1.0f), glm::vec3(scl))
                            * glm::translate(glm::mat4(1.0f), -center);
            apush.mvp = proj * camera.view() * model;
            apush.model = model;

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, modelPipeline);
            VkDeviceSize mOff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &aircraftAsset.vbo.buffer, &mOff);
            vkCmdBindIndexBuffer(cmd, aircraftAsset.ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdPushConstants(cmd, modelPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                0, sizeof(apush), &apush);
            for (const ModelSubmeshGPU& sm : aircraftAsset.submeshes) {
                if (sm.materialIndex < 0 || sm.materialIndex >= (int)aircraftAsset.materials.size()) continue;
                VkDescriptorSet ds = aircraftAsset.materials[sm.materialIndex].descriptor;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    modelPipelineLayout, 0, 1, &ds, 0, nullptr);
                vkCmdDrawIndexed(cmd, sm.indexCount, 1, sm.indexOffset, 0, 0);
            }
        } else if (rs.visible && aircraftIndexCount > 0) {
            glm::mat4 model = glm::translate(glm::mat4(1.0f), rs.position)
                            * glm::mat4_cast(rs.rotation)
                            * glm::scale(glm::mat4(1.0f), glm::vec3(rs.halfExtents.x));
            apush.mvp = proj * camera.view() * model;
            apush.model = model;

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, aircraftPipeline);
            VkDeviceSize aOff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &aircraftVertexBuffer.buffer, &aOff);
            vkCmdBindIndexBuffer(cmd, aircraftIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdPushConstants(cmd, aircraftPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                0, sizeof(apush), &apush);
            vkCmdDrawIndexed(cmd, aircraftIndexCount, 1, 0, 0, 0);
        }
    }

    // ── Visible landing-gear struts (live spring-damper compression) ──
    if (gearResourcesReady && gearIndexCount > 0 && aircraftIndexCount >= 0) {
        struct AircraftPush { glm::mat4 mvp; glm::mat4 model; } gpush;
        gpush.model = glm::mat4(1.0f);          // gear verts are already in world space
        gpush.mvp = proj * camera.view();
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, aircraftPipeline);
        VkDeviceSize gOff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &gearVertexBuffer.buffer, &gOff);
        vkCmdBindIndexBuffer(cmd, gearIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdPushConstants(cmd, aircraftPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
            0, sizeof(gpush), &gpush);
        vkCmdDrawIndexed(cmd, gearIndexCount, 1, 0, 0, 0);
    }

    // ── Aerodynamic visualization (pressure overlay + airflow streamlines) ──
    recordAeroRender(cmd, proj * camera.view());

    // ── Editor scene objects (placed model/primitive instances) ──
    drawSceneObjects(cmd, proj, camera.view());

    if (recordCallCount <= 2) std::cout << "    [record] step I: ending main rendering..." << std::endl << std::flush;

    vkCmdEndRendering(cmd);

    writeTs(4);  // after main pass (terrain+ocean+sky+foliage+aircraft+aero+scene)

    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    //  Volumetric cloud pass
    //  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    //  Depth attachment is finished writing; transition it to
    //  DEPTH_READ_ONLY_OPTIMAL so the cloud raymarch can sample
    //  it via a regular sampler2D. The cloud pass writes to the
    //  swapchain colour via alpha blending (pre-multiplied).
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    {
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        b.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        b.image = depth.image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Cloud params were already updated this frame in updateUniformBuffer()
    // so terrain cloud-shadows and the visible clouds share one state.
    drawCloudsPass(cmd, imageIndex);

    // Return depth to DEPTH_ATTACHMENT_OPTIMAL for next frame.
    {
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        b.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        b.image = depth.image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Return skyview LUT to GENERAL so next frame's compute can write it.
    {
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.image = skyviewLUT.image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        VkDependencyInfo skyDep{};
        skyDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        skyDep.imageMemoryBarrierCount = 1;
        skyDep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &skyDep);
    }

    // â”€â”€â”€ ImGui overlay pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    //   Legacy render-pass-based draw onto the existing swapchain image
    //   (LOADs current contents, renders ImGui, stores result).
    writeTs(5);  // after cloud pass (+ depth/skyview restore barriers)

    if (imguiInitialised) {
        SettingsPanel::renderToImage(cmd, imageIndex, swapchainExtent);
    }

    writeTs(6);  // after ImGui overlay

    // Transition swapchain image for presentation
    colorBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    colorBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    colorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    colorBarrier.dstAccessMask = VK_ACCESS_2_NONE;
    colorBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    if (recordCallCount <= 2) std::cout << "    [record] step J: all rendering done, ending command buffer..." << std::endl << std::flush;

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        throw std::runtime_error("failed to record command buffer");

    if (recordCallCount <= 2) std::cout << "    [record] EXIT recordRenderCommand OK" << std::endl << std::flush;
}

void VulkanEngine::createGpuProfiler() {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    gpuTimestampPeriodNs = props.limits.timestampPeriod;  // ns per tick
    if (gpuTimestampPeriodNs <= 0.0f || props.limits.timestampComputeAndGraphics == VK_FALSE) {
        std::cout << "  [gpu-profiler] timestamps unsupported on this device — disabled" << std::endl;
        gpuProfilingEnabled = false;
        return;
    }

    VkQueryPoolCreateInfo qpci{};
    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = GPU_TS_PER_FRAME * MAX_FRAMES_IN_FLIGHT;
    if (vkCreateQueryPool(device, &qpci, nullptr, &gpuQueryPool) != VK_SUCCESS) {
        std::cout << "  [gpu-profiler] query pool creation failed — disabled" << std::endl;
        gpuProfilingEnabled = false;
        return;
    }
    gpuProfilingEnabled = true;
    std::cout << "  [gpu-profiler] enabled (timestampPeriod=" << gpuTimestampPeriodNs
              << " ns)" << std::endl;
}

void VulkanEngine::readGpuTimestamps(uint32_t frameSlot) {
    if (!gpuProfilingEnabled || !gpuSlotWritten[frameSlot]) return;

    uint64_t ts[GPU_TS_PER_FRAME] = {};
    const uint32_t first = frameSlot * GPU_TS_PER_FRAME;
    VkResult r = vkGetQueryPoolResults(device, gpuQueryPool, first, GPU_TS_PER_FRAME,
        sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) return;  // VK_NOT_READY — skip this sample

    auto ms = [&](int a, int b) -> double {
        if (ts[b] <= ts[a]) return 0.0;
        return double(ts[b] - ts[a]) * double(gpuTimestampPeriodNs) * 1e-6;
    };
    const double inst[6] = { ms(0,1), ms(1,2), ms(2,3), ms(3,4), ms(4,5), ms(5,6) };
    const double total = ms(0,6);
    for (int i = 0; i < 6; ++i)
        gpuPassMs[i] = (gpuPassMs[i] <= 0.0) ? inst[i] : (gpuPassMs[i] + (inst[i] - gpuPassMs[i]) * 0.1);
    gpuTotalMs = (gpuTotalMs <= 0.0) ? total : (gpuTotalMs + (total - gpuTotalMs) * 0.1);
}

void VulkanEngine::createSyncObjects() {
    // Semaphores: one pair per swapchain image (for correct semaphore reuse)
    size_t swapchainImageCount = swapchainImages.size();
    imageAvailable.resize(swapchainImageCount);
    renderFinished.resize(swapchainImageCount);
    imagesInFlight.resize(swapchainImageCount, VK_NULL_HANDLE);

    // Fences: MAX_FRAMES_IN_FLIGHT for CPU-GPU frame pacing
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < swapchainImageCount; i++) {
        if (vkCreateSemaphore(device, &semInfo, nullptr, &imageAvailable[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semInfo, nullptr, &renderFinished[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create sync objects");
    }
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create fences");
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Frustum culling (Phase 3)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

Frustum extractFrustum(const glm::mat4& projView) {
    Frustum f;

    // Left:  row3 + row0
    f.planes[0].normal = glm::vec3(projView[0][3] + projView[0][0],
                                    projView[1][3] + projView[1][0],
                                    projView[2][3] + projView[2][0]);
    f.planes[0].distance = projView[3][3] + projView[3][0];

    // Right: row3 - row0
    f.planes[1].normal = glm::vec3(projView[0][3] - projView[0][0],
                                    projView[1][3] - projView[1][0],
                                    projView[2][3] - projView[2][0]);
    f.planes[1].distance = projView[3][3] - projView[3][0];

    // Bottom: row3 + row1
    f.planes[2].normal = glm::vec3(projView[0][3] + projView[0][1],
                                    projView[1][3] + projView[1][1],
                                    projView[2][3] + projView[2][1]);
    f.planes[2].distance = projView[3][3] + projView[3][1];

    // Top: row3 - row1
    f.planes[3].normal = glm::vec3(projView[0][3] - projView[0][1],
                                    projView[1][3] - projView[1][1],
                                    projView[2][3] - projView[2][1]);
    f.planes[3].distance = projView[3][3] - projView[3][1];

    // Near: row2 (Vulkan depth 0â†’1)
    f.planes[4].normal = glm::vec3(projView[0][2],
                                    projView[1][2],
                                    projView[2][2]);
    f.planes[4].distance = projView[3][2];

    // Far: row3 - row2
    f.planes[5].normal = glm::vec3(projView[0][3] - projView[0][2],
                                    projView[1][3] - projView[1][2],
                                    projView[2][3] - projView[2][2]);
    f.planes[5].distance = projView[3][3] - projView[3][2];

    // Normalize all planes
    for (int i = 0; i < 6; i++) {
        float len = glm::length(f.planes[i].normal);
        f.planes[i].normal /= len;
        f.planes[i].distance /= len;
    }

    return f;
}

bool isAabbInFrustum(const Frustum& f, const glm::vec3& min, const glm::vec3& max) {
    for (int i = 0; i < 6; i++) {
        glm::vec3 p(
            f.planes[i].normal.x > 0 ? max.x : min.x,
            f.planes[i].normal.y > 0 ? max.y : min.y,
            f.planes[i].normal.z > 0 ? max.z : min.z
        );
        if (glm::dot(f.planes[i].normal, p) + f.planes[i].distance < 0)
            return false;
    }
    return true;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  LOD selection (Phase 6)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Chunk management
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Erosion (Phase 4)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Hydrology (Phase 9) â€” D8 flow + river extraction
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Particle erosion (Phase 10) â€” SimpleHydrology GPU implementation
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Vegetation system (SimpleHydrology vegetation.h integration)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::initVegetationSystem() {
    std::cout << "  [vegetation] initializing plant system..." << std::endl;

    // Initialize MasterWorld with terrain data
    // Build discharge from the hydrology flow accumulation
    uint32_t N = settings.masterRes * settings.masterRes;
    std::vector<float> dischargeMap(N, 0.0f);
    std::vector<float> momX(N, 0.0f), momY(N, 0.0f);

    // Convert flow accumulation to discharge (normalized)
    float maxAcc = (float)std::max(1, masterFlowData.maxAccumulation);
    for (uint32_t i = 0; i < N; i++) {
        float normalizedAcc = (float)masterFlowData.flowAccumulation[i] / maxAcc;
        dischargeMap[i] = normalizedAcc;

        // Derive momentum from flow direction
        float azimuth = masterFlowData.flowDirection[i];
        if (!std::isnan(azimuth)) {
            momX[i] = std::cos(azimuth) * normalizedAcc;
            momY[i] = std::sin(azimuth) * normalizedAcc;
        }
    }

    vegetationWorld.init(settings.masterRes, settings.worldSize,
        masterHeightData, dischargeMap, momX, momY);

    // Initialize vegetation with seed from settings (randomized if 0)
    uint32_t vegSeed = settings.activeVegetationSeed();
    vegetation = VegetationEngine::Vegetation(vegSeed);
    vegetation.maxPlants = settings.maxPlants;

    // Apply plant-level settings from configuration
    VegetationEngine::Plant::maxSize        = settings.plantMaxSize;
    VegetationEngine::Plant::growRate       = settings.plantGrowRate;
    VegetationEngine::Plant::maxSteep       = settings.plantMaxSteep;
    VegetationEngine::Plant::maxDischarge   = settings.plantMaxDischarge;
    VegetationEngine::Plant::maxTreeHeight  = settings.plantMaxTreeHeight;
    VegetationEngine::Plant::rootRadius     = settings.plantRootRadius;

    // Pre-seed some vegetation
    for (int i = 0; i < settings.initialPlants; i++) {
        vegetation.spawnOne(vegetationWorld, settings.masterRes);
    }

    // Apply root density from initial spawning
    vegetation.applyAllRoots(vegetationWorld);

    std::cout << "  [vegetation] initialized with " << vegetation.count()
              << " plants" << std::endl;
}

void VulkanEngine::updateVegetation() {
    if (vegetationWorld.R == 0) return;  // not initialized yet

    // Grow vegetation (includes spawn, grow, die, spread)
    vegetation.grow(vegetationWorld, settings.masterRes);

    // Build root density map for GPU upload
    vegetation.applyAllRoots(vegetationWorld);
}

// â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Terrain blur â€” Gaussian smoothing for heightmaps
// â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Helpers (continued)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

uint32_t VulkanEngine::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("failed to find suitable memory type");
}

void VulkanEngine::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags props, Buffer& buf)
{
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &info, nullptr, &buf.buffer) != VK_SUCCESS)
        throw std::runtime_error("failed to create buffer");

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, buf.buffer, &reqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = reqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(reqs.memoryTypeBits, props);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &buf.memory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate buffer memory");

    vkBindBufferMemory(device, buf.buffer, buf.memory, 0);
}

void VulkanEngine::createImage(uint32_t w, uint32_t h, VkFormat fmt,
                                 VkImageUsageFlags usage, VkImageAspectFlags aspect,
                                 Image& img)
{
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.extent = {w, h, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.format = fmt;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.usage = usage;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(device, &info, nullptr, &img.image) != VK_SUCCESS)
        throw std::runtime_error("failed to create image");

    VkMemoryRequirements reqs;
    vkGetImageMemoryRequirements(device, img.image, &reqs);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = reqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &img.memory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate image memory");
    vkBindImageMemory(device, img.image, img.memory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = img.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = fmt;
    viewInfo.subresourceRange.aspectMask = aspect;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &viewInfo, nullptr, &img.view) != VK_SUCCESS)
        throw std::runtime_error("failed to create image view");
}

VkCommandBuffer VulkanEngine::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    return cmd;
}

void VulkanEngine::endSingleTimeCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

VkShaderModule VulkanEngine::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module;
    if (vkCreateShaderModule(device, &info, nullptr, &module) != VK_SUCCESS)
        throw std::runtime_error("failed to create shader module");
    return module;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Global foliage â€” maps VegetationEngine plants to GPU buffers
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void VulkanEngine::generateGlobalFoliage() {
    const auto& plants = vegetation.getPlants();
    globalFoliageCount = (uint32_t)plants.size();
    if (globalFoliageCount == 0) return;

    std::cout << "  [foliage] generating " << globalFoliageCount
              << " tree instances for GPU..." << std::endl;

    // Build instance data: vec4 posScale + vec4 color per tree
    std::vector<glm::vec4> posScale(globalFoliageCount);
    std::vector<glm::vec4> colors(globalFoliageCount);

    for (uint32_t i = 0; i < globalFoliageCount; i++) {
        float worldX = plants[i].pos.x;
        float worldZ = plants[i].pos.y;
        float groundH = sampleWorldHeight(worldX, worldZ);
        float size = plants[i].size;

        posScale[i] = glm::vec4(worldX, groundH, worldZ, glm::clamp(size, 0.1f, 3.0f));

        // Color varies by height: darker green at low elevation, lighter at high
        float heightTint = glm::clamp((groundH + 5.0f) / 25.0f, 0.0f, 1.0f);
        colors[i] = glm::vec4(
            0.08f + heightTint * 0.05f,
            0.25f + heightTint * 0.20f,
            0.08f + heightTint * 0.05f,
            1.0f
        );
    }

    // Create instance buffers
    VkDeviceSize posSize = sizeof(glm::vec4) * globalFoliageCount;
    VkDeviceSize colSize = sizeof(glm::vec4) * globalFoliageCount;

    createBuffer(posSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, globalFoliageInstanceBuffer);

    createBuffer(colSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, globalFoliageColorBuffer);

    // Upload via staging
    VkCommandBuffer cmd = beginSingleTimeCommands();

    Buffer staging;
    createBuffer(posSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
    void* mapped;
    vkMapMemory(device, staging.memory, 0, posSize, 0, &mapped);
    memcpy(mapped, posScale.data(), posSize);
    vkUnmapMemory(device, staging.memory);
    VkBufferCopy copy{};
    copy.size = posSize;
    vkCmdCopyBuffer(cmd, staging.buffer, globalFoliageInstanceBuffer.buffer, 1, &copy);
    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);

    createBuffer(colSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
    vkMapMemory(device, staging.memory, 0, colSize, 0, &mapped);
    memcpy(mapped, colors.data(), colSize);
    vkUnmapMemory(device, staging.memory);
    copy.size = colSize;
    vkCmdCopyBuffer(cmd, staging.buffer, globalFoliageColorBuffer.buffer, 1, &copy);
    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);

    endSingleTimeCommands(cmd);
}

void VulkanEngine::updateGlobalFoliageBuffers() {
    const auto& plants = vegetation.getPlants();
    uint32_t count = (uint32_t)plants.size();
    if (count == 0) return;

    // Rebuild if count changed significantly
    if (count != globalFoliageCount) {
        // The old foliage buffers may still be referenced by in-flight render
        // command buffers; wait for the GPU to finish before freeing them.
        if (globalFoliageInstanceBuffer.buffer != VK_NULL_HANDLE ||
            globalFoliageColorBuffer.buffer != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
        // Destroy old buffers
        if (globalFoliageInstanceBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, globalFoliageInstanceBuffer.buffer, nullptr);
            vkFreeMemory(device, globalFoliageInstanceBuffer.memory, nullptr);
            globalFoliageInstanceBuffer = {};
        }
        if (globalFoliageColorBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, globalFoliageColorBuffer.buffer, nullptr);
            vkFreeMemory(device, globalFoliageColorBuffer.memory, nullptr);
            globalFoliageColorBuffer = {};
        }
        globalFoliageCount = count;

        std::vector<glm::vec4> posScale(count);
        std::vector<glm::vec4> colors(count);
        for (uint32_t i = 0; i < count; i++) {
            float groundH = sampleWorldHeight(plants[i].pos.x, plants[i].pos.y);
            posScale[i] = glm::vec4(plants[i].pos.x, groundH, plants[i].pos.y,
                                    glm::clamp(plants[i].size, 0.1f, 3.0f));
            float ht = glm::clamp((groundH + 5.0f) / 25.0f, 0.0f, 1.0f);
            colors[i] = glm::vec4(0.08f + ht * 0.05f, 0.25f + ht * 0.20f, 0.08f + ht * 0.05f, 1.0f);
        }

        VkDeviceSize posSize = sizeof(glm::vec4) * count;
        VkDeviceSize colSize = sizeof(glm::vec4) * count;

        createBuffer(posSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, globalFoliageInstanceBuffer);
        createBuffer(colSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, globalFoliageColorBuffer);

        // Use two separate staging buffers so both copies remain valid for
        // the entire single-time command buffer. Staging buffers must only be
        // destroyed AFTER the command buffer has been submitted and completed
        // (endSingleTimeCommands waits on the queue) — destroying them while the
        // command buffer is still recording invalidates it.
        VkCommandBuffer cmd = beginSingleTimeCommands();

        Buffer posStaging;
        createBuffer(posSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, posStaging);
        void* mapped;
        vkMapMemory(device, posStaging.memory, 0, posSize, 0, &mapped);
        memcpy(mapped, posScale.data(), posSize);
        vkUnmapMemory(device, posStaging.memory);
        VkBufferCopy copy{}; copy.size = posSize;
        vkCmdCopyBuffer(cmd, posStaging.buffer, globalFoliageInstanceBuffer.buffer, 1, &copy);

        Buffer colStaging;
        createBuffer(colSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, colStaging);
        vkMapMemory(device, colStaging.memory, 0, colSize, 0, &mapped);
        memcpy(mapped, colors.data(), colSize);
        vkUnmapMemory(device, colStaging.memory);
        copy.size = colSize;
        vkCmdCopyBuffer(cmd, colStaging.buffer, globalFoliageColorBuffer.buffer, 1, &copy);

        endSingleTimeCommands(cmd);

        vkDestroyBuffer(device, posStaging.buffer, nullptr);
        vkFreeMemory(device, posStaging.memory, nullptr);
        vkDestroyBuffer(device, colStaging.buffer, nullptr);
        vkFreeMemory(device, colStaging.memory, nullptr);
    }
}

std::vector<char> VulkanEngine::readFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open file: " + path);
    size_t size = file.tellg();
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), size);
    file.close();
    return buffer;
}
