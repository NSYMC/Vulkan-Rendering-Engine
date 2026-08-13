cmake_minimum_required(VERSION 3.21)
project(TerrainEngine VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan
find_package(Vulkan REQUIRED)

include(FetchContent)

# GLM
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glm)

# GLFW
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Jolt Physics 3D
set(JPH_BUILD_SHARED_LIBRARY OFF CACHE BOOL "" FORCE)
# Match the dynamic CRT (/MD) used by the executable + GLFW, otherwise Jolt
# defaults to the static CRT (/MT) and the link fails with LNK2038/LNK2005.
set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)
set(PROFILER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(JoltPhysics
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
    GIT_TAG        v5.2.0
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  Build
)
FetchContent_MakeAvailable(JoltPhysics)

# Shader compilation
set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_BIN_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BIN_DIR})

function(add_shader TARGET SHADER_PATH)
    get_filename_component(SHADER_NAME ${SHADER_PATH} NAME)
    string(REPLACE "." "_" SHADER_TARGET ${SHADER_NAME})
    set(SPV ${SHADER_BIN_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -I${SHADER_SRC_DIR} -o ${SPV} ${SHADER_PATH}
        DEPENDS ${SHADER_PATH}
        COMMENT "Compiling ${SHADER_NAME} -> SPIR-V"
    )
    add_custom_target(${SHADER_TARGET}_spv DEPENDS ${SPV})
    add_dependencies(${TARGET} ${SHADER_TARGET}_spv)
    target_sources(${TARGET} PRIVATE ${SPV})
endfunction()

# ImGui (sources copied into src/imgui)
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/src/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)

# ImGuizmo (transform gizmos) — depends on the in-tree ImGui above.
set(IMGUIZMO_DIR ${CMAKE_SOURCE_DIR}/src/third_party/ImGuizmo)
set(IMGUIZMO_SOURCES ${IMGUIZMO_DIR}/ImGuizmo.cpp)

# Engine executable
add_executable(TerrainEngine
    src/main.cpp
    src/VulkanEngine.cpp
    src/PhysicsWorld.cpp
    src/AerodynamicBody.cpp
    src/AutonomousAircraft.cpp
    src/Model.cpp
    src/third_party/ufbx.c
    ${IMGUI_SOURCES}
    ${IMGUIZMO_SOURCES}
)

# ufbx: enable threaded loading is off by default; keep it simple/portable.
set_source_files_properties(src/third_party/ufbx.c PROPERTIES LANGUAGE C)

add_shader(TerrainEngine ${SHADER_SRC_DIR}/terrain.vert)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/terrain.frag)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/ocean.vert)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/ocean.frag)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/airflow_advect.comp)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/airflow_render.vert)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/airflow_render.frag)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/aero_surface.vert)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/aero_surface.frag)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/aero_panel_viz.vert)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/aero_panel_viz.frag)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/streamline.vert)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/streamline.frag)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/texgen.comp)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/shadow.vert)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/foliage.vert)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/foliage.frag)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/aircraft.vert)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/aircraft.frag)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/model.vert)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/model.frag)

# Bruneton / Hillaire atmospheric scattering LUTs + sky pass
# Ported from atmosphere_bac_src (Matej Sakmary's BSc thesis).
add_shader(TerrainEngine ${SHADER_SRC_DIR}/bruneton_transmittance.comp)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/bruneton_multiscatter.comp)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/bruneton_skyview.comp)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/bruneton_sky.vert)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/bruneton_sky.frag)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/cloud_noise_gen.comp)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/bruneton_clouds.vert)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/bruneton_clouds.frag)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/cloud_temporal.frag)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/cloud_composite.frag)
add_shader(TerrainEngine ${SHADER_SRC_DIR}/cloud_shadow.comp)

target_include_directories(TerrainEngine PRIVATE
    src
    src/third_party
    ${IMGUIZMO_DIR}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
target_link_libraries(TerrainEngine PRIVATE glm::glm glfw Vulkan::Vulkan Jolt)

# Absolute path to the source tree so large assets (models/) can be loaded
# in-place without copying hundreds of MB next to the executable each build.
target_compile_definitions(TerrainEngine PRIVATE TE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Jolt headers live one directory above Build/
FetchContent_GetProperties(JoltPhysics POPULATED jolt_pop SOURCE_DIR jolt_src)
if (jolt_pop)
    target_include_directories(TerrainEngine PRIVATE "${jolt_src}")
endif()

# ImGui needs imm32 (IME) and dwmapi (DWM) on Windows.
if(WIN32)
    target_link_libraries(TerrainEngine PRIVATE imm32 dwmapi)
endif()

# Copy SPV files next to the executable
add_custom_command(TARGET TerrainEngine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${SHADER_BIN_DIR} $<TARGET_FILE_DIR:TerrainEngine>/shaders
)

# Copy settings file next to the executable
add_custom_command(TARGET TerrainEngine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/terrain_settings.json
    $<TARGET_FILE_DIR:TerrainEngine>/terrain_settings.json
)

set_target_properties(TerrainEngine PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:TerrainEngine>"
)
