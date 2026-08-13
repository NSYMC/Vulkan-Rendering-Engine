@echo off
cd /d "%~dp0"

echo ============================================
echo  TerrainEngine - Build ^& Run
echo ============================================

:: Add MinGW and Vulkan SDK to PATH
set "PATH=C:\Program Files\CodeBlocks\MinGW\bin;%PATH%"
if exist "C:\VulkanSDK\1.4.335.0\Bin" (
    set "PATH=C:\VulkanSDK\1.4.335.0\Bin;%PATH%"
) else if defined VULKAN_SDK (
    set "PATH=%VULKAN_SDK%\Bin;%PATH%"
)

:: Build
echo.
echo [1/2] Building...
cmake --build build_cb
if %ERRORLEVEL% neq 0 (
    echo.
    echo BUILD FAILED  --  see errors above
    pause
    exit /b 1
)
echo Build OK.

:: Copy settings file to build output
echo.
echo Copying terrain_settings.json...
copy /y "%~dp0terrain_settings.json" "%~dp0build_cb\terrain_settings.json" >nul

:: Run  (must be in build_cb so the exe finds shaders/ folder)
echo.
echo [2/2] Launching TerrainEngine.exe ...
echo.
cd build_cb
TerrainEngine.exe

pause
