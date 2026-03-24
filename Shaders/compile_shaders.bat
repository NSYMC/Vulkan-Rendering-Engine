@echo off
REM Run from Shaders folder so .spv files are written here. App loads Shaders/vert.spv etc. (relative to exe working dir).
cd /d "%~dp0"

if defined VULKAN_SDK (
  set "VAL=%VULKAN_SDK%\Bin\glslangValidator.exe"
) else (
  set "VAL=C:\VulkanSDK\1.4.335.0\Bin\glslangValidator.exe"
)
if not exist "%VAL%" (
  echo glslangValidator not found at: %VAL%
  echo Install Vulkan SDK or set VULKAN_SDK to your SDK root.
  pause
  exit /b 1
)

"%VAL%" -o vert.spv -V shader.vert
if errorlevel 1 goto err
"%VAL%" -o frag.spv -V shader.frag
if errorlevel 1 goto err
"%VAL%" -o shadow_vert.spv -V shadow.vert
if errorlevel 1 goto err
"%VAL%" -o shadow_frag.spv -V shadow.frag
if errorlevel 1 goto err
"%VAL%" -o second_vert.spv -V second.vert
if errorlevel 1 goto err
"%VAL%" -o second_frag.spv -V second.frag
if errorlevel 1 goto err
"%VAL%" -o line_vert.spv -V line.vert
if errorlevel 1 goto err
"%VAL%" -o line_frag.spv -V line.frag
if errorlevel 1 goto err
"%VAL%" -o sky_vert.spv -V sky.vert
if errorlevel 1 goto err
"%VAL%" -o sky_frag.spv -V sky.frag
if errorlevel 1 goto err
"%VAL%" -o star_vert.spv -V star.vert
if errorlevel 1 goto err
"%VAL%" -o star_frag.spv -V star.frag
if errorlevel 1 goto err

echo All shaders compiled to %CD%
echo Run the app from the project dir that contains this Shaders folder.
pause
exit /b 0
:err
echo Compile failed. Fix errors above.
pause
exit /b 1
