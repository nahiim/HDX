
@echo off
setlocal

:: Define the shader directory
set SHADER_DIR=shaders

set OUTPUT_DIR=Debug/res/shaders

:: Compile vertex shaders
for %%f in (%SHADER_DIR%\*.vert) do (
    glslc "%%f" -o "%%~dpnxf.spv"
)

:: Compile fragment shaders
for %%f in (%SHADER_DIR%\*.frag) do (
    glslc "%%f" -o "%%~dpnxf.spv"
)

:: Compile compute shaders
for %%f in (%SHADER_DIR%\*.comp) do (
    glslc "%%f" -o "%%~dpnxf.spv"
)

endlocal

:: Move all .spv files to output directory
for %%f in (shaders\*.spv) do (
    move "%%f" "%OUTPUT_DIR%"
)
copy *.spv Debug\res\shaders
move *.spv res/shaders

pause