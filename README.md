🚀 Build & Run Guide (Step-by-step)

This project uses CMake + CUDA + VTK + OpenGL, so building requires a few setup steps.
Here is the full process explained simply, especially for Windows users using Visual Studio + CMake GUI + CLion.

✅ Prerequisites

Before building, make sure you have installed:

NVIDIA GPU drivers

CUDA Toolkit

Visual Studio 2022

CMake (GUI recommended)

VTK 9.x (built or installed through vcpkg)

GLAD + GLFW for OpenGL rendering

⚙️ 1. Create and configure CMakeLists.txt

Your main CMakeLists.txt should include:

CUDA language enable

Proper GPU architecture setting
(this is important or CUDA kernels won't run efficiently)

Set the architecture flag depending on your GPU:

GPU Generation	Example GPUs	CMake Compute Arch
RTX-40 series	4090 / 4080 / 4070 / 4050	89
RTX-30 series	3090 / 3080 / 3070 / 3060	86
RTX-20 series	2080 / 2070 / 2060	75
GTX-10 series	1080 / 1070 / 1060	61
