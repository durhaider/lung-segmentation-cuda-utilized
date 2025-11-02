cmake_minimum_required(VERSION 3.20)
project(cuda_vtk_lung LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- NEW: CUDA Toolkit (gives headers + cudart target)
find_package(CUDAToolkit REQUIRED)          # <--- add this

# ---- VTK ----
find_package(VTK REQUIRED COMPONENTS
  CommonCore CommonDataModel IOImage ImagingCore InteractionStyle
  RenderingCore RenderingOpenGL2 RenderingVolume RenderingVolumeOpenGL2
  FiltersCore FiltersGeneral IOGeometry IOExport
)

find_package(glfw3 CONFIG REQUIRED)
find_package(glm   CONFIG REQUIRED)
find_package(glad  CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(lung_viewer
  main.cxx
  preprocess.cu
)

# --- REMOVE this line so the profile flag controls the arch
# set_property(TARGET lung_viewer PROPERTY CUDA_ARCHITECTURES 89)

# --- NEW: expose CUDA headers to cl.exe and link cudart
target_include_directories(lung_viewer PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(lung_viewer PRIVATE CUDA::cudart)

# (rest unchanged)
target_compile_options(lung_viewer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

if (MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  target_compile_definitions(lung_viewer PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

target_link_libraries(lung_viewer PRIVATE
  VTK::CommonCore VTK::CommonDataModel VTK::IOImage VTK::ImagingCore
  VTK::InteractionStyle VTK::RenderingCore VTK::RenderingOpenGL2
  VTK::RenderingVolume VTK::RenderingVolumeOpenGL2 VTK::FiltersCore
  VTK::FiltersGeneral VTK::IOGeometry VTK::IOExport
  glad::glad glfw glm::glm OpenGL::GL
)

vtk_module_autoinit(
  TARGETS lung_viewer
  MODULES
    VTK::CommonCore VTK::CommonDataModel VTK::IOImage VTK::ImagingCore
    VTK::InteractionStyle VTK::RenderingCore VTK::RenderingOpenGL2
    VTK::RenderingVolume VTK::RenderingVolumeOpenGL2 VTK::FiltersCore
    VTK::FiltersGeneral VTK::IOGeometry VTK::IOExport
)
