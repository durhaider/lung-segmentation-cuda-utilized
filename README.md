# Lung Segmentation with CUDA Acceleration

## Overview

This project implements GPU-accelerated lung segmentation for CT scan processing using custom CUDA kernels. The system achieves a **75% reduction in processing time** (from ~30 seconds to 7-8 seconds) while maintaining full  accuracy.

## Performance Achievements

- **Processing Time**: 75% reduction (30s → 7-8s)
- **Architecture**: Custom CUDA kernels with complete GPU offloading

## Technical Implementation

### GPU Acceleration Features

- **Custom CUDA Kernels**: Purpose-built for lung tissue isolation and iterative segmentation
- **DICOM Support**: Full compatibility with medical imaging data formats

## Prerequisites

- **NVIDIA GPU**: CUDA-enabled (compute capability 8.0+ recommended)
- **CUDA Toolkit**: Version 12.0 or later
- **CMake**: Version 3.15 or later
- **Visual Studio**: 2019 or 2022 with C++ desktop development workload
- **DICOM Dataset**: CT scan data in DICOM format

## Build Instructions

### Step 1: Configure GPU Architecture

Edit `CMakeLists.txt` to match your GPU generation:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)  # Modify based on your GPU
```

**Architecture Values:**
- RTX 40 Series (Ada Lovelace): `89`
- RTX 30 Series (Ampere): `86`
- RTX 20 Series (Turing): `75`
- GTX 10 Series (Pascal): `61`

### Step 2: CMake Configuration

1. Open **CMake GUI**
2. Set source directory to your project folder
3. Set build directory to `build` subfolder
4. Click **Configure** → Select Visual Studio version and x64 platform
5. Click **Generate** to create the `.sln` file in the `build` folder

### Step 3: Build in Visual Studio

1. Open `build/LungSegmentation.sln` in Visual Studio
2. Select **Debug** or **Release** configuration (x64 platform)
3. Right-click solution → **Build Solution** (Ctrl+Shift+B)
4. Executable will be generated in `build/Debug/` or `build/Release/`

### Step 4: Locate the Executable

After successful build, find `lung_viewer.exe` in:
- `build/Debug/lung_viewer.exe` (Debug build)
- `build/Release/lung_viewer.exe` (Release build)

## Running the Application

### Command Line Execution

Open Command Prompt and navigate to the executable directory, then run:

```bash
lung_viewer.exe -DICOM "C:\path\to\your\dicom\folder" -slice 12
```

### Parameters

- **`-DICOM`**: Path to DICOM dataset folder (use quotes for paths with spaces)
- **`-slice`**: CT slice index to process (e.g., 12 for the 12th slice)

### Example Usage

```bash
# Process slice 15 from a specific dataset
lung_viewer.exe -DICOM "D:\Medical_Data\Patient_001\CT_Scan" -slice 15

# Process slice 8 from another dataset
lung_viewer.exe -DICOM "C:\Research\DICOM_Files\Lung_Study_042" -slice 8
```

### Expected Output

The application will:
1. Load the specified DICOM dataset
2. Initialize CUDA kernels and GPU memory
3. Perform GPU-accelerated segmentation on the specified slice
4. Display processing time and performance metrics
5. Output segmentation results

## Project Structure

```
lung-segmentation-cuda-utilized/
├── src/
│   ├── main.cpp              # Application entry point
│   ├── process.cu            # CUDA kernel implementations
│   └── *.cuh                 # CUDA header files
├── build/                    # CMake build directory (generated)
│   └── Debug/Release/
│       └── lung_viewer.exe   # Executable
├── CMakeLists.txt            # CMake configuration
└── README.md                 # Documentation
```
- You may put cmakelists.txt into src (recommended)
## Troubleshooting

### Build Issues

**CUDA Architecture Mismatch:**
- Update `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` to match your GPU

**Missing CUDA Toolkit:**
- Install NVIDIA CUDA Toolkit and ensure it's in system PATH

**Visual Studio Errors:**
- Verify C++ desktop development workload is installed

### Runtime Issues

**DICOM Path Not Found:**
- Use absolute paths with quotes for folders containing spaces
- Verify folder contains valid DICOM files

**Out of Memory:**
- Monitor GPU memory with `nvidia-smi`
- Reduce dataset size or processing resolution

**Invalid Slice Index:**
- Ensure slice number exists in your DICOM dataset



---

**Note**: This project demonstrates practical GPU acceleration in medical imaging, achieving significant performance improvements while maintaining clinical accuracy standards.
