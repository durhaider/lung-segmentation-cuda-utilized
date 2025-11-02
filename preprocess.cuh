#pragma once
#include <cuda_runtime.h>

void preprocess_volume(
    float* d_in,        // in-place working buffer on device (nx*ny*nz)
    float* d_tmp,       // scratch buffer on device
    float* d_out,       // scratch buffer on device
    const float* d_k,   // gaussian kernel on device (len 2*r+1)
    int nx, int ny, int nz,
    int r,              // kernel radius
    float loHU, float hiHU,
    cudaStream_t stream = nullptr);

// ---- GPU lung helpers (BFS exterior + connected components + morphology) ----
#include <stdint.h>

// exterior = flood-fill of air from volume borders, blocked by body
cudaError_t gpu_exterior_air(
    const uint8_t* d_air, const uint8_t* d_body, uint8_t* d_exterior,
    int nx, int ny, int nz, cudaStream_t stream = nullptr);

// Keep only two largest 6-connected components from an input binary mask.
// Writes resulting mask (0/1) to d_keep. Optionally returns the kept label ids to host.
cudaError_t gpu_label_keep_two(
    const uint8_t* d_mask, uint8_t* d_keep,
    int nx, int ny, int nz,
    int* h_kept_labels /*[2]*/ = nullptr,
    cudaStream_t stream = nullptr);

// In-place binary closing (dilate then erode) using 6-neighborhood.
cudaError_t gpu_close3d(uint8_t* d_mask, int nx, int ny, int nz, int iterations,
                        cudaStream_t stream = nullptr);

// Elementwise out = a & (~b)
cudaError_t gpu_andnot(const uint8_t* d_a, const uint8_t* d_b, uint8_t* d_out, size_t n,
                       cudaStream_t stream = nullptr);
