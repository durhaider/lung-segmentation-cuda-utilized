#include "preprocess.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <algorithm>

static __device__ __forceinline__ int idx(int x, int y, int z, int nx, int ny) {
    return x + y * nx + z * nx * ny;
}

__global__
void gaussian1D_x(const float* in, float* out, int nx, int ny, int nz,
    const float* k, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    float s = 0.f;
    for (int t = -r; t <= r; ++t) {
        int xx = max(0, min(nx - 1, x + t));
        s += k[t + r] * in[idx(xx, y, z, nx, ny)];
    }
    out[idx(x, y, z, nx, ny)] = s;
}

__global__
void gaussian1D_y(const float* in, float* out, int nx, int ny, int nz,
    const float* k, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    float s = 0.f;
    for (int t = -r; t <= r; ++t) {
        int yy = max(0, min(ny - 1, y + t));
        s += k[t + r] * in[idx(x, yy, z, nx, ny)];
    }
    out[idx(x, y, z, nx, ny)] = s;
}

__global__
void gaussian1D_z(const float* in, float* out, int nx, int ny, int nz,
    const float* k, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    float s = 0.f;
    for (int t = -r; t <= r; ++t) {
        int zz = max(0, min(nz - 1, z + t));
        s += k[t + r] * in[idx(x, y, zz, nx, ny)];
    }
    out[idx(x, y, z, nx, ny)] = s;
}

__global__
void thresholdHU(float* vol, int nx, int ny, int nz, float loHU, float hiHU) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int i = x + y * nx + z * nx * ny;
    float v = vol[i];

    // No-op pass-through to keep symbol used and avoid warnings.
    // If you want to enable thresholding on GPU, uncomment the next line:
    // vol[i] = (v >= loHU && v <= hiHU) ? v : -1024.0f;

    vol[i] = v;
}



void preprocess_volume(
    float* d_in, float* d_tmp, float* d_out, const float* d_k,
    int nx, int ny, int nz, int r, float loHU, float hiHU, cudaStream_t stream)
{
    dim3 block(16, 8, 4);
    dim3 grid((nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z);

    gaussian1D_x << <grid, block, 0, stream >> > (d_in, d_tmp, nx, ny, nz, d_k, r);
    gaussian1D_y << <grid, block, 0, stream >> > (d_tmp, d_out, nx, ny, nz, d_k, r);
    gaussian1D_z << <grid, block, 0, stream >> > (d_out, d_in, nx, ny, nz, d_k, r);
    thresholdHU << <grid, block, 0, stream >> > (d_in, nx, ny, nz, loHU, hiHU);
}

// ====== GPU helpers for exterior flood, CCL (keep-2), and morphology ======
#include <stdint.h>

static __device__ __forceinline__ size_t idx3d(int x, int y, int z, int nx, int ny, int nz) {
    return (size_t)x + (size_t)nx * ((size_t)y + (size_t)ny * (size_t)z);
}

__global__ void k_andnot(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (a[i] && !b[i]) ? 1u : 0u;
}

cudaError_t gpu_andnot(const uint8_t* d_a, const uint8_t* d_b, uint8_t* d_out, size_t n,
                       cudaStream_t stream) {
    dim3 block(256), grid((unsigned)((n + block.x - 1) / block.x));
    k_andnot<<<grid, block, 0, stream>>>(d_a, d_b, d_out, n);
    return cudaGetLastError();
}

// ---- Exterior flood (frontier-based BFS) ----
__global__ void k_clear(uint8_t* a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0;
}

__global__ void k_seed_border(const uint8_t* air, const uint8_t* body, uint8_t* exterior,
                              uint8_t* frontier, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    bool is_border = (x == 0) || (x == nx - 1) || (y == 0) || (y == ny - 1) || (z == 0) || (z == nz - 1);
    if (!is_border) return;
    size_t id = idx3d(x, y, z, nx, ny, nz);
    if (air[id] && !body[id]) {
        exterior[id] = 1;
        frontier[id] = 1;
    }
}

__global__ void k_expand_frontier(const uint8_t* air, const uint8_t* body,
                                  uint8_t* exterior, const uint8_t* frontier,
                                  uint8_t* nextF, int nx, int ny, int nz, int* changed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    size_t id = idx3d(x, y, z, nx, ny, nz);
    if (!frontier[id]) return;
    const int dx[6] = {+1,-1,0,0,0,0};
    const int dy[6] = {0,0,+1,-1,0,0};
    const int dz[6] = {0,0,0,0,+1,-1};
    for (int k = 0; k < 6; ++k) {
        int xx = x + dx[k], yy = y + dy[k], zz = z + dz[k];
        if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) continue;
        size_t nid = idx3d(xx, yy, zz, nx, ny, nz);
        if (!exterior[nid] && air[nid] && !body[nid]) {
            exterior[nid] = 1;
            nextF[nid] = 1;
            atomicExch(changed, 1);
        }
    }
}

cudaError_t gpu_exterior_air(const uint8_t* d_air, const uint8_t* d_body, uint8_t* d_exterior,
                             int nx, int ny, int nz, cudaStream_t stream) {
    size_t vox = (size_t)nx * ny * nz;
    dim3 block3(8, 8, 8);
    dim3 grid3((nx + block3.x - 1) / block3.x,
               (ny + block3.y - 1) / block3.y,
               (nz + block3.z - 1) / block3.z);
    dim3 block1(256), grid1((unsigned)((vox + block1.x - 1) / block1.x));

    // exterior = 0
    k_clear<<<grid1, block1, 0, stream>>>(d_exterior, vox);

    // allocate frontier buffers and change flag
    uint8_t *d_frontier = nullptr, *d_next = nullptr;
    int *d_changed = nullptr;
    cudaMalloc(&d_frontier, vox);
    cudaMalloc(&d_next, vox);
    cudaMalloc(&d_changed, sizeof(int));
    k_clear<<<grid1, block1, 0, stream>>>(d_frontier, vox);
    k_clear<<<grid1, block1, 0, stream>>>(d_next, vox);

    // seed from borders
    k_seed_border<<<grid3, block3, 0, stream>>>(d_air, d_body, d_exterior, d_frontier, nx, ny, nz);

    // wavefront expand
    int h_changed = 0;
    int it = 0, maxIt = nx + ny + nz + 64;
    do {
        cudaMemsetAsync(d_changed, 0, sizeof(int), stream);
        k_expand_frontier<<<grid3, block3, 0, stream>>>(d_air, d_body, d_exterior, d_frontier, d_next, nx, ny, nz, d_changed);
        // swap
        k_clear<<<grid1, block1, 0, stream>>>(d_frontier, vox);
        std::swap(d_frontier, d_next);
        cudaMemcpyAsync(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        ++it;
        // clear next frontier for next pass
        k_clear<<<grid1, block1, 0, stream>>>(d_next, vox);
    } while (h_changed && it < maxIt);

    cudaFree(d_frontier);
    cudaFree(d_next);
    cudaFree(d_changed);
    return cudaGetLastError();
}

// ---- Connected components via iterative min-propagation ----
__global__ void k_init_labels(const uint8_t* mask, int* labels, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) labels[i] = mask[i] ? (int)i : -1;
}

__global__ void k_propagate_labels(const uint8_t* mask, const int* labels_in, int* labels_out,
                                   int nx, int ny, int nz, int* changed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    size_t id = idx3d(x, y, z, nx, ny, nz);
    int lab = labels_in[id];
    if (!mask[id] || lab < 0) { labels_out[id] = -1; return; }

    int best = lab;
    const int dx[6] = {+1,-1,0,0,0,0};
    const int dy[6] = {0,0,+1,-1,0,0};
    const int dz[6] = {0,0,0,0,+1,-1};
    for (int k = 0; k < 6; ++k) {
        int xx = x + dx[k], yy = y + dy[k], zz = z + dz[k];
        if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) continue;
        size_t nid = idx3d(xx, yy, zz, nx, ny, nz);
        int nl = labels_in[nid];
        if (nl >= 0 && nl < best) best = nl;
    }
    labels_out[id] = best;
    if (best != lab) atomicExch(changed, 1);
}

__global__ void k_hist_sizes(const int* labels, int* sizes, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int L = labels[i];
        if (L >= 0) atomicAdd(&sizes[L], 1);
    }
}

__global__ void k_keep_mask_from_labels(const int* labels, uint8_t* out, size_t n, int keepA, int keepB) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int L = labels[i];
        out[i] = (L == keepA || L == keepB) ? 1u : 0u;
    }
}

cudaError_t gpu_label_keep_two(const uint8_t* d_mask, uint8_t* d_keep,
                               int nx, int ny, int nz, int* h_kept_labels, cudaStream_t stream)
{
    size_t vox = (size_t)nx * ny * nz;
    dim3 block3(8, 8, 8);
    dim3 grid3((nx + block3.x - 1) / block3.x,
               (ny + block3.y - 1) / block3.y,
               (nz + block3.z - 1) / block3.z);
    dim3 block1(256), grid1((unsigned)((vox + block1.x - 1) / block1.x));

    int *d_labA = nullptr, *d_labB = nullptr, *d_sizes = nullptr, *d_changed = nullptr;
    cudaMalloc(&d_labA, vox * sizeof(int));
    cudaMalloc(&d_labB, vox * sizeof(int));
    cudaMalloc(&d_sizes, vox * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    k_init_labels<<<grid1, block1, 0, stream>>>(d_mask, d_labA, vox);

    int h_changed = 0, it = 0, maxIt = nx + ny + nz + 64;
    do {
        cudaMemsetAsync(d_changed, 0, sizeof(int), stream);
        k_propagate_labels<<<grid3, block3, 0, stream>>>(d_mask, d_labA, d_labB, nx, ny, nz, d_changed);
        std::swap(d_labA, d_labB);
        cudaMemcpyAsync(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        ++it;
    } while (h_changed && it < maxIt);

    // histogram sizes
    cudaMemsetAsync(d_sizes, 0, vox * sizeof(int), stream);
    k_hist_sizes<<<grid1, block1, 0, stream>>>(d_labA, d_sizes, vox);

    // bring sizes to host to get top-2
    std::vector<int> h_sizes(vox);
    cudaMemcpyAsync(h_sizes.data(), d_sizes, vox * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int keepA = -1, keepB = -1;
    long long bestA = 0, bestB = 0;
    for (size_t i = 0; i < vox; ++i) {
        int s = h_sizes[i];
        if (s > bestA) { bestB = bestA; keepB = keepA; bestA = s; keepA = (int)i; }
        else if (s > bestB) { bestB = s; keepB = (int)i; }
    }
    if (h_kept_labels) { h_kept_labels[0] = keepA; h_kept_labels[1] = keepB; }

    k_keep_mask_from_labels<<<grid1, block1, 0, stream>>>(d_labA, d_keep, vox, keepA, keepB);

    cudaFree(d_labA);
    cudaFree(d_labB);
    cudaFree(d_sizes);
    cudaFree(d_changed);
    return cudaGetLastError();
}

// ---- Morphology: 6-neighborhood dilate/erode ----
__global__ void k_dilate6(const uint8_t* in, uint8_t* out, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    size_t id = idx3d(x, y, z, nx, ny, nz);
    if (in[id]) { out[id] = 1; return; }
    const int dx[6] = {+1,-1,0,0,0,0};
    const int dy[6] = {0,0,+1,-1,0,0};
    const int dz[6] = {0,0,0,0,+1,-1};
    uint8_t v = 0;
    for (int k = 0; k < 6; ++k) {
        int xx = x + dx[k], yy = y + dy[k], zz = z + dz[k];
        if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) continue;
        v |= in[idx3d(xx, yy, zz, nx, ny, nz)];
        if (v) break;
    }
    out[id] = v;
}

__global__ void k_erode6(const uint8_t* in, uint8_t* out, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    size_t id = idx3d(x, y, z, nx, ny, nz);
    if (!in[id]) { out[id] = 0; return; }
    const int dx[6] = {+1,-1,0,0,0,0};
    const int dy[6] = {0,0,+1,-1,0,0};
    const int dz[6] = {0,0,0,0,+1,-1};
    for (int k = 0; k < 6; ++k) {
        int xx = x + dx[k], yy = y + dy[k], zz = z + dz[k];
        if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) { out[id] = 0; return; }
        if (!in[idx3d(xx, yy, zz, nx, ny, nz)]) { out[id] = 0; return; }
    }
    out[id] = 1;
}

cudaError_t gpu_close3d(uint8_t* d_mask, int nx, int ny, int nz, int iterations,
                        cudaStream_t stream) {
    size_t vox = (size_t)nx * ny * nz;
    uint8_t* d_tmp = nullptr;
    cudaMalloc(&d_tmp, vox);
    dim3 block3(8, 8, 8);
    dim3 grid3((nx + block3.x - 1) / block3.x,
               (ny + block3.y - 1) / block3.y,
               (nz + block3.z - 1) / block3.z);

    for (int it = 0; it < iterations; ++it) {
        k_dilate6<<<grid3, block3, 0, stream>>>(d_mask, d_tmp, nx, ny, nz);
        k_erode6 <<<grid3, block3, 0, stream>>>(d_tmp, d_mask, nx, ny, nz);
    }
    cudaFree(d_tmp);
    return cudaGetLastError();
}

