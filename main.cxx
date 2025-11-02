#include <vtkDICOMImageReader.h>
#include <vtkImageData.h>
#include <vtkImageImport.h>
#include <vtkImageSliceMapper.h>
#include <vtkImageSlice.h>
#include <vtkImageProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <vtkImageAlgorithm.h>
#include <vtkGPUVolumeRayCastMapper.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkVolume.h>

#include <vtkFlyingEdges3D.h>
#include <vtkCleanPolyData.h>
#include <vtkDecimatePro.h>
#include <vtkWindowedSincPolyDataFilter.h>
#include <vtkPolyDataNormals.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkOBJWriter.h>
#include <vtkOBJExporter.h>
#include <vtkImageCast.h>

#include <cuda_runtime.h>

#include <vector>
#include <queue>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>
#include <filesystem>
#include "preprocess.cuh"

// viewer_opengl.cpp
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE 0
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// RAII for CUDA buffers
struct CudaBuf { void* p = nullptr; size_t n = 0; ~CudaBuf() { if (p) cudaFree(p); } };

static inline size_t idx3(int x, int y, int z, int nx, int ny, int nz) {
    return size_t(x) + size_t(nx) * (size_t(y) + size_t(ny) * size_t(z));
}

static void show2D(vtkImageAlgorithm* src, int slice, int nz,
    const char* title, double win, double lvl)
{
    const int sliceIndex = std::clamp(slice, 0, nz - 1);

    vtkNew<vtkImageSliceMapper> sMapper;
    sMapper->SetInputConnection(src->GetOutputPort());
    sMapper->SetOrientationToZ();
    sMapper->SetSliceNumber(sliceIndex);

    vtkNew<vtkImageSlice> sliceActor;
    sliceActor->SetMapper(sMapper);
    sliceActor->GetProperty()->SetColorWindow(win);
    sliceActor->GetProperty()->SetColorLevel(lvl);

    vtkNew<vtkRenderer> ren2D;
    ren2D->AddViewProp(sliceActor);
    ren2D->SetBackground(0.10, 0.10, 0.12);

    vtkNew<vtkRenderWindow> win2D;
    win2D->AddRenderer(ren2D);
    win2D->SetSize(800, 800);
    win2D->SetWindowName(title);

    vtkNew<vtkRenderWindowInteractor> iren2D;
    iren2D->SetRenderWindow(win2D);
    vtkNew<vtkInteractorStyleImage> style;
    iren2D->SetInteractorStyle(style);

    win2D->Render();
    ren2D->ResetCamera();
    win2D->Render();

    iren2D->Initialize();
    iren2D->Start();
}

// Keep only the two largest connected components (lungs)
static void keepLargestTwoComponents(std::vector<uint8_t>& mask, int nx, int ny, int nz)
{
    std::vector<int> label(mask.size(), -1);
    std::vector<size_t> sizes;
    const int dx[6] = { +1,-1, 0, 0, 0, 0 };
    const int dy[6] = { 0, 0,+1,-1, 0, 0 };
    const int dz[6] = { 0, 0, 0, 0,+1,-1 };

    int curLabel = 0;
    for (size_t s = 0; s < mask.size(); ++s) {
        if (!mask[s] || label[s] >= 0) continue;
        std::queue<size_t> q;
        q.push(s); label[s] = curLabel; size_t cnt = 1;
        while (!q.empty()) {
            size_t id = q.front(); q.pop();
            int z = int(id / (size_t(nx) * ny));
            size_t rem = id - size_t(z) * nx * ny;
            int y = int(rem / nx);
            int x = int(rem - size_t(y) * nx);
            for (int k = 0; k < 6; ++k) {
                int xx = x + dx[k], yy = y + dy[k], zz = z + dz[k];
                if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) continue;
                size_t nid = idx3(xx, yy, zz, nx, ny, nz);
                if (mask[nid] && label[nid] < 0) { label[nid] = curLabel; q.push(nid); ++cnt; }
            }
        }
        sizes.push_back(cnt);
        ++curLabel;
    }

    int keepA = -1, keepB = -1;
    size_t bestA = 0, bestB = 0;
    for (int i = 0; i < curLabel; ++i) {
        size_t s = sizes[i];
        if (s > bestA) { bestB = bestA; keepB = keepA; bestA = s; keepA = i; }
        else if (s > bestB) { bestB = s; keepB = i; }
    }

    for (size_t i = 0; i < mask.size(); ++i) {
        int L = label[i];
        mask[i] = (L == keepA || L == keepB) ? 1u : 0u;
    }
}

// Binary dilation (6-neighborhood)
static void dilate3D(std::vector<uint8_t>& m, int nx, int ny, int nz, int iterations = 1) {
    const int dx[6] = { +1,-1,0,0,0,0 }, dy[6] = { 0,0,+1,-1,0,0 }, dz[6] = { 0,0,0,0,+1,-1 };
    for (int it = 0; it < iterations; ++it) {
        std::vector<uint8_t> out = m;
        for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
            size_t id = idx3(x, y, z, nx, ny, nz);
            if (m[id]) continue;
            for (int k = 0; k < 6; ++k) {
                int xx = x + dx[k], yy = y + dy[k], zz = z + dz[k];
                if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) continue;
                if (m[idx3(xx, yy, zz, nx, ny, nz)]) { out[id] = 1; break; }
            }
        }
        m.swap(out);
    }
}

// Erosion (6-neighborhood)
static void erode3D(std::vector<uint8_t>& m, int nx, int ny, int nz, int iterations = 1) {
    const int dx[6] = { +1,-1,0,0,0,0 }, dy[6] = { 0,0,+1,-1,0,0 }, dz[6] = { 0,0,0,0,+1,-1 };
    for (int it = 0; it < iterations; ++it) {
        std::vector<uint8_t> out = m;
        for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
            size_t id = idx3(x, y, z, nx, ny, nz);
            if (!m[id]) continue;
            bool keep = true;
            for (int k = 0; k < 6; ++k) {
                int xx = x + dx[k], yy = y + dy[k], zz = z + dz[k];
                if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) { keep = false; break; }
                if (!m[idx3(xx, yy, zz, nx, ny, nz)]) { keep = false; break; }
            }
            out[id] = keep ? 1u : 0u;
        }
        m.swap(out);
    }
}




// Closing = dilate then erode
static void close3D(std::vector<uint8_t>& m, int nx, int ny, int nz, int iterations = 1) {
    dilate3D(m, nx, ny, nz, iterations);
    erode3D(m, nx, ny, nz, iterations);
}

struct StageTimer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0 = Clock::now();
    std::string mode;  // "gpu" or "cpu"
    std::vector<std::pair<std::string, double>> stamps;

    explicit StageTimer(std::string m) : mode(std::move(m)) {}

    void mark(const std::string& tag) {
        auto now = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - t0).count();
        stamps.emplace_back(tag, ms);
        std::cout << "[TIME] " << std::setw(8) << std::fixed << std::setprecision(3)
            << ms << " ms   " << tag << "\n";
    }

    void write_csv(const std::string& path,
        const std::string& dicomDir, int nx, int ny, int nz) {
        bool newFile = !std::ifstream(path).good();
        std::ofstream f(path, std::ios::app);
        if (!f) return;

        if (newFile) {
            f << "iso8601,mode,dicom,nx,ny,nz,total_ms";
            for (auto& p : stamps) f << "," << p.first;
            f << "\n";
        }

        auto n = std::chrono::system_clock::now();
        std::time_t tt = std::chrono::system_clock::to_time_t(n);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &tt);
#else
        localtime_r(&tt, &tm);
#endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);

        double total = stamps.empty() ? 0.0 : stamps.back().second;
        f << buf << "," << mode << "," << std::quoted(dicomDir) << ","
            << nx << "," << ny << "," << nz << ","
            << std::fixed << std::setprecision(3) << total;
        for (auto& p : stamps) f << "," << p.second;
        f << "\n";
    }
};

// Minimal shader helpers
static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0'); glGetShaderInfoLog(s, len, nullptr, log.data());
        throw std::runtime_error("Shader compile failed:\n" + log);
    }
    return s;
}
static GLuint link(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0'); glGetProgramInfoLog(p, len, nullptr, log.data());
        throw std::runtime_error("Program link failed:\n" + log);
    }
    return p;
}

// Unit cube (0..1)^3
static void makeCube(GLuint& vao, GLuint& vbo, GLuint& ibo, GLsizei& indexCount) {
    struct V { float p[3]; };
    const V verts[] = {
        {{0,0,0}},{{1,0,0}},{{1,1,0}},{{0,1,0}},
        {{0,0,1}},{{1,0,1}},{{1,1,1}},{{0,1,1}}
    };
    const uint16_t idx[] = {
        // front (z=0)
        0,1,2,  0,2,3,
        // back (z=1)
        4,6,5,  4,7,6,
        // left (x=0)
        0,3,7,  0,7,4,
        // right (x=1)
        1,5,6,  1,6,2,
        // bottom (y=0)
        0,4,5,  0,5,1,
        // top (y=1)
        3,2,6,  3,6,7
    };
    glCreateVertexArrays(1, &vao);
    glCreateBuffers(1, &vbo);
    glNamedBufferData(vbo, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexArrayVertexBuffer(vao, 0, vbo, 0, sizeof(V));
    glEnableVertexArrayAttrib(vao, 0);
    glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vao, 0, 0);

    glCreateBuffers(1, &ibo);
    glNamedBufferData(ibo, sizeof(idx), idx, GL_STATIC_DRAW);
    glVertexArrayElementBuffer(vao, ibo);

    indexCount = (GLsizei)(sizeof(idx) / sizeof(idx[0]));
}


static GLuint upload3D_R32F(int X, int Y, int Z, const float* data) {
    GLuint tex = 0;
    glCreateTextures(GL_TEXTURE_3D, 1, &tex);
    glTextureStorage3D(tex, 1, GL_R32F, X, Y, Z);
    glTextureSubImage3D(tex, 0, 0, 0, 0, X, Y, Z, GL_RED, GL_FLOAT, data);
    glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    return tex;
}


// Vertex & Fragment shaders
static const char* kVS = R"GLSL(
#version 450 core
layout(location=0) in vec3 inPos;   // object-space, cube [0,1]^3
uniform mat4 uModel, uView, uProj;
out vec3 vPosObj;                   // object-space position on the cube surface
out vec3 vPosWorld;                 // world-space (for debugging if needed)
void main(){
    vPosObj = inPos;
    vec4 wp = uModel * vec4(inPos,1.0);
    vPosWorld = wp.xyz;
    gl_Position = uProj * uView * wp;
}
)GLSL";

static const char* kFS = R"GLSL(
#version 450 core
in vec3 vPosObj;      // surface point on cube in object-space [0,1]^3
in vec3 vPosWorld;
out vec4 frag;

uniform sampler3D  uTexHU;
uniform usampler3D uTexMask;
uniform int  uUseMask;      // 0/1
uniform vec3 uSpacing;      // (dx,dy,dz) or (dz,dy,dx) — we’ll use as step scaling
uniform vec3 uCamPosObj;    // camera position in object space
uniform float uStepWorld;   // base world step size (scaled by spacing min)
uniform vec3 uTexSize;   // (nx, ny, nz)

uniform mat4 uModel;

// Axis-aligned box intersection for [0,1]^3
bool rayBox(vec3 ro, vec3 rd, out float t0, out float t1){
    vec3 inv = 1.0 / rd;
    vec3 tbot = (vec3(0.0)-ro) * inv;
    vec3 ttop = (vec3(1.0)-ro) * inv;
    vec3 tmin = min(tbot, ttop);
    vec3 tmax = max(tbot, ttop);
    t0 = max(max(tmin.x, tmin.y), tmin.z);
    t1 = min(min(tmax.x, tmax.y), tmax.z);
    return t1 > max(t0, 0.0);
}

// Simple transfer function: map normalized intensity to color/alpha
vec4 TF(float x){
    float a = smoothstep(0.30, 0.70, x);   // slightly stronger opacity ramp
    vec3  c = vec3(x);                     // grayscale
    return vec4(c, a);
}

// Optional window normalization if h_seg is raw HU-like float
float windowNorm(float hu, float center, float width){
    return clamp((hu - (center - width*0.5)) / width, 0.0, 1.0);
}

void main(){

    // Ray origin & direction in object space
    vec3 ro = uCamPosObj;
    vec3 rd = normalize(vPosObj - uCamPosObj);

    float tEnter, tExit;
    if (!rayBox(ro, rd, tEnter, tExit)) discard;  // miss the volume

    // March from entry
    float t = max(tEnter, 0.0);
    vec4 accum = vec4(0.0);
    // Scale step by minimal spacing so sampling density follows voxel size
   float maxDim = max(uTexSize.x, max(uTexSize.y, uTexSize.z));
float step = uStepWorld / maxDim;   // ~ texel-sized step


    // Jitter to reduce banding
    t += fract(sin(dot(gl_FragCoord.xy , vec2(12.9898,78.233))) * 43758.5453) * step;

    for (; t < tExit && accum.a < 0.98; t += step) {
        vec3 p = ro + t*rd;   // object-space sample position in [0,1]^3

        if (uUseMask == 1) {
            uint m = texture(uTexMask, p).r;
            if (m == 0u) continue;
        }

        // If your h_seg is already normalized [0..1], read directly:
        float hu = texture(uTexHU, p).r;

        // If h_seg is HU-like (e.g., -1024..~400), uncomment and set your window:
       
float v = texture(uTexHU, p).r;
        vec4 col = TF(v);

        // Front-to-back compositing
        accum.rgb += (1.0 - accum.a) * col.a * col.rgb;
        accum.a   += (1.0 - accum.a) * col.a;
    }

    // Tone map / gamma if desired
    frag = vec4(pow(accum.rgb, vec3(1.0/2.2)), accum.a);
}
)GLSL";


struct Camera {
    float yaw = 0.7f, pitch = 0.5f, dist = 2.0f;
    glm::mat4 proj{ 1.0f }, view{ 1.0f };
    glm::vec3 eye{ 0.0f, 0.0f, 2.0f };
    void update(int w, int h) {
        float aspect = (float)w / (float)h;
        proj = glm::perspective(glm::radians(45.0f), aspect, 0.05f, 10.0f);
        glm::vec3 target(0.5f, 0.5f, 0.5f);
        glm::vec3 dir(cosf(pitch) * cosf(yaw), sinf(pitch), cosf(pitch) * sinf(yaw));
        eye = target - dir * dist;
        view = glm::lookAt(eye, target, glm::vec3(0, 1, 0));
    }
};

// Upload 3D textures
static GLuint upload3D_R16F(int X, int Y, int Z, const float* data) {
    GLuint tex = 0;
    glCreateTextures(GL_TEXTURE_3D, 1, &tex);
    glTextureStorage3D(tex, 1, GL_R16F, X, Y, Z);         // GL_R32F if you prefer
    glTextureSubImage3D(tex, 0, 0, 0, 0, X, Y, Z, GL_RED, GL_FLOAT, data);
    glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    return tex;
}
static GLuint upload3D_R8UI(int X, int Y, int Z, const uint8_t* data) {
    GLuint tex = 0;
    glCreateTextures(GL_TEXTURE_3D, 1, &tex);
    glTextureStorage3D(tex, 1, GL_R8UI, X, Y, Z);
    glTextureSubImage3D(tex, 0, 0, 0, 0, X, Y, Z, GL_RED_INTEGER, GL_UNSIGNED_BYTE, data);
    glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    return tex;
}


int main(int argc, char** argv)
{// Decide which mode label to stamp into the CSV.
// If you compile two variants, change this to "cpu" for the CPU build.
    std::string mode = "gpu";
    StageTimer T(mode);
    T.mark("start");

    // ---------- defaults you want ----------
    std::string dicomDirStr = R"(C:\Users\Dur\OneDrive\Documents\FYP\4.000000-CHEST LUNG-98563)";
    int         userSlice = 61;
    bool only2D = false, showOrig = false;
    std::string objOutPath, scenePrefix;

    // Parse CLI to allow overrides (optional)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-DICOM" && i + 1 < argc)        dicomDirStr = argv[++i];
        else if (a == "-slice" && i + 1 < argc)   userSlice = std::atoi(argv[++i]);
        else if (a == "-only2D")                  only2D = true;
        else if (a == "-showorig")                showOrig = true;
        else if (a == "-exportObj" && i + 1 < argc)   objOutPath = argv[++i];
        else if (a == "-exportScene" && i + 1 < argc) scenePrefix = argv[++i];
        else if (a == "-h" || a == "--help") {
            std::cout
                << "Usage: app -DICOM <folder> [-slice <index>] [-only2D] [-showorig]\n"
                << "Defaults (used if flags omitted):\n"
                << "  -DICOM \"" << dicomDirStr << "\"\n"
                << "  -slice " << userSlice << "\n";
            return 0;
        }
    }

    // Optional safety: check the folder exists (comment out if you prefer)
    if (!std::filesystem::exists(dicomDirStr)) {
        std::cerr << "Error: DICOM folder not found: " << dicomDirStr << "\n";
        return 1;
    }
    // 1) Read DICOM
    vtkNew<vtkDICOMImageReader> reader;
    //reader->SetDirectoryName(dicomDir);
    reader->SetDirectoryName(dicomDirStr.c_str());

    reader->Update();
    T.mark("dicom_loaded");
    vtkImageData* img = reader->GetOutput();

    int nx = 0, ny = 0, nz = 0;
    int dims[3]; img->GetDimensions(dims);
    double spacing[3]; img->GetSpacing(spacing);
    double ori[3]; img->GetOrigin(ori);
    nx = dims[0], ny = dims[1], nz = dims[2];
    size_t vox = size_t(nx) * ny * nz;

    if (vox == 0) { std::cerr << "Empty volume.\n"; return 1; }
    int st = img->GetScalarType();
    if (st != VTK_SHORT && st != VTK_UNSIGNED_SHORT) {
        std::cerr << "Need 16-bit (short/ushort) volume. Got VTK scalar type: " << st << "\n";
        return 1;
    }

    // 2) Host buffers (pinned): blurred original + segmented display
    float* h_blur = nullptr;
    float* h_seg = nullptr;
    cudaHostAlloc(&h_blur, vox * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_seg, vox * sizeof(float), cudaHostAllocDefault);

    T.mark("memory allocated for volume (slice converted 3D volume) in Device");

    if (st == VTK_SHORT) {
        auto* p = static_cast<short*>(img->GetScalarPointer());
        for (size_t i = 0; i < vox; ++i) h_blur[i] = float(p[i]);
    }
    else {
        auto* p = static_cast<unsigned short*>(img->GetScalarPointer());
        for (size_t i = 0; i < vox; ++i) h_blur[i] = float(p[i]);
    }
    
    // 3) Device buffers + Gaussian kernel
    CudaBuf d_in, d_tmp, d_out, d_k;
    cudaMalloc(&d_in.p, vox * sizeof(float)); d_in.n = vox * sizeof(float);
    cudaMalloc(&d_tmp.p, vox * sizeof(float)); d_tmp.n = vox * sizeof(float);
    cudaMalloc(&d_out.p, vox * sizeof(float)); d_out.n = vox * sizeof(float);
    cudaMemcpy(d_in.p, h_blur, d_in.n, cudaMemcpyHostToDevice);

    

    int r = 2; float sigma = 1.0f;
    std::vector<float> hk(2 * r + 1); float sum = 0.f;
    for (int i = -r; i <= r; ++i) { float w = std::exp(-(i * i) / (2 * sigma * sigma)); hk[i + r] = w; sum += w; }
    for (auto& w : hk) w /= sum;
    cudaMalloc(&d_k.p, hk.size() * sizeof(float)); d_k.n = hk.size() * sizeof(float);
    cudaMemcpy(d_k.p, hk.data(), d_k.n, cudaMemcpyHostToDevice);
    T.mark("volume (slice converted 3D volume) sent over to the GPU");
    // 4) CUDA preprocess
    preprocess_volume(static_cast<float*>(d_in.p),
        static_cast<float*>(d_tmp.p),
        static_cast<float*>(d_out.p),
        static_cast<const float*>(d_k.p),
        nx, ny, nz, r, -650.f, -600.f, nullptr);
    cudaDeviceSynchronize();

    // 5) Download blurred
    cudaMemcpy(h_blur, d_out.p, d_out.n, cudaMemcpyDeviceToHost);
    T.mark("blurring applied to volume (slice converted 3D volume) for lung on GPU");

    float AIRLIKE_THRESH = -600.0f;   // voxels <= this are air-like
    float BODY_THRESH = -500.0f;   // voxels >  this are body barrier (loosened)

    std::vector<uint8_t> airMask(vox, 0), bodyMask(vox, 0);
    for (size_t i = 0; i < vox; ++i) {
        float v = h_blur[i];
        airMask[i] = (v <= AIRLIKE_THRESH) ? 1u : 0u;
        bodyMask[i] = (v > BODY_THRESH) ? 1u : 0u;
    }

    // Make the body barrier thicker so exterior flood can't slip inside
    dilate3D(bodyMask, nx, ny, nz, /*iterations=*/1);
  

    // ==== GPU accelerated exterior flood + largest-two components + closing ====
    // Allocate device masks
    CudaBuf d_air, d_body, d_exterior, d_lungs, d_tmpMask;
    cudaMalloc(&d_air.p, vox);        d_air.n = vox;
    cudaMalloc(&d_body.p, vox);       d_body.n = vox;
    cudaMalloc(&d_exterior.p, vox);   d_exterior.n = vox;
    cudaMalloc(&d_lungs.p, vox);      d_lungs.n = vox;
    // Upload air/body masks
    cudaMemcpy(d_air.p, airMask.data(), vox, cudaMemcpyHostToDevice);
    cudaMemcpy(d_body.p, bodyMask.data(), vox, cudaMemcpyHostToDevice);

    // 1) exterior = flood from border through air \ body
    gpu_exterior_air(static_cast<const uint8_t*>(d_air.p),
        static_cast<const uint8_t*>(d_body.p),
        static_cast<uint8_t*>(d_exterior.p),
        nx, ny, nz, nullptr);
    T.mark("flood exterior wall fill applied to blurred and thresholded volume (slice converted 3D volume) for lung on GPU");
    // 2) lungsCandidate = air & ~exterior
    CudaBuf d_lungCand; cudaMalloc(&d_lungCand.p, vox); d_lungCand.n = vox;
    gpu_andnot(static_cast<const uint8_t*>(d_air.p),
        static_cast<const uint8_t*>(d_exterior.p),
        static_cast<uint8_t*>(d_lungCand.p), vox, nullptr);
    T.mark("air & ~exterior applied to volume (slice converted 3D volume) for lung on GPU, which provides only lung segmented");
    // 3) Keep only the two largest components
    gpu_label_keep_two(static_cast<const uint8_t*>(d_lungCand.p),
        static_cast<uint8_t*>(d_lungs.p),
        nx, ny, nz, nullptr, nullptr);
    T.mark("Keep two largest components applied to volume (slice converted 3D volume) for lung on GPU, which removes unconnected objects");
    // 4) Morphological closing to seal small gaps (3 iters)
    gpu_close3d(static_cast<uint8_t*>(d_lungs.p), nx, ny, nz, 3, nullptr);
    T.mark("erosion & dilation applied to volume (slice converted 3D volume) for lung on GPU");
    // Download to host lungMask
    //std::vector<uint8_t> lungMask(vox, 0);
    std::vector<uint8_t> lungMask;
    lungMask.assign(vox, 0);

    cudaMemcpy(lungMask.data(), d_lungs.p, vox, cudaMemcpyDeviceToHost);
    /*  size_t lungCnt = 0;
      for (auto v : lungMask) lungCnt += (v != 0);
      std::cerr << "[DEBUG] air=" << airCnt
          << " body=" << bodyCnt
          << " lungs=" << lungCnt << " vox\n";
      if (lungCnt == 0) {
          std::cerr << "[WARN] Lung mask is empty. "
              "Increase body dilation or relax BODY_THRESH (e.g., -200).\n";
      }*/
    T.mark("segmented volume copied back from GPU to CPU");
    

    // Build HU-masked volume for 2D/3D rendering: keep HU in lungs, set outside to air
    for (size_t i = 0; i < vox; ++i) {
        h_seg[i] = lungMask[i] ? h_blur[i] : -1024.0f;  // air outside lungs
    }
    T.mark("lung_mask_ready and displaying");
    // 2D preview inputs
    vtkNew<vtkImageImport> importBlur;
    importBlur->CopyImportVoidPointer(h_blur, static_cast<vtkIdType>(vox * sizeof(float)));
    importBlur->SetDataScalarTypeToFloat();
    importBlur->SetNumberOfScalarComponents(1);
    importBlur->SetWholeExtent(0, nx - 1, 0, ny - 1, 0, nz - 1);
    importBlur->SetDataExtent(0, nx - 1, 0, ny - 1, 0, nz - 1);
    importBlur->SetDataSpacing(spacing[0], spacing[1], spacing[2]);
    importBlur->Update();
    //std::cerr << "now 2D blur imported.\n";

    vtkNew<vtkImageImport> importSeg;
    importSeg->CopyImportVoidPointer(h_seg, static_cast<vtkIdType>(vox * sizeof(float)));
    importSeg->SetDataScalarTypeToFloat();
    importSeg->SetNumberOfScalarComponents(1);
    importSeg->SetWholeExtent(0, nx - 1, 0, ny - 1, 0, nz - 1);
    importSeg->SetDataExtent(0, nx - 1, 0, ny - 1, 0, nz - 1);
    importSeg->SetDataSpacing(spacing[0], spacing[1], spacing[2]);
    importSeg->Update();

    //std::cerr << "now 2D importseg done .\n";

    int slice = (userSlice >= 0 && userSlice < nz) ? userSlice : nz / 2;
    if (showOrig) show2D(importBlur, slice, nz, "Original slice (blurred)", 1500, -600);
    show2D(importSeg, slice, nz, "Segmented slice (lungs only, top view)", 1500, -600);

    if (only2D) {
        cudaFreeHost(h_blur); cudaFreeHost(h_seg);
        return 0;
    }



    try {
        T.mark("lung_mask sending over to OpenGL for texture conversion");
    // ---- OpenGL init (after VTK/segmentation is done) ----
    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(1200, 900, "3D Textured Lungs (Ray March)", nullptr, nullptr);
    if (!win) { fprintf(stderr, "Failed to create window\n"); return -1; }
    glfwMakeContextCurrent(win);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to load GL\n"); return -1;
    }
    glEnable(GL_DEPTH_TEST);

    // set scroll callback ONCE
    static double g_scroll = 0.0;
    glfwSetScrollCallback(win, [](GLFWwindow*, double, double yoff) { g_scroll += yoff; });
    T.mark("compiling and linking shaders");
    // --- Build pipeline ---
    GLuint vs = compile(GL_VERTEX_SHADER, kVS);
    GLuint fs = compile(GL_FRAGMENT_SHADER, kFS);
    GLuint prog = link(vs, fs);
    glDeleteShader(vs); glDeleteShader(fs);

    GLuint vao, vbo, ibo; GLsizei indexCount = 0;
    makeCube(vao, vbo, ibo, indexCount);

    // --- Upload textures ---
    T.mark("Uploading textures");
    if (!h_seg) { fprintf(stderr, "h_seg is null!\n"); return -2; }
    const float center = -600.f, width = 1500.f;
    const float lo = center - 0.5f * width, hi = center + 0.5f * width;
    for (size_t i = 0; i < vox; ++i) {
        float hu = h_seg[i];                        // HU (lungs kept, outside -1024)
        float v = (hu - lo) / (hi - lo);
        h_seg[i] = std::clamp(v, 0.0f, 1.0f);       // overwrite with [0..1]
    }

    GLuint texHU = upload3D_R16F(nx, ny, nz, h_seg);

    GLuint texMask = 0;
    int useMask = !lungMask.empty() ? 1 : 0;
    if (useMask) texMask = upload3D_R8UI(nx, ny, nz, lungMask.data());

    // --- Uniform locations ---
    GLint uModelLoc = glGetUniformLocation(prog, "uModel");
    GLint uViewLoc = glGetUniformLocation(prog, "uView");
    GLint uProjLoc = glGetUniformLocation(prog, "uProj");
    GLint uTexHULoc = glGetUniformLocation(prog, "uTexHU");
    GLint uTexMaskLoc = glGetUniformLocation(prog, "uTexMask");
    GLint uUseMaskLoc = glGetUniformLocation(prog, "uUseMask");
    GLint uSpacingLoc = glGetUniformLocation(prog, "uSpacing");
    GLint uCamPosLoc = glGetUniformLocation(prog, "uCamPosObj");
    GLint uStepLoc = glGetUniformLocation(prog, "uStepWorld");

    // Model is identity; cube is [0,1]^3

    Camera cam;
    int W = 1200, H = 900; glfwGetFramebufferSize(win, &W, &H);
    cam.update(W, H);


    // ---- before the render loop ----
    glm::vec3 phys((float)nx * (float)spacing[0],
        (float)ny * (float)spacing[1],
        (float)nz * (float)spacing[2]);
    float m = std::max(phys.x, std::max(phys.y, phys.z));
    glm::mat4 model = glm::scale(glm::mat4(1.0f), phys / m);
    glm::mat4 invModel = glm::inverse(model);

    GLint uTexSizeLoc = glGetUniformLocation(prog, "uTexSize");
    
    // Bind sampler units once
    glUseProgram(prog);
    glUniform1i(uTexHULoc, 0);
    glUniform1i(uTexMaskLoc, 1);
    glUniform3f(uTexSizeLoc, (float)nx, (float)ny, (float)nz);

    glUseProgram(0);



    T.mark("running the fragment shader for each fragment");
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        glfwGetFramebufferSize(win, &W, &H);
        glViewport(0, 0, W, H);

        // simple orbit (LMB drag)
        static double ox = -1, oy = -1; static bool dragging = false;
        if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            double x, y; glfwGetCursorPos(win, &x, &y);
            if (!dragging) { dragging = true; ox = x; oy = y; }
            else {
                cam.yaw += float((x - ox) * 0.005);
                cam.pitch += float((y - oy) * -0.005);
                cam.pitch = glm::clamp(cam.pitch, -1.2f, 1.2f);
                ox = x; oy = y;
            }
        }
        else dragging = false;

        // zoom via scroll
        cam.dist = glm::clamp(cam.dist * powf(0.9f, (float)g_scroll), 0.5f, 6.0f);
        g_scroll = 0.0;

        cam.update(W, H);

        glClearColor(0.07f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);
        glBindVertexArray(vao);

        glUniformMatrix4fv(uModelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(uViewLoc, 1, GL_FALSE, glm::value_ptr(cam.view));
        glUniformMatrix4fv(uProjLoc, 1, GL_FALSE, glm::value_ptr(cam.proj));

        // camera in object space
        glm::vec3 camPosObj = glm::vec3(invModel * glm::vec4(cam.eye, 1));
        glUniform3fv(uCamPosLoc, 1, glm::value_ptr(camPosObj));

        // spacing uniform (rename to avoid clash with double spacing[3])
        glm::vec3 spacingVec((float)spacing[0], (float)spacing[1], (float)spacing[2]);
        glUniform3fv(uSpacingLoc, 1, glm::value_ptr(spacingVec));
        glUniform1f(uStepLoc, 1.5f);     // quality knob
        glUniform1i(uUseMaskLoc, 0);



        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, texHU);
        if (useMask) { glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_3D, texMask); }

        glDisable(GL_CULL_FACE);
       
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_SHORT, 0);
        
        glfwSwapBuffers(win);
    }

    glfwDestroyWindow(win);
    glfwTerminate();
  
}

    catch (const std::exception& e) {
        fprintf(stderr, "[FATAL] %s\n", e.what());
        return 1;
    }
    // pinned buffers
    T.mark("closed window and Freed memory ");
    cudaFreeHost(h_blur);
    cudaFreeHost(h_seg);
    return 0;


 
}
