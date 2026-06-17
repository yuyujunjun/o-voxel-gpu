#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#define FDG_BLOCK_SIZE 256
#define FDG_EMPTY_KEY UINT64_MAX
#define FDG_EMPTY_VAL UINT32_MAX

// GPU-side float3/int3 structs (no Eigen dependency)
struct fdg_float3 { float x, y, z; __device__ float& operator[](int i) { return (&x)[i]; } };
struct fdg_int3   { int x, y, z;   __device__ int&   operator[](int i) { return (&x)[i]; } };

// ---------------------------------------------------------------------------
// Voxel key encoding — pack 3 int coords into uint64_t (21 bits each)
// ---------------------------------------------------------------------------
__forceinline__ __device__ uint64_t encode_voxel_key(int x, int y, int z) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 42) |
           (static_cast<uint64_t>(static_cast<uint32_t>(y)) << 21) |
           (static_cast<uint64_t>(static_cast<uint32_t>(z)));
}

__forceinline__ __device__ void decode_voxel_key(uint64_t key, int& x, int& y, int& z) {
    x = static_cast<int>(key >> 42);
    y = static_cast<int>((key >> 21) & 0x1FFFFF);
    z = static_cast<int>(key & 0x1FFFFF);
}

// ---------------------------------------------------------------------------
// 64-bit Murmur3 finalizer (avalanche mixing for hash table)
// ---------------------------------------------------------------------------
__forceinline__ __device__ uint64_t murmur3_fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

__forceinline__ __device__ uint32_t hash_key(uint64_t key, uint32_t capacity) {
    return static_cast<uint32_t>(murmur3_fmix64(key) % capacity);
}

// ---------------------------------------------------------------------------
// GPU hash table: get-or-create a voxel entry
// Returns the voxel index (new or existing).
// hash_keys:   [capacity] uint64_t, initialised to FDG_EMPTY_KEY
// hash_vals:   [capacity] uint32_t, initialised to FDG_EMPTY_VAL
// voxel_count: pointer to single uint32_t atomic counter (init 0)
// ---------------------------------------------------------------------------
__forceinline__ __device__ uint32_t get_or_create_voxel(
    uint64_t key, int x, int y, int z,
    uint64_t* hash_keys, uint32_t* hash_vals,
    uint32_t* voxel_count, int32_t* voxel_coords,
    uint32_t capacity)
{
    uint32_t slot = hash_key(key, capacity);
    uint32_t probe = 0;
    while (probe < capacity) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long*>(&hash_keys[slot]),
            static_cast<unsigned long long>(FDG_EMPTY_KEY),
            static_cast<unsigned long long>(key));
        if (prev == FDG_EMPTY_KEY) {
            // Won the slot — allocate a fresh voxel index
            uint32_t idx = atomicAdd(voxel_count, 1u);
            voxel_coords[idx * 3 + 0] = x;
            voxel_coords[idx * 3 + 1] = y;
            voxel_coords[idx * 3 + 2] = z;
            // Fence to ensure coords and value are visible before other threads read them
            __threadfence();
            hash_vals[slot] = idx;
            return idx;
        } else if (prev == key) {
            // Key already exists — read the stored index
            // Use volatile pointer to prevent compiler from optimizing away the re-read
            volatile uint32_t* vptr = &hash_vals[slot];
            uint32_t val = *vptr;
            while (val == FDG_EMPTY_VAL) {
                val = *vptr;
            }
            return val;
        }
        // Collision — linear probe
        slot = (slot + 1u) % capacity;
        ++probe;
    }
    // Hash table full — return sentinel (caller must check)
    return FDG_EMPTY_VAL;
}

// ---------------------------------------------------------------------------
// GPU hash table: read-only lookup (for face_qef / boundry_qef)
// Returns voxel index or FDG_EMPTY_VAL if not found.
// ---------------------------------------------------------------------------
__forceinline__ __device__ uint32_t hashmap_lookup_voxel(
    uint64_t key,
    const uint64_t* hash_keys, const uint32_t* hash_vals,
    uint32_t capacity)
{
    uint32_t slot = hash_key(key, capacity);
    uint32_t probe = 0;
    while (probe < capacity) {
        uint64_t k = hash_keys[slot];
        if (k == FDG_EMPTY_KEY) return FDG_EMPTY_VAL;
        if (k == key) return hash_vals[slot];
        slot = (slot + 1u) % capacity;
        ++probe;
    }
    return FDG_EMPTY_VAL;
}

// ---------------------------------------------------------------------------
// Closed-form 3x3 linear solver: A * x = b
// A stored as a[9] row-major. b[3]. Solution written to x[3].
// Returns true on success, false if singular (det < eps).
// ---------------------------------------------------------------------------
__forceinline__ __device__ bool solve_3x3(const float* A, const float* b, float* x, float eps = 1e-12f) {
    // Cofactor expansion
    float a00 = A[4] * A[8] - A[5] * A[7];
    float a01 = A[5] * A[6] - A[3] * A[8];
    float a02 = A[3] * A[7] - A[4] * A[6];
    float det = A[0] * a00 + A[1] * a01 + A[2] * a02;
    if (fabsf(det) < eps) return false;

    float a10 = A[2] * A[7] - A[1] * A[8];
    float a11 = A[0] * A[8] - A[2] * A[6];
    float a12 = A[1] * A[6] - A[0] * A[7];

    float a20 = A[1] * A[5] - A[2] * A[4];
    float a21 = A[2] * A[3] - A[0] * A[5];
    float a22 = A[0] * A[4] - A[1] * A[3];

    float inv_det = 1.0f / det;
    // x = A^{-1} * b  (cofactor transpose / det * b)
    x[0] = (a00 * b[0] + a10 * b[1] + a20 * b[2]) * inv_det;
    x[1] = (a01 * b[0] + a11 * b[1] + a21 * b[2]) * inv_det;
    x[2] = (a02 * b[0] + a12 * b[1] + a22 * b[2]) * inv_det;
    return true;
}

// ---------------------------------------------------------------------------
// Closed-form 2x2 linear solver: A * x = b
// A stored as a[4] row-major: [a00 a01; a10 a11]
// Returns true on success.
// ---------------------------------------------------------------------------
__forceinline__ __device__ bool solve_2x2(const float* A, const float* b, float* x, float eps = 1e-12f) {
    float det = A[0] * A[3] - A[1] * A[2];
    if (fabsf(det) < eps) return false;
    float inv_det = 1.0f / det;
    x[0] = (A[3] * b[0] - A[1] * b[1]) * inv_det;
    x[1] = (A[0] * b[1] - A[2] * b[0]) * inv_det;
    return true;
}

// ---------------------------------------------------------------------------
// QEF error evaluation: err = p^T * Q * p  (p is [x,y,z,1]^T, Q is 4x4)
// ---------------------------------------------------------------------------
__forceinline__ __device__ float eval_qef_error(const float* Q, const fdg_float3& p) {
    float v[4] = {p.x, p.y, p.z, 1.0f};
    // v^T * Q * v
    float row0 = Q[ 0]*v[0] + Q[ 1]*v[1] + Q[ 2]*v[2] + Q[ 3]*v[3];
    float row1 = Q[ 4]*v[0] + Q[ 5]*v[1] + Q[ 6]*v[2] + Q[ 7]*v[3];
    float row2 = Q[ 8]*v[0] + Q[ 9]*v[1] + Q[10]*v[2] + Q[11]*v[3];
    float row3 = Q[12]*v[0] + Q[13]*v[1] + Q[14]*v[2] + Q[15]*v[3];
    return v[0]*row0 + v[1]*row1 + v[2]*row2 + v[3]*row3;
}

// ---------------------------------------------------------------------------
// Add QEF matrix contribution: dest += weight * Q
// Q and dest are 16-element float arrays (row-major 4x4).
// ---------------------------------------------------------------------------
__forceinline__ __device__ void add_qef(float* dest, const float* Q, float weight) {
    for (int i = 0; i < 16; ++i) {
        atomicAdd(&dest[i], weight * Q[i]);
    }
}

// ---------------------------------------------------------------------------
// Compute face QEF matrix from three vertices
// Q = plane * plane^T where plane = [n.x, n.y, n.z, -n·v0]
// ---------------------------------------------------------------------------
__forceinline__ __device__ void compute_face_qef(
    const fdg_float3& v0, const fdg_float3& v1, const fdg_float3& v2,
    float* Q)
{
    // Edge vectors
    float e0x = v1.x - v0.x, e0y = v1.y - v0.y, e0z = v1.z - v0.z;
    float e1x = v2.x - v1.x, e1y = v2.y - v1.y, e1z = v2.z - v1.z;

    // Normal = e0 x e1
    float nx = e0y * e1z - e0z * e1y;
    float ny = e0z * e1x - e0x * e1z;
    float nz = e0x * e1y - e0y * e1x;
    float inv_len = rsqrtf(nx*nx + ny*ny + nz*nz + 1e-30f);
    nx *= inv_len; ny *= inv_len; nz *= inv_len;

    float nw = -(nx * v0.x + ny * v0.y + nz * v0.z);

    // Q = plane * plane^T (outer product, stored row-major)
    Q[ 0] = nx*nx; Q[ 1] = nx*ny; Q[ 2] = nx*nz; Q[ 3] = nx*nw;
    Q[ 4] = ny*nx; Q[ 5] = ny*ny; Q[ 6] = ny*nz; Q[ 7] = ny*nw;
    Q[ 8] = nz*nx; Q[ 9] = nz*ny; Q[10] = nz*nz; Q[11] = nz*nw;
    Q[12] = nw*nx; Q[13] = nw*ny; Q[14] = nw*nz; Q[15] = nw*nw;
}

// ---------------------------------------------------------------------------
// Compute boundary edge QEF matrix
// Q = [I - d*d^T,  -(I - d*d^T)*v0;  -v0^T*(I-d*d^T),  v0^T*(I-d*d^T)*v0]
// ---------------------------------------------------------------------------
__forceinline__ __device__ void compute_boundary_qef(
    const fdg_float3& v0, const fdg_float3& v1,
    float* Q)
{
    float dx = v1.x - v0.x, dy = v1.y - v0.y, dz = v1.z - v0.z;
    float len2 = dx*dx + dy*dy + dz*dz;
    if (len2 < 1e-12f) { for (int i=0;i<16;++i) Q[i]=0.0f; return; }
    float inv_len = rsqrtf(len2);
    dx *= inv_len; dy *= inv_len; dz *= inv_len;

    // A = I - d*d^T (3x3 projection matrix)
    float A00 = 1.0f - dx*dx, A01 = -dx*dy,     A02 = -dx*dz;
    float A10 = -dy*dx,      A11 = 1.0f - dy*dy, A12 = -dy*dz;
    float A20 = -dz*dx,      A21 = -dz*dy,       A22 = 1.0f - dz*dz;

    // b = -A * v0
    float b0 = -(A00*v0.x + A01*v0.y + A02*v0.z);
    float b1 = -(A10*v0.x + A11*v0.y + A12*v0.z);
    float b2 = -(A20*v0.x + A21*v0.y + A22*v0.z);

    // c = v0^T * A * v0
    float Av0_x = A00*v0.x + A01*v0.y + A02*v0.z;
    float Av0_y = A10*v0.x + A11*v0.y + A12*v0.z;
    float Av0_z = A20*v0.x + A21*v0.y + A22*v0.z;
    float c = v0.x*Av0_x + v0.y*Av0_y + v0.z*Av0_z;

    Q[ 0]=A00; Q[ 1]=A01; Q[ 2]=A02; Q[ 3]=b0;
    Q[ 4]=A10; Q[ 5]=A11; Q[ 6]=A12; Q[ 7]=b1;
    Q[ 8]=A20; Q[ 9]=A21; Q[10]=A22; Q[11]=b2;
    Q[12]=b0;  Q[13]=b1;  Q[14]=b2;  Q[15]=c;
}

// ---------------------------------------------------------------------------
// Context struct for scan_half device function (avoids lambda in kernel)
// ---------------------------------------------------------------------------
struct ScanHalfCtx {
    float voxel_size_ax0, voxel_size_ax1, voxel_size_ax2;
    int grid_min_ax0, grid_max_ax0, grid_min_ax2, grid_max_ax2;
    int grid_min_x, grid_min_y, grid_min_z;
    int grid_max_x, grid_max_y, grid_max_z;
    int ax0, ax1, ax2;
    uint64_t* hash_keys;
    uint32_t* hash_vals;
    uint32_t* voxel_count;
    int32_t* voxel_coords;
    float* voxel_means;
    float* voxel_cnt;
    uint32_t* voxel_intersected;
    float* voxel_qefs;
    float* Q_face;
    uint32_t capacity;
};

__device__ void scan_half(
    int row_start, int row_end,
    double t0x, double t0y, double t0z,
    double t1x, double t1y, double t1z,
    double t2x, double t2y, double t2z,
    const ScanHalfCtx& ctx)
{
    for (int y_idx = row_start; y_idx < row_end; ++y_idx) {
        double y = (y_idx + 1) * ctx.voxel_size_ax1;

        double t3x, t3z;
        if (fabs(t0y - t1y) < 1e-12) {
            t3x = t0x; t3z = t0z;
        } else {
            double alpha = (y - t0y) / (t1y - t0y);
            t3x = (1.0 - alpha) * t0x + alpha * t1x;
            t3z = (1.0 - alpha) * t0z + alpha * t1z;
        }

        double t4x, t4z;
        if (fabs(t0y - t2y) < 1e-12) {
            t4x = t0x; t4z = t0z;
        } else {
            double alpha = (y - t0y) / (t2y - t0y);
            t4x = (1.0 - alpha) * t0x + alpha * t2x;
            t4z = (1.0 - alpha) * t0z + alpha * t2z;
        }

        if (t3x > t4x) { double tmp; tmp = t3x; t3x = t4x; t4x = tmp; tmp = t3z; t3z = t4z; t4z = tmp; }

        int line_start = max(min(int(t3x / ctx.voxel_size_ax0), ctx.grid_max_ax0 - 1), ctx.grid_min_ax0);
        int line_end   = max(min(int(t4x / ctx.voxel_size_ax0), ctx.grid_max_ax0 - 1), ctx.grid_min_ax0);

        for (int x_idx = line_start; x_idx < line_end; ++x_idx) {
            double x = (x_idx + 1) * ctx.voxel_size_ax0;
            double z;
            if (fabs(t4x - t3x) < 1e-12) {
                z = t3z;
            } else {
                double alpha = (x - t3x) / (t4x - t3x);
                z = (1.0 - alpha) * t3z + alpha * t4z;
            }
            int z_idx = int(z / ctx.voxel_size_ax2);
            if (z_idx < ctx.grid_min_ax2 || z_idx >= ctx.grid_max_ax2) continue;

            for (int dx = 0; dx < 2; ++dx) {
                for (int dy = 0; dy < 2; ++dy) {
                    int cx = 0, cy = 0, cz = 0;
                    if (ctx.ax0 == 0) cx = x_idx + dx; else if (ctx.ax0 == 1) cy = x_idx + dx; else cz = x_idx + dx;
                    if (ctx.ax1 == 0) cx = y_idx + dy; else if (ctx.ax1 == 1) cy = y_idx + dy; else cz = y_idx + dy;
                    if (ctx.ax2 == 0) cx = z_idx; else if (ctx.ax2 == 1) cy = z_idx; else cz = z_idx;

                    if (cx < ctx.grid_min_x || cx >= ctx.grid_max_x) continue;
                    if (cy < ctx.grid_min_y || cy >= ctx.grid_max_y) continue;
                    if (cz < ctx.grid_min_z || cz >= ctx.grid_max_z) continue;

                    uint64_t key = encode_voxel_key(cx, cy, cz);
                    uint32_t idx = get_or_create_voxel(
                        key, cx, cy, cz,
                        ctx.hash_keys, ctx.hash_vals, ctx.voxel_count,
                        ctx.voxel_coords, ctx.capacity);

                    if (idx == FDG_EMPTY_VAL) continue;

                    float intersect_coord[3];
                    intersect_coord[ctx.ax0] = (float)x;
                    intersect_coord[ctx.ax1] = (float)y;
                    intersect_coord[ctx.ax2] = (float)z;
                    atomicAdd(&ctx.voxel_means[idx * 3 + 0], intersect_coord[0]);
                    atomicAdd(&ctx.voxel_means[idx * 3 + 1], intersect_coord[1]);
                    atomicAdd(&ctx.voxel_means[idx * 3 + 2], intersect_coord[2]);
                    atomicAdd(&ctx.voxel_cnt[idx], 1.0f);

                    if (dx == 0 && dy == 0) {
                        atomicOr(&ctx.voxel_intersected[idx], 1u << ctx.ax2);
                    }

                    add_qef(ctx.voxel_qefs + idx * 16, ctx.Q_face, 1.0f);
                }
            }
        }
    }
}

// ===========================================================================
// Kernel declarations
// ===========================================================================

__global__ void prepare_triangles_kernel(
    int F,
    const float* vertices, const int32_t* faces,
    float* triangles, int32_t* tri_indices);

__global__ void intersect_qef_cuda_kernel(
    int N_tri,
    const float* triangles,
    float voxel_size_x, float voxel_size_y, float voxel_size_z,
    int grid_min_x, int grid_min_y, int grid_min_z,
    int grid_max_x, int grid_max_y, int grid_max_z,
    uint64_t* hash_keys, uint32_t* hash_vals,
    uint32_t* voxel_count,
    int32_t* voxel_coords,
    float* voxel_means,
    float* voxel_cnt,
    uint32_t* voxel_intersected,
    float* voxel_qefs,
    uint32_t capacity);

__global__ void face_qef_cuda_kernel(
    int N_tri,
    const float* triangles,
    float voxel_size_x, float voxel_size_y, float voxel_size_z,
    int grid_min_x, int grid_min_y, int grid_min_z,
    int grid_max_x, int grid_max_y, int grid_max_z,
    float face_weight,
    const uint64_t* hash_keys, const uint32_t* hash_vals,
    float* voxel_qefs,
    uint32_t capacity);

__global__ void boundry_qef_cuda_kernel(
    int N_edges,
    const float* boundries,
    float voxel_size_x, float voxel_size_y, float voxel_size_z,
    int grid_min_x, int grid_min_y, int grid_min_z,
    int grid_max_x, int grid_max_y, int grid_max_z,
    float boundary_weight,
    const uint64_t* hash_keys, const uint32_t* hash_vals,
    float* voxel_qefs,
    uint32_t capacity);

__global__ void qef_solve_cuda_kernel(
    uint32_t num_voxels,
    const int32_t* voxel_coords,
    const float* voxel_means,
    const float* voxel_cnt,
    float voxel_size_x, float voxel_size_y, float voxel_size_z,
    float regularization_weight,
    const float* voxel_qefs,
    float* dual_vertices,
    uint32_t* voxel_intersected);

__global__ void extract_edges_kernel(
    int F,
    const int32_t* faces,
    uint64_t* edge_keys, int32_t* edge_v0, int32_t* edge_v1);

__global__ void filter_boundary_edges_kernel(
    int N_edges_total,
    const uint64_t* edge_keys, const int32_t* edge_v0, const int32_t* edge_v1,
    const float* vertices, int32_t* boundry_count, float* boundries);
