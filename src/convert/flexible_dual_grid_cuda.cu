#include "flexible_dual_grid.cuh"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/zip_function.h>

// ===========================================================================
// prepare_triangles_kernel — gather triangle vertices from indexed mesh
// ===========================================================================
__global__ void prepare_triangles_kernel(
    int F,
    const float* vertices, const int32_t* faces,
    float* triangles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    for (int v = 0; v < 3; ++v) {
        int vi = faces[tid * 3 + v];  // vertex index
        triangles[(tid * 3 + v) * 3 + 0] = vertices[vi * 3 + 0];
        triangles[(tid * 3 + v) * 3 + 1] = vertices[vi * 3 + 1];
        triangles[(tid * 3 + v) * 3 + 2] = vertices[vi * 3 + 2];
    }
}

// ===========================================================================
// intersect_qef_cuda_kernel
// One thread per triangle.  Scan-line algorithm in 3 axis directions.
// ===========================================================================
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
    uint32_t capacity)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_tri) return;

    float voxel_size[3] = {voxel_size_x, voxel_size_y, voxel_size_z};
    int grid_min[3] = {grid_min_x, grid_min_y, grid_min_z};
    int grid_max[3] = {grid_max_x, grid_max_y, grid_max_z};

    // Load triangle vertices
    const float* tv = triangles + tid * 9;  // 3 vertices * 3 floats
    fdg_float3 v0{tv[0], tv[1], tv[2]};
    fdg_float3 v1{tv[3], tv[4], tv[5]};
    fdg_float3 v2{tv[6], tv[7], tv[8]};

    // Compute face normal and Q matrix
    float Q_face[16];
    compute_face_qef(v0, v1, v2, Q_face);

    // Scan-line fill from three axis directions
    for (int ax2 = 0; ax2 < 3; ++ax2) {
        int ax0 = (ax2 + 1) % 3;
        int ax1 = (ax2 + 2) % 3;

        // Project vertices to (ax0, ax1, ax2) space using double precision
        double t[3][3] = {
            {(double)v0[ax0], (double)v0[ax1], (double)v0[ax2]},
            {(double)v1[ax0], (double)v1[ax1], (double)v1[ax2]},
            {(double)v2[ax0], (double)v2[ax1], (double)v2[ax2]}
        };

        // Sort by y (ax1 coordinate) using simple index sort
        int order[3] = {0, 1, 2};
        if (t[order[0]][1] > t[order[1]][1]) { int tmp = order[0]; order[0] = order[1]; order[1] = tmp; }
        if (t[order[1]][1] > t[order[2]][1]) { int tmp = order[1]; order[1] = order[2]; order[2] = tmp; }
        if (t[order[0]][1] > t[order[1]][1]) { int tmp = order[0]; order[0] = order[1]; order[1] = tmp; }

        double* t0 = t[order[0]];
        double* t1 = t[order[1]];
        double* t2 = t[order[2]];

        int start = max(min(int(t0[1] / voxel_size[ax1]), grid_max[ax1] - 1), grid_min[ax1]);
        int mid   = max(min(int(t1[1] / voxel_size[ax1]), grid_max[ax1] - 1), grid_min[ax1]);
        int end   = max(min(int(t2[1] / voxel_size[ax1]), grid_max[ax1] - 1), grid_min[ax1]);

        ScanHalfCtx ctx;
        ctx.voxel_size_ax0 = voxel_size[ax0];
        ctx.voxel_size_ax1 = voxel_size[ax1];
        ctx.voxel_size_ax2 = voxel_size[ax2];
        ctx.grid_min_ax0 = grid_min[ax0];
        ctx.grid_max_ax0 = grid_max[ax0];
        ctx.grid_min_ax2 = grid_min[ax2];
        ctx.grid_max_ax2 = grid_max[ax2];
        ctx.grid_min_x = grid_min_x;
        ctx.grid_min_y = grid_min_y;
        ctx.grid_min_z = grid_min_z;
        ctx.grid_max_x = grid_max_x;
        ctx.grid_max_y = grid_max_y;
        ctx.grid_max_z = grid_max_z;
        ctx.ax0 = ax0;
        ctx.ax1 = ax1;
        ctx.ax2 = ax2;
        ctx.hash_keys = hash_keys;
        ctx.hash_vals = hash_vals;
        ctx.voxel_count = voxel_count;
        ctx.voxel_coords = voxel_coords;
        ctx.voxel_means = voxel_means;
        ctx.voxel_cnt = voxel_cnt;
        ctx.voxel_intersected = voxel_intersected;
        ctx.voxel_qefs = voxel_qefs;
        ctx.Q_face = Q_face;
        ctx.capacity = capacity;

        // First half: t0 → t1 along t0→t2
        scan_half(start, mid, t0[0], t0[1], t0[2], t1[0], t1[1], t1[2], t2[0], t2[1], t2[2], ctx);
        // Second half: t2 → t1 along t2→t0  (inverted triangle)
        scan_half(mid, end, t2[0], t2[1], t2[2], t1[0], t1[1], t1[2], t0[0], t0[1], t0[2], ctx);
    }
}

// ===========================================================================
// face_qef_cuda_kernel
// One thread per triangle.  Plane-vs-AABB overlap test.
// Only updates voxels already in the hash table.
// ===========================================================================
__global__ void face_qef_cuda_kernel(
    int N_tri,
    const float* triangles,
    float voxel_size_x, float voxel_size_y, float voxel_size_z,
    int grid_min_x, int grid_min_y, int grid_min_z,
    int grid_max_x, int grid_max_y, int grid_max_z,
    float face_weight,
    const uint64_t* hash_keys, const uint32_t* hash_vals,
    float* voxel_qefs,
    uint32_t capacity)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_tri) return;

    float vs[3] = {voxel_size_x, voxel_size_y, voxel_size_z};
    int gmin[3] = {grid_min_x, grid_min_y, grid_min_z};
    int gmax[3] = {grid_max_x, grid_max_y, grid_max_z};

    const float* tv = triangles + tid * 9;
    fdg_float3 v0{tv[0], tv[1], tv[2]};
    fdg_float3 v1{tv[3], tv[4], tv[5]};
    fdg_float3 v2{tv[6], tv[7], tv[8]};

    // Edge vectors and normal
    float e0x = v1.x - v0.x, e0y = v1.y - v0.y, e0z = v1.z - v0.z;
    float e1x = v2.x - v1.x, e1y = v2.y - v1.y, e1z = v2.z - v1.z;
    float e2x = v0.x - v2.x, e2y = v0.y - v2.y, e2z = v0.z - v2.z;

    float nx = e0y * e1z - e0z * e1y;
    float ny = e0z * e1x - e0x * e1z;
    float nz = e0x * e1y - e0y * e1x;
    float inv_len = rsqrtf(nx*nx + ny*ny + nz*nz + 1e-30f);
    nx *= inv_len; ny *= inv_len; nz *= inv_len;

    // Q matrix
    float Q_face[16];
    compute_face_qef(v0, v1, v2, Q_face);

    // Bounding box in voxel coordinates
    auto fmin = [](float a, float b) { return a < b ? a : b; };
    auto fmax = [](float a, float b) { return a > b ? a : b; };

    float bb_min_f_x = fmin(fmin(v0.x, v1.x), v2.x) / vs[0];
    float bb_min_f_y = fmin(fmin(v0.y, v1.y), v2.y) / vs[1];
    float bb_min_f_z = fmin(fmin(v0.z, v1.z), v2.z) / vs[2];
    float bb_max_f_x = fmax(fmax(v0.x, v1.x), v2.x) / vs[0];
    float bb_max_f_y = fmax(fmax(v0.y, v1.y), v2.y) / vs[1];
    float bb_max_f_z = fmax(fmax(v0.z, v1.z), v2.z) / vs[2];

    int bb_min_x = max(int(bb_min_f_x), gmin[0]);
    int bb_min_y = max(int(bb_min_f_y), gmin[1]);
    int bb_min_z = max(int(bb_min_f_z), gmin[2]);
    int bb_max_x = min(int(bb_max_f_x + 1.0f), gmax[0]);
    int bb_max_y = min(int(bb_max_f_y + 1.0f), gmax[1]);
    int bb_max_z = min(int(bb_max_f_z + 1.0f), gmax[2]);

    // Plane-test setup
    float c_x = nx > 0.0f ? vs[0] : 0.0f;
    float c_y = ny > 0.0f ? vs[1] : 0.0f;
    float c_z = nz > 0.0f ? vs[2] : 0.0f;
    float d1 = nx * (c_x - v0.x) + ny * (c_y - v0.y) + nz * (c_z - v0.z);
    float d2 = nx * (vs[0] - c_x - v0.x) + ny * (vs[1] - c_y - v0.y) + nz * (vs[2] - c_z - v0.z);

    // XY projection test parameters
    int mul_xy = (nz < 0.0f) ? -1 : 1;
    float n_xy_e0_x = -mul_xy * e0y, n_xy_e0_y =  mul_xy * e0x;
    float n_xy_e1_x = -mul_xy * e1y, n_xy_e1_y =  mul_xy * e1x;
    float n_xy_e2_x = -mul_xy * e2y, n_xy_e2_y =  mul_xy * e2x;

    float d_xy_e0 = -(n_xy_e0_x * v0.x + n_xy_e0_y * v0.y)
                    + fmaxf(n_xy_e0_x, 0.0f) * vs[0] + fmaxf(n_xy_e0_y, 0.0f) * vs[1];
    float d_xy_e1 = -(n_xy_e1_x * v1.x + n_xy_e1_y * v1.y)
                    + fmaxf(n_xy_e1_x, 0.0f) * vs[0] + fmaxf(n_xy_e1_y, 0.0f) * vs[1];
    float d_xy_e2 = -(n_xy_e2_x * v2.x + n_xy_e2_y * v2.y)
                    + fmaxf(n_xy_e2_x, 0.0f) * vs[0] + fmaxf(n_xy_e2_y, 0.0f) * vs[1];

    // YZ projection test parameters
    int mul_yz = (nx < 0.0f) ? -1 : 1;
    float n_yz_e0_x = -mul_yz * e0z, n_yz_e0_y = mul_yz * e0y;
    float n_yz_e1_x = -mul_yz * e1z, n_yz_e1_y = mul_yz * e1y;
    float n_yz_e2_x = -mul_yz * e2z, n_yz_e2_y = mul_yz * e2y;

    float d_yz_e0 = -(n_yz_e0_x * v0.y + n_yz_e0_y * v0.z)
                    + fmaxf(n_yz_e0_x, 0.0f) * vs[1] + fmaxf(n_yz_e0_y, 0.0f) * vs[2];
    float d_yz_e1 = -(n_yz_e1_x * v1.y + n_yz_e1_y * v1.z)
                    + fmaxf(n_yz_e1_x, 0.0f) * vs[1] + fmaxf(n_yz_e1_y, 0.0f) * vs[2];
    float d_yz_e2 = -(n_yz_e2_x * v2.y + n_yz_e2_y * v2.z)
                    + fmaxf(n_yz_e2_x, 0.0f) * vs[1] + fmaxf(n_yz_e2_y, 0.0f) * vs[2];

    // ZX projection test parameters
    int mul_zx = (ny < 0.0f) ? -1 : 1;
    float n_zx_e0_x = -mul_zx * e0x, n_zx_e0_y = mul_zx * e0z;
    float n_zx_e1_x = -mul_zx * e1x, n_zx_e1_y = mul_zx * e1z;
    float n_zx_e2_x = -mul_zx * e2x, n_zx_e2_y = mul_zx * e2z;

    float d_zx_e0 = -(n_zx_e0_x * v0.z + n_zx_e0_y * v0.x)
                    + fmaxf(n_zx_e0_x, 0.0f) * vs[2] + fmaxf(n_zx_e0_y, 0.0f) * vs[0];
    float d_zx_e1 = -(n_zx_e1_x * v1.z + n_zx_e1_y * v1.x)
                    + fmaxf(n_zx_e1_x, 0.0f) * vs[2] + fmaxf(n_zx_e1_y, 0.0f) * vs[0];
    float d_zx_e2 = -(n_zx_e2_x * v2.z + n_zx_e2_y * v2.x)
                    + fmaxf(n_zx_e2_x, 0.0f) * vs[2] + fmaxf(n_zx_e2_y, 0.0f) * vs[0];

    // Iterate over bounding box voxels
    for (int z = bb_min_z; z < bb_max_z; ++z) {
        for (int y = bb_min_y; y < bb_max_y; ++y) {
            for (int x = bb_min_x; x < bb_max_x; ++x) {
                // Voxel corner position
                float px = x * vs[0];
                float py = y * vs[1];
                float pz = z * vs[2];

                // Plane-box test
                float nDOTp = nx * px + ny * py + nz * pz;
                if ((nDOTp + d1) * (nDOTp + d2) > 0.0f) continue;

                // XY projection test
                if (n_xy_e0_x * px + n_xy_e0_y * py + d_xy_e0 < 0.0f) continue;
                if (n_xy_e1_x * px + n_xy_e1_y * py + d_xy_e1 < 0.0f) continue;
                if (n_xy_e2_x * px + n_xy_e2_y * py + d_xy_e2 < 0.0f) continue;

                // YZ projection test
                if (n_yz_e0_x * py + n_yz_e0_y * pz + d_yz_e0 < 0.0f) continue;
                if (n_yz_e1_x * py + n_yz_e1_y * pz + d_yz_e1 < 0.0f) continue;
                if (n_yz_e2_x * py + n_yz_e2_y * pz + d_yz_e2 < 0.0f) continue;

                // ZX projection test
                if (n_zx_e0_x * pz + n_zx_e0_y * px + d_zx_e0 < 0.0f) continue;
                if (n_zx_e1_x * pz + n_zx_e1_y * px + d_zx_e1 < 0.0f) continue;
                if (n_zx_e2_x * pz + n_zx_e2_y * px + d_zx_e2 < 0.0f) continue;

                // Passed all tests — look up voxel and add QEF
                uint64_t key = encode_voxel_key(x, y, z);
                uint32_t idx = hashmap_lookup_voxel(key, hash_keys, hash_vals, capacity);
                if (idx == FDG_EMPTY_VAL) continue;

                add_qef(voxel_qefs + idx * 16, Q_face, face_weight);
            }
        }
    }
}

// ===========================================================================
// boundry_qef_cuda_kernel
// One thread per boundary edge.  DDA traversal.
// ===========================================================================
__global__ void boundry_qef_cuda_kernel(
    int N_edges,
    const float* boundries,
    float voxel_size_x, float voxel_size_y, float voxel_size_z,
    int grid_min_x, int grid_min_y, int grid_min_z,
    int grid_max_x, int grid_max_y, int grid_max_z,
    float boundary_weight,
    const uint64_t* hash_keys, const uint32_t* hash_vals,
    float* voxel_qefs,
    uint32_t capacity)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_edges) return;

    const float* bv = boundries + tid * 6;
    fdg_float3 v0{bv[0], bv[1], bv[2]};
    fdg_float3 v1{bv[3], bv[4], bv[5]};

    // Edge QEF matrix
    float Q_edge[16];
    compute_boundary_qef(v0, v1, Q_edge);

    float vs[3] = {voxel_size_x, voxel_size_y, voxel_size_z};
    int gmin[3] = {grid_min_x, grid_min_y, grid_min_z};
    int gmax[3] = {grid_max_x, grid_max_y, grid_max_z};

    // Direction and length
    double dir_x = (double)v1.x - (double)v0.x;
    double dir_y = (double)v1.y - (double)v0.y;
    double dir_z = (double)v1.z - (double)v0.z;
    double seg_len = sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    if (seg_len < 1e-6) return;
    dir_x /= seg_len; dir_y /= seg_len; dir_z /= seg_len;

    // Starting voxel
    int cur_x = int(floorf(v0.x / vs[0]));
    int cur_y = int(floorf(v0.y / vs[1]));
    int cur_z = int(floorf(v0.z / vs[2]));
    int end_x = int(floorf(v1.x / vs[0]));
    int end_y = int(floorf(v1.y / vs[1]));
    int end_z = int(floorf(v1.z / vs[2]));

    if (fabs(dir_x) + fabs(dir_y) + fabs(dir_z) < 1e-12) return;

    // Step direction
    int step_x = (dir_x > 0.0) ? 1 : -1;
    int step_y = (dir_y > 0.0) ? 1 : -1;
    int step_z = (dir_z > 0.0) ? 1 : -1;

    // tMax and tDelta
    double tMax_x, tMax_y, tMax_z;
    double tDelta_x, tDelta_y, tDelta_z;

    if (fabs(dir_x) < 1e-12) {
        tMax_x = 1e300; tDelta_x = 1e300;
    } else {
        float border = vs[0] * (cur_x + (step_x > 0 ? 1 : 0));
        tMax_x = (border - v0.x) / dir_x;
        tDelta_x = vs[0] / fabs(dir_x);
    }
    if (fabs(dir_y) < 1e-12) {
        tMax_y = 1e300; tDelta_y = 1e300;
    } else {
        float border = vs[1] * (cur_y + (step_y > 0 ? 1 : 0));
        tMax_y = (border - v0.y) / dir_y;
        tDelta_y = vs[1] / fabs(dir_y);
    }
    if (fabs(dir_z) < 1e-12) {
        tMax_z = 1e300; tDelta_z = 1e300;
    } else {
        float border = vs[2] * (cur_z + (step_z > 0 ? 1 : 0));
        tMax_z = (border - v0.z) / dir_z;
        tDelta_z = vs[2] / fabs(dir_z);
    }

    // Accumulate for starting voxel
    if (cur_x >= gmin[0] && cur_x < gmax[0] &&
        cur_y >= gmin[1] && cur_y < gmax[1] &&
        cur_z >= gmin[2] && cur_z < gmax[2]) {
        uint64_t key = encode_voxel_key(cur_x, cur_y, cur_z);
        uint32_t idx = hashmap_lookup_voxel(key, hash_keys, hash_vals, capacity);
        if (idx != FDG_EMPTY_VAL) {
            add_qef(voxel_qefs + idx * 16, Q_edge, boundary_weight);
        }
    }

    // DDA traversal
    while (true) {
        int axis;
        if (tMax_x < tMax_y) {
            axis = (tMax_x < tMax_z) ? 0 : 2;
        } else {
            axis = (tMax_y < tMax_z) ? 1 : 2;
        }

        if      (axis == 0 && tMax_x > seg_len) break;
        else if (axis == 1 && tMax_y > seg_len) break;
        else if (axis == 2 && tMax_z > seg_len) break;

        if      (axis == 0) { cur_x += step_x; tMax_x += tDelta_x; }
        else if (axis == 1) { cur_y += step_y; tMax_y += tDelta_y; }
        else                { cur_z += step_z; tMax_z += tDelta_z; }

        if (cur_x < gmin[0] || cur_x >= gmax[0]) continue;
        if (cur_y < gmin[1] || cur_y >= gmax[1]) continue;
        if (cur_z < gmin[2] || cur_z >= gmax[2]) continue;

        uint64_t key = encode_voxel_key(cur_x, cur_y, cur_z);
        uint32_t idx = hashmap_lookup_voxel(key, hash_keys, hash_vals, capacity);
        if (idx != FDG_EMPTY_VAL) {
            add_qef(voxel_qefs + idx * 16, Q_edge, boundary_weight);
        }
    }
}

// ===========================================================================
// qef_solve_cuda_kernel
// One thread per voxel.  Solve QEF with constraints.
// ===========================================================================
__global__ void qef_solve_cuda_kernel(
    uint32_t num_voxels,
    const int32_t* voxel_coords,
    const float* voxel_means,
    const float* voxel_cnt,
    float voxel_size_x, float voxel_size_y, float voxel_size_z,
    float regularization_weight,
    const float* voxel_qefs,
    float* dual_vertices,
    uint32_t* voxel_intersected)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_voxels) return;

    float vs[3] = {voxel_size_x, voxel_size_y, voxel_size_z};

    int cx = voxel_coords[tid * 3 + 0];
    int cy = voxel_coords[tid * 3 + 1];
    int cz = voxel_coords[tid * 3 + 2];

    float min_corner[3] = {cx * vs[0], cy * vs[1], cz * vs[2]};
    float max_corner[3] = {(cx + 1) * vs[0], (cy + 1) * vs[1], (cz + 1) * vs[2]};

    // Load Q matrix
    float Q[16];
    const float* src_Q = voxel_qefs + tid * 16;
    for (int i = 0; i < 16; ++i) Q[i] = src_Q[i];

    // Add regularization
    if (regularization_weight > 0.0f && voxel_cnt[tid] > 0.0f) {
        float inv_cnt = 1.0f / voxel_cnt[tid];
        float px = voxel_means[tid * 3 + 0] * inv_cnt;
        float py = voxel_means[tid * 3 + 1] * inv_cnt;
        float pz = voxel_means[tid * 3 + 2] * inv_cnt;
        float weight = regularization_weight * voxel_cnt[tid];

        Q[ 0] += weight;  Q[ 1] += 0.0f;   Q[ 2] += 0.0f;   Q[ 3] -= weight * px;
        Q[ 4] += 0.0f;    Q[ 5] += weight;  Q[ 6] += 0.0f;   Q[ 7] -= weight * py;
        Q[ 8] += 0.0f;    Q[ 9] += 0.0f;   Q[10] += weight;  Q[11] -= weight * pz;
        Q[12] -= weight * px; Q[13] -= weight * py; Q[14] -= weight * pz;
        Q[15] += weight * (px*px + py*py + pz*pz);
    }

    // Unconstrained solve: A * x = b where A = Q.topLeftCorner<3,3>(), b = -Q.block<3,1>(0,3)
    float A[9] = {Q[0], Q[1], Q[2], Q[4], Q[5], Q[6], Q[8], Q[9], Q[10]};
    float b[3] = {-Q[3], -Q[7], -Q[11]};

    fdg_float3 v_new;
    float best = 1e30f;
    bool found = false;

    float x_sol[3];
    if (solve_3x3(A, b, x_sol)) {
        if (x_sol[0] >= min_corner[0] && x_sol[0] <= max_corner[0] &&
            x_sol[1] >= min_corner[1] && x_sol[1] <= max_corner[1] &&
            x_sol[2] >= min_corner[2] && x_sol[2] <= max_corner[2]) {
            v_new = fdg_float3{x_sol[0], x_sol[1], x_sol[2]};
            found = true;
            // No need to check error — unconstrained solution is optimal
        }
    }

    if (!found) {
        // Single-constraint: fix one axis to min or max, solve 2x2
        for (int fixed_axis = 0; fixed_axis < 3; ++fixed_axis) {
            int ax1 = (fixed_axis + 1) % 3;
            int ax2 = (fixed_axis + 2) % 3;

            float A2[4] = {Q[fixed_axis*4 + ax1],
                           Q[fixed_axis*4 + ax2],
                           Q[ax1*4 + fixed_axis],
                           Q[ax1*4 + ax2]};
            // Wait, let me recompute: A2 is the 2x2 submatrix for ax1, ax2
            // A2 = [[Q(ax1,ax1), Q(ax1,ax2)], [Q(ax2,ax1), Q(ax2,ax2)]]
            A2[0] = Q[ax1*4 + ax1]; A2[1] = Q[ax1*4 + ax2];
            A2[2] = Q[ax2*4 + ax1]; A2[3] = Q[ax2*4 + ax2];

            float B2[4]; // [Q(ax1, fixed_axis), Q(ax1, 3); Q(ax2, fixed_axis), Q(ax2, 3)]
            B2[0] = Q[ax1*4 + fixed_axis]; B2[1] = Q[ax1*4 + 3];
            B2[2] = Q[ax2*4 + fixed_axis]; B2[3] = Q[ax2*4 + 3];

            for (int bound_type = 0; bound_type < 2; ++bound_type) {
                float q0 = bound_type ? min_corner[fixed_axis] : max_corner[fixed_axis];
                float b2[2] = {-(B2[0] * q0 + B2[1] * 1.0f),
                               -(B2[2] * q0 + B2[3] * 1.0f)};
                float x2[2];
                if (solve_2x2(A2, b2, x2)) {
                    if (x2[0] >= min_corner[ax1] && x2[0] <= max_corner[ax1] &&
                        x2[1] >= min_corner[ax2] && x2[1] <= max_corner[ax2]) {
                        fdg_float3 p;
                        p[fixed_axis] = q0;
                        p[ax1] = x2[0];
                        p[ax2] = x2[1];
                        float err = eval_qef_error(Q, p);
                        if (err < best) { best = err; v_new = p; found = true; }
                    }
                }
            }
        }

        // Two-constraint: fix two axes, solve 1x1 for the free axis
        for (int free_axis = 0; free_axis < 3; ++free_axis) {
            int ax1 = (free_axis + 1) % 3;
            int ax2 = (free_axis + 2) % 3;

            float a_diag = Q[free_axis*4 + free_axis];
            if (fabsf(a_diag) < 1e-12f) continue;
            float b_coeff[3] = {Q[free_axis*4 + ax1], Q[free_axis*4 + ax2], Q[free_axis*4 + 3]};

            for (int b1 = 0; b1 < 2; ++b1) {
                for (int b2 = 0; b2 < 2; ++b2) {
                    float q0 = b1 ? min_corner[ax1] : max_corner[ax1];
                    float q1 = b2 ? min_corner[ax2] : max_corner[ax2];
                    float x = -(b_coeff[0] * q0 + b_coeff[1] * q1 + b_coeff[2] * 1.0f) / a_diag;
                    if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
                        fdg_float3 p;
                        p[free_axis] = x;
                        p[ax1] = q0;
                        p[ax2] = q1;
                        float err = eval_qef_error(Q, p);
                        if (err < best) { best = err; v_new = p; found = true; }
                    }
                }
            }
        }

        // Three-constraint: all 8 corners
        for (int bx = 0; bx < 2; ++bx) {
            for (int by = 0; by < 2; ++by) {
                for (int bz = 0; bz < 2; ++bz) {
                    fdg_float3 p;
                    p.x = bx ? min_corner[0] : max_corner[0];
                    p.y = by ? min_corner[1] : max_corner[1];
                    p.z = bz ? min_corner[2] : max_corner[2];
                    float err = eval_qef_error(Q, p);
                    if (err < best) { best = err; v_new = p; found = true; }
                }
            }
        }
    }

    // Write result
    dual_vertices[tid * 3 + 0] = v_new.x;
    dual_vertices[tid * 3 + 1] = v_new.y;
    dual_vertices[tid * 3 + 2] = v_new.z;

    // intersected flag (already set in intersect_qef kernel)
}

// ===========================================================================
// extract_edges_kernel — one thread per face, writes 3 canonical edges
// ===========================================================================
__global__ void extract_edges_kernel(
    int F,
    const int32_t* faces,
    uint64_t* edge_keys, int32_t* edge_v0, int32_t* edge_v1)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int v[3] = {faces[tid * 3], faces[tid * 3 + 1], faces[tid * 3 + 2]};
    for (int e = 0; e < 3; ++e) {
        int a = v[e], b = v[(e + 1) % 3];
        if (a > b) { int t = a; a = b; b = t; }
        int out = tid * 3 + e;
        edge_keys[out] = ((uint64_t)(uint32_t)a << 32) | (uint64_t)(uint32_t)b;
        edge_v0[out] = a;
        edge_v1[out] = b;
    }
}

// ===========================================================================
// filter_boundary_edges_kernel — after sort, edges unique to a run of 1 are boundaries
// ===========================================================================
__global__ void filter_boundary_edges_kernel(
    int N_edges_total,
    const uint64_t* edge_keys, const int32_t* edge_v0, const int32_t* edge_v1,
    const float* vertices, int32_t* boundry_count, float* boundries)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_edges_total) return;

    bool left_diff  = (i == 0)                   || (edge_keys[i] != edge_keys[i - 1]);
    bool right_diff = (i == N_edges_total - 1)   || (edge_keys[i] != edge_keys[i + 1]);

    if (left_diff && right_diff) {
        int out = atomicAdd(boundry_count, 1);
        int v0 = edge_v0[i], v1 = edge_v1[i];
        boundries[out * 6 + 0] = vertices[v0 * 3 + 0];
        boundries[out * 6 + 1] = vertices[v0 * 3 + 1];
        boundries[out * 6 + 2] = vertices[v0 * 3 + 2];
        boundries[out * 6 + 3] = vertices[v1 * 3 + 0];
        boundries[out * 6 + 4] = vertices[v1 * 3 + 1];
        boundries[out * 6 + 5] = vertices[v1 * 3 + 2];
    }
}

// ===========================================================================
// Host function
// ===========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mesh_to_flexible_dual_grid_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range,
    float face_weight,
    float boundary_weight,
    float regularization_weight,
    bool timing)
{
    int V = vertices.size(0);
    int F = faces.size(0);

    // Ensure CUDA tensors
    auto vertices_cuda = vertices.cuda().contiguous();
    auto faces_cuda = faces.cuda().contiguous();
    auto voxel_size_cuda = voxel_size.cuda().contiguous();
    auto grid_range_cuda = grid_range.cuda().contiguous();

    TORCH_CHECK(vertices_cuda.is_cuda(), "vertices must be on CUDA");
    TORCH_CHECK(faces_cuda.is_cuda(), "faces must be on CUDA");
    TORCH_CHECK(voxel_size_cuda.is_cuda(), "voxel_size must be on CUDA");
    TORCH_CHECK(grid_range_cuda.is_cuda(), "grid_range must be on CUDA");

    // Copy scalars to host using PyTorch ops
    auto vs_cpu = voxel_size_cuda.cpu();
    auto gr_cpu = grid_range_cuda.cpu();
    float vs_x = vs_cpu[0].item<float>();
    float vs_y = vs_cpu[1].item<float>();
    float vs_z = vs_cpu[2].item<float>();
    int gmin_x = gr_cpu[0][0].item<int>();
    int gmin_y = gr_cpu[0][1].item<int>();
    int gmin_z = gr_cpu[0][2].item<int>();
    int gmax_x = gr_cpu[1][0].item<int>();
    int gmax_y = gr_cpu[1][1].item<int>();
    int gmax_z = gr_cpu[1][2].item<int>();

    // Capacity estimation
    int64_t total_cells = (int64_t)(gmax_x - gmin_x) * (gmax_y - gmin_y) * (gmax_z - gmin_z);
    int64_t est_voxels = total_cells;
    if (est_voxels > (int64_t)F * 64) est_voxels = (int64_t)F * 64;
    if (est_voxels < 65536) est_voxels = 65536;
    int64_t capacity = est_voxels * 2;
    int64_t max_voxels = est_voxels;

    if (est_voxels > INT32_MAX) {
        throw std::runtime_error("Estimated voxel count exceeds INT32_MAX, grid too large for CUDA path");
    }

    cudaStream_t stream = 0;

    // Allocate GPU buffers
    auto options_uint64 = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA);
    auto options_uint32 = torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);
    auto options_int32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto options_float  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto hash_keys     = torch::full({capacity}, (uint64_t)FDG_EMPTY_KEY, options_uint64);
    auto hash_vals     = torch::full({capacity}, (uint32_t)FDG_EMPTY_VAL, options_uint32);
    auto d_voxel_count = torch::zeros({1}, options_uint32);

    auto d_coords      = torch::zeros({max_voxels * 3}, options_int32);
    auto d_means       = torch::zeros({max_voxels * 3}, options_float);
    auto d_cnt         = torch::zeros({max_voxels}, options_float);
    auto d_intersected = torch::zeros({max_voxels}, options_uint32);
    auto d_qefs        = torch::zeros({max_voxels * 16}, options_float);

    // Prepare triangles array: [F * 3, 3]
    auto triangles = torch::empty({F * 3, 3}, options_float);
    {
        int threads = FDG_BLOCK_SIZE;
        int blocks = (F + threads - 1) / threads;
        prepare_triangles_kernel<<<blocks, threads, 0, stream>>>(
            F,
            vertices_cuda.data_ptr<float>(),
            faces_cuda.data_ptr<int32_t>(),
            triangles.data_ptr<float>());
    }

    cudaEvent_t evt_start, evt_stop;
    if (timing) {
        cudaEventCreate(&evt_start);
        cudaEventCreate(&evt_stop);
    }

    // Phase 1: intersect_qef
    if (timing) cudaEventRecord(evt_start, stream);
    {
        int threads = FDG_BLOCK_SIZE;
        int blocks = (F + threads - 1) / threads;
        intersect_qef_cuda_kernel<<<blocks, threads, 0, stream>>>(
            F,
            triangles.data_ptr<float>(),
            vs_x, vs_y, vs_z,
            gmin_x, gmin_y, gmin_z,
            gmax_x, gmax_y, gmax_z,
            (uint64_t*)hash_keys.data_ptr(),
            (uint32_t*)hash_vals.data_ptr(),
            (uint32_t*)d_voxel_count.data_ptr(),
            (int32_t*)d_coords.data_ptr(),
            d_means.data_ptr<float>(),
            d_cnt.data_ptr<float>(),
            (uint32_t*)d_intersected.data_ptr(),
            d_qefs.data_ptr<float>(),
            (uint32_t)capacity);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("intersect_qef kernel failed: ") + cudaGetErrorString(err));
        }
    }
    if (timing) {
        cudaEventRecord(evt_stop, stream);
        cudaEventSynchronize(evt_stop);
        float ms;
        cudaEventElapsedTime(&ms, evt_start, evt_stop);
        printf("Intersect QEF (CUDA): %.3f ms\n", ms);
    }

    // Read voxel count
    uint32_t num_voxels;
    cudaMemcpy(&num_voxels, d_voxel_count.data_ptr(), sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (num_voxels == 0) {
        auto empty_coords = torch::zeros({0, 3}, torch::kInt32);
        auto empty_verts = torch::zeros({0, 3}, torch::kFloat32);
        auto empty_inter = torch::zeros({0, 3}, torch::kBool);
        return std::make_tuple(empty_coords, empty_verts, empty_inter);
    }
    if (num_voxels > max_voxels) {
        throw std::runtime_error("Voxel count exceeded capacity. Increase capacity estimate.");
    }

    // Phase 2: face_qef
    if (face_weight > 0.0f) {
        if (timing) cudaEventRecord(evt_start, stream);
        {
            int threads = FDG_BLOCK_SIZE;
            int blocks = (F + threads - 1) / threads;
            face_qef_cuda_kernel<<<blocks, threads, 0, stream>>>(
                F,
                triangles.data_ptr<float>(),
                vs_x, vs_y, vs_z,
                gmin_x, gmin_y, gmin_z,
                gmax_x, gmax_y, gmax_z,
                face_weight,
                (const uint64_t*)hash_keys.data_ptr(),
                (const uint32_t*)hash_vals.data_ptr(),
                d_qefs.data_ptr<float>(),
                (uint32_t)capacity);
        }
        if (timing) {
            cudaEventRecord(evt_stop, stream);
            cudaEventSynchronize(evt_stop);
            float ms;
            cudaEventElapsedTime(&ms, evt_start, evt_stop);
            printf("Face QEF (CUDA): %.3f ms\n", ms);
        }
    }

    // Phase 3: boundry_qef
    if (boundary_weight > 0.0f) {
        int N_edges_total = F * 3;
        auto d_edge_keys = torch::empty({N_edges_total}, options_uint64);
        auto d_edge_v0  = torch::empty({N_edges_total}, options_int32);
        auto d_edge_v1  = torch::empty({N_edges_total}, options_int32);

        // Extract edges from faces (GPU)
        {
            int threads = FDG_BLOCK_SIZE;
            int blocks = (F + threads - 1) / threads;
            extract_edges_kernel<<<blocks, threads, 0, stream>>>(F,
                faces_cuda.data_ptr<int32_t>(),
                (uint64_t*)d_edge_keys.data_ptr(),
                (int32_t*)d_edge_v0.data_ptr(),
                (int32_t*)d_edge_v1.data_ptr());
        }

        // Sort edges by key (thrust, GPU)
        {
            auto keys_ptr = (uint64_t*)d_edge_keys.data_ptr();
            auto v0_ptr  = (int32_t*)d_edge_v0.data_ptr();
            auto v1_ptr  = (int32_t*)d_edge_v1.data_ptr();
            auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(v0_ptr, v1_ptr));
            thrust::sort_by_key(thrust::device, keys_ptr, keys_ptr + N_edges_total, zip_begin);
        }

        // Filter: edges that appear exactly once are boundaries
        auto d_boundry_count = torch::zeros({1}, options_int32);
        auto d_boundries = torch::empty({N_edges_total * 6}, options_float);
        {
            int threads = FDG_BLOCK_SIZE;
            int blocks = (N_edges_total + threads - 1) / threads;
            filter_boundary_edges_kernel<<<blocks, threads, 0, stream>>>(
                N_edges_total,
                (const uint64_t*)d_edge_keys.data_ptr(),
                (const int32_t*)d_edge_v0.data_ptr(),
                (const int32_t*)d_edge_v1.data_ptr(),
                vertices_cuda.data_ptr<float>(),
                (int32_t*)d_boundry_count.data_ptr(),
                d_boundries.data_ptr<float>());
        }

        // Read boundary edge count
        int N_edges;
        cudaMemcpy(&N_edges, d_boundry_count.data_ptr(), sizeof(int), cudaMemcpyDeviceToHost);

        if (N_edges > 0) {
            if (timing) cudaEventRecord(evt_start, stream);
            {
                int threads = FDG_BLOCK_SIZE;
                int blocks = (N_edges + threads - 1) / threads;
                boundry_qef_cuda_kernel<<<blocks, threads, 0, stream>>>(
                    N_edges,
                    d_boundries.data_ptr<float>(),
                    vs_x, vs_y, vs_z,
                    gmin_x, gmin_y, gmin_z,
                    gmax_x, gmax_y, gmax_z,
                    boundary_weight,
                    (const uint64_t*)hash_keys.data_ptr(),
                    (const uint32_t*)hash_vals.data_ptr(),
                    d_qefs.data_ptr<float>(),
                    (uint32_t)capacity);
            }
            if (timing) {
                cudaEventRecord(evt_stop, stream);
                cudaEventSynchronize(evt_stop);
                float ms;
                cudaEventElapsedTime(&ms, evt_start, evt_stop);
                printf("Boundary QEF (CUDA): %.3f ms\n", ms);
            }
        }
    }

    // Phase 4: QEF solve
    auto d_dual_vertices = torch::empty({max_voxels * 3}, options_float);
    {
        if (timing) cudaEventRecord(evt_start, stream);
        int threads = FDG_BLOCK_SIZE;
        int blocks = (num_voxels + threads - 1) / threads;
        qef_solve_cuda_kernel<<<blocks, threads, 0, stream>>>(
            num_voxels,
            (const int32_t*)d_coords.data_ptr(),
            d_means.data_ptr<float>(),
            d_cnt.data_ptr<float>(),
            vs_x, vs_y, vs_z,
            regularization_weight,
            d_qefs.data_ptr<float>(),
            d_dual_vertices.data_ptr<float>(),
            (uint32_t*)d_intersected.data_ptr());
        if (timing) {
            cudaEventRecord(evt_stop, stream);
            cudaEventSynchronize(evt_stop);
            float ms;
            cudaEventElapsedTime(&ms, evt_start, evt_stop);
            printf("QEF Solve (CUDA): %.3f ms\n", ms);
        }
    }

    // Slice to actual voxel count
    auto coords_out = d_coords.slice(0, 0, num_voxels * 3).view({(int64_t)num_voxels, 3}).to(torch::kInt32).contiguous();
    auto dual_out   = d_dual_vertices.slice(0, 0, num_voxels * 3).view({(int64_t)num_voxels, 3}).contiguous();

    // Convert intersected from uint32 bitmask to bool3 on GPU
    auto d_inter_slice = d_intersected.slice(0, 0, num_voxels).to(torch::kInt32);
    // Bit 0 -> channel 0, bit 1 -> channel 1, bit 2 -> channel 2
    auto inter_b0 = (d_inter_slice.bitwise_and(1)).not_equal(0);
    auto inter_b1 = (d_inter_slice.bitwise_and(2)).not_equal(0);
    auto inter_b2 = (d_inter_slice.bitwise_and(4)).not_equal(0);
    auto intersected_out = torch::stack({inter_b0, inter_b1, inter_b2}, 1);

    if (timing) {
        cudaEventDestroy(evt_start);
        cudaEventDestroy(evt_stop);
    }

    return std::make_tuple(coords_out, dual_out, intersected_out);
}
