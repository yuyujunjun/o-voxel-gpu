# O-Voxel-GPU

CUDA-accelerated fork of [o-voxel](https://github.com/microsoft/TRELLIS.2) from [TRELLIS.2](https://github.com/microsoft/TRELLIS.2). `mesh_to_flexible_dual_grid` runs entirely on GPU with output **100% identical** to the CPU path (verified voxel-by-voxel on cube, sphere, and complex meshes). On `rec_helmet.glb` (297K vertices, 593K faces, 512³ grid) the GPU path achieves **~15x cold / ~170x warmed** speedup over CPU on an NVIDIA A100.

## What's Different from Upstream

This fork adds one new function: **`subdivide_mesh_gpu`** — an optional GPU pre-subdivision step that splits large triangles before voxelization. It does not alter the voxelization algorithm itself; with or without subdivision, the final output is numerically identical.

**Why it exists.** The triangle-parallel scanline algorithm processes one triangle per CUDA thread. When a mesh contains large triangles (common in CAD, architectural models, or low-poly assets), a single thread may traverse hundreds of voxels while the rest of the warp idles — **warp divergence**. This pushes voxelization from ~30ms to hundreds of milliseconds.

`subdivide_mesh_gpu` eliminates the imbalance by splitting large triangles via GPU longest-edge bisection. After subdivision all triangles are uniformly small, and the scanline algorithm runs at full efficiency — large-triangle geometry drops from several hundred milliseconds to **under 10ms**.

The subdivision is external to the voxelization pipeline — callers can subdivide once and voxelize many times (e.g. animated meshes). Default threshold is `2e-5` (optimal for 512³).

## Installation

```bash
git clone --recursive https://github.com/yuyujunjun/o-voxel-gpu.git
cd o-voxel-gpu
pip install . --no-build-isolation
```

To build with limited RAM: `MAX_JOBS=2 pip install . --no-build-isolation`.

Requirements: CUDA Toolkit ≥ 11.8, PyTorch ≥ 2.0, C++17 compiler.

## API

### mesh_to_flexible_dual_grid

Auto-dispatches to CUDA or CPU based on tensor device.

```python
o_voxel.convert.mesh_to_flexible_dual_grid(
    vertices: torch.Tensor,       # (V, 3) float32
    faces: torch.Tensor,          # (F, 3) int32
    voxel_size: float | [3] = None,  # one of voxel_size or grid_size required
    grid_size: int | [3] = None,
    aabb: (2, 3) float = None,    # auto-computed if None
    face_weight: float = 1.0,
    boundary_weight: float = 1.0,
    regularization_weight: float = 0.1,
    timing: bool = False,
) -> tuple[voxel_indices (N,3) int32, dual_vertices (N,3) float32, intersected (N,3) bool]
```

### mesh_to_flexible_dual_grid_cuda

Explicit GPU path. Tensors must be on CUDA. Same signature as above.

```python
o_voxel.convert.mesh_to_flexible_dual_grid_cuda(
    vertices, faces, voxel_size=..., grid_size=..., ...
) -> tuple[voxel_indices, dual_vertices, intersected]
```

### subdivide_mesh_gpu

Pre-subdivide large triangles for efficient GPU voxelization.

```python
from o_voxel.convert.mesh_subdivide import subdivide_mesh_gpu

subdivide_mesh_gpu(
    verts: torch.Tensor,              # (V, 3) float32, GPU
    faces: torch.Tensor,              # (F, 3) int64, GPU
    area_threshold: float = 2e-5,     # optimal for 512³; use 5e-6 for 1024³
    max_iters: int = 12,
) -> tuple[new_verts (V',3) float32, new_faces (F',3) int64]
```

Threshold formula: `area_threshold ≈ 5 / grid_size²`.

### flexible_dual_grid_to_mesh

Reconstruct a triangle mesh from voxel grid + dual vertices.

```python
o_voxel.convert.flexible_dual_grid_to_mesh(
    coords: torch.Tensor,            # (N, 3) int
    dual_vertices: torch.Tensor,     # (N, 3) float
    intersected_flag: torch.Tensor,  # (N, 3) bool
    split_weight: torch.Tensor | None,
    aabb: (2, 3) float,
    voxel_size: ... = None,
    grid_size: ... = None,
    train: bool = False,
) -> tuple[vertices (M,3) float, faces (K,3) int]
```

## Quick Start

```python
import torch, trimesh, o_voxel
from o_voxel.convert.mesh_subdivide import subdivide_mesh_gpu

# Load and upload mesh
mesh = trimesh.load("rec_helmet.glb", process=False)
v = torch.from_numpy(mesh.vertices).float().cuda()
f = torch.from_numpy(mesh.faces).long().cuda()

# Optional: subdivide large triangles (~10x faster at 512³)
v, f = subdivide_mesh_gpu(v, f)

# Voxelize
vi, dv, inter = o_voxel.convert.mesh_to_flexible_dual_grid_cuda(
    vertices=v, faces=f.int(), grid_size=512,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
)

# Save VXZ
vid = o_voxel.serialize.encode_seq(vi)
m = torch.argsort(vid)
vi, dv, inter = vi[m], dv[m], inter[m]
dv = torch.clamp((dv * 512 - vi) * 255, 0, 255).byte()
inter = (inter[:, 0] + 2 * inter[:, 1] + 4 * inter[:, 2]).byte().unsqueeze(1)
o_voxel.io.write_vxz("output.vxz", vi.cpu(),
    {"vertices": dv.cpu(), "intersected": inter.cpu()})
```

## Performance

NVIDIA A100, 512³ grid.

### `rec_helmet.glb` (297K vertices, 593K faces)

This mesh has uniformly small triangles — pre-subdivision provides no additional benefit. The GPU path alone delivers the full speedup:

| | CPU | CUDA (cold) | CUDA (warmed) |
|---|---|---|---|
| **Total** | 389ms | 26ms | 2.27ms |

### Large-triangle geometry (CAD / low-poly, 512³)

Meshes with large faces expose warp divergence in the triangle-parallel scanline. We evaluated three approaches:

| Approach | Voxelization time |
|---|---|
| Triangle-parallel (original) | 236ms |
| Voxel-parallel blockface | 78.8ms |
| Triangle-parallel + pre-subdivision | **7.7ms** |

**Why voxel-parallel was abandoned.** The blockface launched one thread per voxel — 134M at 512³. While ~92% of those voxels are empty and bail out immediately, the few that do intersect a mesh triangle may need to test against every triangle in their 8³ block. The 8³ granularity is too coarse to isolate triangle-voxel pairs efficiently: finer blocks mean more blocks (more overhead), coarser blocks mean larger triangle lists per block. Pre-subdivision eliminates large triangles at the source, and at 7.7ms there was no reason to keep tuning the voxel-parallel approach.

## Appendix: Voxel-Parallel Approach (Explored)

We experimented with a fully **voxel-parallel** alternative to the triangle-parallel scanline. The idea: instead of one thread per triangle, assign one thread per voxel (or block of voxels) for both intersection and face-QEF.

### Design

1. **Block construction.** The grid is partitioned into 8³-voxel blocks. Each block accumulates a list of triangles whose bounding boxes intersect the block, using a CUDA hash table indexed by block ID.

2. **Per-voxel intersection (70ms).** One thread per voxel tests overlap against the block's triangle list via AABB-plane separation tests. Voxels that intersect the mesh are recorded in the hash table for downstream QEF accumulation.

3. **Per-voxel face-QEF (5ms).** For each occupied voxel, test overlap with each triangle in the block's list, atomically accumulate the QEF matrix.

4. **QEF solve.** Identical to the triangle-parallel path — one thread per voxel, closed-form linear solve.

### Result

The voxel-parallel pipeline achieves 78.8ms (70ms intersect + 5ms face-QEF), a solid improvement over raw triangle-parallel 236ms. However, the intersect step suffers from the sheer number of voxel threads (134M at 512³, mostly empty) combined with the 8³ block granularity: too coarse to narrow triangle-voxel pairs, too fine to avoid massive thread-launch overhead. A finer granularity (e.g. 16³) would reduce block count but inflate per-block triangle lists, and vice versa. Given that pre-subdivision already delivers 7.7ms with a simpler implementation, further tuning the voxel-parallel approach was not justified.

## Files Added vs Upstream

| Component | File |
|---|---|
| CUDA device functions | `src/convert/flexible_dual_grid.cuh` |
| CUDA kernels + host launcher | `src/convert/flexible_dual_grid_cuda.cu` |
| Python GPU entry point | `o_voxel/convert/flexible_dual_grid.py` |
| Mesh pre-subdivision | `o_voxel/convert/mesh_subdivide.py` |
| Build integration | `setup.py` |

All other modules (I/O, serialization, rasterization) are unchanged from upstream.

## License

Same as upstream [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).
