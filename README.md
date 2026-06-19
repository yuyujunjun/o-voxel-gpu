# O-Voxel-GPU

CUDA-accelerated fork of [o-voxel](https://github.com/microsoft/TRELLIS.2) from [TRELLIS.2](https://github.com/microsoft/TRELLIS.2). `mesh_to_flexible_dual_grid` runs entirely on GPU with output **100% identical** to the CPU path (verified voxel-by-voxel on cube, sphere, and complex meshes). On `rec_helmet.glb` (297K vertices, 593K faces, 512³ grid) the GPU path achieves **~15x cold / ~170x warmed** speedup over CPU on an NVIDIA A100.

## What's Different from Upstream

The triangle-parallel scanline algorithm processes one triangle per CUDA thread. When a mesh contains large triangles (common in CAD, architectural models, or low-poly assets), a single thread may traverse hundreds of voxels while the rest of the warp idles — **warp divergence**. This pushes voxelization from ~30ms to hundreds of milliseconds.

This fork adds an optional **pre-subdivision** step: `subdivide_mesh_gpu` splits large triangles via GPU longest-edge bisection **before** voxelization. With a default threshold of `2e-5` (optimal for 512³), large-triangle geometry drops from several hundred milliseconds to **under 30ms**.

The subdivision is external to the voxelization pipeline — callers can subdivide once and voxelize many times (e.g. animated meshes).

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

NVIDIA A100, `rec_helmet.glb` (297K vertices, 593K faces), 512³ grid:

| | CPU | CUDA (cold) | CUDA (warmed) |
|---|---|---|---|
| **Total** | 389ms | 26ms | 2.27ms |

With pre-subdivision (large-triangle geometry, 512³):

| No subdivide | With subdivide (2e-5) | Speedup |
|---|---|---|
| 78.8ms | 7.7ms | **10.3x** |

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
