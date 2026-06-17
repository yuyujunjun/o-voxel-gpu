# O-Voxel-GPU: CUDA-Accelerated Voxelization

**O-Voxel-GPU** is a CUDA-accelerated fork of the original [o-voxel](https://github.com/microsoft/TRELLIS.2) library from the [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) project. It adds a GPU implementation of `mesh_to_flexible_dual_grid`, achieving **~170x speedup** over the CPU path (2.3ms vs 389ms on an NVIDIA H100).

## Key Features

- **GPU Voxelization**: Full GPU pipeline for `mesh_to_flexible_dual_grid` — triangle-voxel intersection, face/edge QEF accumulation, and dual vertex solving all run on CUDA.
- **Boundary Edge Detection on GPU**: Replaces the CPU `std::map` boundary edge step (~200ms) with a CUDA thrust-based sort+filter (~0.04ms).
- **Drop-in Replacement**: Same Python API as upstream — tensors on GPU automatically use the CUDA code path.
- **Exact Output Match**: Verified voxel-coordinate identical to CPU output across cube, sphere, and complex meshes.

## Requirements

- CUDA Toolkit ≥ 11.8
- PyTorch ≥ 2.0
- C++17 compiler (gcc ≥ 9, or equivalent)

## Installation

```bash
git clone --recursive https://github.com/yuyujunjun/o-voxel-gpu.git
cd o-voxel-gpu
pip install . --no-build-isolation
```

To build in an environment with limited RAM:

```bash
MAX_JOBS=2 pip install . --no-build-isolation
```

To force a CUDA (non-ROCm) build when both toolchains are present:

```bash
BUILD_TARGET=cuda pip install . --no-build-isolation
```

## Quick Start

### GPU Voxelization

```python
import torch
import o_voxel

# Load mesh
import trimesh
mesh = trimesh.load("model.glb", process=False)
vertices = torch.from_numpy(mesh.vertices).float().cuda()
faces = torch.from_numpy(mesh.faces).long().cuda()

# GPU voxelization (automatic when tensors are on GPU)
voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
    vertices=vertices,
    faces=faces,
    grid_size=512,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    face_weight=1.0,
    boundary_weight=0.2,
    regularization_weight=1e-2,
)
```

### Save VXZ

```python
vid = o_voxel.serialize.encode_seq(voxel_indices)
mapping = torch.argsort(vid)
voxel_indices = voxel_indices[mapping]
dual_vertices = dual_vertices[mapping]
intersected = intersected[mapping]

dual_vertices = torch.clamp((dual_vertices * 512 - voxel_indices) * 255, 0, 255).byte()
intersected = (intersected[:, 0] + 2 * intersected[:, 1] + 4 * intersected[:, 2]).byte().unsqueeze(1)

o_voxel.io.write_vxz("output.vxz", voxel_indices.cpu(),
    {"vertices": dual_vertices.cpu(), "intersected": intersected.cpu()})
```

### Reconstruct Mesh from VXZ

```python
coords, attr = o_voxel.io.read_vxz("output.vxz")
dual_vertices = attr["vertices"].float().cuda() / 255.0
intersected = torch.cat([
    attr["intersected"] % 2,
    attr["intersected"] // 2 % 2,
    attr["intersected"] // 4 % 2,
], dim=-1).bool().cuda()

verts, faces = o_voxel.convert.flexible_dual_grid_to_mesh(
    coords.cuda(), dual_vertices, intersected,
    split_weight=None,
    grid_size=512,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
)
```

### Full Pipeline Verification

The `verify_pipeline.py` script runs end-to-end: GLB → GPU voxelization → VXZ → TRELLIS latent encode → decode → GLB. Requires `trellis2`.

```bash
python verify_pipeline.py --input model.glb --output decoded.glb
```

## Relationship to Upstream

This fork is derived from [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2). The GPU implementation adds:

| Component | File |
|---|---|
| CUDA kernels (7 kernels + device functions) | `src/convert/flexible_dual_grid.cuh` |
| Host-side launcher | `src/convert/flexible_dual_grid_cuda.cu` |
| Python GPU dispatch | `o_voxel/convert/flexible_dual_grid.py` |
| Build integration | `setup.py` |

All other modules (I/O, serialization, rasterization, postprocessing) are unchanged from upstream.

## Performance

Measured on an NVIDIA H100 with `rec_helmet.glb` (297K vertices, 593K faces, 512³ grid):

| Stage | CPU | CUDA | Speedup |
|---|---|---|---|
| intersect_qef | 297ms | 17.1ms | 17x |
| face_qef | 37ms | 5.3ms | 7x |
| boundry_qef | 204ms | 0.04ms | 5100x |
| qef_solve | 10ms | 0.04ms | 250x |
| **Total** | **389ms** | **2.27ms** | **170x** |

## License

Same as upstream [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).
