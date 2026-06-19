"""Evaluate subdivide_mesh_gpu area_threshold for 512 and 1024 resolutions.

Goal: find threshold that balances subdivision cost vs voxelization speedup.
"""
import sys, time, torch, numpy as np, trimesh
sys.path.insert(0, '.')
import o_voxel
from o_voxel.convert.mesh_subdivide import subdivide_mesh_gpu

glb_path = "test_scene.glb"
THRESHOLDS = [2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
RESOLUTIONS = [512, 1024]

print(f"Loading: {glb_path}")
scene = trimesh.load(glb_path, process=False)
if isinstance(scene, trimesh.Scene):
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    mesh = trimesh.util.concatenate(meshes)
else:
    mesh = scene
v = torch.from_numpy(np.asarray(mesh.vertices)).float()
f = torch.from_numpy(np.asarray(mesh.faces)).long()
print(f"Mesh: {v.shape[0]} verts, {f.shape[0]} faces")

vmin, vmax = v.min(0).values, v.max(0).values
center = (vmin + vmax) / 2
scale = 0.99999 / (vmax - vmin).max()
v = ((v - center) * scale).cuda().contiguous()
f = f.int().cuda().contiguous()
print(f"Normalized to unit cube")

# --- Pre-subdivide with different thresholds ---
print("\n=== SUBDIVISION TEST ===")
print(f"{'threshold':>10s}  {'faces':>8s}  {'BB_avg':>8s}  {'BB_max':>8s}  {'subdiv_ms':>10s}")
for th in THRESHOLDS:
    s_v, s_f = v.clone(), f.clone()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    sv, sf = subdivide_mesh_gpu(s_v, s_f, area_threshold=th)
    torch.cuda.synchronize()
    subdiv_ms = (time.perf_counter() - t0) * 1000

    # Estimate BB coverage in voxels at 512
    vs_1d = 1.0 / 512
    sv_cpu = sv.cpu()
    bb_min = sv_cpu[sf.cpu()].min(dim=1).values
    bb_max = sv_cpu[sf.cpu()].max(dim=1).values
    bb_vox = ((bb_max - bb_min) / vs_1d).max(dim=1).values
    print(f"{th:>10.1e}  {sf.shape[0]:>8d}  {bb_vox.mean():>8.2f}  {bb_vox.max():>8.0f}  {subdiv_ms:>10.1f}")
    del sv, sf

# --- Key comparison: no-subdivide vs subdivide @ 512 and 1024 ---
print("\n=== VOXELIZATION TIMING ===")
for res in RESOLUTIONS:
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32, device='cuda')
    grid = torch.tensor([res, res, res], dtype=torch.int32, device='cuda')
    vs_t = (aabb[1] - aabb[0]) / grid
    gr = torch.stack([torch.zeros_like(grid), grid], dim=0).int()
    v_shift = v - aabb[0].reshape(1, 3)

    # Baseline: no subdivision
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    vi, dv, inter = o_voxel._C.mesh_to_flexible_dual_grid_cuda(
        v_shift, f, vs_t, gr, 1.0, 0.2, 1e-2, False, True)
    torch.cuda.synchronize()
    base_ms = (time.perf_counter() - t0) * 1000
    print(f"res={res:4d}  no-subdiv:  {base_ms:>7.1f}ms  voxels={vi.shape[0]:>8d}")

    # With subdivision at various thresholds
    for th in [2e-5, 1e-5, 5e-6, 2e-6, 1e-6]:
        sv, sf = subdivide_mesh_gpu(v.clone(), f.clone(), area_threshold=th)
        sv_shift = sv - aabb[0].reshape(1, 3)

        t0 = time.perf_counter()
        torch.cuda.synchronize()
        vi, _, _ = o_voxel._C.mesh_to_flexible_dual_grid_cuda(
            sv_shift, sf.int(), vs_t, gr, 1.0, 0.2, 1e-2, False, True)
        torch.cuda.synchronize()
        vx_ms = (time.perf_counter() - t0) * 1000
        speedup = base_ms / vx_ms if vx_ms > 0 else 0
        print(f"res={res:4d}  th={th:.0e}  F={sf.shape[0]:>7d}  {vx_ms:>7.1f}ms  voxels={vi.shape[0]:>8d}  {speedup:.1f}x")
        del sv, sf, sv_shift
