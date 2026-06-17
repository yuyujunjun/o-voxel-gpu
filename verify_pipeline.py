"""
Verification script: GLB → GPU voxelization → VXZ → latent encode → decode → GLB.

Tests the full TRELLIS.2 shape pipeline end-to-end using the CUDA voxelization path.
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trellis2.models as models
import trellis2.modules.sparse as sp
import o_voxel


def load_and_normalize_mesh(glb_path, device="cuda"):
    """Load a GLB mesh and normalize to [-0.5, 0.5]^3 unit cube."""
    mesh = trimesh.load(glb_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = []
        for name, geom in mesh.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if not meshes:
            raise ValueError("No trimesh geometries found in scene")
        mesh = trimesh.util.concatenate(meshes)

    vertices = torch.from_numpy(np.asarray(mesh.vertices)).float()
    faces = torch.from_numpy(np.asarray(mesh.faces)).long()

    vmin = vertices.min(dim=0).values
    vmax = vertices.max(dim=0).values
    extent = (vmax - vmin).max()
    if extent < 1e-8:
        raise ValueError(f"Degenerate mesh: bounding box extent {extent:.2e}")
    center = (vmin + vmax) / 2
    scale = 0.99999 / extent
    vertices = (vertices - center) * scale

    assert vertices.min() >= -0.5 and vertices.max() <= 0.5, \
        f"Normalisation failed: range [{vertices.min():.4f}, {vertices.max():.4f}]"

    return vertices.to(device).contiguous(), faces.to(device).contiguous()


@torch.no_grad()
def voxelize_and_save_vxz(vertices, faces, grid_size, vxz_path):
    """Run GPU voxelization and save VXZ file."""
    t0 = time.perf_counter()
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32, device=vertices.device)
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices,
        faces=faces,
        grid_size=grid_size,
        aabb=aabb,
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=False,
    )
    torch.cuda.synchronize()
    t_voxel = time.perf_counter() - t0
    print(f"[voxel] GPU voxelization: {t_voxel*1000:.1f}ms, {voxel_indices.shape[0]} voxels")

    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]
    dual_vertices = torch.clamp((dual_vertices * grid_size - voxel_indices) * 255, 0, 255).byte()
    intersected = (intersected[:, 0] + 2 * intersected[:, 1] + 4 * intersected[:, 2]).byte().unsqueeze(1)

    os.makedirs(os.path.dirname(vxz_path) or ".", exist_ok=True)
    o_voxel.io.write_vxz(
        vxz_path, voxel_indices.cpu(),
        {"vertices": dual_vertices.cpu(), "intersected": intersected.cpu()},
    )
    print(f"[voxel] Saved VXZ: {vxz_path}")
    return voxel_indices, dual_vertices, intersected


@torch.no_grad()
def encode_to_latent(vxz_path, fdg_enc, ss_enc, device, ss_res):
    """Load VXZ, encode to latent space."""
    coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)

    vertices = sp.SparseTensor(
        (attr["vertices"] / 255.0).float(),
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
    )
    intersected = vertices.replace(torch.cat([
        attr["intersected"] % 2,
        attr["intersected"] // 2 % 2,
        attr["intersected"] // 4 % 2,
    ], dim=-1).bool())

    z_shape = fdg_enc(vertices.cuda(), intersected.cuda())
    s_coords = z_shape.coords

    ss = torch.zeros(1, ss_res, ss_res, ss_res, dtype=torch.long, device=device)
    ss[:, s_coords[:, 1], s_coords[:, 2], s_coords[:, 3]] = 1
    ss_latent = ss_enc(ss.float()[None])

    torch.cuda.synchronize()
    if not torch.isfinite(z_shape.feats).all() or not torch.isfinite(ss_latent).all():
        raise RuntimeError("Non-finite values in encoded latents")

    return z_shape, ss_latent


@torch.no_grad()
def decode_to_mesh(z_shape, ss_latent, fdg_dec, ss_dec, grid_size, device):
    """Decode latents back to mesh. Returns shape meshes only (no texture)."""
    decoded = ss_dec(ss_latent) > 0
    ss_coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    z_shape.coords = ss_coords

    fdg_dec.set_resolution(grid_size)
    shape_meshes = fdg_dec(z_shape, return_subs=False)
    return shape_meshes


def main():
    parser = argparse.ArgumentParser(description="GLB → VXZ → Latent → GLB pipeline test")
    parser.add_argument("--input", type=str, required=True, help="Path to input GLB file")
    parser.add_argument("--output", type=str, default=None, help="Path to output GLB file (default: {input_stem}_decoded.glb in cwd)")
    parser.add_argument("--grid-size", type=int, default=512, help="Voxel grid resolution")
    parser.add_argument("--vxz", type=str, default=None, help="Path to save intermediate VXZ (default: {input_stem}.vxz in cwd)")
    parser.add_argument("--npz", type=str, default=None, help="Path to save intermediate latent NPZ (default: {input_stem}_latent.npz in cwd)")
    parser.add_argument("--skip-voxel", action="store_true", help="Skip voxelization, use existing VXZ")
    parser.add_argument("--skip-encode", action="store_true", help="Skip encoding, use existing NPZ")
    args = parser.parse_args()

    device = "cuda"
    grid_size = args.grid_size
    ss_res = 32 if grid_size <= 512 else 64

    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    vxz_path = args.vxz or f"{input_stem}.vxz"
    npz_path = args.npz or f"{input_stem}_latent.npz"
    out_glb = args.output or f"{input_stem}_decoded.glb"

    # ── Stage 1: Voxelization ──────────────────────────────────────────
    if not args.skip_voxel:
        print(f"[stage 1] Loading mesh: {args.input}")
        vertices, faces = load_and_normalize_mesh(args.input, device=device)
        print(f"[stage 1] Vertices: {vertices.shape}, Faces: {faces.shape}")

        voxelize_and_save_vxz(vertices, faces, grid_size, vxz_path)
        del vertices, faces
    else:
        print(f"[stage 1] Skipping voxelization, using existing VXZ: {vxz_path}")

    # ── Stage 2: Load models ───────────────────────────────────────────
    print("[stage 2] Loading TRELLIS models...")
    t0 = time.perf_counter()

    fdg_enc = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
    ).to(device).eval()
    fdg_dec = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
    ).to(device).eval()
    ss_enc = models.from_pretrained(
        "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
    ).to(device).eval()
    ss_dec = models.from_pretrained(
        "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
    ).to(device).eval()

    torch.cuda.synchronize()
    print(f"[stage 2] Models loaded in {(time.perf_counter() - t0):.1f}s")

    # ── Stage 3: Encode to latent ──────────────────────────────────────
    if not args.skip_encode:
        print("[stage 3] Encoding to latent...")
        t0 = time.perf_counter()

        z_shape, ss_latent = encode_to_latent(vxz_path, fdg_enc, ss_enc, device, ss_res)

        # Save NPZ
        pack = {
            "latent_feats": z_shape.feats.cpu().numpy().astype(np.float32),
            "tex_latent_feats": np.array([], dtype=np.float32),
            "ss_latent": ss_latent.cpu().numpy().astype(np.float32),
            "coords": z_shape.coords[:, 1:].cpu().numpy().astype(np.uint8),
        }
        np.savez_compressed(npz_path, **pack)
        torch.cuda.synchronize()
        print(f"[stage 3] Latent saved to {npz_path} in {(time.perf_counter() - t0):.1f}s")
    else:
        print(f"[stage 3] Loading existing NPZ: {npz_path}")
        data = np.load(npz_path)
        latent_feats = torch.from_numpy(data["latent_feats"]).to(device)
        coords_np = data["coords"]
        ss_latent = torch.from_numpy(data["ss_latent"]).to(device)

        batch_coords = torch.cat([
            torch.zeros(len(coords_np), 1, dtype=torch.int32, device=device),
            torch.from_numpy(coords_np).to(device),
        ], dim=1)
        z_shape = sp.SparseTensor(latent_feats, batch_coords)

    # ── Stage 4: Decode to mesh ────────────────────────────────────────
    print("[stage 4] Decoding to mesh...")
    t0 = time.perf_counter()

    shape_meshes = decode_to_mesh(
        z_shape, ss_latent, fdg_dec, ss_dec, grid_size, device
    )

    torch.cuda.synchronize()
    print(f"[stage 4] Decoded {len(shape_meshes)} mesh(es) in {(time.perf_counter() - t0):.1f}s")

    # ── Stage 5: Export GLB ────────────────────────────────────────────
    print("[stage 5] Exporting GLB...")
    for i, m in enumerate(shape_meshes):
        out_mesh = trimesh.Trimesh(
            vertices=m.vertices.cpu(),
            faces=m.faces.cpu(),
            process=False,
        )
        suffix = f"_{i}" if len(shape_meshes) > 1 else ""
        out_path = out_glb.replace(".glb", f"{suffix}.glb")
        out_mesh.export(out_path)
        print(f"[stage 5] Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
