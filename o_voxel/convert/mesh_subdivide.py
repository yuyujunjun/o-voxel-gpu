"""GPU longest-edge bisection for meshes with large triangles.

Place subdivide_mesh_gpu OUTSIDE the dual-grid pipeline so callers
can subdivide once and voxelize many times (e.g. animated meshes).

Usage:
    from o_voxel.convert.mesh_subdivide import subdivide_mesh_gpu

    verts, faces = subdivide_mesh_gpu(verts, faces,
        area_threshold=1.0 / (grid_size ** 2))
"""

import torch
from pytorch3d.structures import Meshes


def subdivide_mesh_gpu(verts: torch.Tensor, faces: torch.Tensor,
                       area_threshold: float = 2e-5, max_iters: int = 12):
    """GPU longest-edge bisection. Subdivides faces > area_threshold.

    Uses correct column-to-vertex-pair mapping for pytorch3d's
    faces_packed_to_edges_packed() which sorts columns by global edge ID,
    NOT by vertex-pair order.

    Args:
        verts: (V, 3) float32 tensor on GPU
        faces: (F, 3) int64 tensor on GPU
        area_threshold: split faces larger than this
        max_iters: max subdivision passes

    Returns:
        (new_verts, new_faces) — (V', 3) float32, (F', 3) int64, same device
    """
    device = verts.device
    cur = Meshes(verts=[verts], faces=[faces])

    for _ in range(max_iters):
        cv = cur.verts_packed()
        cf = cur.faces_packed()
        edges = cur.edges_packed()
        f2e = cur.faces_packed_to_edges_packed()

        areas = cur.faces_areas_packed()
        large = areas > area_threshold
        if not large.any():
            break

        edge_len = torch.norm(cv[edges[:, 0]] - cv[edges[:, 1]], dim=1)
        _, local_max = torch.max(edge_len[f2e], dim=1)
        max_edge_ids = f2e[torch.arange(cf.shape[0], device=device), local_max]

        target = torch.zeros(edges.shape[0], dtype=torch.bool, device=device)
        target[max_edge_ids[large]] = True
        if not target.any():
            break

        mid = (cv[edges[target, 0]] + cv[edges[target, 1]]) * 0.5
        new_verts = torch.cat([cv, mid], dim=0)
        n_old = cv.shape[0]

        e2new = torch.zeros(edges.shape[0], dtype=torch.long, device=device)
        e2new[target] = torch.arange(n_old, n_old + target.sum().item(), device=device)

        split_counts = target[f2e].sum(dim=1)

        # Match f2e columns -> vertex pairs
        v0, v1, v2 = cf[:, 0], cf[:, 1], cf[:, 2]
        F = cf.shape[0]

        p01_a = torch.minimum(v0, v1); p01_b = torch.maximum(v0, v1)
        p12_a = torch.minimum(v1, v2); p12_b = torch.maximum(v1, v2)
        p20_a = torch.minimum(v2, v0); p20_b = torch.maximum(v2, v0)

        e_act = edges[f2e]
        match01 = (e_act[:, :, 0] == p01_a.unsqueeze(1)) & (e_act[:, :, 1] == p01_b.unsqueeze(1))
        match12 = (e_act[:, :, 0] == p12_a.unsqueeze(1)) & (e_act[:, :, 1] == p12_b.unsqueeze(1))
        match20 = (e_act[:, :, 0] == p20_a.unsqueeze(1)) & (e_act[:, :, 1] == p20_b.unsqueeze(1))

        col01 = match01.long().argmax(dim=1)
        col12 = match12.long().argmax(dim=1)
        col20 = match20.long().argmax(dim=1)

        # 0 split: keep
        keep = cf[split_counts == 0]
        out = [keep]

        # 1 split: bisect
        m1 = (split_counts == 1)
        if m1.any():
            f1 = cf[m1]; fe1 = f2e[m1]
            M = f1.shape[0]
            v0_1, v1_1, v2_1 = f1[:, 0], f1[:, 1], f1[:, 2]

            split_col = target[fe1].long().argmax(dim=1)
            c01_m1 = col01[m1]; c12_m1 = col12[m1]; c20_m1 = col20[m1]
            is_split_01 = (split_col == c01_m1)
            is_split_12 = (split_col == c12_m1)
            is_split_20 = (split_col == c20_m1)

            split_eid = fe1[torch.arange(M, device=device), split_col]
            vn = e2new[split_eid]

            s1 = torch.zeros(M, 3, dtype=torch.long, device=device)
            s2 = torch.zeros(M, 3, dtype=torch.long, device=device)

            s1[is_split_01] = torch.stack([v0_1[is_split_01], vn[is_split_01], v2_1[is_split_01]], dim=1)
            s2[is_split_01] = torch.stack([vn[is_split_01], v1_1[is_split_01], v2_1[is_split_01]], dim=1)
            s1[is_split_12] = torch.stack([v0_1[is_split_12], v1_1[is_split_12], vn[is_split_12]], dim=1)
            s2[is_split_12] = torch.stack([v0_1[is_split_12], vn[is_split_12], v2_1[is_split_12]], dim=1)
            s1[is_split_20] = torch.stack([v0_1[is_split_20], v1_1[is_split_20], vn[is_split_20]], dim=1)
            s2[is_split_20] = torch.stack([vn[is_split_20], v1_1[is_split_20], v2_1[is_split_20]], dim=1)

            out.extend([s1, s2])

        # 2+ splits: subdivide to 4
        mm = (split_counts >= 2)
        if mm.any():
            fm = cf[mm]; fem = f2e[mm]
            M2 = fm.shape[0]
            v0_m, v1_m, v2_m = fm[:, 0], fm[:, 1], fm[:, 2]

            c01_m = col01[mm]; c12_m = col12[mm]; c20_m = col20[mm]

            eid01 = fem[torch.arange(M2, device=device), c01_m]
            e01_is_split = target[eid01]
            e01_v = torch.where(e01_is_split, e2new[eid01], v0_m)

            eid12 = fem[torch.arange(M2, device=device), c12_m]
            e12_is_split = target[eid12]
            e12_v = torch.where(e12_is_split, e2new[eid12], v1_m)

            eid20 = fem[torch.arange(M2, device=device), c20_m]
            e20_is_split = target[eid20]
            e20_v = torch.where(e20_is_split, e2new[eid20], v2_m)

            mf1 = torch.stack([v0_m, e01_v, e20_v], dim=1)
            mf2 = torch.stack([e01_v, v1_m, e12_v], dim=1)
            mf3 = torch.stack([e20_v, e12_v, v2_m], dim=1)
            mf4 = torch.stack([e01_v, e12_v, e20_v], dim=1)
            out.extend([mf1, mf2, mf3, mf4])

        final_faces = torch.cat(out, dim=0)
        cur = Meshes(verts=[new_verts], faces=[final_faces])

    return cur.verts_packed(), cur.faces_packed()
