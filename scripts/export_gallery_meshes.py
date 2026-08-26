"""scripts/export_gallery_meshes.py — label surfaces for the website Gallery, built for latency.

WHY MESHES AND NOT A VOLUME VIEWER. The site already has a NiiVue slice viewer, and it is
the wrong tool for this. Every point the Gallery makes is a SHAPE fact -- a thirteenth rib
on a lumbar body, a cage bridging an interspace, a transitional vertebra part-fused to the
ala -- and shape is what a surface shows. A surface also costs a tenth of a volume: a whole
spine as decimated meshes is a few hundred kB against ~1.7MB of NIfTI, and it rotates at
display rate on a phone.

PER-STRUCTURE, because the interaction has to carry the argument. Each vertebra, rib and
implant is its own object, so the page can isolate the 13th rib and its vertebra, fade the
rest, and let the viewer spin exactly the thing the case is an example of. A slice viewer
cannot do that.

THE LATENCY BUDGET, and where it goes:

  decimate      to a target triangle count per structure, not a fixed ratio -- a rib and a
                sacrum need very different budgets to look the same on screen
  smooth AFTER  decimating, or the result is visibly faceted; Taubin rather than Laplacian
                because Laplacian shrinks a small bone noticeably
  quantise      positions to uint16 within the case's own bounding box, normals to int8.
                Roughly 4x smaller than float32 and indistinguishable at display size
  ONE BLOB      all structures concatenated into a single .bin with a JSON index, so a case
                is one request instead of thirty

Colours come from the ITK-SNAP descriptor, so the Gallery renders in the same scheme the
segmentations were drawn in -- a lumbar rib is teal in the tool and teal on the website.

    python scripts/export_gallery_meshes.py --labels data/v5_final \\
        --cases 0431,0033,1035 --descriptor data/itksnap_v5_labels.txt --out gallery_meshes
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import label_scheme as LS                                          # noqa: E402

# Triangle budget per structure class. High enough that the fallback decimator is a
# light touch: vertex clustering tears thin shells when it has to work hard, and a rib
# is the thinnest shell here.
# BUDGETS HALVED, BECAUSE THEY WERE SIZED FOR A SCREEN NOBODY USES.
# A vertebra renders about 200 px tall in the gallery. At 9000 triangles each one covers
# roughly four pixels, which is far past the point where more triangles change what the
# viewer sees -- a smooth-shaded organic surface stops improving somewhere around ten to
# fifty pixels per triangle. The old budgets cost download time and frame time for
# geometry finer than the display can resolve.
#
# Halving them takes a case from ~246k triangles to ~120k, which is a smaller file AND a
# cheaper frame. The floor matters more than the ceiling here: ribs and thin shells keep
# proportionally more of their budget, because vertex clustering tears them when it has
# to work hard and a torn rib is visible where a slightly coarser vertebra is not.
BUDGET = {"vertebra": 4500, "rib": 3000, "pelvis": 7000, "hardware": 4000,
          "other": 3000}
MIN_VOX = 150
# NO MASK DOWNSAMPLING. At a third of resolution a rib is two or three voxels thick in
# the downsampled grid and the 0.5 isolevel erodes it to nothing -- which is where the
# patchy holes came from. Full resolution costs triangles, and the budget above spends
# them where the geometry is thin.
DOWNSAMPLE = 1


def kind(vid: int) -> str:
    if 1 <= vid <= 25:
        return "vertebra"
    if vid in (26, 29):
        return "vertebra"
    if vid in (30, 31, 32, 33):
        return "pelvis"
    if 34 <= vid <= 57 or vid in (74, 75):
        return "rib"
    if 76 <= vid <= 79:
        return "hardware"
    return "other"


def read_colours(path):
    out = {}
    if not path or not Path(path).exists():
        return out
    for line in Path(path).read_text(errors="replace").splitlines():
        t = line.split()
        if len(t) < 8 or not t[0].isdigit():
            continue
        out[int(t[0])] = [int(t[1]), int(t[2]), int(t[3])]
    return out



def open_edge_fraction(faces):
    """Fraction of edges used by exactly one triangle -- i.e. how holed the surface is.

    A closed surface has every edge shared by two triangles. This is the number that
    decides whether a render reads as a bone or as something with bites taken out of it,
    and it is cheap enough to compute for every structure on every build.
    """
    if len(faces) == 0:
        return 1.0
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return float((counts == 1).sum()) / max(1, len(counts))


# WHICH DECIMATOR ACTUALLY RAN. The two are not interchangeable: measured against the
# full-resolution surface at an identical triangle budget, clustering leaves a hip bone with
# a mean error of 0.433 mm and a worst case of 2.00 mm, while quadric decimation gives 0.313
# and 0.82. Two millimetres of deviation on a cortical surface is what reads as patchiness in
# the viewer. This set is printed at the end of every build so a machine without open3d
# cannot quietly ship the worse mesh again.
DECIMATOR = set()


def decimate(verts, faces, target):
    """Quadric decimation if available, vertex-clustering if not.

    open3d/pymeshlab are not guaranteed in the container, and a Gallery that only builds on
    one machine is not much use. Clustering is cruder but dependency-free and, at these
    triangle counts on bone, visually close.
    """
    if len(faces) <= target:
        return verts, faces
    try:
        import open3d as o3d                                        # noqa: PLC0415
        DECIMATOR.add("quadric")
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts.astype(np.float64)),
            o3d.utility.Vector3iVector(faces.astype(np.int32)))
        m = m.simplify_quadric_decimation(int(target))
        m.remove_degenerate_triangles(); m.remove_duplicated_vertices()
        return np.asarray(m.vertices, np.float32), np.asarray(m.triangles, np.int64)
    except Exception:                                               # noqa: BLE001
        DECIMATOR.add("clustering")
    # THE FALLBACK NOW AIMS AT THE TARGET INSTEAD OF GUESSING.
    #
    # It used to pick a clustering grid from a closed-form expression containing a bare
    # /24, and the triangle count that came out bore no reliable relation to `target`.
    # That made BUDGET inert: halving every budget produced MORE triangles, not fewer,
    # which is how the knob was discovered to be disconnected. open3d is not installed in
    # the container -- nor trimesh, vtk, pymeshlab or pyvista -- so the quadric path above
    # never runs here and this fallback is what everyone actually gets.
    #
    # Bisection on the grid spacing converges in a handful of tries because triangle count
    # falls monotonically as the cells grow. And it stops early if the mesh starts to TEAR:
    # clustering merges vertices across a thin shell, and a rib two voxels thick opens into
    # holes long before a vertebral body would. Overshooting the budget is a cosmetic cost;
    # a torn rib is a visible defect, so the tear check wins ties.
    def _cluster(cell):
        lo, hi = verts.min(0), verts.max(0)
        grid = np.maximum(2, np.ceil((hi - lo) / max(cell, 1e-6)).astype(int))
        idx = np.clip(((verts - lo) / (hi - lo + 1e-9) * (grid - 1)).round().astype(int),
                      0, grid - 1)
        key = (idx[:, 0] * grid[1] + idx[:, 1]) * grid[2] + idx[:, 2]
        uk, inv = np.unique(key, return_inverse=True)
        nv = np.zeros((len(uk), 3), np.float64)
        cnt = np.zeros(len(uk))
        np.add.at(nv, inv, verts)
        np.add.at(cnt, inv, 1)
        nv /= np.maximum(cnt[:, None], 1)
        nf = inv[faces]
        nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
        return nv.astype(np.float32), nf.astype(np.int64)

    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    baseline = open_edge_fraction(faces)
    lo_c, hi_c = diag * 1e-4, diag * 0.10
    best = None
    for _ in range(12):
        cell = 0.5 * (lo_c + hi_c)
        nv, nf = _cluster(cell)
        if len(nf) < 12:
            hi_c = cell
            continue
        torn = open_edge_fraction(nf) > max(0.04, baseline + 0.03)
        if torn:
            hi_c = cell                       # too coarse: it is opening holes
            continue
        best = (nv, nf)
        if len(nf) > target:
            lo_c = cell                       # still too many: cluster harder
        else:
            hi_c = cell                       # under budget: back off toward detail
        if abs(len(nf) - target) <= 0.05 * target:
            break
    return best if best is not None else (verts, faces)


def taubin(verts, faces, iters=4, lam=0.5, mu=-0.53):
    """Taubin smoothing: alternating positive/negative passes, so it does NOT shrink.

    Plain Laplacian smoothing visibly deflates a small bone -- a rib loses its cortex and a
    facet rounds off -- which matters here because the shapes ARE the content.
    """
    if len(faces) == 0:
        return verts
    n = len(verts)
    adj = [[] for _ in range(n)]
    f = faces
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for x, y in zip(f[:, a], f[:, b]):
            adj[x].append(y)
            adj[y].append(x)
    nbr = [np.unique(np.array(v, dtype=np.int64)) if v else np.array([], np.int64)
           for v in adj]
    v = verts.astype(np.float64).copy()
    for it in range(iters):
        step = lam if it % 2 == 0 else mu
        nv = v.copy()
        for i, ns in enumerate(nbr):
            if len(ns):
                nv[i] = v[i] + step * (v[ns].mean(0) - v[i])
        v = nv
    return v.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--descriptor", default="data/itksnap_v5_labels.txt")
    ap.add_argument("--out", default="gallery_meshes")
    a = ap.parse_args()

    try:
        from skimage import measure                                 # noqa: PLC0415
    except ImportError:
        print("  ! scikit-image is required for marching cubes")
        return 1

    colours = read_colours(a.descriptor)
    names = {v: k for k, v in LS.label_dict().items()}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    index = []

    for stem in [c.strip() for c in a.cases.split(",") if c.strip()]:
        fp = Path(a.labels) / f"{stem}_label.nii.gz"
        if not fp.exists():
            print(f"  ! missing {fp.name}")
            continue
        img = nib.as_closest_canonical(nib.load(str(fp)))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        sp = np.array(img.header.get_zooms()[:3], float)

        ids = [int(v) for v in np.unique(lab) if v and int(v) != LS.IGNORE_LABEL
               and (lab == v).sum() >= MIN_VOX]
        # one bounding box for the whole case, so every structure quantises into the same
        # frame and the page can place them without per-structure transforms
        occ = np.isin(lab, ids)
        idx = np.argwhere(occ)
        lo = (idx.min(0) * sp).astype(np.float32)
        hi = (idx.max(0) * sp).astype(np.float32)
        span = np.maximum(hi - lo, 1e-3)

        blob, meta, off = bytearray(), [], 0
        holed = []          # structures still carrying boundary edges
        # every structure's first section starts 4-aligned because each
        # section is padded up to 4 on the way in
        for vid in ids:
            m = (lab == vid)
            # DOWNSAMPLE THE MASK FIRST. Decimating a finished mesh is the wrong lever:
            # open3d is absent from the container so quadric decimation is unavailable, and
            # the clustering fallback left 280k-570k triangles per case -- 20x the budget.
            # Marching cubes on a mask at half resolution produces roughly a quarter the
            # triangles directly, is far cheaper, and on bone at display size is
            # indistinguishable. Anti-aliasing the boolean first stops the coarser grid
            # from turning smooth cortex into stairs.
            # CROP TO THE STRUCTURE FIRST. Blurring and marching the WHOLE volume for
            # every one of thirty structures allocated a 570 MiB float32 copy each time --
            # which is both why the export took a quarter of an hour a case and why running
            # several at once died with a memory error. A vertebra occupies well under a
            # hundredth of the volume, and marching cubes inside its own bounding box gives
            # the same surface; the vertices are shifted back afterwards.
            bb = []
            for ax in range(3):
                hit = np.nonzero(m.any(axis=tuple(i for i in range(3) if i != ax)))[0]
                if not len(hit):
                    break
                pad = 3
                bb.append(slice(max(0, int(hit[0]) - pad),
                                min(m.shape[ax], int(hit[-1]) + pad + 1)))
            if len(bb) != 3:
                continue
            sub = m[tuple(bb)]
            origin = np.array([q.start for q in bb], float) * np.asarray(sp, float)

            step = DOWNSAMPLE
            if step == 1:
                # a light blur, not a resample: it takes the staircase off the cortex
                # without moving the surface, and gives the decimator a smoother field
                sm = ndimage.gaussian_filter(sub.astype(np.float32), 0.6)
                eff = tuple(np.asarray(sp, float))
            else:
                sm = ndimage.zoom(sub.astype(np.float32), 1.0 / step, order=1)
                eff = tuple(np.asarray(sp, float) * step)
            if sm.size == 0 or sm.max() < 0.5:
                continue
            try:
                v, f, _, _ = measure.marching_cubes(sm, level=0.5, spacing=eff)
            except Exception:                                       # noqa: BLE001
                continue
            v = v + origin
            if len(f) == 0:
                continue
            v, f = decimate(v.astype(np.float32), f.astype(np.int64),
                            BUDGET.get(kind(vid), 1500))
            if len(f) == 0:
                continue
            v = taubin(v, f)
            _open = open_edge_fraction(f)
            if _open > 0.02:
                holed.append((int(vid), round(_open, 3)))
            # normals from face geometry, averaged per vertex
            tri = v[f]
            fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            vn = np.zeros_like(v)
            for c in range(3):
                np.add.at(vn, f[:, c], fn)
            ln = np.linalg.norm(vn, axis=1, keepdims=True)
            vn = np.divide(vn, np.maximum(ln, 1e-9))

            q = np.clip(((v - lo) / span * 65535.0).round(), 0, 65535).astype("<u2")
            nq = np.clip((vn * 127.0).round(), -127, 127).astype("<i1")
            fi = f.astype("<u4" if len(v) > 65535 else "<u2")

            # ALIGNMENT. A TypedArray view must begin on a multiple of its element size:
            # Uint16Array needs an even byteOffset, Uint32Array a multiple of four. The
            # normals are INT8, so an odd-length run pushes whatever follows onto an odd
            # offset and the browser throws a RangeError -- which killed every case at its
            # first misaligned structure (141 of them across five cases). Pad each section
            # to 4 bytes; the cost is at most three bytes per array.
            def _put(arr):
                nonlocal blob
                start = len(blob)
                blob += arr.tobytes()
                pad = (-len(blob)) % 4
                if pad:
                    blob += b"\x00" * pad
                return [start, int(arr.nbytes)]

            # NORMALS ARE NOT SHIPPED. They are computed here as face normals averaged
            # per vertex and normalised -- which is precisely what three.js's
            # computeVertexNormals() does, from data the viewer already has. Sending them
            # spends 14% of the payload on something the client can reconstruct exactly.
            #
            # The one thing that changes is that the client recomputes from DEQUANTISED
            # positions rather than the float ones used here. At 16 bits across the case's
            # own bounding box that error is a fraction of a micron, which no shading
            # difference survives.
            rec = {"id": int(vid), "name": names.get(vid, str(vid)),
                   "kind": kind(vid), "color": colours.get(vid, [200, 200, 200]),
                   "nverts": int(len(v)), "ntris": int(len(f)),
                   "pos": _put(q), "idx": _put(fi),
                   "idx_bytes": int(fi.dtype.itemsize)}
            off = len(blob)
            meta.append(rec)

        raw = bytes(blob)
        (out / f"{stem}.bin").write_bytes(raw)
        # AND A GZIPPED COPY. The payload is quantised integers, so gzip only returns
        # about 1.27x -- there is little redundancy left to find. It is still a quarter
        # off the wire for the cost of one file and DecompressionStream in the browser,
        # which is native and needs no library. The viewer asks for .bin.gz and falls
        # back to .bin, so an old client and a new file still work together.
        import gzip as _gz
        (out / f"{stem}.bin.gz").write_bytes(_gz.compress(raw, 9))
        # THE AXIS CODES TRAVEL WITH THE MESH. Vertices are in the label array's own
        # axes, and these volumes are ('P','I','R') -- the first mesh axis runs
        # posterior, not right. Without this the viewer has to assume a patient frame,
        # and its "anterior" view looked at the back. Recording what nibabel reports
        # means nothing downstream has to guess.
        head = {"case": stem, "bbox_lo": lo.tolist(), "bbox_hi": hi.tolist(),
                "quant": 65535, "axcodes": list(nib.aff2axcodes(img.affine)),
                "mm_per_unit": 1.0,
                # tells the viewer to reconstruct normals rather than look for a stream
                # that is no longer there; absent in files written before this change,
                # which is exactly the right default for them
                "normals": False,
                "structures": meta}
        (out / f"{stem}.json").write_text(json.dumps(head))
        kb = len(raw) / 1024
        gz_kb = len(_gz.compress(raw, 9)) / 1024
        tris = sum(m["ntris"] for m in meta)
        print(f"  {stem}: {len(meta)} structures, {tris} tris, "
              f"{kb:.0f} kB raw, {gz_kb:.0f} kB gzipped")
        if DECIMATOR:
            which = ", ".join(sorted(DECIMATOR))
            print(f"  decimator: {which}")
            if "clustering" in DECIMATOR:
                print("  ! vertex clustering was used for at least one structure. Install "
                      "open3d: it halves the surface error at the same triangle count.")
        if holed:
            worst = sorted(holed, key=lambda t: -t[1])[:4]
            print("      holes: " + ", ".join(f"id{v} {o:.1%}" for v, o in worst)
                  + (f"  (+{len(holed) - len(worst)} more)" if len(holed) > 4 else ""))
        else:
            print("      holes: none above 2% of edges")
        index.append({"case": stem, "structures": len(meta), "kB": round(kb)})

    # THE INDEX DESCRIBES THE DIRECTORY, NOT THIS RUN. Writing only the cases just
    # exported replaced an eight-case index with a one-case index the first time a single
    # case was regenerated, and deck.js reads it, so the rest of the set disappeared.
    # Rebuilt from what is on disk: exporting one case can no longer remove seven.
    everything = []
    for j in sorted(out.glob("*.json")):
        if j.stem in ("index", "distributions"):
            continue
        try:
            h = json.loads(j.read_text(encoding="utf-8"))
        except ValueError:
            continue
        b = out / f"{j.stem}.bin"
        everything.append({
            "case": j.stem,
            "structures": len(h.get("structures", [])),
            "kB": round(b.stat().st_size / 1024) if b.exists() else 0,
        })
    # A BUILD STAMP OVER THE MESH PAYLOAD, for the same reason the measure figures carry
    # one: the .bin and .bin.gz names never change, so a browser holding them keeps them and
    # a rebuilt gallery reaches nobody. viewer.js appends this to every data URL.
    import hashlib
    h = hashlib.sha256()
    for q in sorted(out.glob("*.bin")):
        h.update(q.read_bytes())
    build = h.hexdigest()[:8]
    (out / "index.json").write_text(
        json.dumps({"build": build, "cases": everything}, indent=1) + "\n",
        encoding="utf-8")
    print(f"  mesh build {build}")
    print(f"\n  wrote {out}/  ({len(index)} case(s) this run, "
          f"{len(everything)} in the index)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
