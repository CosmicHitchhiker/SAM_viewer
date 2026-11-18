"""Build the sidecar index the viewer needs for an ASTRAEUS treefile.

    python build_index.py tree_0.dat [tree_1.dat ...]

The treefile is only ever opened READ-ONLY; nothing in it is changed. The script
writes one new file next to it, ``<treefile>.viewidx.npz``, holding

* the snapshot number of every galaxy,
* the largest Mvir of every tree,
* min / smallest positive / max of every column,
* the scale factor of every snapshot,
* ``dMstar`` = Mstar - sum(Mstar of direct progenitors) of every galaxy
  (the viewer divides it by the time between snapshots to get the SFR).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from astraeus_io import memmap_treefile  # noqa: E402

INDEX_SUFFIX = ".viewidx.npz"
INDEX_VERSION = 1
CHUNK = 500_000  # galaxies read per step (~100 MB)
SKIP_RANGE = ("Padding",)


def index_path(treefile: str | Path) -> Path:
    return Path(str(treefile) + INDEX_SUFFIX)


def build_index(treefile: str | Path) -> Path:
    treefile = Path(treefile)
    stat = treefile.stat()
    ngal, mm = memmap_treefile(treefile)
    ntrees, n = len(ngal), len(mm)
    offsets = np.concatenate([[0], np.cumsum(ngal)])
    cols = [c for c in mm.dtype.names if c not in SKIP_RANGE]
    print(f"{treefile}: {ntrees:,} trees, {n:,} galaxies, {mm.dtype.itemsize} bytes per galaxy")

    snap = np.empty(n, dtype=np.int16)
    dmstar = np.empty(n, dtype=np.float32)
    tree_mass = np.zeros(ntrees, dtype=np.float32)
    col_min = np.full(len(cols), np.inf)
    col_minpos = np.full(len(cols), np.inf)
    col_max = np.full(len(cols), -np.inf)
    aexp_of_snap: dict[int, float] = {}

    # chunk boundaries fall on tree boundaries so progenitor sums never straddle two chunks
    bounds = np.unique(np.append(np.searchsorted(offsets, np.arange(0, n, CHUNK)), ntrees))
    t_start = time.time()
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        a, b = int(offsets[t0]), int(offsets[t1])
        if b == a:
            continue
        rec = np.array(mm[a:b])  # private in-memory copy of this chunk
        counts = ngal[t0:t1]
        starts = (offsets[t0:t1] - a)[counts > 0]
        first = np.repeat(offsets[t0:t1] - a, counts)  # chunk position of each galaxy's tree

        if not np.array_equal(rec["localID"], np.arange(b - a) - first):
            raise ValueError(f"{treefile}: localID is not the position of the galaxy inside its tree "
                             f"(trees {t0}..{t1}); this layout is not supported")

        snap[a:b] = rec["snapnum"]
        u, i = np.unique(rec["snapnum"], return_index=True)
        aexp_of_snap.update(zip(u.tolist(), rec["aexp"][i].tolist()))
        tree_mass[t0:t1][counts > 0] = np.maximum.reduceat(rec["Mvir"], starts)

        mstar = rec["Mstar"].astype(np.float64)
        desc = rec["localDescID"]
        has_desc = desc >= 0
        prog_sum = np.bincount(first[has_desc] + desc[has_desc], weights=mstar[has_desc], minlength=b - a)
        dmstar[a:b] = mstar - prog_sum

        for k, c in enumerate(cols):
            v = rec[c]
            v = v[np.isfinite(v)]
            if len(v):
                col_min[k] = min(col_min[k], v.min())
                col_max[k] = max(col_max[k], v.max())
                pos = v[v > 0]
                if len(pos):
                    col_minpos[k] = min(col_minpos[k], pos.min())
        print(f"\r  {b:,} / {n:,} galaxies ({100 * b / n:.0f} %, {time.time() - t_start:.0f} s)",
              end="", flush=True)
    print()

    snap_ids = np.array(sorted(aexp_of_snap), dtype=np.int64)
    out = index_path(treefile)
    with open(out, "wb") as f:
        np.savez(f, version=INDEX_VERSION, src_size=stat.st_size, src_mtime_ns=stat.st_mtime_ns,
                 snap=snap, dMstar=dmstar, tree_mass=tree_mass,
                 snap_ids=snap_ids, snap_aexp=np.array([aexp_of_snap[s] for s in snap_ids.tolist()]),
                 col_names=np.array(cols), col_min=col_min, col_minpos=col_minpos, col_max=col_max)

    after = treefile.stat()
    assert (after.st_size, after.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns), "treefile changed?!"
    print(f"  wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB); {treefile.name} untouched")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("treefiles", nargs="+", type=Path, help="ASTRAEUS tree_*.dat files")
    for treefile in parser.parse_args().treefiles:
        build_index(treefile)


if __name__ == "__main__":
    main()
