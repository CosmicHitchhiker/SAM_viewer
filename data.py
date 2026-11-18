"""Reading of ASTRAEUS treefiles (``tree_*.dat``) and merger-tree bookkeeping.

The treefile is memory-mapped READ-ONLY and never loaded as a whole: a tree is read
when it is selected, whole-file plots use a random sample of galaxies. The sidecar
index written by ``build_index.py`` supplies everything that would need a full scan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from astraeus_io import memmap_treefile
from build_index import INDEX_VERSION, index_path

# Columns describing the tree structure; they are not offered as plot parameters.
STRUCTURE_COLS = ("treeID", "localID", "localDescID", "ID", "descID")
RENAME = {"snapnum": "snapnumber", "aexp": "scalefactor"}  # ASTRAEUS name -> viewer name
MAX_SCATTER_POINTS = 100_000

GYR_PER_INV_H0 = 9.7779  # 1/H0 in Gyr for h = 1
DEFAULT_COSMOLOGY = {"Omega_m": 0.307115, "Omega_L": 0.692885, "h": 0.6777}  # VSMDPL (Planck 2013)


def load_cosmology(config: str | Path | None) -> dict:
    """Read the [cosmology] table of a parameter file (TOML); VSMDPL values without one."""

    if config and Path(config).is_file():
        import tomllib
        with open(config, "rb") as f:
            return {**DEFAULT_COSMOLOGY, **tomllib.load(f).get("cosmology", {})}
    return dict(DEFAULT_COSMOLOGY)


def cosmic_age_gyr(a: np.ndarray, cosmo: dict) -> np.ndarray:
    """Age of the universe at scale factor ``a`` (flat LCDM, no radiation)."""

    om, ol, h = cosmo["Omega_m"], cosmo["Omega_L"], cosmo["h"]
    return 2.0 / (3.0 * np.sqrt(ol)) * GYR_PER_INV_H0 / h * np.arcsinh(np.sqrt(ol / om) * np.power(a, 1.5))


def _ratio(num, den) -> np.ndarray:
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


# name -> (columns it needs, formula); masses keep the file's h^-1 units, ratios are unit-free
DERIVED = {
    "sSFR": (("SFR", "Mstar"), lambda d: _ratio(d["SFR"], d["Mstar"])),
    "Z_gas": (("MZ", "Mgas"), lambda d: _ratio(d["MZ"], d["Mgas"])),
    "Z_O": (("MZ_O", "Mgas"), lambda d: _ratio(d["MZ_O"], d["Mgas"])),
    "O_Fe": (("MZ_O", "MZ_Fe"), lambda d: _ratio(d["MZ_O"], d["MZ_Fe"])),
    "f_gas": (("Mgas", "Mstar"), lambda d: _ratio(d["Mgas"], d["Mgas"] + d["Mstar"])),
    "f_star": (("Mstar", "Mvir"), lambda d: _ratio(d["Mstar"], d["Mvir"])),
    "f_baryon": (("Mgas", "Mstar", "Mvir"), lambda d: _ratio(d["Mgas"] + d["Mstar"], d["Mvir"])),
    "dust_to_gas": (("Mdust", "Mgas"), lambda d: _ratio(d["Mdust"], d["Mgas"])),
    "dust_to_metal": (("Mdust", "MZ"), lambda d: _ratio(d["Mdust"], d["MZ"])),
    "f_ej": (("Mgas_ej", "Mgas_ini"), lambda d: _ratio(d["Mgas_ej"], d["Mgas_ini"])),
}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    for name, (needs, formula) in DERIVED.items():
        if all(c in df for c in needs):
            df[name] = formula(df)
    return df


class TreeData:
    """One merger tree: galaxy table, parent/child links and a drawing layout."""

    def __init__(self, df: pd.DataFrame, mass_col: str):
        self.df = df.reset_index(drop=True)
        self.mass_col = mass_col
        n = len(self.df)

        local = self.df["localID"].to_numpy()
        desc_local = self.df["localDescID"].to_numpy()
        lookup = np.full(int(max(local.max(), desc_local.max())) + 2, -1, dtype=np.int64)
        lookup[local] = np.arange(n)
        self.row_of_local = lookup
        # descendant row (-1 for tree roots)
        self.desc = np.where(desc_local >= 0, lookup[np.clip(desc_local, 0, None)], -1)

        mass = self.df[mass_col].to_numpy(dtype=float) if mass_col in self.df else np.zeros(n)
        mass = np.nan_to_num(mass, nan=0.0)
        self.mass = mass

        # children grouped by parent, most massive first
        has_parent = np.flatnonzero(self.desc >= 0)
        order = has_parent[np.lexsort((-mass[has_parent], self.desc[has_parent]))]
        self.child_idx = order
        counts = np.bincount(self.desc[order], minlength=n)
        self.child_ptr = np.concatenate([[0], np.cumsum(counts)])

        self.roots = np.flatnonzero(self.desc < 0)
        self._compute_layout()

    # ---- structure -------------------------------------------------------
    def children(self, i: int) -> np.ndarray:
        return self.child_idx[self.child_ptr[i]:self.child_ptr[i + 1]]

    def row(self, local_id: int) -> int:
        if 0 <= local_id < len(self.row_of_local):
            return int(self.row_of_local[local_id])
        return -1

    def descendants(self, i: int) -> list[int]:
        """Chain of descendants from ``i`` (excluded) down to the root."""

        chain = []
        j = int(self.desc[i])
        while j >= 0:
            chain.append(j)
            j = int(self.desc[j])
        return chain

    def progenitors(self, i: int, main_only: bool = False) -> list[int]:
        """All progenitors of ``i`` (excluded), or only the most massive branch."""

        if main_only:
            chain = []
            kids = self.children(i)
            while len(kids):
                chain.append(int(kids[0]))
                kids = self.children(int(kids[0]))
            return chain

        found, stack = [], list(self.children(i))
        while stack:
            j = int(stack.pop())
            found.append(j)
            stack.extend(self.children(j))
        return found

    # ---- layout ----------------------------------------------------------
    def _compute_layout(self) -> None:
        """Assign a breadth coordinate ``y`` to every galaxy.

        Leaves get consecutive slots; a galaxy sits at the position of its most
        massive progenitor, so every main branch is a straight line and mergers
        show up as branches joining it.
        """

        n = len(self.df)
        y = np.zeros(n)
        slot = 0
        roots = self.roots[np.argsort(-self.mass[self.roots], kind="stable")]
        for root in roots:
            stack = [(int(root), False)]
            while stack:
                node, expanded = stack.pop()
                kids = self.children(node)
                if not len(kids):
                    y[node] = slot
                    slot += 1
                elif expanded:
                    y[node] = y[kids[0]]
                else:
                    stack.append((node, True))
                    # reversed so the most massive child is visited (and placed) first
                    stack.extend((int(k), False) for k in kids[::-1])
            slot += 1  # gap between separate roots
        self.y = y


class OutputFile:
    """One ASTRAEUS treefile: read-only memory map + the index from ``build_index.py``."""

    time_cols = ["snapnumber", "scalefactor", "redshift", "cosmic_time", "lookback_time"]
    mass_col = "Mvir"

    def __init__(self, path: Path, cosmology: dict):
        self.path = path
        self.cosmo = cosmology
        self.ngal, self.mm = memmap_treefile(path)
        self.offsets = np.concatenate([[0], np.cumsum(self.ngal)])

        idx_file, stat = index_path(path), path.stat()
        hint = f"run `python build_index.py {path}` first"
        if not idx_file.is_file():
            raise FileNotFoundError(f"no index file {idx_file.name}; {hint}")
        with np.load(idx_file) as idx:
            if (int(idx["version"]) != INDEX_VERSION or int(idx["src_size"]) != stat.st_size
                    or int(idx["src_mtime_ns"]) != stat.st_mtime_ns):
                raise ValueError(f"{idx_file.name} does not belong to this version of the file; {hint}")
            self.snap = idx["snap"]
            self.dmstar = idx["dMstar"]
            self.tree_mass = idx["tree_mass"]
            self.steps = idx["snap_ids"]
            self.step_aexp = idx["snap_aexp"]
            self._extent = {RENAME.get(str(c), str(c)): np.array([lo, pos, hi]) for c, lo, pos, hi in zip(
                idx["col_names"], idx["col_min"], idx["col_minpos"], idx["col_max"])}

        self.step_age = cosmic_age_gyr(self.step_aexp, cosmology)  # Gyr
        self.age_now = float(cosmic_age_gyr(np.array(1.0), cosmology))
        # years since the previous snapshot (unknown for the first one)
        self.step_dt = np.concatenate([[np.nan], np.diff(self.step_age)]) * 1e9
        self._trees: dict[int, TreeData] = {}
        self._samples: dict = {}
        first = int(np.argmax(self.ngal > 0))
        frame = self._frame(slice(int(self.offsets[first]), int(self.offsets[first]) + 1))
        self.params = [c for c in frame.columns if c not in STRUCTURE_COLS and not c.startswith("_")]

    # ---- reading ---------------------------------------------------------
    def _frame(self, rows) -> pd.DataFrame:
        """Galaxy table for a slice or a sorted array of global row numbers."""

        df = pd.DataFrame(np.array(self.mm[rows]))
        df = df.drop(columns=[c for c in ("Padding",) if c in df]).rename(columns=RENAME)
        if isinstance(rows, slice):
            rows = np.arange(rows.start, rows.stop)
        # side="right" skips empty trees, which share their offset with the next tree
        df["treeID"] = np.searchsorted(self.offsets, rows, side="right") - 1
        step = np.searchsorted(self.steps, self.snap[rows])
        df["_step"] = step
        a = df["scalefactor"].to_numpy(dtype=float)
        df["redshift"] = 1.0 / a - 1.0
        df["cosmic_time"] = cosmic_age_gyr(a, self.cosmo)
        df["lookback_time"] = self.age_now - df["cosmic_time"]
        df["SFR"] = self.dmstar[rows] / self.step_dt[step]
        return add_derived(df)

    def has_tree(self, tree_id: int) -> bool:
        return 0 <= tree_id < len(self.ngal) and self.ngal[tree_id] > 0

    def tree_summary(self) -> pd.DataFrame:
        return pd.DataFrame({"treeID": np.arange(len(self.ngal)), "n": self.ngal, "mass": self.tree_mass})

    def tree(self, tree_id: int) -> TreeData:
        tree_id = int(tree_id)
        if tree_id not in self._trees:
            if len(self._trees) > 8:  # keep memory bounded
                self._trees.pop(next(iter(self._trees)))
            rows = slice(int(self.offsets[tree_id]), int(self.offsets[tree_id + 1]))
            self._trees[tree_id] = TreeData(self._frame(rows), self.mass_col)
        return self._trees[tree_id]

    def sample(self, step: int | None = None, n: int = MAX_SCATTER_POINTS) -> tuple[pd.DataFrame, int]:
        """Up to ``n`` random galaxies of the whole file, or of one snapshot (index into ``steps``).

        Returns the table and the number of galaxies it was drawn from.
        """

        key = (step, n)
        if key not in self._samples:
            if len(self._samples) >= 3:
                self._samples.pop(next(iter(self._samples)))
            rng = np.random.default_rng(0)
            if step is None:
                total = len(self.mm)
                rows = rng.choice(total, n, replace=False) if total > n else np.arange(total)
            else:
                rows = np.flatnonzero(self.snap == self.steps[step])
                total = len(rows)
                if total > n:
                    rows = rng.choice(rows, n, replace=False)
            self._samples[key] = (self._frame(np.sort(rows)), total)
        return self._samples[key]

    def extent(self, col: str) -> np.ndarray:
        """Values spanning the range of ``col`` over the whole file (exact for stored columns,
        from the random sample for derived ones)."""

        if col in self._extent:
            return self._extent[col]
        return self.sample()[0][col].to_numpy(dtype=float)

    def step_label(self, idx: int) -> str:
        a = self.step_aexp[idx]
        return f"snapshot {self.steps[idx]} (z = {1 / a - 1:.3g}, t = {self.step_age[idx]:.3g} Gyr)"


_COSMOLOGY: dict = dict(DEFAULT_COSMOLOGY)
_CACHE: dict[str, tuple[float, OutputFile]] = {}


def get_output(path: str | Path, cosmology: dict | None = None) -> OutputFile:
    """Load ``path``, reusing the cached copy while the file is unchanged."""

    path = Path(path)
    mtime = path.stat().st_mtime
    hit = _CACHE.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    if len(_CACHE) >= 2:
        _CACHE.pop(next(iter(_CACHE)))
    out = OutputFile(path, cosmology or _COSMOLOGY)
    _CACHE[str(path)] = (mtime, out)
    return out


def set_cosmology(cosmology: dict) -> None:
    """Cosmology used to derive cosmic/lookback time for files loaded from now on."""

    _COSMOLOGY.clear()
    _COSMOLOGY.update(cosmology)
    _CACHE.clear()


def list_outputs(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("tree_*.dat"))
