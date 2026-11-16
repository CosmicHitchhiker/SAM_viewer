"""Loading of simulation output JSON files and merger-tree bookkeeping.

The viewer never hard-codes galaxy properties: every numeric field found in the
file becomes a plottable parameter, and list-valued fields (``pos``, ``vel``, ...)
are split into ``pos_x, pos_y, pos_z`` (or ``name_0, name_1, ...``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:  # optional, much faster for big files
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

# Columns describing the tree structure; they are not offered as plot parameters.
STRUCTURE_COLS = ("treeID", "localID", "localDescID")
VECTOR_SUFFIXES = {3: ("x", "y", "z")}


GYR_PER_INV_H0 = 9.7779  # 1/H0 in Gyr for h = 1
DEFAULT_COSMOLOGY = {"Omega_m": 0.315, "Omega_L": 0.685, "h": 0.674}


def load_cosmology(config: str | Path | None) -> dict:
    """Read the [cosmology] table of a simulation parameter file (TOML)."""

    if config and Path(config).is_file():
        try:
            import tomllib
        except ModuleNotFoundError:  # Python < 3.11
            import tomli as tomllib
        with open(config, "rb") as f:
            return {**DEFAULT_COSMOLOGY, **tomllib.load(f).get("cosmology", {})}
    return dict(DEFAULT_COSMOLOGY)


def cosmic_age_gyr(a: np.ndarray, cosmo: dict) -> np.ndarray:
    """Age of the universe at scale factor ``a`` (flat LCDM, no radiation)."""

    om, ol, h = cosmo["Omega_m"], cosmo["Omega_L"], cosmo["h"]
    return 2.0 / (3.0 * np.sqrt(ol)) * GYR_PER_INV_H0 / h * np.arcsinh(np.sqrt(ol / om) * np.power(a, 1.5))


def _read_json(path: Path) -> dict:
    if orjson is not None:
        return orjson.loads(path.read_bytes())
    with path.open() as f:
        return json.load(f)


def flatten_galaxies(data: dict) -> pd.DataFrame:
    """Turn ``{"trees": [{"treeID", "galaxies": [...]}]}`` into one flat table."""

    rows = []
    for t_index, tree in enumerate(data["trees"]):
        tree_id = tree.get("treeID", t_index)
        for galaxy in tree["galaxies"]:
            if "treeID" not in galaxy:
                galaxy = {**galaxy, "treeID": tree_id}
            rows.append(galaxy)

    df = pd.DataFrame(rows)
    for col in list(df.columns):
        sample = df[col].dropna()
        if sample.empty:
            continue
        first = sample.iloc[0]
        if isinstance(first, (list, tuple)):
            arr = np.array(df[col].tolist(), dtype=float)
            suffixes = VECTOR_SUFFIXES.get(arr.shape[1], range(arr.shape[1]))
            for k, suffix in enumerate(suffixes):
                df[f"{col}_{suffix}"] = arr[:, k]
            df = df.drop(columns=col)
        elif isinstance(first, dict):
            df = df.drop(columns=col)

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    df = df[numeric].astype(float, errors="ignore")
    for col in STRUCTURE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(np.int64)
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


@dataclass
class OutputFile:
    path: Path
    df: pd.DataFrame
    mass_col: str
    params: list[str] = field(default_factory=list)
    time_cols: list[str] = field(default_factory=list)
    steps: np.ndarray = field(default_factory=lambda: np.array([]))
    _trees: dict = field(default_factory=dict)
    _groups: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, cosmology: dict | None = None) -> "OutputFile":
        path = Path(path)
        df = flatten_galaxies(_read_json(path))
        if "treeID" not in df:
            df["treeID"] = 0
        if "scalefactor" in df:
            a = df["scalefactor"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                df["redshift"] = np.where(a > 0, 1.0 / a - 1.0, np.nan)
            cosmo = cosmology or DEFAULT_COSMOLOGY
            df["cosmic_time"] = np.where(a > 0, cosmic_age_gyr(np.clip(a, 0, None), cosmo), np.nan)
            df["lookback_time"] = float(cosmic_age_gyr(np.array(1.0), cosmo)) - df["cosmic_time"]

        time_cols = [c for c in ("snapnumber", "scalefactor", "redshift", "cosmic_time", "lookback_time") if c in df]
        if not time_cols:
            raise ValueError("Output needs at least one of snapnumber / scalefactor")
        key = "snapnumber" if "snapnumber" in df else "scalefactor"
        steps = np.sort(df[key].unique())
        df["_step"] = np.searchsorted(steps, df[key].to_numpy())

        mass_col = "Mvir" if "Mvir" in df else next(
            (c for c in df.columns if c not in STRUCTURE_COLS), df.columns[0])
        params = [c for c in df.columns if c not in STRUCTURE_COLS and not c.startswith("_")]
        groups = df.groupby("treeID", sort=True).indices
        return cls(path, df, mass_col, params, time_cols, steps, {}, groups)

    @property
    def tree_ids(self) -> list[int]:
        return [int(t) for t in self._groups]

    def tree_summary(self) -> pd.DataFrame:
        g = self.df.groupby("treeID")
        out = pd.DataFrame({"n": g.size(), "mass": g[self.mass_col].max()})
        return out.reset_index()

    def tree(self, tree_id: int) -> TreeData:
        tree_id = int(tree_id)
        if tree_id not in self._trees:
            if len(self._trees) > 8:  # keep memory bounded
                self._trees.pop(next(iter(self._trees)))
            sub = self.df.iloc[self._groups[tree_id]]
            self._trees[tree_id] = TreeData(sub, self.mass_col)
        return self._trees[tree_id]


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
    out = OutputFile.load(path, cosmology or _COSMOLOGY)
    _CACHE[str(path)] = (mtime, out)
    return out


def set_cosmology(cosmology: dict) -> None:
    """Cosmology used to derive cosmic/lookback time for files loaded from now on."""

    _COSMOLOGY.clear()
    _COSMOLOGY.update(cosmology)
    _CACHE.clear()


def list_outputs(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("*.json"))
