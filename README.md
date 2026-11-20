# ASTRAEUS tree viewer

An interactive web app for exploring the galaxy merger trees (`tree_*.dat` files)
produced by the ASTRAEUS semi-analytic galaxy
evolution model. It runs locally in your browser and is built with
[Dash](https://dash.plotly.com) and Plotly.

With it you can:

* draw a merger tree as a graph and step through it in time,
* inspect every stored parameter of any galaxy and walk to its progenitors and descendant,
* plot the evolution of any parameter along a galaxy's history,
* make scatter plots of any parameter against any other, for one tree or the whole file,
* look at the full galaxy population of a single snapshot.

Treefiles of several GB (millions of galaxies) are handled without loading them into
memory, and the treefile itself is never modified.

## Requirements

* Python 3.9 or newer (3.11 or newer for the `--config` option, which uses `tomllib`)
* `dash`, `plotly`, `pandas`, `numpy` (see `requirements.txt`)

## Installation

```bash
git clone <repository-url>
cd viewer
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Usage

### 1. Build the index (once per treefile)

```bash
python build_index.py path/to/tree_0.dat [path/to/tree_1.dat ...]
```

This reads the treefile once and writes a small sidecar file next to it,
`tree_0.dat.viewidx.npz` (about 110 MB for 18 million galaxies; takes roughly
10–30 s). It holds information that would otherwise require scanning the whole file
on every start: the snapshot number of every galaxy, the largest `Mvir` of every tree,
the min/max of every column, the scale factor of every snapshot and
`Mstar − Σ Mstar(progenitors)` of every galaxy (used for the SFR).

The treefile is opened read-only. If the treefile changes later (different size or
modification time), the viewer reports that the index is out of date; just run
`build_index.py` again.

### 2. Start the viewer

```bash
python app.py --dir path/to/treefiles
```

and open <http://127.0.0.1:8050> in a browser.

| Option | Default | Meaning |
| --- | --- | --- |
| `--dir DIR` | `.` | Directory scanned for `tree_*.dat` files. Files added later appear after clicking *Refresh file list*. |
| `--port PORT` | `8050` | Port of the web server. |
| `--host HOST` | `127.0.0.1` | Interface to listen on (use `0.0.0.0` to reach the viewer from another machine). |
| `--config FILE.toml` | VSMDPL cosmology | Cosmology to use, see [Cosmology](#cosmology). |
| `--debug` | off | Run Dash in debug mode (auto-reload, error pop-ups). |

The treefile layout (with/without metals, with/without Rockstar IDs) is detected
automatically from the record size.

## Features

| Tab | What it shows |
| --- | --- |
| **Merger tree** | The tree as a graph (nodes = galaxies, edges = progenitor → descendant), with time on the x or y axis. Nodes can be coloured and sized by any parameter. A **time slider** with a play button either grows the tree snapshot by snapshot or highlights one snapshot. Clicking a node selects that galaxy and highlights its descendants and main progenitor branch. |
| **Galaxy** | All parameters of the selected galaxy, with buttons to jump to its direct progenitors or its descendant. |
| **Evolution** | Any number of parameters vs. time for the selected galaxy, its progenitors (all or only the main branch) and its descendants. |
| **Parameter vs parameter** | Scatter plot of any parameter against any other, optionally coloured by a third, for the selected lineage, the whole tree, or the whole file (random sample of 100,000 galaxies). Clicking a point from another tree switches to that tree. |
| **Snapshot population** | All galaxies of one snapshot (random sample of 100,000 if there are more), for the whole file or the selected tree. Options: log axes, axes and colour scale kept fixed while moving through time, binned-median line. Clicking a point selects that galaxy, switching trees if necessary. |

**Choosing a tree:** a file can contain a million trees, so the dropdown lists only
the 500 trees with the most galaxies. Any other tree can be opened with *Go to treeID*
or by clicking one of its galaxies in a whole-file scatter plot.

**Large trees:** the *hide galaxies below log10 mass* slider thins out the tree
drawing; by default only the 20,000 most massive galaxies are shown.

## Parameters

Every field stored in the treefile can be plotted, **with values exactly as stored**
(masses in h⁻¹ M☉ etc.; nothing is rescaled). `snapnum` and `aexp` are shown as
`snapnumber` and `scalefactor`. The following quantities are derived on the fly:

| Name | Definition |
| --- | --- |
| `redshift`, `cosmic_time`, `lookback_time` | from `aexp`; times in Gyr for flat ΛCDM (radiation neglected) |
| `SFR` | (Mstar − Σ Mstar of direct progenitors) / time since the previous snapshot [h⁻¹ M☉ / yr]; undefined at the first snapshot |
| `sSFR` | SFR / Mstar [1/yr] |
| `Z_gas`, `Z_O` | MZ / Mgas, MZ_O / Mgas |
| `O_Fe` | MZ_O / MZ_Fe |
| `f_gas` | Mgas / (Mgas + Mstar) |
| `f_star`, `f_baryon` | Mstar / Mvir, (Mgas + Mstar) / Mvir |
| `dust_to_gas`, `dust_to_metal` | Mdust / Mgas, Mdust / MZ |
| `f_ej` | Mgas_ej / Mgas_ini |

Ratios with a zero denominator are left empty. "Keep axes fixed" uses the exact
whole-file range for stored columns and the range of the random sample for derived
ones.

New derived quantities can be added with one line each in the `DERIVED` table in
`data.py`.

## Cosmology

The default cosmology is VSMDPL (h = 0.6777, Ωm = 0.307115, ΩΛ = 0.692885). To use a
different one, pass a TOML file with a `[cosmology]` table:

```toml
[cosmology]
h = 0.6777
Omega_m = 0.307115
Omega_L = 0.692885
```

```bash
python app.py --dir path/to/treefiles --config my_cosmology.toml
```

## Project layout

| File | Contents |
| --- | --- |
| `app.py` | The Dash app (layout and callbacks); entry point. |
| `build_index.py` | Builds the `.viewidx.npz` index for a treefile. |
| `data.py` | Loading trees, derived quantities, cosmology. |
| `figures.py` | Plotly figures. |
| `astraeus_io.py` | Plain readers for ASTRAEUS output (`read_treefile`, `memmap_treefile`, `read_snapshot`, ...), usable on their own, e.g. in Jupyter notebooks. |
