# ASTRAEUS tree viewer

A [Dash](https://dash.plotly.com) web app for exploring the galaxy merger trees
(`tree_*.dat`) written by ASTRAEUS.

```bash
pip install -r requirements.txt
python build_index.py tree_0.dat        # once per treefile, see below
python app.py --dir . --port 8050       # then open http://127.0.0.1:8050
```

`--dir` is scanned for `tree_*.dat` files; use *Refresh file list* after adding one.

## The treefile is never modified

A treefile is several GB (millions of galaxies), so it is **memory-mapped read-only**
and never loaded as a whole: a tree is read when you select it, whole-file plots use
a random sample of galaxies.

Everything that would need a scan of the whole file is prepared once by
`build_index.py`. It only *reads* the treefile and writes one new file next to it,
`tree_0.dat.viewidx.npz` (~110 MB for 18 M galaxies, ~10–30 s), containing the
snapshot number of every galaxy, the largest `Mvir` of every tree, the min/max of
every column, the scale factor of every snapshot and `Mstar − Σ Mstar(progenitors)`
of every galaxy. The app itself writes nothing. If the index is missing, or the
treefile's size or modification time changed since it was built, the app says so and
asks you to run the script.

The layout variant (with/without metals, with/without rockstar IDs) is detected from
the record size.

## What it does

| Tab | Features |
| --- | --- |
| **Merger tree** | Tree drawn as a graph (nodes = galaxies, edges = progenitor → descendant). Time on the x or y axis. Colour/size nodes by any parameter. **Time scroller**: slider + play button, either *growing tree* or *highlight snapshot*. Click a node to select it; its descendants and main progenitor branch are highlighted. |
| **Galaxy** | All parameters of the selected galaxy, with buttons to jump to its direct progenitors / descendant. |
| **Evolution** | Any number of parameters vs. time for the selected galaxy, its progenitors (or only the main branch) and its descendants. |
| **Parameter vs parameter** | Scatter of any parameter against any other, optionally coloured by a third; for the selected lineage, the whole tree, or the whole file (random 100,000 galaxies). Clicking a point of another tree switches to that tree. |
| **Snapshot population** | All galaxies of one snapshot (random 100,000 if there are more), for the whole file or the selected tree. Options: log axes, axes and colour scale fixed while scrolling through time, binned-median line. Clicking a point selects that galaxy, switching to its tree if needed. |

**Choosing a tree:** a file can hold a million trees, so the dropdown lists only the
500 with the most galaxies. Any other tree is reached with *Go to treeID* or by
clicking a galaxy in a whole-file scatter plot.

## Parameters

All fields stored in the treefile are plottable, **with the values exactly as stored**
(masses in h⁻¹ M☉ etc. — nothing is rescaled). `snapnum` and `aexp` are shown as
`snapnumber` and `scalefactor`. Derived on the fly:

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

Ratios with a zero denominator are left empty. To add a quantity, add a line to
`DERIVED` in `data.py`.

The cosmology defaults to VSMDPL (h = 0.6777, Ωm = 0.307115, ΩΛ = 0.692885); pass
`--config file.toml` with a `[cosmology]` table (`Omega_m`, `Omega_L`, `h`) to change it.

## Notes

* **Large trees:** the *hide galaxies below log10 mass* slider thins the tree (it
  defaults to the 20,000 most massive galaxies).
* "Keep axes fixed" uses the exact whole-file range for stored columns and the range
  of the random sample for derived ones.
* `astraeus_io.py` holds the plain readers (`read_treefile`, ...) for use in notebooks.
