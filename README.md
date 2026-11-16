# simpleSAM output viewer

A [Dash](https://dash.plotly.com) web app for exploring the `*.json` files that
`scripts/run_simulation.py` writes to `outputs/`.

```bash
pip install -r viewer/requirements.txt
python viewer/app.py --dir outputs --port 8050     # then open http://127.0.0.1:8050
```

To produce an output file from a merger-tree file:

```bash
python scripts/run_simulation.py trees/VSMDPL/<run>/tree_localhorizontal_0.dat -o outputs/galaxies
```

`--dir` is scanned for `*.json` files; use *Refresh file list* after a new run.
`--config` (default `config/fiducial.toml`) supplies the cosmology used for cosmic time.

## What it does

| Tab | Features |
| --- | --- |
| **Merger tree** | Tree drawn as a graph (nodes = galaxies, edges = progenitor → descendant). Time on the x or y axis; time axis can be snapshot, scale factor or redshift. Colour/size nodes by any parameter. **Time scroller**: slider + play button, either *growing tree* (only galaxies that exist up to the chosen snapshot) or *highlight snapshot*. Click a node to select it; the selected galaxy's descendants and main progenitor branch are highlighted. |
| **Galaxy** | All parameters of the selected galaxy, with buttons to jump to its direct progenitors / descendant. |
| **Evolution** | Any number of parameters vs. time for the selected galaxy, all its progenitors (or only the main branch) and its descendants, drawn along the tree links. |
| **Parameter vs parameter** | Scatter plot of any parameter against any other, optionally coloured by a third; for the selected lineage, the whole tree, or the whole file. Log axes available. |
| **Snapshot population** | Many galaxies at one moment in time: pick a snapshot with the time scroller (label shows its redshift and cosmic time, play button available) and scatter any parameter against any other, optionally coloured by a third, for the whole file or the selected tree. Options: log axes, keep axes (and colour scale) fixed while scrolling through time, and a binned-median trend line. Clicking a point selects that galaxy, switching to its tree if needed. |

## Time axes

Every time axis dropdown (merger tree, evolution) can be switched between
snapshot, scale factor, redshift, cosmic time [Gyr] and lookback time [Gyr].
Cosmic time is the age of the universe computed from the scale factor for a flat
ΛCDM cosmology (`Omega_m`, `Omega_L`, `h` from the `[cosmology]` table of the
config file; radiation neglected).

## Notes

* **New parameters need no viewer changes.** Every numeric field in the JSON
  becomes selectable; list fields are split (`pos` → `pos_x, pos_y, pos_z`,
  other lengths → `name_0, name_1, ...`). A `redshift` column is derived from
  `scalefactor`.
* Required fields: `treeID`, `localID`, `localDescID` and `snapnumber` or
  `scalefactor`. The mass used to order branches and for the mass cut is `Mvir`
  (falling back to the first numeric parameter).
* **Large trees:** the *hide galaxies below log10 mass* slider thins the tree
  (it defaults to the 20,000 most massive galaxies). Scatter plots are capped at
  100,000 randomly chosen points. The whole JSON file is read into memory once
  and cached; trees are laid out lazily when selected.
