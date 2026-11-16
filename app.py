"""Dash web app for exploring simpleSAM output files.

Run from the repository root:

    python viewer/app.py --dir outputs --port 8050
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import get_output, list_outputs, load_cosmology, set_cosmology  # noqa: E402
from figures import evolution_figure, relation_figure, snapshot_figure, tree_figure  # noqa: E402

MAX_TREE_NODES = 20_000  # default mass cut keeps at most this many galaxies on screen
TIME_LABELS = {"snapnumber": "Snapshot", "scalefactor": "Scale factor", "redshift": "Redshift",
               "cosmic_time": "Cosmic time [Gyr]", "lookback_time": "Lookback time [Gyr]"}


def opts(values):
    return [{"label": v, "value": v} for v in values]


def custom_item(point: dict, i: int = 0):
    """``customdata[i]`` of a clicked point, whether it arrives as a list or an {"i": ..} dict."""

    cd = point.get("customdata")
    if cd is None:
        return None
    if isinstance(cd, dict):
        return cd.get(str(i))
    return cd[i] if isinstance(cd, (list, tuple)) and len(cd) > i else (cd if i == 0 else None)


def step_index(out, value) -> int:
    """Index into ``out.steps`` for a slider value (a snapshot number, or an index without snapnumber)."""

    n = len(out.steps)
    if "snapnumber" not in out.df:
        return int(min(max(value, 0), n - 1))
    return int(max(np.searchsorted(out.steps, value, side="right") - 1, 0))


def build_app(directory: Path) -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True, title="simpleSAM viewer")

    def labelled(text, component):
        return html.Div([html.Label(text, className="lbl"), component], className="field")

    sidebar = html.Div([
        html.H3("simpleSAM viewer"),
        labelled("Output file", html.Div([
            dcc.Dropdown(id="file", clearable=False),
            html.Button("Refresh file list", id="refresh", n_clicks=0, className="small"),
        ])),
        labelled("Merger tree", dcc.Dropdown(id="tree", clearable=False)),
        html.Hr(),
        html.H4("Selected galaxy"),
        html.Div(id="selection-label"),
        labelled("Jump to localID", dcc.Input(id="jump-id", type="number", debounce=True, step=1,
                                              style={"width": "100%"})),
        html.Div("Click a galaxy in the tree to select it.", className="hint"),
        dcc.Store(id="selected"),
        html.Div(id="error", className="error"),
    ], className="sidebar")

    tree_tab = html.Div([
        html.Div([
            labelled("Layout", dcc.RadioItems(id="orient", value="h", inline=True, options=[
                {"label": "Time on x", "value": "h"}, {"label": "Time on y", "value": "v"}])),
            labelled("Time axis", dcc.Dropdown(id="time-col", clearable=False)),
            labelled("Time scroller", dcc.RadioItems(id="tmode", value="cumulative", inline=True, options=[
                {"label": "Growing tree", "value": "cumulative"},
                {"label": "Highlight snapshot", "value": "highlight"}])),
            labelled("Colour by", dcc.Dropdown(id="color-by")),
            labelled("Size by", dcc.Dropdown(id="size-by")),
            labelled("", dcc.Checklist(id="log-color", value=["log"], options=[
                {"label": "log colour", "value": "log"}])),
        ], className="controls"),
        html.Div([
            html.Button("▶ Play", id="play", n_clicks=0),
            dcc.Slider(id="step", min=0, max=1, step=1, value=1, updatemode="drag",
                       tooltip={"always_visible": False}),
            dcc.Dropdown(id="speed", clearable=False, value=300, style={"width": "110px"}, options=[
                {"label": "slow", "value": 800}, {"label": "medium", "value": 300},
                {"label": "fast", "value": 100}]),
        ], className="scroller"),
        html.Div(id="step-label", className="hint"),
        html.Div([html.Label("Hide galaxies below log10 mass (declutters big trees)", className="lbl"),
                  dcc.Slider(id="mass-cut", min=0, max=1, step=0.05, value=0, updatemode="mouseup")],
                 className="field"),
        dcc.Graph(id="tree-graph", style={"height": "68vh"}, config={"scrollZoom": True}),
        dcc.Interval(id="tick", interval=300, disabled=True),
    ])

    galaxy_tab = html.Div(id="galaxy-panel")

    evolution_tab = html.Div([
        html.Div([
            labelled("Parameters", dcc.Dropdown(id="evo-params", multi=True)),
            labelled("Include", dcc.RadioItems(id="evo-scope", value="all", inline=True, options=[
                {"label": "All progenitors + descendants", "value": "all"},
                {"label": "Main branch + descendants", "value": "main"}])),
            labelled("x axis", dcc.Dropdown(id="evo-time", clearable=False)),
            labelled("", dcc.Checklist(id="evo-log", value=[], options=[{"label": "log y", "value": "log"}])),
        ], className="controls"),
        dcc.Graph(id="evo-graph"),
    ])

    relation_tab = html.Div([
        html.Div([
            labelled("X", dcc.Dropdown(id="rel-x", clearable=False)),
            labelled("Y", dcc.Dropdown(id="rel-y", clearable=False)),
            labelled("Colour", dcc.Dropdown(id="rel-color")),
            labelled("Galaxies", dcc.RadioItems(id="rel-scope", value="lineage", inline=True, options=[
                {"label": "Selected lineage", "value": "lineage"},
                {"label": "This tree", "value": "tree"},
                {"label": "Whole file", "value": "all"}])),
            labelled("", dcc.Checklist(id="rel-log", value=[], options=[
                {"label": "log x", "value": "x"}, {"label": "log y", "value": "y"}], inline=True)),
        ], className="controls"),
        dcc.Graph(id="rel-graph"),
    ])

    snapshot_tab = html.Div([
        html.Div([
            labelled("X", dcc.Dropdown(id="snap-x", clearable=False)),
            labelled("Y", dcc.Dropdown(id="snap-y", clearable=False)),
            labelled("Colour", dcc.Dropdown(id="snap-color")),
            labelled("Galaxies", dcc.RadioItems(id="snap-scope", value="all", inline=True, options=[
                {"label": "Whole file", "value": "all"}, {"label": "This tree", "value": "tree"}])),
            labelled("", dcc.Checklist(id="snap-opts", value=["fix"], options=[
                {"label": "log x", "value": "logx"}, {"label": "log y", "value": "logy"},
                {"label": "keep axes fixed over time", "value": "fix"},
                {"label": "binned median", "value": "median"}])),
        ], className="controls"),
        html.Div([
            html.Button("▶ Play", id="snap-play", n_clicks=0),
            dcc.Slider(id="snap-step", min=0, max=1, step=1, value=1, updatemode="drag"),
            dcc.Dropdown(id="snap-speed", clearable=False, value=500, style={"width": "110px"}, options=[
                {"label": "slow", "value": 1200}, {"label": "medium", "value": 500},
                {"label": "fast", "value": 150}]),
        ], className="scroller"),
        html.Div(id="snap-label", className="hint"),
        dcc.Graph(id="snap-graph"),
        dcc.Interval(id="snap-tick", interval=500, disabled=True),
        dcc.Store(id="pending"),
    ])

    app.layout = html.Div([
        sidebar,
        html.Div([
            dcc.Tabs(id="tabs", value="tree-tab", children=[
                dcc.Tab(label="Merger tree", value="tree-tab"),
                dcc.Tab(label="Galaxy", value="galaxy-tab"),
                dcc.Tab(label="Evolution", value="evo-tab"),
                dcc.Tab(label="Parameter vs parameter", value="rel-tab"),
                dcc.Tab(label="Snapshot population", value="snap-tab"),
            ]),
            # all panes stay mounted so controls keep their state while switching tabs
            html.Div(tree_tab, id="pane-tree-tab"),
            html.Div(galaxy_tab, id="pane-galaxy-tab", style={"display": "none"}),
            html.Div(evolution_tab, id="pane-evo-tab", style={"display": "none"}),
            html.Div(relation_tab, id="pane-rel-tab", style={"display": "none"}),
            html.Div(snapshot_tab, id="pane-snap-tab", style={"display": "none"}),
            html.Div(id="resize-dummy", style={"display": "none"}),
        ], className="main"),
    ], className="page")

    panes = ["tree-tab", "galaxy-tab", "evo-tab", "rel-tab", "snap-tab"]

    @app.callback([Output(f"pane-{p}", "style") for p in panes], Input("tabs", "value"))
    def show_pane(tab):
        return [{} if p == tab else {"display": "none"} for p in panes]

    # plots drawn while hidden have no size; make plotly re-measure once a pane is shown
    app.clientside_callback(
        "function(v){setTimeout(function(){window.dispatchEvent(new Event('resize'));}, 80);"
        "return window.dash_clientside.no_update;}",
        Output("resize-dummy", "children"), Input("tabs", "value"))

    # ---------------------------------------------------------------- files
    @app.callback(Output("file", "options"), Output("file", "value"),
                  Input("refresh", "n_clicks"), State("file", "value"))
    def file_list(_, current):
        files = [str(p) for p in list_outputs(directory)]
        options = [{"label": Path(p).name, "value": p} for p in files]
        value = current if current in files else (files[0] if files else None)
        return options, value

    @app.callback(
        Output("tree", "options"), Output("tree", "value"), Output("time-col", "options"),
        Output("time-col", "value"), Output("color-by", "options"), Output("color-by", "value"),
        Output("size-by", "options"), Output("evo-params", "options"), Output("evo-params", "value"),
        Output("evo-time", "options"), Output("evo-time", "value"), Output("rel-x", "options"),
        Output("rel-x", "value"), Output("rel-y", "options"), Output("rel-y", "value"),
        Output("rel-color", "options"), Output("error", "children"),
        Input("file", "value"))
    def on_file(path):
        if not path:
            msg = f"No *.json files found in {directory}"
            return ([], None, [], None, [], None, [], [], None, [], None, [], None, [], None, [], msg)
        try:
            out = get_output(path)
        except Exception as exc:  # show the problem in the UI instead of a blank page
            return ([], None, [], None, [], None, [], [], None, [], None, [], None, [], None, [], f"Cannot read {path}: {exc}")

        summary = out.tree_summary().sort_values("n", ascending=False)
        tree_opts = [{"label": f"tree {r.treeID} — {r.n} galaxies, max {out.mass_col} {r.mass:.2e}",
                      "value": int(r.treeID)} for r in summary.itertuples()]
        time_opts = [{"label": TIME_LABELS[c], "value": c} for c in out.time_cols]
        params = out.params
        first = out.time_cols[0]
        px = "Mstar" if "Mstar" in params else params[0]
        py = out.mass_col
        evo_default = [p for p in (out.mass_col, "Mgas", "Mstar") if p in params][:2]
        color_default = out.mass_col if out.mass_col in params else None
        return (tree_opts, tree_opts[0]["value"], time_opts, first, opts(params), color_default,
                opts(params), opts(params), evo_default, time_opts, first, opts(params), px,
                opts(params), py, opts(params), "")

    # ---------------------------------------------------------------- tree selection
    def slider_props(path):
        """(min, max, value, marks) of a time slider over the file's snapshot numbers.

        Without a snapnumber column the slider runs over step indices, labelled with the scale factor.
        """

        if not path:
            return 0, 1, 1, {}
        out = get_output(path)
        steps = out.steps
        if "snapnumber" in out.df:
            lo, hi = int(steps[0]), int(steps[-1])
            every = max(1, int(np.ceil((hi - lo) / 10)))
            marks = {v: str(v) for v in range(lo, hi + 1, every)}
            marks[hi] = str(hi)
            return lo, hi, hi, marks
        n = len(steps)
        every = max(1, n // 8)
        marks = {i: f"{steps[i]:.3g}" for i in range(0, n, every)}
        marks[n - 1] = f"{steps[-1]:.3g}"
        return 0, n - 1, n - 1, marks

    @app.callback(
        Output("step", "min"), Output("step", "max"), Output("step", "value"), Output("step", "marks"),
        Input("file", "value"))
    def slider_range(path):
        return slider_props(path)

    @app.callback(
        Output("mass-cut", "min"), Output("mass-cut", "max"),
        Output("mass-cut", "value"), Output("mass-cut", "marks"),
        Input("tree", "value"), State("file", "value"))
    def on_tree(tree_id, path):
        if tree_id is None or not path:
            return 0, 1, 0, {}
        out = get_output(path)
        tree = out.tree(tree_id)
        positive = tree.mass[tree.mass > 0]
        lo = float(np.log10(positive.min())) if len(positive) else 0.0
        hi = float(np.log10(positive.max())) if len(positive) else 1.0
        lo, hi = np.floor(lo * 10) / 10, np.ceil(hi * 10) / 10
        if len(positive) > MAX_TREE_NODES:
            default = float(np.log10(np.sort(positive)[-MAX_TREE_NODES]))
        else:
            default = lo
        mark_cut = {round(lo, 1): f"{lo:.1f}", round(hi, 1): f"{hi:.1f}"}
        return lo, hi, round(default, 2), mark_cut

    @app.callback(Output("selected", "data"),
                  Input("tree", "value"), Input("tree-graph", "clickData"), Input("jump-id", "value"),
                  Input({"type": "jump", "local": ALL}, "n_clicks"), Input("rel-graph", "clickData"),
                  Input("evo-graph", "clickData"), Input("snap-graph", "clickData"),
                  State("file", "value"), State("selected", "data"), State("pending", "data"),
                  prevent_initial_call=False)
    def select(tree_id, click, jump, _jumps, rel_click, evo_click, snap_click, path, current, pending):
        if tree_id is None or not path:
            return None
        out = get_output(path)
        tree = out.tree(tree_id)
        trig = ctx.triggered_id

        def pack(row):
            return {"tree": int(tree_id), "local": int(tree.df["localID"].iloc[row])}

        if trig == "tree-graph" and click:
            local = custom_item(click["points"][0], 0)
            if local is not None and tree.row(int(local)) >= 0:
                return pack(tree.row(int(local)))
        elif trig == "evo-graph" and evo_click:
            local = custom_item(evo_click["points"][0], 0)
            if local is not None and tree.row(int(local)) >= 0:
                return pack(tree.row(int(local)))
        elif trig in ("rel-graph", "snap-graph") and (rel_click if trig == "rel-graph" else snap_click):
            point = (rel_click if trig == "rel-graph" else snap_click)["points"][0]
            tid, local = custom_item(point, 0), custom_item(point, 1)
            if local is not None and int(tid) == int(tree_id) and tree.row(int(local)) >= 0:
                return pack(tree.row(int(local)))
        elif trig == "jump-id" and jump is not None and tree.row(int(jump)) >= 0:
            return pack(tree.row(int(jump)))
        elif isinstance(trig, dict) and trig.get("type") == "jump":
            row = tree.row(int(trig["local"]))
            if row >= 0 and any(n for n in _jumps if n):
                return pack(row)
        if trig == "tree" and pending and pending.get("tree") == tree_id and tree.row(int(pending["local"])) >= 0:
            return pack(tree.row(int(pending["local"])))  # arrived here by clicking a galaxy of another tree
        if trig == "tree" or current is None or current.get("tree") != tree_id:
            return pack(int(tree.roots[np.argmax(tree.mass[tree.roots])]))
        return no_update

    def selected_row(sel, tree_id, tree):
        if not sel or sel.get("tree") != tree_id:
            return None
        row = tree.row(sel["local"])
        return row if row >= 0 else None

    @app.callback(Output("selection-label", "children"),
                  Input("selected", "data"), State("file", "value"))
    def sel_label(sel, path):
        if not sel or not path:
            return "none"
        return f"tree {sel['tree']}, localID {sel['local']}"

    # ---------------------------------------------------------------- tree view
    @app.callback(
        Output("tree-graph", "figure"), Output("step-label", "children"),
        Input("file", "value"), Input("tree", "value"), Input("orient", "value"),
        Input("time-col", "value"), Input("tmode", "value"), Input("step", "value"),
        Input("color-by", "value"), Input("log-color", "value"), Input("size-by", "value"),
        Input("mass-cut", "value"), Input("selected", "data"))
    def draw_tree(path, tree_id, orient, time_col, tmode, step, color_by, log_color, size_by, cut, sel):
        if not path or tree_id is None or time_col is None or step is None:
            return no_update, no_update
        out = get_output(path)
        tree = out.tree(tree_id)
        if time_col not in tree.df:
            return no_update, no_update
        idx = step_index(out, step)
        fig = tree_figure(
            out, tree, orient=orient, time_col=time_col, mode=tmode, step=idx,
            color_by=color_by, log_color=bool(log_color), size_by=size_by,
            mass_cut=10 ** cut if cut else None, selected=selected_row(sel, tree_id, tree))
        key = "snapnumber" if "snapnumber" in out.df else "scalefactor"
        n_now = int((tree.df["_step"] <= idx).sum())
        extra = ""
        when = tree.df[tree.df["_step"] == idx]
        if len(when) == 0:
            when = out.df[out.df["_step"] == idx]
        if len(when) and "redshift" in when:
            extra = f" (z = {when['redshift'].iloc[0]:.3g}"
            if "cosmic_time" in when:
                extra += f", t = {when['cosmic_time'].iloc[0]:.3g} Gyr"
            extra += ")"
        label = f"{key} {out.steps[idx]:.4g}{extra} — {n_now} of {len(tree.df)} galaxies exist by now"
        return fig, label

    @app.callback(Output("tick", "disabled"), Output("tick", "interval"), Output("play", "children"),
                  Input("play", "n_clicks"), Input("speed", "value"), State("tick", "disabled"),
                  prevent_initial_call=True)
    def play(_, speed, disabled):
        if ctx.triggered_id == "play":
            disabled = not disabled
        return disabled, speed, "▶ Play" if disabled else "⏸ Pause"

    def next_step(path, value, vmin):
        """Next snapshot that exists in the file after ``value``, wrapping around at the end."""

        if not path or value is None:
            return no_update
        out = get_output(path)
        if "snapnumber" not in out.df:
            return vmin if value >= len(out.steps) - 1 else value + 1
        later = out.steps[out.steps > value]
        return int(later[0]) if len(later) else vmin

    @app.callback(Output("step", "value", allow_duplicate=True), Input("tick", "n_intervals"),
                  State("step", "value"), State("step", "min"), State("file", "value"),
                  prevent_initial_call=True)
    def advance(_, value, vmin, path):
        return next_step(path, value, vmin)

    # ---------------------------------------------------------------- snapshot population
    @app.callback(
        Output("snap-x", "options"), Output("snap-x", "value"), Output("snap-y", "options"),
        Output("snap-y", "value"), Output("snap-color", "options"), Output("snap-color", "value"),
        Output("snap-step", "min"), Output("snap-step", "max"), Output("snap-step", "value"),
        Output("snap-step", "marks"), Input("file", "value"))
    def snap_controls(path):
        if not path:
            return [], None, [], None, [], None, 0, 1, 1, {}
        out = get_output(path)
        params = out.params
        x = out.mass_col if out.mass_col in params else params[0]
        y = "Mstar" if "Mstar" in params else params[-1]
        return (opts(params), x, opts(params), y, opts(params), None, *slider_props(path))

    @app.callback(
        Output("snap-graph", "figure"), Output("snap-label", "children"),
        Input("file", "value"), Input("tree", "value"), Input("snap-step", "value"),
        Input("snap-x", "value"), Input("snap-y", "value"), Input("snap-color", "value"),
        Input("snap-scope", "value"), Input("snap-opts", "value"), Input("tabs", "value"))
    def draw_snapshot(path, tree_id, step, x, y, color, scope, flags, tab):
        if tab != "snap-tab" or not path or step is None or tree_id is None:
            return no_update, no_update
        out = get_output(path)
        idx = step_index(out, step)
        flags = flags or []
        fig, n = snapshot_figure(out, out.tree(tree_id), idx, x, y, color, scope, "logx" in flags,
                                 "logy" in flags, "fix" in flags, "median" in flags)
        key = "snapnumber" if "snapnumber" in out.df else "scalefactor"
        rows = out.df[out.df["_step"] == idx]
        label = f"{key} {out.steps[idx]:.4g}"
        if len(rows) and "redshift" in rows:
            label += f" (z = {rows['redshift'].iloc[0]:.3g}"
            if "cosmic_time" in rows:
                label += f", t = {rows['cosmic_time'].iloc[0]:.3g} Gyr"
            label += ")"
        return fig, f"{label} — {n:,} galaxies. Click a point to select that galaxy."

    @app.callback(Output("snap-tick", "disabled"), Output("snap-tick", "interval"), Output("snap-play", "children"),
                  Input("snap-play", "n_clicks"), Input("snap-speed", "value"), State("snap-tick", "disabled"),
                  prevent_initial_call=True)
    def snap_play(_, speed, disabled):
        if ctx.triggered_id == "snap-play":
            disabled = not disabled
        return disabled, speed, "▶ Play" if disabled else "⏸ Pause"

    @app.callback(Output("snap-step", "value", allow_duplicate=True), Input("snap-tick", "n_intervals"),
                  State("snap-step", "value"), State("snap-step", "min"), State("file", "value"),
                  prevent_initial_call=True)
    def snap_advance(_, value, vmin, path):
        return next_step(path, value, vmin)

    @app.callback(Output("tree", "value", allow_duplicate=True), Output("pending", "data"),
                  Input("snap-graph", "clickData"), State("tree", "value"), prevent_initial_call=True)
    def snap_click_other_tree(click, tree_id):
        """A point from another tree: switch to that tree, remembering which galaxy to select."""

        if not click:
            return no_update, no_update
        point = click["points"][0]
        tid, local = custom_item(point, 0), custom_item(point, 1)
        if tid is None or local is None or int(tid) == int(tree_id):
            return no_update, no_update
        return int(tid), {"tree": int(tid), "local": int(local)}

    # ---------------------------------------------------------------- galaxy panel
    @app.callback(Output("galaxy-panel", "children"),
                  Input("selected", "data"), Input("tabs", "value"), State("file", "value"))
    def galaxy_panel(sel, tab, path):
        if tab != "galaxy-tab" or not sel or not path:
            return no_update
        out = get_output(path)
        tree = out.tree(sel["tree"])
        row = selected_row(sel, sel["tree"], tree)
        if row is None:
            return "No galaxy selected."

        def jump_btn(r):
            local = int(tree.df["localID"].iloc[r])
            return html.Button(str(local), id={"type": "jump", "local": local}, n_clicks=0, className="small")

        rec = tree.df.iloc[row]
        rows = [html.Tr([html.Td(k), html.Td(f"{v:.6g}")]) for k, v in rec.items() if not k.startswith("_")]
        progs = [jump_btn(int(r)) for r in tree.children(row)]
        desc = tree.descendants(row)
        return html.Div([
            html.H4(f"Galaxy {sel['local']} (tree {sel['tree']})"),
            html.Div([html.B("Direct progenitors: "), *(progs or ["none (leaf)"])], className="links"),
            html.Div([html.B("Descendant: "), jump_btn(desc[0]) if desc else "none (root)"], className="links"),
            html.Div(f"{len(tree.progenitors(row))} progenitors in total, "
                     f"{len(desc)} galaxies downstream.", className="hint"),
            html.Table(rows, className="props"),
        ], className="galaxy")

    # ---------------------------------------------------------------- evolution
    @app.callback(Output("evo-graph", "figure"),
                  Input("selected", "data"), Input("evo-params", "value"), Input("evo-time", "value"),
                  Input("evo-scope", "value"), Input("evo-log", "value"), Input("tabs", "value"),
                  State("file", "value"))
    def evolution(sel, params, time_col, scope, log, tab, path):
        if tab != "evo-tab" or not sel or not path or not time_col:
            return no_update
        tree = get_output(path).tree(sel["tree"])
        row = selected_row(sel, sel["tree"], tree)
        if row is None:
            return no_update
        return evolution_figure(tree, row, params or [], time_col, scope, bool(log))

    # ---------------------------------------------------------------- relation
    @app.callback(Output("rel-graph", "figure"),
                  Input("selected", "data"), Input("rel-x", "value"), Input("rel-y", "value"),
                  Input("rel-color", "value"), Input("rel-scope", "value"), Input("rel-log", "value"),
                  Input("tabs", "value"), State("file", "value"))
    def relation(sel, x, y, color, scope, log, tab, path):
        if tab != "rel-tab" or not path:
            return no_update
        out = get_output(path)
        tree = row = None
        if sel:
            tree = out.tree(sel["tree"])
            row = selected_row(sel, sel["tree"], tree)
        if scope != "all" and row is None:
            return no_update
        return relation_figure(out, tree, row, x, y, color, scope, "x" in (log or []), "y" in (log or []))

    return app


CSS = """
body { margin: 0; font-family: system-ui, sans-serif; }
.page { display: flex; height: 100vh; }
.sidebar { width: 280px; min-width: 280px; padding: 12px; background: #f4f5f7; overflow-y: auto; }
.main { flex: 1; padding: 8px 12px; overflow-y: auto; }
.field { margin-bottom: 8px; }
.lbl { font-size: 12px; font-weight: 600; color: #444; }
.controls { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 4px 14px; padding: 8px 0; }
.scroller { display: flex; align-items: center; gap: 12px; }
.scroller > div:nth-child(2) { flex: 1; }
.hint { font-size: 12px; color: #666; margin: 4px 0; }
.error { color: #b00020; font-size: 13px; }
.small { font-size: 11px; padding: 1px 6px; margin: 2px; }
.props td { padding: 2px 14px 2px 0; font-family: monospace; font-size: 13px; }
.props td:first-child { color: #555; }
.links { margin: 6px 0; }
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=Path("outputs"), help="directory with *.json outputs")
    parser.add_argument("--config", type=Path, default=Path("config/fiducial.toml"),
                        help="simulation parameter file; its [cosmology] table converts scale factor to cosmic time")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    set_cosmology(load_cosmology(args.config))
    app = build_app(args.dir)
    app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
