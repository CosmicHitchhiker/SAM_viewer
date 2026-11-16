"""Plotly figure builders for the viewer."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import OutputFile, TreeData

MAX_SCATTER_POINTS = 100_000
TIME_LABELS = {"snapnumber": "snapshot", "scalefactor": "scale factor a", "redshift": "redshift z",
               "cosmic_time": "cosmic time [Gyr]", "lookback_time": "lookback time [Gyr]"}
REVERSED_TIME = ("redshift", "lookback_time")  # early times on the left/bottom


def _edges(x, y, desc, mask):
    """Line segments child -> descendant for galaxies selected by ``mask``."""

    idx = np.flatnonzero(mask & (desc >= 0))
    idx = idx[mask[desc[idx]]]
    nan = np.full(len(idx), np.nan)
    ex = np.column_stack([x[idx], x[desc[idx]], nan]).ravel()
    ey = np.column_stack([y[idx], y[desc[idx]], nan]).ravel()
    return ex, ey


def _values(tree: TreeData, col: str, log: bool) -> np.ndarray:
    v = tree.df[col].to_numpy(dtype=float)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(v > 0, np.log10(v), np.nan)
    return v


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def tree_figure(out: OutputFile, tree: TreeData, *, orient="h", time_col="snapnumber",
                mode="cumulative", step=0, color_by=None, log_color=True, size_by=None,
                mass_cut=None, selected=None) -> go.Figure:
    df = tree.df
    t = df[time_col].to_numpy(dtype=float)
    br = tree.y
    horizontal = orient == "h"
    x, y = (t, br) if horizontal else (br, t)

    visible = np.ones(len(df), dtype=bool)
    if mass_cut is not None:
        visible &= tree.mass >= mass_cut
    steps = df["_step"].to_numpy()
    if mode == "cumulative":
        visible &= steps <= step
    visible &= np.isfinite(t)

    fig = go.Figure()
    # invisible corner points pin the axis ranges so the view does not jump while scrolling time
    finite_t = t[np.isfinite(t)]
    if len(finite_t):
        corners_t = [finite_t.min(), finite_t.max()]
        corners_b = [br.min(), br.max()]
        cx, cy = ((corners_t, corners_b) if horizontal else (corners_b, corners_t))
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", opacity=0, hoverinfo="skip",
                                 showlegend=False))

    ex, ey = _edges(x, y, tree.desc, visible)
    fig.add_trace(go.Scattergl(x=ex, y=ey, mode="lines", line=dict(color="rgba(120,120,120,0.55)", width=1),
                               hoverinfo="skip", showlegend=False, name="links"))

    if selected is not None and visible[selected]:
        lineage = np.zeros(len(df), dtype=bool)
        lineage[[selected] + tree.descendants(selected) + tree.progenitors(selected, main_only=True)] = True
        lx, ly = _edges(x, y, tree.desc, lineage & visible)
        fig.add_trace(go.Scattergl(x=lx, y=ly, mode="lines", line=dict(color="#e8710a", width=2.5),
                                   hoverinfo="skip", showlegend=False, name="lineage"))

    cvals = _values(tree, color_by, log_color) if color_by else np.zeros(len(df))
    sel = np.flatnonzero(visible)
    if size_by:
        s = _values(tree, size_by, False)[sel]
        lo, hi = np.nanmin(s) if len(s) else 0, np.nanmax(s) if len(s) else 1
        sizes = 3 + 9 * (s - lo) / (hi - lo) if hi > lo else np.full(len(sel), 7.0)
        sizes = np.nan_to_num(sizes, nan=5.0)
    else:
        sizes = 7 if len(sel) < 3000 else 4

    custom = np.column_stack([
        df["localID"].to_numpy()[sel], df[time_col].to_numpy()[sel], cvals[sel],
    ]).tolist()  # plain lists: numpy 2-D arrays reach click events as {"0": .., "1": ..} objects
    color_label = (f"log10 {color_by}" if log_color else color_by) if color_by else ""
    opacity = 1.0
    if mode == "highlight":
        opacity = np.where(steps[sel] == step, 1.0, 0.35)
    fig.add_trace(go.Scattergl(
        x=x[sel], y=y[sel], mode="markers", customdata=custom, showlegend=False, name="galaxies",
        marker=dict(size=sizes, color=cvals[sel] if color_by else "#1f77b4", colorscale="Viridis",
                    opacity=opacity, showscale=bool(color_by),
                    colorbar=dict(title=color_label, thickness=12),
                    cmin=np.nanmin(cvals) if color_by and np.isfinite(cvals).any() else None,
                    cmax=np.nanmax(cvals) if color_by and np.isfinite(cvals).any() else None),
        hovertemplate=("localID %{customdata[0]:d}<br>" + TIME_LABELS.get(time_col, time_col)
                       + " %{customdata[1]:.4g}<br>" + (color_label + " %{customdata[2]:.4g}" if color_by else "")
                       + "<extra></extra>"),
    ))

    if mode == "highlight":
        now = np.flatnonzero(visible & (steps == step))
        if len(now):
            fig.add_trace(go.Scattergl(x=x[now], y=y[now], mode="markers", hoverinfo="skip", showlegend=False,
                                       marker=dict(size=14, symbol="circle-open", color="red",
                                                   line=dict(width=2))))
    if selected is not None and visible[selected]:
        fig.add_trace(go.Scatter(x=[x[selected]], y=[y[selected]], mode="markers", hoverinfo="skip",
                                 showlegend=False,
                                 marker=dict(size=17, symbol="circle-open", color="black", line=dict(width=3))))

    time_axis = dict(title=TIME_LABELS.get(time_col, time_col), autorange="reversed" if time_col in REVERSED_TIME else True)
    branch_axis = dict(visible=False)
    fig.update_layout(
        xaxis=time_axis if horizontal else branch_axis,
        yaxis=branch_axis if horizontal else time_axis,
        margin=dict(l=40, r=10, t=10, b=40), uirevision=f"{tree.df['treeID'].iloc[0]}-{orient}-{time_col}",
        plot_bgcolor="white", clickmode="event", hovermode="closest",
    )
    return fig


def _lineage(tree: TreeData, sel: int, scope: str):
    prog = tree.progenitors(sel, main_only=(scope == "main"))
    desc = tree.descendants(sel)
    role = np.full(len(tree.df), "", dtype=object)
    role[prog] = "progenitor"
    role[desc] = "descendant"
    role[sel] = "selected"
    return np.flatnonzero(role != ""), role


def evolution_figure(tree: TreeData, sel: int, params: list[str], time_col: str, scope: str,
                     log_y: bool) -> go.Figure:
    if not params:
        return _empty("Pick at least one parameter")
    members, role = _lineage(tree, sel, scope)
    inlin = np.zeros(len(tree.df), dtype=bool)
    inlin[members] = True
    t = tree.df[time_col].to_numpy(dtype=float)

    fig = make_subplots(rows=len(params), cols=1, shared_xaxes=True, vertical_spacing=0.04)
    colors = {"progenitor": "#1f77b4", "descendant": "#d62728", "selected": "black"}
    for r, p in enumerate(params, start=1):
        v = tree.df[p].to_numpy(dtype=float)
        ex, ey = _edges(t, v, tree.desc, inlin)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(120,120,120,0.6)", width=1),
                                 hoverinfo="skip", showlegend=False), row=r, col=1)
        for name in ("progenitor", "descendant", "selected"):
            idx = np.flatnonzero(role == name)
            if not len(idx):
                continue
            fig.add_trace(go.Scatter(
                x=t[idx], y=v[idx], mode="markers", name=name, legendgroup=name, showlegend=(r == 1),
                marker=dict(color=colors[name], size=11 if name == "selected" else 6,
                            symbol="star" if name == "selected" else "circle"),
                customdata=tree.df["localID"].to_numpy()[idx].tolist(),
                hovertemplate="localID %{customdata}<br>" + f"{p}" + "=%{y:.4g}<extra></extra>"),
                row=r, col=1)
        fig.update_yaxes(title_text=p, type="log" if log_y else "linear", row=r, col=1)
    fig.update_xaxes(title_text=TIME_LABELS.get(time_col, time_col), row=len(params), col=1)
    if time_col in REVERSED_TIME:
        fig.update_xaxes(autorange="reversed")
    fig.update_layout(height=max(380, 260 * len(params)), margin=dict(l=60, r=10, t=10, b=40),
                      legend=dict(orientation="h", y=1.04))
    return fig


def relation_figure(out: OutputFile, tree: TreeData | None, sel: int | None, x: str, y: str,
                    color: str | None, scope: str, log_x: bool, log_y: bool) -> go.Figure:
    if not x or not y:
        return _empty("Pick X and Y parameters")
    if scope == "all":
        df = out.df
    elif scope == "tree":
        df = tree.df
    else:  # lineage
        members, _ = _lineage(tree, sel, "all")
        df = tree.df.iloc[members]

    note = ""
    if len(df) > MAX_SCATTER_POINTS:
        note = f"showing a random {MAX_SCATTER_POINTS:,} of {len(df):,} points"
        df = df.sample(MAX_SCATTER_POINTS, random_state=0)

    marker = dict(size=5 if len(df) < 5000 else 3, opacity=0.8)
    if color:
        marker.update(color=df[color], colorscale="Viridis", showscale=True,
                      colorbar=dict(title=color, thickness=12))
    fig = go.Figure(go.Scattergl(
        x=df[x], y=df[y], mode="markers", marker=marker,
        customdata=np.column_stack([df["treeID"], df["localID"]]).tolist(),
        hovertemplate=f"tree %{{customdata[0]}} / ID %{{customdata[1]}}<br>{x}=%{{x:.4g}}<br>{y}=%{{y:.4g}}<extra></extra>"))
    fig.update_layout(
        xaxis=dict(title=x, type="log" if log_x else "linear"),
        yaxis=dict(title=y, type="log" if log_y else "linear"),
        margin=dict(l=60, r=10, t=30, b=50), title=dict(text=note, font=dict(size=12)),
        height=620)
    return fig


def _axis_range(values: np.ndarray, log: bool):
    """Padded axis range covering ``values`` (in log10 units for log axes)."""

    v = values[np.isfinite(values)]
    if log:
        v = v[v > 0]
        if not len(v):
            return None
        lo, hi = np.log10(v.min()), np.log10(v.max())
    else:
        if not len(v):
            return None
        lo, hi = v.min(), v.max()
    pad = 0.05 * (hi - lo) if hi > lo else 0.5
    return [lo - pad, hi + pad]


def _binned_median(x: np.ndarray, y: np.ndarray, log_x: bool, n_bins: int = 20, min_count: int = 3):
    ok = np.isfinite(x) & np.isfinite(y) & ((x > 0) if log_x else True)
    x, y = x[ok], y[ok]
    if len(x) < min_count:
        return np.array([]), np.array([])
    edges = np.geomspace(x.min(), x.max(), n_bins + 1) if log_x and x.max() > x.min() else np.linspace(
        x.min(), x.max() + 1e-12, n_bins + 1)
    which = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    cx, cy = [], []
    for b in range(n_bins):
        sel = which == b
        if sel.sum() >= min_count:
            cx.append(np.median(x[sel]))
            cy.append(np.median(y[sel]))
    return np.array(cx), np.array(cy)


def snapshot_figure(out: OutputFile, tree: TreeData | None, step: int, x: str, y: str,
                    color: str | None, scope: str, log_x: bool, log_y: bool,
                    fix_axes: bool, median: bool) -> tuple[go.Figure, int]:
    """Scatter of ``y`` vs ``x`` for all galaxies of one snapshot. Returns (figure, galaxy count)."""

    if not x or not y:
        return _empty("Pick X and Y parameters"), 0
    pool = out.df if scope == "all" else tree.df
    df = pool[pool["_step"] == step]
    n_total = len(df)
    if n_total == 0:
        return _empty("No galaxies at this snapshot"), 0

    if n_total > MAX_SCATTER_POINTS:  # the caller reports the full count; the plot shows a random subset
        df = df.sample(MAX_SCATTER_POINTS, random_state=0)

    marker = dict(size=6 if len(df) < 3000 else 4, opacity=0.75)
    if color:
        marker.update(color=df[color], colorscale="Viridis", showscale=True,
                      colorbar=dict(title=color, thickness=12))
        if fix_axes:  # keep the colour scale fixed too while scrolling through time
            vals = pool[color].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                marker.update(cmin=float(vals.min()), cmax=float(vals.max()))
    fig = go.Figure(go.Scattergl(
        x=df[x], y=df[y], mode="markers", marker=marker, name="galaxies", showlegend=False,
        customdata=np.column_stack([df["treeID"], df["localID"]]).tolist(),
        hovertemplate=f"tree %{{customdata[0]}} / ID %{{customdata[1]}}<br>{x}=%{{x:.4g}}<br>{y}=%{{y:.4g}}<extra></extra>"))

    if median:
        mx, my = _binned_median(df[x].to_numpy(dtype=float), df[y].to_numpy(dtype=float), log_x)
        if len(mx):
            fig.add_trace(go.Scatter(x=mx, y=my, mode="lines+markers", name="binned median",
                                     line=dict(color="#d62728", width=2), marker=dict(size=6),
                                     hoverinfo="skip"))

    xaxis = dict(title=x, type="log" if log_x else "linear")
    yaxis = dict(title=y, type="log" if log_y else "linear")
    if fix_axes:
        rx = _axis_range(pool[x].to_numpy(dtype=float), log_x)
        ry = _axis_range(pool[y].to_numpy(dtype=float), log_y)
        if rx:
            xaxis["range"] = rx
        if ry:
            yaxis["range"] = ry
    fig.update_layout(xaxis=xaxis, yaxis=yaxis, margin=dict(l=60, r=10, t=30, b=50), height=620,
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.04),
                      uirevision=f"snap-{x}-{y}-{scope}-{log_x}-{log_y}-{fix_axes}")
    return fig, n_total
