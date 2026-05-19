"""
Behavioral Investment Intelligence — Özet Dashboard
Alım · Satım · İşlem · Bakiye  |  Dinamik eksenler, heatmap-öncelikli görsel sistem
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics_engine import (
    DONEM_ARALIK,
    DONEM_SIRASI,
    SEGMENT_SIRASI,
    SEG_RENK,
    URUNLER,
    URUN_RENKLER,
    build_composite_net,
    build_analysis_bundle,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yatırım Analitik Özeti",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
C_BLUE   = "#0F4C81"
C_RED    = "#8B1C1C"
C_GREEN  = "#1A6B4A"
C_AMBER  = "#C4922B"
C_MUTED  = "#64748B"
C_DARK   = "#0D1B2A"
C_MID    = "#334155"
C_GRID   = "#E5EAF0"
FONT     = "'Segoe UI', system-ui, sans-serif"

# Colorscale sözlüğü — tip başına birer tane, tutarlı kullanım
CS = {
    "blue":  [[0.0, "#CFE5F5"], [0.5, "#4F94C9"], [1.0, "#0F4C81"]],
    "red":   [[0.0, "#F6D2CB"], [0.5, "#C96B4F"], [1.0, "#8B1C1C"]],
    "net":   [[0.0, "#8B1C1C"], [0.3, "#F5C4BB"], [0.5, "#F4F6F9"],
              [0.7, "#AACFED"], [1.0, "#0F4C81"]],
    "teal":  [[0.0, "#D9F0E8"], [0.5, "#3DBE99"], [1.0, "#0E6655"]],
    "amber": [[0.0, "#FBE9BE"], [0.5, "#D4A849"], [1.0, "#7A4A00"]],
    "score": [[0.0, "#F7E1A0"], [0.4, "#F4C242"], [0.75, "#27AE60"],
              [1.0, "#1F618D"]],
}


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(155deg,#EEF3F9 0%,#F4F6F9 60%,#EDF1EF 100%); }
    .block-container { padding-top:.9rem; padding-bottom:2rem; max-width:1520px; }

    /* KPI */
    .kpi { background:rgba(255,255,255,.96); border:1px solid rgba(15,23,42,.07);
           border-radius:14px; padding:.85rem 1rem; box-shadow:0 4px 16px rgba(15,23,42,.06); }
    .kpi-lbl { font-size:.72rem; font-weight:700; letter-spacing:.09em;
               text-transform:uppercase; color:#64748B; margin-bottom:.35rem; }
    .kpi-val { font-size:1.65rem; font-weight:800; color:#0D1B2A; margin-bottom:.2rem; }
    .kpi-sub { font-size:.83rem; color:#475569; }

    /* Section header */
    .sh { border-left:3px solid #0F4C81; padding:.2rem 0 .2rem .6rem; margin:.3rem 0 .5rem; }
    .sh b { font-size:.94rem; color:#0D1B2A; display:block; margin-bottom:.1rem; }
    .sh small { font-size:.78rem; color:#52606d; }

    /* Tabs */
    div[data-baseweb="tab-list"]    { gap:.25rem; }
    button[data-baseweb="tab"]      { background:rgba(255,255,255,.65);
                                      border-radius:999px!important;
                                      border:1px solid rgba(15,23,42,.07)!important;
                                      padding:.3rem .9rem; font-size:.86rem; color:#334155; }
    button[data-baseweb="tab"][aria-selected="true"]
                                    { background:linear-gradient(135deg,#0F4C81,#1A5D9A)!important;
                                      color:white!important; border-color:transparent!important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#0D1B2A,#162638);
                                border-right:1px solid rgba(255,255,255,.05); }
    [data-testid="stSidebar"] *     { color:#D6E4F5!important; }
    [data-testid="stSidebar"] label { color:#8AAFC8!important; font-size:.8rem; }
    </style>""", unsafe_allow_html=True)


inject_css()

# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI: BASE LAYOUT  (tüm grafiklere uygulanır)
# ─────────────────────────────────────────────────────────────────────────────
def _lay(height: int = 380, title_text: str = "", **extra) -> dict:
    """Dinamik eksen, automargin, tutarlı font."""
    base = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.55)",
        font=dict(family=FONT, color=C_MID, size=12),
        title=dict(text=title_text,
                   font=dict(family=FONT, color=C_DARK, size=13, weight="bold"),
                   x=0.01, xanchor="left", pad=dict(l=4, b=6)),
        margin=dict(l=10, r=16, t=50, b=10),
        xaxis=dict(gridcolor=C_GRID, linecolor="#CBD5E1",
                   tickfont=dict(size=11), automargin=True, tickangle=-30),
        yaxis=dict(gridcolor=C_GRID, linecolor="#CBD5E1",
                   tickfont=dict(size=11), automargin=True),
        legend=dict(bgcolor="rgba(255,255,255,.88)", bordercolor="#DDE3EC",
                    borderwidth=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", bordercolor="#CBD5E1",
                        font=dict(color=C_DARK, size=12, family=FONT)),
        coloraxis_colorbar=dict(thickness=11, len=.72, tickfont=dict(size=9)),
        uniformtext=dict(minsize=8, mode="hide"),
    )
    base.update(extra)
    return base


def _ap(fig: go.Figure, height: int = 380, title: str = "", **extra) -> go.Figure:
    fig.update_layout(**_lay(height, title_text=title, **extra))
    return fig


def _chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _sh(title: str, note: str = "") -> None:
    note_html = f"<small>{note}</small>" if note else ""
    st.markdown(f'<div class="sh"><b>{title}</b>{note_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# GRAFİK FABRİKALARI
# ─────────────────────────────────────────────────────────────────────────────
def _clean(s: str) -> str:
    return str(s).replace("_", " ")


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index   = [_clean(i) for i in df.index]
    df.columns = [_clean(c) for c in df.columns]
    return df


def _heat_bounds(values: np.ndarray, zmid: float | None = None) -> tuple[float | None, float | None]:
    flat = values.astype(float).ravel()
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return None, None

    if zmid is not None:
        span = float(np.nanquantile(np.abs(flat - zmid), 0.95))
        if span <= 0:
            span = float(np.nanmax(np.abs(flat - zmid)))
        if span <= 0:
            span = 1.0
        return zmid - span, zmid + span

    zmin = float(np.nanquantile(flat, 0.05))
    zmax = float(np.nanquantile(flat, 0.95))
    if np.isclose(zmin, zmax):
        zmin = float(np.nanmin(flat))
        zmax = float(np.nanmax(flat))
    if np.isclose(zmin, zmax):
        zmax = zmin + 1.0
    return zmin, zmax


def heatmap(
    df: pd.DataFrame,
    title: str,
    cs: list | None = None,
    zmid: float | None = None,
    fmt: str = ".1f",
    unit: str = "",
) -> go.Figure:
    if cs is None:
        cs = CS["blue"]
    df   = _clean_df(df)
    vals = df.to_numpy(dtype=float)
    rows = len(df.index)
    h    = max(280, rows * 34 + 130)   # dinamik satır yüksekliği
    text = np.vectorize(lambda v: f"{v:{fmt}}{unit}")(vals)
    zmin, zmax = _heat_bounds(vals, zmid=zmid)
    fig  = go.Figure(go.Heatmap(
        z=vals,
        x=list(df.columns),
        y=list(df.index),
        colorscale=cs, zmid=zmid, zmin=zmin, zmax=zmax,
        xgap=1, ygap=1,
        text=text, texttemplate="%{text}",
        textfont=dict(size=10, color="#0D1B2A"),
        hovertemplate="<b>%{y}</b> · <b>%{x}</b><br>%{z}<extra></extra>",
        colorbar=dict(thickness=11, len=.72, tickfont=dict(size=9)),
    ))
    fig.update_xaxes(showgrid=False, automargin=True, tickangle=-30, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(size=11))
    fig.update_layout(
        height=h, paper_bgcolor="rgba(255,255,255,0.97)", plot_bgcolor="rgba(255,255,255,0.97)",
        font=dict(family=FONT, color=C_MID, size=12),
        title=dict(text=title,
                   font=dict(family=FONT, color=C_DARK, size=13, weight="bold"),
                   x=0.01, xanchor="left", pad=dict(l=4, b=6)),
        margin=dict(l=10, r=16, t=50, b=10),
        hoverlabel=dict(bgcolor="white", bordercolor="#CBD5E1",
                        font=dict(color=C_DARK, size=12, family=FONT)),
    )
    return fig


def bar_h(
    series: pd.Series,
    title: str,
    color_map: dict | None = None,
) -> go.Figure:
    df = pd.DataFrame({"lbl": [_clean(i) for i in series.index], "val": series.values})
    df = df.sort_values("val")
    colors = [
        (color_map.get(str(r.lbl).replace(" ", "_"), C_BLUE) if color_map else C_BLUE)
        for r in df.itertuples()
    ]
    fig = go.Figure(go.Bar(
        x=df["val"], y=df["lbl"], orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b>: %{x:.2f}<extra></extra>",
        texttemplate="%{x:.1f}", textposition="outside",
        textfont=dict(size=10),
    ))
    h = max(260, len(df) * 32 + 80)
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    fig.update_xaxes(automargin=True, tickfont=dict(size=11), tickangle=0)
    return _ap(fig, h, title=title, showlegend=False)


def bar_group(
    df_long: pd.DataFrame,
    x: str, y: str, color: str,
    title: str,
    color_map: dict | None = None,
) -> go.Figure:
    fig = px.bar(df_long, x=x, y=y, color=color, barmode="group",
                 color_discrete_map=color_map or {}, template="none")
    fig.update_xaxes(automargin=True, tickangle=-30, tickfont=dict(size=11))
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    fig.update_traces(hovertemplate="<b>%{x}</b> · %{data.name}<br>%{y:.1f}<extra></extra>")
    return _ap(fig, 340, title=title)


def line_trend(df: pd.DataFrame, title: str) -> go.Figure:
    tidy = (df.reset_index()
              .rename(columns={df.index.name or "index": "segment"})
              .melt(id_vars="segment", var_name="Dönem", value_name="Değer"))
    tidy["seg"] = tidy["segment"].astype(str).str.replace("_", " ", regex=False)
    cmap = {s.replace("_", " "): SEG_RENK[s] for s in SEGMENT_SIRASI if s in df.index}
    fig = px.line(tidy, x="Dönem", y="Değer", color="seg", markers=True,
                  color_discrete_map=cmap, template="none")
    fig.update_traces(line_width=2, marker_size=7)
    fig.update_xaxes(automargin=True, tickangle=-20, tickfont=dict(size=11))
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    fig.update_layout(legend_title_text="Segment")
    return _ap(fig, 340, title=title)


def line_daily(series_dict: dict[str, pd.Series], title: str) -> go.Figure:
    colors = [C_BLUE, C_RED, C_GREEN, C_AMBER]
    fig = go.Figure()
    for i, (name, s) in enumerate(series_dict.items()):
        fig.add_trace(go.Scatter(
            x=s.index.astype(int), y=s.values, name=name, mode="lines+markers",
            line=dict(width=2, color=colors[i % len(colors)]),
            marker=dict(size=6),
            hovertemplate="Gün %{x}: %{y:.1f}<extra>" + name + "</extra>",
        ))
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color=C_MUTED)
    fig.add_annotation(x=0.02, y=0.97, xref="paper", yref="paper",
                       text="▲ Event", showarrow=False,
                       font=dict(size=10, color=C_MUTED))
    fig.update_xaxes(automargin=True, title_text="Gün Farkı", tickfont=dict(size=11))
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    return _ap(fig, 320, title=title, legend_title_text="")


# ─────────────────────────────────────────────────────────────────────────────
# HESAPLAMA YARDIMCILARI
# ─────────────────────────────────────────────────────────────────────────────
def _fi(df: pd.DataFrame, segs: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    valid = [s for s in segs if s in df.index]
    return df.loc[valid] if valid else df


def _window_pivot(wdf: pd.DataFrame, event_type: str, agg: str = "mean") -> pd.DataFrame:
    sub = wdf[wdf["event_type"] == event_type]
    if sub.empty:
        return pd.DataFrame()
    pv = sub.pivot_table(index="gun_fark", columns="urun_grubu",
                         values="islem_tutari", aggfunc=agg, observed=True)
    pv = pv.reindex(columns=URUNLER, fill_value=0).fillna(0)
    pv = pv.loc[sorted(pv.index)]
    if agg == "mean":
        pv = (pv / 1e3).round(1)
    return pv


def _seg_urun(islem_df: pd.DataFrame, segs: list[str]) -> pd.DataFrame:
    sub = islem_df[islem_df["musteri_segmenti"].isin(segs)]
    pv  = sub.groupby(["musteri_segmenti", "urun_grubu"], observed=True).size().unstack(fill_value=0)
    return pv.reindex(index=SEGMENT_SIRASI, columns=URUNLER, fill_value=0).dropna(how="all")


def _yon_urun(islem_df: pd.DataFrame, segs: list[str]) -> pd.DataFrame:
    sub = islem_df[islem_df["musteri_segmenti"].isin(segs)]
    pv  = sub.groupby(["islem_yonu", "urun_grubu"], observed=True).size().unstack(fill_value=0)
    return pv.reindex(columns=URUNLER, fill_value=0).fillna(0)


def _seg_yon(islem_df: pd.DataFrame, segs: list[str]) -> pd.DataFrame:
    sub = islem_df[islem_df["musteri_segmenti"].isin(segs)]
    pv  = (sub.groupby(["musteri_segmenti", "islem_yonu"], observed=True)["islem_tutari"]
              .sum().div(1e6).unstack(fill_value=0))
    return (pv.reindex(index=SEGMENT_SIRASI, fill_value=0)
              .dropna(how="all").round(2))


def _clean_combo(value: str, multiline: bool = False) -> str:
    parts = [_clean(part) for part in str(value).split("|")]
    return ("<br>" if multiline else " | ").join(parts)


def _prep_islem_df(islem_df: pd.DataFrame) -> pd.DataFrame:
    if islem_df is None or islem_df.empty:
        return pd.DataFrame()
    df = islem_df.copy()
    if "urun_tur_yon" not in df.columns:
        df["urun_tur_yon"] = (
            df["urun_grubu"].astype(str)
            + "|"
            + df["islem_turu"].astype(str)
            + "|"
            + df["islem_yonu"].astype(str)
        )
    if "donem" not in df.columns:
        tarih = pd.to_datetime(df["islem_tarihi"])
        df["donem"] = pd.NA
        for donem, (bas, bit) in DONEM_ARALIK.items():
            mask = tarih.between(pd.Timestamp(bas), pd.Timestamp(bit))
            df.loc[mask, "donem"] = donem
    return df[df["donem"].notna()].copy()


def _top_combo_order(islem_df: pd.DataFrame, segs: list[str], periods: list[str], top_n: int = 10) -> list[str]:
    sub = islem_df[
        islem_df["musteri_segmenti"].isin(segs)
        & islem_df["donem"].isin(periods)
    ]
    if sub.empty:
        return []
    return (
        sub.groupby("urun_tur_yon", observed=True)["islem_tutari"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )


def _segment_combo_share(
    islem_df: pd.DataFrame,
    segs: list[str],
    periods: list[str],
    combos: list[str],
) -> pd.DataFrame:
    sub = islem_df[
        islem_df["musteri_segmenti"].isin(segs)
        & islem_df["donem"].isin(periods)
        & islem_df["urun_tur_yon"].isin(combos)
    ]
    if sub.empty:
        return pd.DataFrame()
    pv = (
        sub.groupby(["musteri_segmenti", "urun_tur_yon"], observed=True)["islem_tutari"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=segs, columns=combos, fill_value=0)
    )
    share = pv.div(pv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).mul(100).round(1)
    share.columns = [_clean_combo(c, multiline=True) for c in share.columns]
    return share


def _donem_combo_volume(
    islem_df: pd.DataFrame,
    segs: list[str],
    periods: list[str],
    combos: list[str],
) -> pd.DataFrame:
    sub = islem_df[
        islem_df["musteri_segmenti"].isin(segs)
        & islem_df["donem"].isin(periods)
        & islem_df["urun_tur_yon"].isin(combos)
    ]
    if sub.empty:
        return pd.DataFrame()
    pv = (
        sub.groupby(["donem", "urun_tur_yon"], observed=True)["islem_tutari"]
        .sum()
        .div(1e6)
        .round(2)
        .unstack(fill_value=0)
        .reindex(index=periods, columns=combos, fill_value=0)
    )
    pv.columns = [_clean_combo(c, multiline=True) for c in pv.columns]
    return pv


def _segment_period_combo_volume(
    islem_df: pd.DataFrame,
    segs: list[str],
    periods: list[str],
    combo: str,
) -> pd.DataFrame:
    sub = islem_df[
        islem_df["musteri_segmenti"].isin(segs)
        & islem_df["donem"].isin(periods)
        & (islem_df["urun_tur_yon"] == combo)
    ]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["musteri_segmenti", "donem"], observed=True)["islem_tutari"]
        .sum()
        .div(1e6)
        .round(2)
        .unstack(fill_value=0)
        .reindex(index=segs)
        .reindex(columns=periods, fill_value=0)
    )


def _segment_period_combo_pop(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or df.shape[1] < 2:
        return pd.DataFrame()
    pop = df.pct_change(axis=1).replace([np.inf, -np.inf], np.nan).mul(100).round(1)
    return pop.iloc[:, 1:].fillna(0)


def _resolve_graph(
    bundle: dict,
    islem_df: pd.DataFrame,
    segs: list[str],
    periods: list[str],
    min_edge: int,
) -> nx.DiGraph:
    all_seg = set(segs) == set(SEGMENT_SIRASI)
    all_per = set(periods) == set(DONEM_SIRASI)

    if all_seg and all_per:
        return bundle.get("G_COMPOSITE", nx.DiGraph()).copy()
    if len(segs) == 1 and all_per:
        return bundle.get("G_COMPOSITE_SEG", {}).get(segs[0], nx.DiGraph()).copy()
    if len(periods) == 1 and all_seg:
        return bundle.get("TEMPORAL_COMPOSITE_GRAPHS", {}).get(periods[0], nx.DiGraph()).copy()

    sub = islem_df[
        islem_df["musteri_segmenti"].isin(segs)
        & islem_df["donem"].isin(periods)
    ].copy()
    graph, _ = build_composite_net(sub, min_edge=min_edge)
    return graph


def _focus_graph(graph: nx.DiGraph, max_nodes: int = 32) -> nx.DiGraph:
    if graph.number_of_nodes() <= max_nodes:
        return graph.copy()

    def score(node: str) -> float:
        freq = float(graph.nodes[node].get("freq", 0))
        flow = sum(d.get("weight", 0) for _, _, d in graph.in_edges(node, data=True))
        flow += sum(d.get("weight", 0) for _, _, d in graph.out_edges(node, data=True))
        return freq + (flow * 5)

    keep = sorted(graph.nodes(), key=score, reverse=True)[:max_nodes]
    return graph.subgraph(keep).copy()


def _graph_stats(graph: nx.DiGraph) -> dict:
    if graph.number_of_edges() == 0:
        return {"nodes": graph.number_of_nodes(), "edges": 0, "top_route": "NA", "route_weight": 0}
    top_edge = max(graph.edges(data=True), key=lambda edge: edge[2].get("weight", 0))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "top_route": f"{_clean_combo(top_edge[0])} → {_clean_combo(top_edge[1])}",
        "route_weight": int(top_edge[2].get("weight", 0)),
    }


def _node_parts(node: str) -> tuple[str, str, str]:
    parts = str(node).split("|")
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


def _top_weighted_graph(graph: nx.DiGraph, edge_limit: int = 80) -> nx.DiGraph:
    if graph.number_of_edges() <= edge_limit:
        return graph.copy()
    top_edges = sorted(graph.edges(data=True), key=lambda edge: edge[2].get("weight", 0), reverse=True)[:edge_limit]
    keep_nodes = set()
    out = nx.DiGraph()
    for src, dst, data in top_edges:
        keep_nodes.add(src)
        keep_nodes.add(dst)
        out.add_edge(src, dst, **data)
    for node in keep_nodes:
        out.add_node(node, **graph.nodes[node])
    return out


def _filter_graph_edges(graph: nx.DiGraph, edge_mode: str) -> nx.DiGraph:
    if edge_mode == "Tümü":
        return graph.copy()
    out = nx.DiGraph()
    for node, attrs in graph.nodes(data=True):
        out.add_node(node, **attrs)
    for src, dst, data in graph.edges(data=True):
        is_self = src == dst
        if edge_mode == "Self-loop" and is_self:
            out.add_edge(src, dst, **data)
        elif edge_mode == "Düğümler arası" and not is_self:
            out.add_edge(src, dst, **data)
    empty_nodes = [node for node in out.nodes() if out.degree(node) == 0]
    out.remove_nodes_from(empty_nodes)
    return out


def _prepare_display_graph(
    graph: nx.DiGraph,
    edge_limit: int = 80,
    edge_mode: str = "Tümü",
    max_nodes: int = 34,
) -> nx.DiGraph:
    focused = _focus_graph(graph, max_nodes=max_nodes)
    weighted = _top_weighted_graph(focused, edge_limit=edge_limit)
    return _filter_graph_edges(weighted, edge_mode=edge_mode)


def _node_centrality(graph: nx.DiGraph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {n: 1.0 for n in graph.nodes()}
    except Exception:
        return {n: float(graph.degree(n)) for n in graph.nodes()}


def _structured_3d_pos(graph: nx.DiGraph, centrality: dict[str, float]) -> dict[str, tuple[float, float, float]]:
    if graph.number_of_nodes() == 0:
        return {}

    products = [urun for urun in URUNLER if any(_node_parts(node)[0] == urun for node in graph.nodes())]
    product_spacing = 13.0
    y_map = {"Çıkış": -8.0, "Giriş": 8.0}
    pos: dict[str, tuple[float, float, float]] = {}

    for p_idx, product in enumerate(products):
        product_nodes = [node for node in graph.nodes() if _node_parts(node)[0] == product]
        x_base = (p_idx - (len(products) - 1) / 2) * product_spacing
        for direction in ["Çıkış", "Giriş"]:
            dir_nodes = [node for node in product_nodes if _node_parts(node)[2] == direction]
            if not dir_nodes:
                continue
            types = sorted({_node_parts(node)[1] for node in dir_nodes})
            z_map = {
                tx: (idx - (len(types) - 1) / 2) * 5.2
                for idx, tx in enumerate(types)
            }
            ranked = sorted(dir_nodes, key=lambda node: centrality.get(node, 0.0), reverse=True)
            for rank, node in enumerate(ranked):
                _, tx_type, tx_dir = _node_parts(node)
                x = x_base + ((rank % 3) - 1) * 1.2
                y = y_map.get(tx_dir, 0.0) + (rank // 3) * 1.1
                z = z_map.get(tx_type, 0.0) + ((rank % 2) - 0.5) * 0.9
                pos[node] = (x, y, z)

    return pos


def _graph_legend_df(graph: nx.DiGraph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()
    centrality = _node_centrality(graph)
    rows = []
    for node in sorted(graph.nodes(), key=lambda item: centrality.get(item, 0.0), reverse=True):
        urun, islem_turu, yon = _node_parts(node)
        rows.append({
            "Node": _clean_combo(node),
            "Ürün": _clean(urun),
            "İşlem Türü": _clean(islem_turu),
            "Yön": _clean(yon),
            "Pagerank": round(float(centrality.get(node, 0.0)), 4),
            "Frekans": int(graph.nodes[node].get("freq", 0)),
        })
    return pd.DataFrame(rows)


def network_3d(graph: nx.DiGraph, title: str) -> go.Figure:
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Bu filtre seti için network oluşmadı.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(size=14, color=C_MUTED),
        )
        fig.update_layout(height=520, paper_bgcolor="rgba(255,255,255,0.97)")
        return fig

    centrality = _node_centrality(graph)
    pos = _structured_3d_pos(graph, centrality)

    edge_weights = [float(data.get("weight", 0)) for _, _, data in graph.edges(data=True)]
    wmin = min(edge_weights) if edge_weights else 0.0
    wmax = max(edge_weights) if edge_weights else 1.0
    wspan = max(wmax - wmin, 1.0)

    node_order = list(graph.nodes())
    cent_vals = np.array([float(centrality.get(n, 0.0)) for n in node_order], dtype=float)
    cmin = float(cent_vals.min()) if len(cent_vals) else 0.0
    cmax = float(cent_vals.max()) if len(cent_vals) else 1.0
    cspan = max(cmax - cmin, 1.0)
    node_size = 12 + ((cent_vals - cmin) / cspan) * 26

    hover_text = {}
    for node in node_order:
        hover_text[node] = (
            f"<b>{_clean_combo(node)}</b><br>Frekans: {int(graph.nodes[node].get('freq', 0)):,}<br>"
            f"Pagerank: {centrality.get(node, 0):.4f}<br>Giriş kenarı: {graph.in_degree(node)}<br>Çıkış kenarı: {graph.out_degree(node)}"
        )

    fig = go.Figure()
    for src, dst, data in sorted(graph.edges(data=True), key=lambda edge: edge[2].get("weight", 0), reverse=True):
        x0, y0, z0 = pos[src]
        x1, y1, z1 = pos[dst]
        weight = float(data.get("weight", 0))
        width = 2.0 + ((weight - wmin) / wspan) * 8.0
        alpha = 0.25 + ((weight - wmin) / wspan) * 0.70
        if src == dst:
            radius = 0.9 + ((weight - wmin) / wspan) * 1.2
            theta = np.linspace(0, 2 * np.pi, 34)
            lx = x0 + radius * np.cos(theta)
            ly = y0 + (radius * 0.35)
            lz = z0 + radius * np.sin(theta)
            fig.add_trace(go.Scatter3d(
                x=lx.tolist(), y=np.full_like(theta, ly).tolist(), z=lz.tolist(),
                mode="lines",
                line=dict(color=f"rgba(15,76,129,{alpha:.2f})", width=width),
                hovertemplate=(
                    f"<b>{_clean_combo(src)}</b><br>Self-loop<br>"
                    f"Geçiş adedi: {int(weight):,}<br>Pay: {data.get('pct', 0)}%<extra></extra>"
                ),
                showlegend=False,
            ))
            continue
        fig.add_trace(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode="lines",
            line=dict(color=f"rgba(15,76,129,{alpha:.2f})", width=width),
            hovertemplate=(
                f"<b>{_clean_combo(src)}</b><br>→ {_clean_combo(dst)}<br>"
                f"Geçiş adedi: {int(weight):,}<br>Pay: {data.get('pct', 0)}%<extra></extra>"
            ),
            showlegend=False,
        ))

    for urun in [u for u in URUNLER if any(_node_parts(node)[0] == u for node in node_order)]:
        urun_nodes = [node for node in node_order if _node_parts(node)[0] == urun]
        if not urun_nodes:
            continue
        xs = [pos[node][0] for node in urun_nodes]
        ys = [pos[node][1] for node in urun_nodes]
        zs = [pos[node][2] for node in urun_nodes]
        sizes = [float(node_size[node_order.index(node)]) for node in urun_nodes]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            name=_clean(urun),
            hovertext=[hover_text[node] for node in urun_nodes],
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(
                size=sizes,
                color=URUN_RENKLER.get(urun, C_BLUE),
                opacity=0.94,
                line=dict(color="white", width=1.1),
            ),
            legendgroup=urun,
        ))

    fig.update_layout(
        height=680,
        paper_bgcolor="rgba(255,255,255,0.97)",
        plot_bgcolor="rgba(255,255,255,0.97)",
        margin=dict(l=0, r=0, t=50, b=0),
        font=dict(family=FONT, color=C_MID, size=12),
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=13, color=C_DARK)),
        legend=dict(
            title="Ürün Grubu",
            x=1.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#DDE3EC",
            borderwidth=1,
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(255,255,255,0)",
            aspectmode="manual",
            aspectratio=dict(x=2.2, y=1.5, z=1.7),
            camera=dict(eye=dict(x=1.95, y=1.8, z=1.35)),
        ),
    )
    return fig


def sankey_from_graph(graph: nx.DiGraph, title: str, edge_limit: int = 40) -> go.Figure:
    graph = _top_weighted_graph(graph, edge_limit=edge_limit)
    if graph.number_of_edges() == 0:
        fig = go.Figure()
        fig.add_annotation(text="Bu filtre seti için akış bulunamadı.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=420, paper_bgcolor="rgba(255,255,255,0.97)")
        return fig

    nodes = list(graph.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes)}
    labels = [_clean_combo(node) for node in nodes]
    colors = [graph.nodes[node].get("color", C_BLUE) for node in nodes]

    sources, targets, values, link_colors, custom = [], [], [], [], []
    for src, dst, data in sorted(graph.edges(data=True), key=lambda edge: edge[2].get("weight", 0), reverse=True):
        sources.append(node_index[src])
        targets.append(node_index[dst])
        values.append(float(data.get("weight", 0)))
        link_colors.append("rgba(15,76,129,0.35)")
        custom.append([_clean_combo(src), _clean_combo(dst), int(data.get("weight", 0)), data.get("pct", 0)])

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=colors, pad=18, thickness=18, line=dict(color="white", width=0.8)),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b><br>→ %{customdata[1]}<br>Geçiş: %{customdata[2]:,}<br>Pay: %{customdata[3]}%<extra></extra>",
        ),
    ))
    fig.update_layout(
        height=460,
        paper_bgcolor="rgba(255,255,255,0.97)",
        plot_bgcolor="rgba(255,255,255,0.97)",
        font=dict(family=FONT, color=C_MID, size=11),
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=13, color=C_DARK)),
        margin=dict(l=8, r=8, t=44, b=8),
    )
    return fig


def live_tx_scatter(df: pd.DataFrame, title: str) -> go.Figure:
    if df.empty:
        return go.Figure()
    view = df.copy().sort_values("islem_tarihi").tail(500)
    view["combo"] = view["urun_tur_yon"].map(_clean_combo)
    view["tutar_k"] = (view["islem_tutari"] / 1e3).round(1)
    fig = px.scatter(
        view,
        x="islem_tarihi",
        y="combo",
        color="musteri_segmenti",
        size="tutar_k",
        size_max=18,
        color_discrete_map=SEG_RENK,
        template="none",
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{x}<br>Tutar: %{marker.size:.1f} K TL<extra>%{fullData.name}</extra>",
        marker=dict(opacity=0.78, line=dict(width=0.5, color="white")),
    )
    fig.update_yaxes(automargin=True, tickfont=dict(size=10))
    fig.update_xaxes(automargin=True, tickfont=dict(size=11))
    return _ap(fig, 420, title=title)


# ─────────────────────────────────────────────────────────────────────────────
# VERİ YÜKLEME
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Analitik paketi hesaplanıyor…")
def load_bundle(seed: int) -> dict:
    return build_analysis_bundle(seed).data


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Filtreler")
    st.divider()
    seed    = st.slider("Simülasyon tohumu", 7, 77, 42, step=1)
    sel_seg = st.multiselect("Segmentler", SEGMENT_SIRASI, default=SEGMENT_SIRASI)
    sel_per = st.multiselect("Dönemler",   DONEM_SIRASI,   default=DONEM_SIRASI)
    st.divider()
    st.caption("Filtreler yalnızca görsel katmanı etkiler.")

bundle        = load_bundle(seed)
sel_seg       = sel_seg or SEGMENT_SIRASI
sel_per       = sel_per or DONEM_SIRASI
islem_df_raw  = bundle.get("islem_df",         pd.DataFrame())
window_df_raw = bundle.get("islem_window_df",  pd.DataFrame())
islem_view_df = _prep_islem_df(islem_df_raw)

# ─────────────────────────────────────────────────────────────────────────────
# BAŞLIK + KPI
# ─────────────────────────────────────────────────────────────────────────────
kpi      = bundle["KPI"]
alert_df = bundle["SEGMENT_ALERT_DF"]

st.markdown(
    f"""<div style="background:linear-gradient(135deg,rgba(255,255,255,.95),rgba(236,243,252,.95));
    border:1px solid rgba(15,23,42,.07);border-radius:18px;padding:1.1rem 1.6rem;
    box-shadow:0 4px 22px rgba(15,23,42,.07);margin-bottom:.8rem;">
    <div style="font-size:.71rem;font-weight:700;letter-spacing:.16em;
    text-transform:uppercase;color:#8A6523;margin-bottom:.35rem;">
    Bankacılık Yatırım Analitik Özeti · Seed {seed}</div>
    <div style="font-size:1.75rem;font-weight:800;color:#0D1B2A;margin-bottom:.2rem;">
    Alım · Satım · İşlem · Bakiye</div>
    <div style="font-size:.9rem;color:#405063;">Segment ve dönem bazlı özet heatmap analizleri.</div>
    </div>""",
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
anom_seg = alert_df["Anomaly_Score"].idxmax().replace("_", " ")
anom_val = alert_df["Anomaly_Score"].max()

with k1:
    st.markdown(
        f'<div class="kpi"><div class="kpi-lbl">Net AUM</div>'
        f'<div class="kpi-val">{kpi["net_aum_m"]:.1f} M TL</div>'
        f'<div class="kpi-sub">Alım {kpi["toplam_alim_m"]:.1f} M · Satım {kpi["toplam_satim_m"]:.1f} M</div></div>',
        unsafe_allow_html=True)
with k2:
    st.markdown(
        f'<div class="kpi"><div class="kpi-lbl">Toplam Olay</div>'
        f'<div class="kpi-val">{kpi["toplam_event"]:,}</div>'
        f'<div class="kpi-sub">Alım {kpi["alim_musteri"]:,} · Satım {kpi["satim_musteri"]:,} müşteri</div></div>',
        unsafe_allow_html=True)
with k3:
    st.markdown(
        f'<div class="kpi"><div class="kpi-lbl">Pencere İşlemi</div>'
        f'<div class="kpi-val">{kpi["window_islem"]:,}</div>'
        f'<div class="kpi-sub">±7 gün event penceresi içinde</div></div>',
        unsafe_allow_html=True)
with k4:
    color = C_RED if anom_val > 60 else C_DARK
    st.markdown(
        f'<div class="kpi"><div class="kpi-lbl">En Yüksek Anomali</div>'
        f'<div class="kpi-val" style="color:{color}">{anom_val:.1f}</div>'
        f'<div class="kpi-sub">Segment: {anom_seg}</div></div>',
        unsafe_allow_html=True)

st.write("")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
t_alim, t_satim, t_islem, t_network, t_bakiye = st.tabs([
    "📈  Alım Analizi",
    "📉  Satım Analizi",
    "🔄  İşlem Hareketleri",
    "🕸️  İşlem Network",
    "💰  Bakiye Analizi",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ALIM
# ══════════════════════════════════════════════════════════════════════════════
with t_alim:
    a1, a2 = st.columns(2)

    with a1:
        _sh("Dönem × Segment Alım Hacmi (M TL)",
            "4 dönem boyunca segment bazlı alım miktarı.")
        _chart(heatmap(
            _fi(bundle["DONEM_ALIM_PIVOT"][sel_per], sel_seg),
            "Alım Hacmi — Dönem × Segment (M TL)", CS["blue"], unit=" M",
        ))

        poc_cols = [p for p in sel_per if p in bundle["DONEM_ALIM_POC"].columns]
        if len(poc_cols) > 1:
            _sh("Alım Dönemsel Değişim PoP (%)",
                "Bir önceki döneme göre alım büyümesi.")
            _chart(heatmap(
                _fi(bundle["DONEM_ALIM_POC"][poc_cols[1:]], sel_seg),
                "Alım PoP Değişim (%)", CS["net"], zmid=0, unit=" %",
            ))

    with a2:
        _sh("Pre-Buy Ürün Bileşimi (%)",
            "Alım eventi öncesinde hangi ürüne yatırım yapılmış.")
        _chart(heatmap(
            _fi(bundle["URUN_PRE_BUY"], sel_seg),
            "Pre-Buy Ürün Dağılımı — Segment × Ürün (%)", CS["blue"], unit=" %",
        ))

        _sh("Post-Buy Ürün Bileşimi (%)",
            "Alım sonrasında ürün kompozisyonu nasıl şekilleniyor.")
        _chart(heatmap(
            _fi(bundle["URUN_POST_BUY"], sel_seg),
            "Post-Buy Ürün Dağılımı — Segment × Ürün (%)", CS["teal"], unit=" %",
        ))

    st.divider()
    a3, a4 = st.columns(2)

    with a3:
        _sh("Alım Ürün Geçiş Matrisi (%)",
            "Event öncesi dominant üründen event sonrasına ürün geçiş oranı.")
        _chart(heatmap(bundle["GECIS_ALIM"], "Alım Geçiş Matrisi (%)", CS["blue"], unit=" %"))

    with a4:
        _sh("Penetrasyon & Çok-Dönem Sadakat (%)")
        pen = _fi(bundle["PENETRASYON"], sel_seg)[["Alim_Pct"]].join(
            _fi(bundle["SADAKAT"][["Cok_Donem_Pct"]], sel_seg), how="left"
        )
        pen.columns = ["Alım Pen. %", "Çok-Dönem %"]
        _chart(heatmap(pen, "Penetrasyon & Sadakat — Segment", CS["score"], unit=" %"))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SATIM
# ══════════════════════════════════════════════════════════════════════════════
with t_satim:
    s1, s2 = st.columns(2)

    with s1:
        _sh("Dönem × Segment Satım Hacmi (M TL)")
        _chart(heatmap(
            _fi(bundle["DONEM_SATIM_PIVOT"][sel_per], sel_seg),
            "Satım Hacmi — Dönem × Segment (M TL)", CS["red"], unit=" M",
        ))

        _sh("Net Akış (Alım − Satım, M TL)",
            "Negatif = net satıcı, pozitif = net alıcı pozisyonu.")
        _chart(heatmap(
            _fi(bundle["DONEM_NET_PIVOT"][sel_per], sel_seg),
            "Net Akış — Dönem × Segment (M TL)", CS["net"], zmid=0, unit=" M",
        ))

    with s2:
        _sh("Pre-Sell Ürün Bileşimi (%)",
            "Satım eventi öncesinde müşterinin tuttuğu ürünler.")
        _chart(heatmap(
            _fi(bundle["URUN_PRE_SELL"], sel_seg),
            "Pre-Sell Ürün Dağılımı — Segment × Ürün (%)", CS["red"], unit=" %",
        ))

        _sh("Post-Sell Ürün Bileşimi (%)",
            "Satım sonrasında fon nereye yönlendi.")
        _chart(heatmap(
            _fi(bundle["URUN_POST_SELL"], sel_seg),
            "Post-Sell Ürün Dağılımı — Segment × Ürün (%)", CS["amber"], unit=" %",
        ))

    st.divider()
    s3, s4 = st.columns(2)

    with s3:
        _sh("Satım Ürün Geçiş Matrisi (%)")
        _chart(heatmap(bundle["GECIS_SATIM"], "Satım Geçiş Matrisi (%)", CS["red"], unit=" %"))

    with s4:
        _sh("Alım vs Satım Penetrasyon Karşılaştırması (%)")
        pen_df = _fi(bundle["PENETRASYON"], sel_seg)[["Alim_Pct", "Satim_Pct"]].reset_index()
        sc = pen_df.columns[0]
        pen_df["Segment"] = pen_df[sc].astype(str).str.replace("_", " ", regex=False)
        long_df = pen_df.melt(id_vars="Segment",
                              value_vars=["Alim_Pct", "Satim_Pct"],
                              var_name="Tür", value_name="Oran")
        _chart(bar_group(long_df, "Segment", "Oran", "Tür",
                         "Alım / Satım Penetrasyonu (%)",
                         {"Alim_Pct": C_BLUE, "Satim_Pct": C_RED}))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — İŞLEM HAREKETLERİ
# ══════════════════════════════════════════════════════════════════════════════
with t_islem:
    if window_df_raw.empty or islem_df_raw.empty:
        st.warning("İşlem pencere verisi bu seed için üretilemedi.")
    else:
        # ─ Satır 1: Pencere heatmap'leri ──────────────────────────────────────
        i1, i2 = st.columns(2)

        with i1:
            _sh("Alım Penceresi — Gün × Ürün (Ort. K TL)",
                "Alım eventi ±7 günündeki ürün bazlı işlem tutarı. "
                "Negatif gün = event öncesi, pozitif = sonrası.")
            pv_a = _window_pivot(window_df_raw, "Alım", "mean")
            if not pv_a.empty:
                _chart(heatmap(pv_a, "Alım Penceresi — Gün × Ürün (K TL)", CS["amber"]))

        with i2:
            _sh("Satım Penceresi — Gün × Ürün (Ort. K TL)")
            pv_s = _window_pivot(window_df_raw, "Satım", "mean")
            if not pv_s.empty:
                _chart(heatmap(pv_s, "Satım Penceresi — Gün × Ürün (K TL)", CS["red"]))

        # ─ Satır 2: Adet heatmap'leri ─────────────────────────────────────────
        i3, i4 = st.columns(2)

        with i3:
            _sh("Alım Penceresi — İşlem Adedi (Gün × Ürün)")
            pv_ac = _window_pivot(window_df_raw, "Alım", "count")
            if not pv_ac.empty:
                _chart(heatmap(pv_ac, "Alım Penceresi — Adet", CS["teal"], fmt=".0f"))

        with i4:
            _sh("Satım Penceresi — İşlem Adedi (Gün × Ürün)")
            pv_sc = _window_pivot(window_df_raw, "Satım", "count")
            if not pv_sc.empty:
                _chart(heatmap(pv_sc, "Satım Penceresi — Adet", CS["blue"], fmt=".0f"))

        st.divider()

        # ─ Satır 3: Birleşik ürün + tür + yön ────────────────────────────────
        i5, i6 = st.columns(2)

        combo_list = _top_combo_order(islem_view_df, sel_seg, sel_per, top_n=10)

        with i5:
            _sh("Segment × Ürün/Tür/Yön Davranış Payı (%)",
                "Vadeli açılış çıkışı ile vadeli kapanış girişi ayrı node olarak ele alınır.")
            if combo_list:
                seg_combo = _segment_combo_share(islem_view_df, sel_seg, sel_per, combo_list)
                if not seg_combo.empty:
                    _chart(heatmap(seg_combo, "Segment × Ürün/Tür/Yön Payı (%)", CS["score"], unit=" %"))

            _sh("Dönem × Ürün/Tür/Yön Hacmi (M TL)",
                "Seçili segmentlerde birleşik kombinasyonların dönemsel hacmi.")
            if combo_list:
                donem_combo = _donem_combo_volume(islem_view_df, sel_seg, sel_per, combo_list)
                if not donem_combo.empty:
                    _chart(heatmap(donem_combo, "Dönem × Ürün/Tür/Yön Hacmi (M TL)", CS["amber"], unit=" M"))

        with i6:
            _sh("Odak Kombinasyon — Segment Dönem Trendi",
                "Segment bazında dönemler arası değişimi tek bir ürün/tür/yön kırılımında izle.")
            if combo_list:
                focus_combo = st.selectbox(
                    "Odak ürün / işlem türü / yön",
                    options=combo_list,
                    format_func=lambda x: _clean_combo(x),
                    key="focus_combo",
                )
                focus_pv = _segment_period_combo_volume(islem_view_df, sel_seg, sel_per, focus_combo)
                if not focus_pv.empty:
                    _chart(line_trend(focus_pv, f"{_clean_combo(focus_combo)} — Segment Trend (M TL)"))

                focus_pop = _segment_period_combo_pop(focus_pv)
                if not focus_pop.empty:
                    _chart(heatmap(
                        focus_pop,
                        f"{_clean_combo(focus_combo)} — Segment PoP Değişim (%)",
                        CS["net"],
                        zmid=0,
                        unit=" %",
                    ))

            _sh("Günlük Ort. İşlem Tutarı — Event Ekseninde (K TL)",
                "Alım ve satım eventleri etrafındaki günlük ortalama tutar.")
            ga = bundle.get("GUNLUK_ALIM", pd.Series(dtype=float))
            gs = bundle.get("GUNLUK_SATIM", pd.Series(dtype=float))
            if not ga.empty:
                _chart(line_daily(
                    {"Alım penceresi": ga / 1e3, "Satım penceresi": gs / 1e3},
                    "Günlük Ort. İşlem Tutarı (K TL)",
                ))

        st.divider()
        _sh("Dönem × Segment İşlem Yoğunluğu (Adet)")
        _chart(heatmap(
            _fi(bundle["DONEM_ISLEM_PIVOT"][sel_per], sel_seg),
            "İşlem Yoğunluğu — Dönem × Segment (Adet)",
            CS["teal"], fmt=".0f",
        ))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — İŞLEM NETWORK / LIVE İNCELEME
# ══════════════════════════════════════════════════════════════════════════════
with t_network:
    if islem_view_df.empty:
        st.warning("Network görselleştirmesi için işlem datası üretilemedi.")
    else:
        network_df = islem_view_df[
            islem_view_df["musteri_segmenti"].isin(sel_seg)
            & islem_view_df["donem"].isin(sel_per)
        ].copy()

        if network_df.empty:
            st.info("Seçili segment ve dönem filtrelerinde network verisi yok.")
        else:
            _sh("3B İşlem Networkü",
                "Node'lar ürün | işlem türü | yön birleşimidir. Kenarlar müşteri bazlı ardışık işlem geçişini gösterir.")

            c1, c2, c3, c4 = st.columns([1.15, 1.15, 1.6, 1.6])
            with c1:
                min_edge = st.slider("Min edge", 2, 20, 6, key="network_min_edge")
            with c2:
                segment_focus = st.selectbox(
                    "Segment görünümü",
                    options=["Seçili segmentlerin toplamı"] + sel_seg,
                    key="network_segment_focus",
                )
            with c3:
                snapshot_mode = st.selectbox(
                    "Graph kapsamı",
                    options=["Filtre özeti", "Tek dönem snapshot"],
                    key="network_scope",
                )
            with c4:
                graph_periods = sel_per
                if snapshot_mode == "Tek dönem snapshot":
                    single_period = st.select_slider("Temporal snapshot", options=sel_per, key="network_single_period")
                    graph_periods = [single_period]

            network_seg_scope = sel_seg if segment_focus == "Seçili segmentlerin toplamı" else [segment_focus]
            edge_limit = st.slider("Rendered edge limiti", 10, 140, 60, step=5, key="network_edge_limit")
            edge_mode = st.selectbox(
                "Edge filtresi",
                options=["Tümü", "Düğümler arası", "Self-loop"],
                key="network_edge_mode",
            )

            graph = _resolve_graph(bundle, islem_view_df, network_seg_scope, graph_periods, min_edge=min_edge)
            display_graph = _prepare_display_graph(graph, edge_limit=edge_limit, edge_mode=edge_mode, max_nodes=36)
            stats = _graph_stats(graph)
            self_loop_count = sum(1 for src, dst in graph.edges() if src == dst)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Node", f"{stats['nodes']:,}")
            k2.metric("Edge", f"{stats['edges']:,}")
            k3.metric("Dominant Route Weight", f"{stats['route_weight']:,}")
            k4.metric("Self-loop", f"{self_loop_count:,}")

            st.caption(f"Dominant route: {stats['top_route']}")
            st.caption("Node boyutu pagerank ile ölçeklenir. Self-loop geçişler halka olarak çizilir. Yazılar node üstünde değil, legend ve tabloya taşındı.")

            n1, n2 = st.columns([1.55, 1.0])
            with n1:
                _chart(network_3d(display_graph, "3B İşlem Akış Networkü"))
            with n2:
                _chart(sankey_from_graph(display_graph, "Akış Sankey", edge_limit=min(40, edge_limit)))

            legend_df = _graph_legend_df(display_graph)
            if not legend_df.empty:
                st.dataframe(legend_df, use_container_width=True, hide_index=True)

            st.divider()
            _sh("Canlı İnceleme / Replay",
                "Gerçek streaming değil; seçili tarih etrafında işlem akışını ileri-geri scrub ederek incelersin.")

            date_options = (
                pd.to_datetime(network_df["islem_tarihi"])
                .dt.normalize()
                .sort_values()
                .drop_duplicates()
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )

            if date_options:
                live_key = "network_live_date"
                current_value = st.session_state.get(live_key, date_options[-1])
                if current_value not in date_options:
                    current_value = date_options[-1]
                st.session_state[live_key] = current_value

                current_idx = date_options.index(current_value)
                p1, p2, p3 = st.columns([1, 1, 4])
                with p1:
                    if st.button("◀ Önceki", key="live_prev") and current_idx > 0:
                        st.session_state[live_key] = date_options[current_idx - 1]
                with p2:
                    if st.button("Sonraki ▶", key="live_next") and current_idx < len(date_options) - 1:
                        st.session_state[live_key] = date_options[current_idx + 1]
                with p3:
                    st.select_slider(
                        "İnceleme tarihi",
                        options=date_options,
                        key=live_key,
                        format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"),
                    )

                live_days = st.slider("İnceleme penceresi (gün)", 1, 21, 5, key="live_window_days")
                live_ts = pd.Timestamp(st.session_state.get(live_key, current_value))
                start_ts = live_ts - pd.Timedelta(days=live_days - 1)
                live_df = network_df[
                    pd.to_datetime(network_df["islem_tarihi"]).between(start_ts, live_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
                ].copy()

                lk1, lk2, lk3, lk4 = st.columns(4)
                lk1.metric("Penceredeki İşlem", f"{len(live_df):,}")
                lk2.metric("Hacim", f"{live_df['islem_tutari'].sum() / 1e6:.1f} M TL")
                lk3.metric("Müşteri", f"{live_df['musteri_id'].nunique():,}")
                lk4.metric("Kombinasyon", f"{live_df['urun_tur_yon'].nunique():,}")

                l1, l2 = st.columns(2)
                with l1:
                    combo_bar = (
                        live_df.groupby("urun_tur_yon", observed=True)["islem_tutari"]
                        .sum().div(1e6).sort_values(ascending=False).head(10)
                    )
                    combo_bar.index = [_clean_combo(i) for i in combo_bar.index]
                    if not combo_bar.empty:
                        _chart(bar_h(combo_bar, "Pencere İçindeki Baskın Akışlar (M TL)"))

                with l2:
                    if not live_df.empty:
                        _chart(live_tx_scatter(live_df, "Zaman Ekseni Üzerinde İşlem Replay"))

                show_cols = [
                    "islem_tarihi",
                    "musteri_id",
                    "musteri_segmenti",
                    "urun_grubu",
                    "islem_turu",
                    "islem_yonu",
                    "islem_tutari",
                ]
                live_table = live_df.sort_values("islem_tarihi", ascending=False)[show_cols].head(80).copy()
                if not live_table.empty:
                    live_table["islem_tarihi"] = pd.to_datetime(live_table["islem_tarihi"]).dt.strftime("%Y-%m-%d %H:%M")
                    live_table["musteri_segmenti"] = live_table["musteri_segmenti"].map(_clean)
                    live_table["urun_grubu"] = live_table["urun_grubu"].map(_clean)
                    live_table["islem_turu"] = live_table["islem_turu"].map(_clean)
                    live_table["islem_yonu"] = live_table["islem_yonu"].map(_clean)
                    live_table["islem_tutari"] = (live_table["islem_tutari"] / 1e3).round(1)
                    live_table.columns = ["İşlem Tarihi", "Müşteri", "Segment", "Ürün", "İşlem Türü", "Yön", "Tutar K TL"]
                    st.dataframe(live_table, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BAKİYE ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════
with t_bakiye:
    b1, b2 = st.columns(2)

    with b1:
        _sh("Alım Olayları — Bakiye Dilimi × Segment (Adet)",
            "Alım öncesi/sonrası pencerede bakiye dilimine düşen işlem sayısı.")
        bd_ac = bundle.get("BD_ALIM_COUNT", pd.DataFrame())
        if not bd_ac.empty:
            df_tmp = bd_ac.copy()
            df_tmp.index   = [_clean(i) for i in df_tmp.index]
            df_tmp.columns = [_clean(c) for c in df_tmp.columns]
            _chart(heatmap(df_tmp, "Alım — Bakiye Dilimi × Segment (Adet)", CS["blue"], fmt=".0f"))

        _sh("Satım Olayları — Bakiye Dilimi × Segment (Adet)")
        bd_sc = bundle.get("BD_SATIM_COUNT", pd.DataFrame())
        if not bd_sc.empty:
            df_tmp = bd_sc.copy()
            df_tmp.index   = [_clean(i) for i in df_tmp.index]
            df_tmp.columns = [_clean(c) for c in df_tmp.columns]
            _chart(heatmap(df_tmp, "Satım — Bakiye Dilimi × Segment (Adet)", CS["red"], fmt=".0f"))

    with b2:
        _sh("Alım Olayları — Bakiye Dilimi × Segment (Ort. K TL)")
        bd_at = bundle.get("BD_ALIM_TUTAR", pd.DataFrame())
        if not bd_at.empty:
            df_tmp = (bd_at / 1e3).round(1).copy()
            df_tmp.index   = [_clean(i) for i in df_tmp.index]
            df_tmp.columns = [_clean(c) for c in df_tmp.columns]
            _chart(heatmap(df_tmp, "Alım — Bakiye Dilimi × Segment (Ort. K TL)", CS["teal"], unit=" K"))

        _sh("Satım Olayları — Bakiye Dilimi × Segment (Ort. K TL)")
        bd_st = bundle.get("BD_SATIM_TUTAR", pd.DataFrame())
        if not bd_st.empty:
            df_tmp = (bd_st / 1e3).round(1).copy()
            df_tmp.index   = [_clean(i) for i in df_tmp.index]
            df_tmp.columns = [_clean(c) for c in df_tmp.columns]
            _chart(heatmap(df_tmp, "Satım — Bakiye Dilimi × Segment (Ort. K TL)", CS["amber"], unit=" K"))

    st.divider()
    b3, b4 = st.columns(2)

    with b3:
        _sh("Net AUM — Segment Dağılımı (M TL)")
        _chart(bar_h((_fi(bundle["AUM"], sel_seg) / 1e6).round(2),
                     "Net AUM — Segment (M TL)", SEG_RENK))

    with b4:
        _sh("Bakiye Dilim Eşikleri (K TL)",
            "Her segment için dönem bazlı p33 / p67 bakiye sınır değerleri.")
        dilim = bundle.get("DILIM_ESLIKLERI", pd.DataFrame())
        if not dilim.empty:
            # MultiIndex (segment, donem) → her segment için dönem bazlı tablo
            dil_filt = dilim.loc[
                dilim.index.get_level_values(0).isin(sel_seg)
            ] if not dilim.empty else dilim
            if not dil_filt.empty:
                # Unstack → segment × (p33_2025.06, p33_2025.09 ...) wide format
                dil_wide = (dil_filt / 1e3).round(1).unstack(level="donem")
                dil_wide.columns = [
                    f"{c[0]} {c[1]}" for c in dil_wide.columns
                ]
                dil_wide.index = [_clean(i) for i in dil_wide.index]
                _chart(heatmap(dil_wide, "Bakiye Dilim Eşikleri — Segment × Dönem (K TL)",
                               CS["score"], unit=" K"))

st.divider()
st.caption("Yatırım Analitik Özeti · Sentetik bankacılık verisi · streamlit run streamlit_app.py")
