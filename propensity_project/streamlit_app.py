"""Streamlit report viewer for the fund propensity project."""
from __future__ import annotations

from pathlib import Path
import io
import os

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.getenv("PROPENSITY_OUTPUT_DIR", PROJECT_ROOT / "propensity_outputs"))
DEFAULT_REPORT_NAMES = (
    "fund_propensity_pipeline_audit_segmented_latest_v2.xlsx",
    "fund_propensity_pipeline_audit_segmented_latest.xlsx",
    "fund_propensity_pipeline_audit_segmented.xlsx",
)

st.set_page_config(
    page_title="Fund Propensity Report",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_workbook(source: str | bytes) -> dict[str, pd.DataFrame]:
    workbook_source = io.BytesIO(source) if isinstance(source, bytes) else source
    workbook = pd.ExcelFile(workbook_source)
    return {sheet: workbook.parse(sheet) for sheet in workbook.sheet_names}


def find_default_report() -> Path | None:
    for name in DEFAULT_REPORT_NAMES:
        candidate = OUTPUT_ROOT / name
        if candidate.exists():
            return candidate
    reports = sorted(OUTPUT_ROOT.glob("*.xlsx"), key=lambda path: path.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def sheet_frame(sheets: dict[str, pd.DataFrame], *names: str) -> pd.DataFrame:
    for name in names:
        if name in sheets:
            return sheets[name].copy()
    return pd.DataFrame()


def status_frame(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    status = sheet_frame(sheets, "20_Model_Status_Summary", "Model_Status_Summary")
    if not status.empty:
        return status
    metrics = sheet_frame(sheets, "11_Model_Metrics", "Model_Metrics")
    if not metrics.empty and "status" in metrics.columns:
        return metrics
    return pd.DataFrame()


def chart_layout(figure, height: int = 420):
    figure.update_layout(
        height=height,
        margin=dict(l=20, r=30, t=70, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.85)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#E2E8F0", automargin=True)
    figure.update_yaxes(showgrid=True, gridcolor="#E2E8F0", automargin=True)
    return figure


def show_status(sheets: dict[str, pd.DataFrame]) -> None:
    status = status_frame(sheets)
    if status.empty:
        st.info("Bu workbook şemasında ayrı model status sheeti bulunmuyor.")
        return
    st.dataframe(status, use_container_width=True, hide_index=True)
    categorical = [column for column in ("status", "model_status", "evaluation_stage", "segment") if column in status]
    if categorical:
        selected_column = st.selectbox("Status kırılımı", categorical, key="status_dimension")
        counts = status[selected_column].fillna("unknown").astype(str).value_counts().rename_axis(selected_column).reset_index(name="count")
        figure = px.bar(counts, x=selected_column, y="count", color=selected_column, title=f"{selected_column.replace('_', ' ').title()} dağılımı")
        st.plotly_chart(chart_layout(figure, 420), use_container_width=True, config={"displayModeBar": False})


def show_performance(sheets: dict[str, pd.DataFrame]) -> None:
    performance = sheet_frame(
        sheets,
        "17_Performance_Summary",
        "16_Performance_Summary",
        "Performance_Summary",
    )
    if performance.empty:
        st.info("Performance summary sheeti bu raporda bulunmuyor veya boş.")
        return

    st.dataframe(performance, use_container_width=True, hide_index=True)
    metric_aliases = {
        "PR-AUC": ("pr_auc", "pr_auc_mean"),
        "ROC-AUC": ("roc_auc", "roc_auc_mean"),
        "Brier": ("brier", "brier_mean"),
        "Prevalence Lift": ("lift_pr_auc", "pr_auc_lift_mean"),
    }
    metric_columns = []
    metric_labels = {}
    for label, aliases in metric_aliases.items():
        column = next((candidate for candidate in aliases if candidate in performance), None)
        if column:
            metric_columns.append(column)
            metric_labels[column] = label
    dimensions = [column for column in ("segment", "chart_label", "evaluation_stage", "split") if column in performance]
    if not metric_columns:
        st.warning("Bu workbook’ta grafik üretmek için uygun performans metrik kolonu bulunamadı.")
        return
    chart_data = performance.dropna(subset=metric_columns, how="all").copy()
    if chart_data.empty:
        return
    chart_data["label"] = chart_data[dimensions].astype(str).agg(" | ".join, axis=1) if dimensions else chart_data.index.astype(str)
    selected_metric = st.selectbox(
        "Metrik",
        metric_columns,
        format_func=lambda value: metric_labels[value],
    )
    figure = px.bar(
        chart_data.sort_values(selected_metric),
        x=selected_metric,
        y="label",
        color="segment" if "segment" in chart_data else None,
        orientation="h",
        title=f"Segment bazlı {metric_labels[selected_metric]}",
        labels={selected_metric: metric_labels[selected_metric], "label": "Model / split"},
    )
    st.plotly_chart(chart_layout(figure, max(420, min(1000, len(chart_data) * 26 + 150))), use_container_width=True, config={"displayModeBar": False})

    if len(metric_columns) >= 2:
        comparison = chart_data[["label", *metric_columns]].melt("label", var_name="metric", value_name="value").dropna()
        comparison["metric"] = comparison["metric"].map(metric_labels)
        figure = px.bar(comparison, x="label", y="value", color="metric", barmode="group", title="PR-AUC / ROC-AUC ve diğer metrik karşılaştırması")
        figure.update_xaxes(tickangle=-45)
        st.plotly_chart(chart_layout(figure, 520), use_container_width=True, config={"displayModeBar": False})

    split_columns = [column for column in ("evaluation_stage", "split") if column in chart_data]
    if split_columns:
        split_column = split_columns[-1]
        split_summary = chart_data.groupby(split_column, dropna=False)[metric_columns].mean().reset_index()
        split_long = split_summary.melt(split_column, var_name="metric", value_name="value").dropna()
        split_long["metric"] = split_long["metric"].map(metric_labels)
        figure = px.line(split_long, x=split_column, y="value", color="metric", markers=True, title=f"{split_column.replace('_', ' ').title()} bazında ortalama metrik")
        st.plotly_chart(chart_layout(figure, 420), use_container_width=True, config={"displayModeBar": False})


def show_overview(sheets: dict[str, pd.DataFrame]) -> None:
    runtime = sheet_frame(sheets, "00_Runtime_Summary", "Runtime_Summary")
    general = sheet_frame(sheets, "18_General_Summary", "17_General_Summary", "General_Summary")
    performance = sheet_frame(
        sheets,
        "17_Performance_Summary",
        "16_Performance_Summary",
        "Performance_Summary",
    )
    status = status_frame(sheets)

    columns = st.columns(4)
    columns[0].metric("Workbook sheet", len(sheets))
    columns[1].metric("Performance satırı", len(performance))
    columns[2].metric("Model status satırı", len(status))
    columns[3].metric("Genel özet satırı", len(general))
    if not runtime.empty:
        st.subheader("Runtime ve pipeline özeti")
        st.dataframe(runtime, use_container_width=True, hide_index=True)
    elif not general.empty:
        st.subheader("Genel pipeline özeti")
        st.dataframe(general, use_container_width=True, hide_index=True)
    if not general.empty and {"metric", "value"}.issubset(general.columns):
        general_chart = general.copy()
        general_chart["value"] = pd.to_numeric(general_chart["value"], errors="coerce")
        general_chart = general_chart.dropna(subset=["value"])
        if not general_chart.empty:
            figure = px.bar(general_chart, x="metric", y="value", color="metric", title="Pipeline genel hacim ve çıktı özeti")
            figure.update_xaxes(tickangle=-35)
            st.plotly_chart(chart_layout(figure, 440), use_container_width=True, config={"displayModeBar": False})


def main() -> None:
    st.title("Fund Propensity Report")
    st.caption("Segment bazlı Newsell / Upsell modelleme ve audit çıktıları")

    with st.sidebar:
        st.header("Rapor")
        default_report = find_default_report()
        uploaded = st.file_uploader("Excel audit workbook", type=["xlsx"])
        if uploaded is not None:
            source: str | bytes = uploaded.getvalue()
            source_name = uploaded.name
        elif default_report is not None:
            source = str(default_report)
            source_name = default_report.name
            st.caption(f"Yerel çıktı: {default_report.name}")
        else:
            st.info("Bir audit workbook yükleyin veya propensity_outputs klasörüne rapor koyun.")
            return

    try:
        sheets = load_workbook(source)
    except Exception as error:
        st.error(f"Workbook okunamadı: {error}")
        return

    st.caption(f"Aktif rapor: {source_name}")
    overview_tab, performance_tab, status_tab, audit_tab = st.tabs(
        ["Genel Özet", "Performance", "Model Status", "Audit Sheetleri"]
    )
    with overview_tab:
        show_overview(sheets)
    with performance_tab:
        show_performance(sheets)
    with status_tab:
        show_status(sheets)
    with audit_tab:
        selected_sheet = st.selectbox("Sheet", list(sheets))
        st.dataframe(sheets[selected_sheet], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
