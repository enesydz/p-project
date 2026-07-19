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


def show_status(sheets: dict[str, pd.DataFrame]) -> None:
    status = sheet_frame(sheets, "20_Model_Status_Summary", "Model_Status_Summary")
    if status.empty:
        st.info("Model status sheeti bu raporda bulunmuyor.")
        return
    st.dataframe(status, use_container_width=True, hide_index=True)


def show_performance(sheets: dict[str, pd.DataFrame]) -> None:
    performance = sheet_frame(sheets, "17_Performance_Summary", "Performance_Summary")
    if performance.empty:
        st.info("Performance summary sheeti bu raporda bulunmuyor veya boş.")
        return

    st.dataframe(performance, use_container_width=True, hide_index=True)
    metric_columns = [column for column in ("pr_auc", "roc_auc", "lift_pr_auc") if column in performance]
    dimensions = [column for column in ("segment", "chart_label", "evaluation_stage", "split") if column in performance]
    if not metric_columns or not dimensions:
        return
    chart_data = performance.dropna(subset=metric_columns, how="all").copy()
    if chart_data.empty:
        return
    chart_data["label"] = chart_data[dimensions].astype(str).agg(" | ".join, axis=1)
    selected_metric = st.selectbox(
        "Metrik",
        metric_columns,
        format_func=lambda value: value.replace("_", " ").upper(),
    )
    figure = px.bar(
        chart_data.sort_values(selected_metric),
        x=selected_metric,
        y="label",
        color="segment" if "segment" in chart_data else None,
        orientation="h",
        title=f"Segment bazlı {selected_metric.replace('_', ' ').upper()}",
        labels={selected_metric: selected_metric.replace("_", " ").upper(), "label": "Model / split"},
    )
    figure.update_layout(height=max(420, min(1000, len(chart_data) * 26 + 150)), showlegend=False)
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def show_overview(sheets: dict[str, pd.DataFrame]) -> None:
    runtime = sheet_frame(sheets, "00_Runtime_Summary", "Runtime_Summary")
    general = sheet_frame(sheets, "18_General_Summary", "General_Summary")
    performance = sheet_frame(sheets, "17_Performance_Summary", "Performance_Summary")
    status = sheet_frame(sheets, "20_Model_Status_Summary", "Model_Status_Summary")

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
