#!/usr/bin/env python3
"""Generate the focused fund propensity modelling notebook."""

from pathlib import Path

import nbformat as nbf


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "fund_propensity_ml.ipynb"


def _cell(cell_type: str, source: str, language: str, cell_id: str):
    cell = nbf.v4.new_markdown_cell(source) if cell_type == "markdown" else nbf.v4.new_code_cell(source)
    cell.metadata.update({"id": cell_id, "language": language})
    return cell


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    })
    notebook.cells = [
        _cell("markdown", "## 0. Chapter - Ornek Veri ile Run\n\nBu bolum, notebook'u denemek icin 150 musterili ve 2025.09-2026.06 donemini kapsayan bes manuel tabloyu `propensity_example_data` klasorune yazar. Kod hucresini calistirdiktan sonra diger hucresileri sirayla calistirabilirsiniz. `PROPENSITY_DATA_ROOT` ortam degiskeni zaten tanimliysa mevcut veri yolu korunur.", "markdown", "cell-00"),
        _cell("code", """import os
from pathlib import Path

import numpy as np
import pandas as pd

EXAMPLE_DATA_ROOT = Path.cwd() / "propensity_example_data"
EXAMPLE_DATA_ROOT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
example_months = pd.period_range("2025-09", "2026-06", freq="M").astype(str)
example_customers = pd.DataFrame({
    "musteri_id": [f"M{i:04d}" for i in range(1, 151)],
})
example_customers["segment"] = ((np.arange(len(example_customers)) % 5) + 1).astype(int)
example_customers["gelir"] = rng.lognormal(mean=10.4, sigma=0.45, size=len(example_customers)).round(2)
example_customers["risk_skoru"] = rng.normal(loc=650, scale=55, size=len(example_customers)).clip(350, 850).round(1)

example_panel = pd.MultiIndex.from_product(
    [example_customers["musteri_id"], example_months],
    names=["musteri_id", "month"],
).to_frame(index=False).merge(example_customers, on="musteri_id", how="left")
example_panel["month_index"] = example_panel["month"].map({month: index for index, month in enumerate(example_months)})
segment_signal = example_panel["segment"].isin([2, 4]).astype(float)
example_panel["ppf_aktif"] = (rng.random(len(example_panel)) < (0.52 + 0.12 * segment_signal)).astype(int)
example_panel["nf_aktif"] = (rng.random(len(example_panel)) < (0.38 + 0.10 * (1 - segment_signal))).astype(int)

input_table = example_panel[["musteri_id", "month", "segment", "gelir", "risk_skoru"]]
activity_table = example_panel[["musteri_id", "month", "ppf_aktif", "nf_aktif"]]

fund_table = example_panel[["musteri_id", "month"]].loc[example_panel.index.repeat(2)].reset_index(drop=True)
fund_table["para_flg"] = np.tile([1, 0], len(example_panel))
fund_table["tutar"] = np.where(
    fund_table["para_flg"].eq(1),
    rng.gamma(shape=3.0, scale=18000, size=len(fund_table)),
    rng.gamma(shape=2.5, scale=12000, size=len(fund_table)),
).round(2)

transaction_table = fund_table[["musteri_id", "month", "para_flg"]].rename(columns={"para_flg": "ppf_flg"})
transaction_table["alim_tutari"] = rng.gamma(shape=1.8, scale=900, size=len(transaction_table)).round(2)
transaction_table["satim_tutari"] = rng.gamma(shape=1.4, scale=650, size=len(transaction_table)).round(2)
transaction_table.loc[transaction_table["month"].eq("2026-06"), "alim_tutari"] *= 1.8

inflation_table = pd.DataFrame({
    "month": example_months,
    "inflation_rate": [0.025, 0.028, 0.030, 0.027, 0.024, 0.022, 0.021, 0.020, 0.019, 0.018],
})

for filename, frame in {
    "input.csv": input_table,
    "aktiflik.csv": activity_table,
    "fon_tutar.csv": fund_table,
    "alim_satim.csv": transaction_table,
    "enflasyon.csv": inflation_table,
}.items():
    frame.to_csv(EXAMPLE_DATA_ROOT / filename, index=False)

os.environ.setdefault("PROPENSITY_DATA_ROOT", str(EXAMPLE_DATA_ROOT))
print("Ornek veri klasoru:", EXAMPLE_DATA_ROOT.resolve())
print("Olusturulan tablolar:", ", ".join(sorted(path.name for path in EXAMPLE_DATA_ROOT.glob("*.csv"))))
print("Ornek satir sayilari:")
display(pd.DataFrame({
    "table": ["input", "activity", "fund", "transaction", "inflation"],
    "rows": [len(input_table), len(activity_table), len(fund_table), len(transaction_table), len(inflation_table)],
}))""", "python", "cell-00-code"),
        _cell("markdown", """# Fon Newsell ve Upsell Propensity

Bu notebook yalnizca bes uygulama adimindan olusur: **data load**, **preprocess**, **feature engineering**, **modelling** ve **reporting**.

Her asamada degisken kalitesi, elenen veya donusturulen degiskenler, satir sayilari ve karar gerekceleri notebook output'una yazdirilir. Ayni tablolar detayli bir Excel workbook icinde sheet sheet kaydedilir.

Model evreni iki urun sinifi (`para_piyasasi`, `nitelikli`) ve iki hedeften (`newsell`, `upsell`) olusur. Model tuning icin Optuna TPE sampler, `MedianPruner` ve LightGBM iterasyon pruning kullanilir.""", "markdown", "cell-01"),
        _cell("markdown", "## 1. Data Load\n\nKullanici config'inde belirtilen input, aktivite, fon tutar, alım-satım ve enflasyon tabloları CSV veya Parquet olarak okunur. Tüm tablolar müşteri ID + ay anahtarıyla 2025.09-2026.06 aylık döneminde bağlanır; sentetik fallback yoktur. Rare-event hedeflerde test ve OOT performansı iki aylık pencerelerde toplanır; her split için minimum pozitif sayısı kontrol edilir.", "markdown", "cell-02"),
        _cell("code", """import os
import time
from pathlib import Path

import pandas as pd

from fund_propensity import (
    PropensityConfig,
    build_canonical_panel,
    build_feature_engineering_audit,
    build_general_summary,
    build_inflation_audit,
    build_model_performance_summary,
    build_model_status_summary,
    build_overfit_audit,
    build_runtime_summary,
    build_feature_table,
    build_pipeline_audit,
    build_target_table,
    load_bundle_from_directory,
    rank_campaign,
    run_optuna_grid,
    create_performance_figures,
    save_performance_charts,
    variable_quality_report,
    validate_sources,
    write_pipeline_excel,
)

RUN_STARTED = time.perf_counter()

def start_stage(stage: str) -> float:
    print(f"[{stage}] başladı")
    return time.perf_counter()

def log_stage(stage: str, started: float) -> None:
    print(f"[{stage}] tamamlandı | süre={time.perf_counter() - started:.2f} sn")

DATA_ROOT = Path(os.getenv("PROPENSITY_DATA_ROOT", "C:/data/fund_propensity"))

# KULLANICI CONFIG: Model grid'ini ve veri yeterlilik kosullarini burada degistirin.
# Her x/y/r degeri test edilir. Dosya yolları ve kolonlar aşağıdaki manuel config'ten değiştirilir.
X_GRID = (1, 3, 6)
Y_GRID = (1, 3)
R_GRID = (0.10, 0.25, 0.50)
PRODUCT_CLASSES = ("para_piyasasi", "nitelikli")
SEGMENT_COLUMN = "segment"
SEGMENT_VALUES = (1, 2, 3, 4, 5)
INPUT_TABLE_FILE = "input.csv"
ACTIVITY_TABLE_FILE = "aktiflik.csv"
FUND_TABLE_FILE = "fon_tutar.csv"
TRANSACTION_TABLE_FILE = "alim_satim.csv"
INFLATION_TABLE_FILE = "enflasyon.csv"
MIN_ELIGIBLE = 100
MIN_POSITIVE = 10
MIN_POSITIVE_TRAIN = 5
MIN_POSITIVE_TEST = 1
MIN_POSITIVE_OOT = 1
TEST_MONTHS = 2
OOT_MONTHS = 2
TOP_K = (0.001, 0.005, 0.01, 0.05)
RARE_EVENT_RATE_THRESHOLD = 0.01
CAMPAIGN_CAPACITY = 1000
RANDOM_SEED = 42
MISSING_THRESHOLD = 0.95
CORRELATION_THRESHOLD = 0.95
CORRELATION_SAMPLE_SIZE = 50000
MAX_CATEGORICAL_LEVELS = 100
MAX_CATEGORICAL_RATIO = 0.50
OUTLIER_LOWER_QUANTILE = 0.01
OUTLIER_UPPER_QUANTILE = 0.99
ADD_MISSING_INDICATORS = True
INFLATION_REFERENCE_MONTH = None
INFLATION_ADJUST_ALL_CONTINUOUS = True
INPUT_TABLE_CUSTOMER_COLUMN = "musteri_id"
INPUT_TABLE_DATE_COLUMN = "month"
INPUT_TABLE_FEATURE_COLUMNS = None
ACTIVITY_TABLE_CUSTOMER_COLUMN = "musteri_id"
ACTIVITY_TABLE_DATE_COLUMN = "month"
ACTIVITY_PPF_FLAG_COLUMN = "ppf_aktif"
ACTIVITY_NF_FLAG_COLUMN = "nf_aktif"
FUND_TABLE_CUSTOMER_COLUMN = "musteri_id"
FUND_TABLE_DATE_COLUMN = "month"
FUND_TABLE_PRODUCT_FLAG_COLUMN = "para_flg"
FUND_TABLE_VALUE_COLUMN = "tutar"
TRANSACTION_TABLE_CUSTOMER_COLUMN = "musteri_id"
TRANSACTION_TABLE_DATE_COLUMN = "month"
TRANSACTION_TABLE_PRODUCT_FLAG_COLUMN = "ppf_flg"
TRANSACTION_TABLE_BUY_COLUMN = "alim_tutari"
TRANSACTION_TABLE_SELL_COLUMN = "satim_tutari"
INFLATION_DATE_COLUMN = "month"
INFLATION_VALUE_COLUMN = "inflation_rate"
FEATURE_WINDOWS = (1, 3, 6, 12)

CONFIG = PropensityConfig(
    x_grid=X_GRID,
    y_grid=Y_GRID,
    r_grid=R_GRID,
    product_classes=PRODUCT_CLASSES,
    segment_column=SEGMENT_COLUMN,
    segment_values=SEGMENT_VALUES,
    top_k=TOP_K,
    rare_event_rate_threshold=RARE_EVENT_RATE_THRESHOLD,
    min_eligible=MIN_ELIGIBLE,
    min_positive=MIN_POSITIVE,
    min_positive_train=MIN_POSITIVE_TRAIN,
    min_positive_test=MIN_POSITIVE_TEST,
    min_positive_oot=MIN_POSITIVE_OOT,
    test_months=TEST_MONTHS,
    oot_months=OOT_MONTHS,
    random_seed=RANDOM_SEED,
    campaign_capacity=CAMPAIGN_CAPACITY,
    missing_threshold=MISSING_THRESHOLD,
    correlation_threshold=CORRELATION_THRESHOLD,
    correlation_sample_size=CORRELATION_SAMPLE_SIZE,
    max_categorical_levels=MAX_CATEGORICAL_LEVELS,
    max_categorical_ratio=MAX_CATEGORICAL_RATIO,
    outlier_lower_quantile=OUTLIER_LOWER_QUANTILE,
    outlier_upper_quantile=OUTLIER_UPPER_QUANTILE,
    add_missing_indicators=ADD_MISSING_INDICATORS,
    inflation_reference_month=INFLATION_REFERENCE_MONTH,
    inflation_adjust_all_continuous=INFLATION_ADJUST_ALL_CONTINUOUS,
    input_table_file=INPUT_TABLE_FILE,
    input_table_customer_column=INPUT_TABLE_CUSTOMER_COLUMN,
    input_table_date_column=INPUT_TABLE_DATE_COLUMN,
    input_table_feature_columns=INPUT_TABLE_FEATURE_COLUMNS,
    activity_table_file=ACTIVITY_TABLE_FILE,
    activity_table_customer_column=ACTIVITY_TABLE_CUSTOMER_COLUMN,
    activity_table_date_column=ACTIVITY_TABLE_DATE_COLUMN,
    activity_ppf_flag_column=ACTIVITY_PPF_FLAG_COLUMN,
    activity_nf_flag_column=ACTIVITY_NF_FLAG_COLUMN,
    fund_table_file=FUND_TABLE_FILE,
    fund_table_customer_column=FUND_TABLE_CUSTOMER_COLUMN,
    fund_table_date_column=FUND_TABLE_DATE_COLUMN,
    fund_table_product_flag_column=FUND_TABLE_PRODUCT_FLAG_COLUMN,
    fund_table_value_column=FUND_TABLE_VALUE_COLUMN,
    transaction_table_file=TRANSACTION_TABLE_FILE,
    transaction_table_customer_column=TRANSACTION_TABLE_CUSTOMER_COLUMN,
    transaction_table_date_column=TRANSACTION_TABLE_DATE_COLUMN,
    transaction_table_product_flag_column=TRANSACTION_TABLE_PRODUCT_FLAG_COLUMN,
    transaction_table_buy_column=TRANSACTION_TABLE_BUY_COLUMN,
    transaction_table_sell_column=TRANSACTION_TABLE_SELL_COLUMN,
    inflation_table_file=INFLATION_TABLE_FILE,
    inflation_date_column=INFLATION_DATE_COLUMN,
    inflation_value_column=INFLATION_VALUE_COLUMN,
    feature_windows=FEATURE_WINDOWS,
)
OPTUNA_TRIALS = int(os.getenv("PROPENSITY_OPTUNA_TRIALS", "30"))
OPTUNA_TIMEOUT_SECONDS = int(os.getenv("PROPENSITY_OPTUNA_TIMEOUT_SECONDS", "600"))
MAX_CONFIGURATIONS = os.getenv("PROPENSITY_MAX_CONFIGURATIONS")
MAX_CONFIGURATIONS = int(MAX_CONFIGURATIONS) if MAX_CONFIGURATIONS else None
stage_started = start_stage("Data Load")
BUNDLE = load_bundle_from_directory(DATA_ROOT, CONFIG)
QUALITY = validate_sources(BUNDLE, CONFIG)
DATA_LOAD_QUALITY = pd.concat([
    variable_quality_report(frame, "data_load", name)
    for name, frame in {
        "customers": BUNDLE.customers,
        "activity": BUNDLE.activity,
        "flows": BUNDLE.flows,
        "monthly_features": BUNDLE.monthly_features,
        "inflation": BUNDLE.inflation,
    }.items() if not frame.empty
], ignore_index=True)
print("Data source:", DATA_ROOT.resolve())
print("Source summary:")
print(pd.DataFrame([QUALITY]).to_string(index=False))
print("Variable quality - data_load:")
display(DATA_LOAD_QUALITY)
log_stage("Data Load", stage_started)""", "python", "cell-03"),
        _cell("markdown", "## 2. Preprocess\n\nKaynaklar tekil musteri-ay-urun anahtarinda birlestirilir. Geniş aktivite tablosundaki `ppf_aktif` ve `nf_aktif` flag'leri `para_piyasasi` ve `nitelikli` satirlarina acilir. Eksik aylardaki pasif durumlar dense panel ile sifirlanir; gelecek penceresi kapanmamis anchor aylar target fabrikasi tarafindan eligible disi birakilir.", "markdown", "cell-04"),
        _cell("code", """stage_started = start_stage("Preprocess")
panel = build_canonical_panel(BUNDLE, CONFIG)
targets = build_target_table(panel, CONFIG)
PREPROCESS_QUALITY = variable_quality_report(panel, "preprocess", "canonical_panel")
PREPROCESS_ACTIONS = pd.DataFrame([
    {"stage": "preprocess", "variable": "ppf_aktif", "action": "transformed", "reason": "para_piyasasi + active_flag satirlarina acildi"},
    {"stage": "preprocess", "variable": "nf_aktif", "action": "transformed", "reason": "nitelikli + active_flag satirlarina acildi"},
    {"stage": "preprocess", "variable": "active_flag", "action": "retained", "reason": "0/1 aylik aktiflik bilgisi"},
    {"stage": "preprocess", "variable": "missing_month_rows", "action": "filled_zero", "reason": "dense customer-month-product panel"},
])
print("Canonical panel:", panel.shape)
print("Target rows:", targets.shape)
print("Panel key duplicates:", panel.duplicated(["musteri_id", "month", "product_class"]).sum())
print("Variable quality - preprocess:")
display(PREPROCESS_QUALITY)
print("Preprocess actions:")
display(PREPROCESS_ACTIONS)
log_stage("Preprocess", stage_started)""", "python", "cell-05"),
        _cell("markdown", "## 3. Feature Engineering\n\nFeature'lar anchor ayi ve geriye donuk rolling pencerelerden uretilir. Future alis, satis, aktiflik ve bakiye bilgileri feature setine dahil edilmez. Target'lar nominal is kuralinda korunurken, continuous feature'lar enflasyon tablosu ile ortak OOT referans tarihine tasinir ve bu donusum audit edilir.", "markdown", "cell-06"),
        _cell("code", """stage_started = start_stage("Feature Engineering")
features, feature_lineage = build_feature_table(panel, CONFIG)
INFLATION_AUDIT = build_inflation_audit(panel, features, CONFIG)
FEATURE_QUALITY = variable_quality_report(features, "feature_engineering", "feature_table")
feature_columns = [
    column for column in features.columns
    if column not in ["musteri_id", "anchor_month", "product_class", CONFIG.segment_column]
]
FEATURE_QUALITY["quality_filter_action"] = FEATURE_QUALITY["constant_flag"].map({True: "eliminated_constant", False: "retained_for_model_review"})
FEATURE_QUALITY["quality_filter_reason"] = FEATURE_QUALITY["constant_flag"].map({True: "tek unique deger", False: "constant degil; model pipeline adayi"})
FEATURE_QUALITY["model_feature_action"] = FEATURE_QUALITY.apply(
    lambda row: "eliminated_from_model" if row["constant_flag"] or row["variable"] in ["musteri_id", "anchor_month", "product_class", CONFIG.segment_column] else "retained_for_model",
    axis=1,
)
FEATURE_QUALITY["model_feature_reason"] = FEATURE_QUALITY["model_feature_action"].map({"eliminated_from_model": "constant veya model key kolonu", "retained_for_model": "quality kontrolunu gecti"})
FEATURE_ELIMINATED = FEATURE_QUALITY[FEATURE_QUALITY["model_feature_action"] == "eliminated_from_model"].copy()
selected_feature_columns = FEATURE_QUALITY.loc[FEATURE_QUALITY["model_feature_action"] == "retained_for_model", "variable"].tolist()

assert not features.duplicated(["musteri_id", "anchor_month", "product_class"]).any()
assert feature_lineage["as_of"].eq("anchor_month").all()
print("Feature table:", features.shape)
print("Inflation reference month:", INFLATION_AUDIT["inflation_reference_month"].iloc[0] if not INFLATION_AUDIT.empty else "not available")
print("Inflation-adjusted continuous columns:", INFLATION_AUDIT["adjusted_continuous_columns"].iloc[0] if not INFLATION_AUDIT.empty else "none")
print("Feature count:", len(feature_columns))
print("Feature lineage:")
display(feature_lineage)
print("Feature quality and elimination decisions:")
display(FEATURE_QUALITY)
print("Eliminated feature count:", len(FEATURE_ELIMINATED))
log_stage("Feature Engineering", stage_started)""", "python", "cell-07"),
    _cell("markdown", "## 4. Modelling\n\nHer segment icin bagimsiz `(model_type, product_class, x, y, r)` modeli kurulur. Segment model girdisine feature olarak verilmez; her segmentte feature contract, outlier clipping, missing imputation, encoding ve correlation filtresi yalnizca o segmentin train doneminde fit edilir. Optuna her trial icin train/test, secilen model icin train/test/OOT metriklerini kaydeder.", "markdown", "cell-08"),
        _cell("code", """stage_started = start_stage("Modelling")
metrics, oot_scores, model_registry, optuna_trials = run_optuna_grid(
    targets=targets,
    features=features,
    config=CONFIG,
    feature_columns=selected_feature_columns,
    n_trials=OPTUNA_TRIALS,
    timeout_per_configuration=OPTUNA_TIMEOUT_SECONDS,
    max_configurations=MAX_CONFIGURATIONS,
)
TARGET_WITH_SEGMENT = targets.merge(features[["musteri_id", "anchor_month", "product_class", CONFIG.segment_column]], on=["musteri_id", "anchor_month", "product_class"], how="left", validate="many_to_one")
TARGET_QUALITY = TARGET_WITH_SEGMENT.groupby([CONFIG.segment_column, "model_type", "product_class", "x_window", "y_window", "r_threshold"], dropna=False).agg(
    total_rows=("target", "size"), eligible_count=("eligible", "sum"), target_non_null=("target", "count"), positive_count=("target", "sum"),
).reset_index()
TARGET_QUALITY["target_rate"] = TARGET_QUALITY["positive_count"] / TARGET_QUALITY["target_non_null"].replace(0, pd.NA)
TARGET_QUALITY["rare_event_flag"] = TARGET_QUALITY["target_rate"] <= CONFIG.rare_event_rate_threshold
TARGET_QUALITY["minimum_positive_required"] = CONFIG.min_positive
TARGET_QUALITY["model_ready"] = (TARGET_QUALITY["eligible_count"] >= CONFIG.min_eligible) & (TARGET_QUALITY["positive_count"] >= CONFIG.min_positive)
TARGET_QUALITY["decision_reason"] = TARGET_QUALITY["model_ready"].map({True: "minimum veri kosulu saglandi", False: "minimum veri kosulu saglanmadi"})
expected_grid_count = len(CONFIG.product_classes) * len(CONFIG.x_grid) * len(CONFIG.y_grid) * (1 + len(CONFIG.r_grid))
expected_segment_count = TARGET_WITH_SEGMENT[CONFIG.segment_column].nunique()
assert len(TARGET_QUALITY) == expected_grid_count * expected_segment_count, f"Segment/grid eksik: beklenen {expected_grid_count * expected_segment_count}, bulunan {len(TARGET_QUALITY)}"
FEATURE_ENGINEERING_AUDIT = build_feature_engineering_audit(features, feature_lineage, CONFIG)
MODEL_FEATURE_AUDIT = pd.concat([
    bundle["feature_audit"] for bundle in model_registry.values() if "feature_audit" in bundle
], ignore_index=True) if model_registry else pd.DataFrame()
print("Segment coverage:", sorted(TARGET_WITH_SEGMENT[CONFIG.segment_column].dropna().unique().tolist()))
print("Grid coverage:", len(TARGET_QUALITY), "/", expected_grid_count * expected_segment_count, "segment/configuration rows")
print("Rare-event target summary:")
display(TARGET_QUALITY.groupby(["model_type", "product_class", "rare_event_flag"], dropna=False).agg(
    eligible_count=("eligible_count", "sum"), positive_count=("positive_count", "sum"),
).reset_index())
print("Target quality and model eligibility:")
display(TARGET_QUALITY)
print("Model metrics:")
display(metrics)
print("Optuna trial quality/status:")
display(optuna_trials)
print("Final split performance summary:")
display(metrics.groupby(["segment", "model_key", "evaluation_stage", "split"], dropna=False).first().reset_index() if not metrics.empty else metrics)
print("Feature engineering transformation and segment distribution audit:")
display(FEATURE_ENGINEERING_AUDIT)
print("Train-only model preprocessing audit:")
display(MODEL_FEATURE_AUDIT)
print("Elimination and trimming summary:")
display(MODEL_FEATURE_AUDIT.groupby(["action", "outlier_method"], dropna=False).agg(
    feature_count=("feature", "nunique"), train_outlier_count=("train_outlier_count", "sum")
).reset_index() if not MODEL_FEATURE_AUDIT.empty else MODEL_FEATURE_AUDIT)
print("Model registry entries:", len(model_registry))
log_stage("Modelling", stage_started)""", "python", "cell-09"),
        _cell("markdown", "## 5. Reporting\n\nTum asama auditleri notebook output'una basilir ve tek Excel workbook icinde ayrik sheet'lere kaydedilir. Workbook'ta veri kalitesi, elenen degiskenler, enflasyon donusumu, target kalitesi, model metrikleri, Optuna trial'lari, OOT skorlar, kampanya listesi ve teknik/genel performans chart verileri bulunur. Teknik ve genel chart'lar notebook output'unda goruntulenir, PNG olarak kaydedilir ve Excel sheet'lerine gomulur.", "markdown", "cell-10"),
        _cell("code", """stage_started = start_stage("Reporting")
OUTPUT_DIR = Path(os.getenv("PROPENSITY_OUTPUT_DIR", "propensity_outputs")); OUTPUT_DIR.mkdir(parents=True, exist_ok=True); REPORT_NAME = os.getenv("PROPENSITY_REPORT_NAME", "fund_propensity_pipeline_audit_segmented.xlsx")

if not oot_scores.empty:
    campaign_scores = rank_campaign(oot_scores, capacity=CONFIG.campaign_capacity)
else:
    campaign_scores = pd.DataFrame()

PERFORMANCE_SUMMARY = build_model_performance_summary(metrics)
MODEL_STATUS_SUMMARY = build_model_status_summary(metrics, TARGET_QUALITY)
OVERFIT_AUDIT = build_overfit_audit(metrics)
GENERAL_SUMMARY = build_general_summary(
    stage_summary=pd.DataFrame(),
    target_quality=TARGET_QUALITY,
    metrics=metrics,
    oot_scores=oot_scores,
)
RUNTIME_SUMMARY = build_runtime_summary(
    config=CONFIG,
    quality=QUALITY,
    panel=panel,
    targets=targets,
    features=features,
    metrics=metrics,
    optuna_trials=optuna_trials,
    model_feature_audit=MODEL_FEATURE_AUDIT,
    overfit_audit=OVERFIT_AUDIT,
    elapsed_seconds=time.perf_counter() - RUN_STARTED,
)
FIGURES = create_performance_figures(PERFORMANCE_SUMMARY, GENERAL_SUMMARY)
display(FIGURES["technical"])
display(FIGURES["general"])
print("Runtime summary:")
display(RUNTIME_SUMMARY)
print("Overfit and rare-event performance audit:")
display(OVERFIT_AUDIT)
print("Segment/model status summary:")
display(MODEL_STATUS_SUMMARY)
CHART_PATHS = save_performance_charts(FIGURES, OUTPUT_DIR)

audit_tables = build_pipeline_audit(
    bundle=BUNDLE,
    panel=panel,
    targets=targets,
    features=features,
    feature_lineage=feature_lineage,
    config=CONFIG,
    selected_feature_columns=selected_feature_columns,
    metrics=metrics,
    optuna_trials=optuna_trials,
    oot_scores=oot_scores,
    campaign_scores=campaign_scores,
    model_feature_audit=MODEL_FEATURE_AUDIT,
    feature_engineering_audit=FEATURE_ENGINEERING_AUDIT,
    inflation_audit=INFLATION_AUDIT,
    performance_summary=PERFORMANCE_SUMMARY,
    general_summary=GENERAL_SUMMARY,
    overfit_audit=OVERFIT_AUDIT,
    runtime_summary=RUNTIME_SUMMARY,
    model_status_summary=MODEL_STATUS_SUMMARY,
)
EXCEL_PATH = write_pipeline_excel(audit_tables, OUTPUT_DIR / REPORT_NAME)

print("Final audit tables:")
for table_name, table in audit_tables.items():
    print(f"\\n--- {table_name} | rows={len(table):,} cols={len(table.columns):,} ---")
    display(table.head(30))
print("Excel workbook:", EXCEL_PATH.resolve())
log_stage("Reporting", stage_started)""", "python", "cell-11"),
    ]
    with OUTPUT.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(f"Notebook olusturuldu: {OUTPUT}")


if __name__ == "__main__":
    build_notebook()
