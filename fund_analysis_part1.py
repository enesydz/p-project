# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 1 — KONFİGÜRASYON · VERİ ÜRETİMİ · ANALİZ MOTORU        ║
# ║  2 Dönem: 2025.09 (Sep) · 2026.03 (Mar)                        ║
# ║  Bu hücreyi bir kez çalıştır — tüm hesaplamalar burada hazırlanır║
# ╚══════════════════════════════════════════════════════════════════╝

# ══════════════════════════════════════════════════════════════════
# § 1 · KÜTÜPHANELER & MASTER CONFIG
# ══════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import math as _math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import seaborn as sns
import networkx as nx
import warnings
from collections import Counter as _Counter

warnings.filterwarnings("ignore")

# ─── 1.1 Analiz Parametreleri ────────────────────────────────────
X_DAYS      = 7
N_MUSTERI   = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─── 1.2 Dönem Tanımları  ▶▶  SADECE 2 DÖNEM ◀◀ ─────────────────
DONEMLER     = ["2025.09", "2026.03"]
DONEM_SIRASI = DONEMLER
DONEM_ARALIK = {
    "2025.09": ("2025-09-01", "2025-09-30"),
    "2026.03": ("2026-03-01", "2026-03-31"),
}
ISLEM_BASLANGIC = pd.Timestamp("2025-08-24")   # 7 gün öncesi pencere
ISLEM_BITIS     = pd.Timestamp("2026-04-07")   # 7 gün sonrası pencere

# ─── 1.3 Segment Konfigürasyonu ──────────────────────────────────
SEGMENT_SIRASI = [
    "Bireysel_Standart", "Bireysel_Premium", "Bireysel_Elite",
    "KOBİ", "KOBİ_Orta", "KOBİ_Büyük",
    "Kurumsal", "Kurumsal_Premium", "Private_Banking", "Ultra_HNW",
]
SEG_ISIMLER = SEGMENT_SIRASI
SEG_ADETLER = [300, 210, 120, 120, 90, 60, 50, 30, 15, 5]

SEG_FON_ORT = {
    "Bireysel_Standart":    15_000,
    "Bireysel_Premium":     75_000,
    "Bireysel_Elite":      300_000,
    "KOBİ":                250_000,
    "KOBİ_Orta":         1_000_000,
    "KOBİ_Büyük":        4_000_000,
    "Kurumsal":          2_000_000,
    "Kurumsal_Premium": 10_000_000,
    "Private_Banking":  25_000_000,
    "Ultra_HNW":        40_000_000,
}
SEG_GUN_ISLEM = {
    "Bireysel_Standart": 0.8,  "Bireysel_Premium": 1.5, "Bireysel_Elite": 2.5,
    "KOBİ": 3.0,               "KOBİ_Orta": 4.5,        "KOBİ_Büyük": 6.0,
    "Kurumsal": 6.0,           "Kurumsal_Premium": 8.0, "Private_Banking": 10.0,
    "Ultra_HNW": 12.0,
}
SINYAL_PROB = 0.35

# ─── 1.4 Ürün & Renk Konfigürasyonu ─────────────────────────────
URUNLER = ["Vadesiz", "Vadeli", "Yatırım", "Döviz", "Kredi"]

_SEG_CLR = ["#2563EB","#7C3AED","#059669","#DC2626","#D97706",
            "#0891B2","#BE185D","#7C2D12","#1D4ED8","#065F46"]
_SEG_CLR_LITE = ["#BFDBFE","#DDD6FE","#A7F3D0","#FECACA","#FDE68A",
                 "#CFFAFE","#FBCFE8","#FED7AA","#DBEAFE","#D1FAE5"]
SEG_RENK      = {s: _SEG_CLR[i]      for i, s in enumerate(SEGMENT_SIRASI)}
SEG_RENK_ACIK = {s: _SEG_CLR_LITE[i] for i, s in enumerate(SEGMENT_SIRASI)}
SEG_RENK_LIST = [SEG_RENK[s] for s in SEGMENT_SIRASI]

URUN_RENKLER = {
    "Vadesiz": "#3B82F6", "Vadeli": "#F59E0B",
    "Yatırım": "#10B981", "Döviz":  "#8B5CF6", "Kredi": "#EF4444",
}
URUN_RENK_LIST = [URUN_RENKLER[u] for u in URUNLER]
YON_RENK = {"Giriş": "#059669", "Çıkış": "#DC2626"}

# ─── 1.5 Bakiye Dilimi ───────────────────────────────────────────
BAKIYE_PERCENTIL        = [0.33, 0.67]
BAKIYE_DILIM_ETIKETLERI = ["Düşük", "Orta", "Yüksek"]
BAKIYE_DILIMLERI        = BAKIYE_DILIM_ETIKETLERI
_BAKIYE_CLR = ["#FCA5A5","#93C5FD","#6EE7B7","#FDE68A","#DDD6FE",
               "#A7F3D0","#FBCFE8","#FED7AA","#BAE6FD","#BBF7D0"]
BAKIYE_RENK = {l: _BAKIYE_CLR[i] for i, l in enumerate(BAKIYE_DILIM_ETIKETLERI)}

# ─── 1.6 Grafik Stil ─────────────────────────────────────────────
FIG_BG = "#FFFFFF"; AXES_BG = "#F8FAFC"; GRID_CLR = "#E2E8F0"
TEXT_CLR = "#1E293B"; SPINE_CLR = "#94A3B8"; TITLE_CLR = "#0F172A"

def style_axes(ax, grid=True, despine=True):
    ax.set_facecolor(AXES_BG)
    if grid:
        ax.grid(True, color=GRID_CLR, linewidth=0.6, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
    if despine:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SPINE_CLR); ax.spines["bottom"].set_color(SPINE_CLR)
    else:
        for sp in ax.spines.values(): sp.set_color(SPINE_CLR); sp.set_linewidth(0.7)
    ax.tick_params(colors=TEXT_CLR, labelsize=8.5)
    ax.xaxis.label.set_color(TEXT_CLR); ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TITLE_CLR)

def style_fig(fig, title=None, subtitle=None):
    fig.patch.set_facecolor(FIG_BG)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", color=TITLE_CLR,
                     x=0.5, y=0.99, va="top")
    if subtitle:
        fig.text(0.5, 0.965, subtitle, ha="center", fontsize=9.5,
                 color=SPINE_CLR, style="italic")

def _short_label(s, maxlen=10):
    for o, n in [("Bireysel_","Bir."),("_Standart","_Std"),("_Premium","_Prm"),
                 ("_Elite","_Elt"),("KOBİ_Orta","KOBI.Ort"),("KOBİ_Büyük","KOBI.Byk"),
                 ("Kurumsal","Kur."),("Private_Banking","Priv.Bnk"),("Ultra_HNW","Ultra")]:
        s = s.replace(o, n)
    return s[:maxlen]

SEG_SHORT = [_short_label(s) for s in SEGMENT_SIRASI]

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,"axes.titlesize":11,
    "axes.titleweight":"bold","axes.titlepad":10,"axes.labelsize":9,
    "xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8.5,
    "legend.framealpha":0.93,"legend.edgecolor":"#94A3B8","legend.borderpad":0.6,
    "figure.dpi":110,"savefig.dpi":150,"figure.facecolor":FIG_BG,
})

SHOW_ANNOTATIONS = True; SHOW_CI = True; SAVE_CHARTS = False; CHART_DIR = "charts"
NET_MIN_EDGE_PCT = 3.0; NET_NODE_SCALE = 5000; NET_EDGE_SCALE = 15
NET_SEQ_MIN_EDGE = 10;  NET_SEQ_MIN_PCT = 2.0
DF_ALIM = "alim_df"; DF_SATIM = "satim_df"; DF_ISLEM = "islem_df"
DF_BAKIYE = "bakiye_df"; DF_MUSTERI = "musteri_df"

print("✅ § 1 Config yüklendi")
print(f"   Dönemler: {DONEM_SIRASI}")
print(f"   Segmentler ({len(SEGMENT_SIRASI)}): {', '.join(SEGMENT_SIRASI)}")


# ══════════════════════════════════════════════════════════════════
# § 2 · VERİ ÜRETİMİ
# ══════════════════════════════════════════════════════════════════
np.random.seed(RANDOM_SEED)

musteri_ids    = [f"MUS{str(i).zfill(5)}" for i in range(1, N_MUSTERI + 1)]
segment_dizisi = np.repeat(SEG_ISIMLER, SEG_ADETLER)
musteri_df     = pd.DataFrame({"musteri_id": musteri_ids, "musteri_segmenti": segment_dizisi})
seg_map        = musteri_df.set_index("musteri_id")["musteri_segmenti"].to_dict()

def lognorm_tutar(ort_dizi, sigma=0.55):
    ort_dizi = np.maximum(ort_dizi, 1.0)
    return np.round(
        np.exp(np.log(ort_dizi) + np.random.randn(len(ort_dizi)) * sigma) / 100) * 100

# ── Alım DF ──────────────────────────────────────────────────────
alim_kayitlar = []
for donem in DONEMLER:
    bas = pd.Timestamp(DONEM_ARALIK[donem][0])
    bit = pd.Timestamp(DONEM_ARALIK[donem][1])
    gun_sayisi = (bit - bas).days + 1
    for seg in SEG_ISIMLER:
        seg_musteriler = musteri_df[musteri_df["musteri_segmenti"] == seg]["musteri_id"].values
        n_katilan = int(len(seg_musteriler) * np.random.uniform(0.55, 0.65))
        katilan   = np.random.choice(seg_musteriler, size=n_katilan, replace=False)
        for m in katilan:
            n_islem  = np.random.randint(1, 5)
            gun_ofset = np.random.randint(0, gun_sayisi, size=n_islem)
            tarihler  = bas + pd.to_timedelta(gun_ofset, unit="D")
            tutarlar  = lognorm_tutar(np.full(n_islem, SEG_FON_ORT[seg]))
            for k in range(n_islem):
                alim_kayitlar.append({"musteri_id": m, "tarih": tarihler[k],
                    "donem": donem, "alim_tutari": tutarlar[k],
                    "islem_adeti": 1, "alim_flg": 1})

alim_df = (pd.DataFrame(alim_kayitlar)
           .assign(tarih=lambda d: pd.to_datetime(d["tarih"]))
           .sort_values(["musteri_id","tarih"]).reset_index(drop=True))

# ── Satım DF ─────────────────────────────────────────────────────
satim_kayitlar = []
for donem in DONEMLER:
    bit       = pd.Timestamp(DONEM_ARALIK[donem][1])
    alim_uniq = (alim_df[alim_df["donem"] == donem]
                 .sort_values("tarih").drop_duplicates("musteri_id", keep="first"))
    satim_yapanlar = alim_uniq.sample(frac=0.40, random_state=RANDOM_SEED)
    for _, row in satim_yapanlar.iterrows():
        m_id = row["musteri_id"]; seg = seg_map[m_id]
        base_t = row["tarih"]; kalan = int((bit - base_t).days)
        if kalan < 1: continue
        max_offset = min(kalan, 45); n_islem = np.random.randint(1, 4)
        tutarlar = lognorm_tutar(
            np.full(n_islem, SEG_FON_ORT[seg] * np.random.uniform(0.60, 0.90)))
        for k in range(n_islem):
            offset = np.random.randint(1, max_offset + 1)
            t = min(base_t + pd.Timedelta(days=int(offset)), bit)
            satim_kayitlar.append({"musteri_id": m_id, "tarih": t, "donem": donem,
                "satim_tutari": tutarlar[k], "islem_adeti": 1, "satim_flg": 1})

satim_df = (pd.DataFrame(satim_kayitlar)
            .assign(tarih=lambda d: pd.to_datetime(d["tarih"]))
            .sort_values(["musteri_id","tarih"]).reset_index(drop=True))

# ── İşlem DF ─────────────────────────────────────────────────────
URUN_ISLEM = pd.DataFrame([
    ("Vadesiz","EFT","Giriş",0.15),("Vadesiz","EFT","Çıkış",0.15),
    ("Vadesiz","Havale","Giriş",0.08),("Vadesiz","Havale","Çıkış",0.08),
    ("Vadesiz","ATM_Çekim","Çıkış",0.06),("Vadesiz","Fatura_Ödeme","Çıkış",0.05),
    ("Vadeli","Vadeli_Açılış","Çıkış",0.04),("Vadeli","Vadeli_Kapanış","Giriş",0.04),
    ("Yatırım","Hisse_Alım","Çıkış",0.05),("Yatırım","Hisse_Satım","Giriş",0.04),
    ("Yatırım","TahvilBono_Alım","Çıkış",0.03),("Yatırım","TahvilBono_Satım","Giriş",0.03),
    ("Yatırım","Repo_Giriş","Giriş",0.03),
    ("Döviz","Döviz_Alım","Çıkış",0.06),("Döviz","Döviz_Satım","Giriş",0.05),
    ("Kredi","Kredi_Ödemesi","Çıkış",0.06),("Kredi","Kredi_Kullanımı","Giriş",0.03),
], columns=["urun_grubu","islem_turu","islem_yonu","agirlik"])

AGIRLIKLAR   = (URUN_ISLEM["agirlik"] / URUN_ISLEM["agirlik"].sum()).values
TOPLAM_GUN   = (ISLEM_BITIS - ISLEM_BASLANGIC).days + 1
TARIH_DIZISI = pd.date_range(ISLEM_BASLANGIC, ISLEM_BITIS, freq="D")

lambda_dizisi = musteri_df["musteri_segmenti"].map(SEG_GUN_ISLEM).values
toplam_islem  = np.random.poisson(lambda_dizisi * TOPLAM_GUN)
musteri_rep   = musteri_df.loc[musteri_df.index.repeat(toplam_islem)].reset_index(drop=True)
N_islem       = len(musteri_rep)

rand_tarih_idx = np.random.randint(0, TOPLAM_GUN, size=N_islem)
rand_tarihler  = TARIH_DIZISI[rand_tarih_idx]
tx_idx         = np.random.choice(len(URUN_ISLEM), size=N_islem, p=AGIRLIKLAR)
seg_ort_arr    = musteri_rep["musteri_segmenti"].map(
    {s: SEG_FON_ORT[s] * 0.10 for s in SEG_FON_ORT}).values
tutarlar_arr   = lognorm_tutar(seg_ort_arr, sigma=0.6)

islem_df = pd.DataFrame({
    "musteri_id":       musteri_rep["musteri_id"].values,
    "musteri_segmenti": musteri_rep["musteri_segmenti"].values,
    "islem_tarihi":     rand_tarihler,
    "islem_yonu":       URUN_ISLEM["islem_yonu"].values[tx_idx],
    "urun_grubu":       URUN_ISLEM["urun_grubu"].values[tx_idx],
    "islem_turu":       URUN_ISLEM["islem_turu"].values[tx_idx],
    "islem_tutari":     tutarlar_arr,
    "islem_adeti":      1,
})

# Davranışsal sinyal
def sinyal_uret(event_df, gun_ofset, yon, urun, tur, tarih_kolon="tarih"):
    tmp = event_df[["musteri_id", tarih_kolon]].copy()
    tmp["islem_tarihi"] = pd.to_datetime(tmp[tarih_kolon]) + pd.Timedelta(days=gun_ofset)
    tmp = tmp[(tmp["islem_tarihi"] >= ISLEM_BASLANGIC) &
              (tmp["islem_tarihi"] <= ISLEM_BITIS)].copy()
    if tmp.empty: return pd.DataFrame()
    mask = np.random.random(len(tmp)) < SINYAL_PROB
    tmp  = tmp[mask].copy()
    if tmp.empty: return pd.DataFrame()
    tmp["musteri_segmenti"] = tmp["musteri_id"].map(seg_map)
    tmp["islem_yonu"] = yon; tmp["urun_grubu"] = urun
    tmp["islem_turu"] = tur; tmp["islem_adeti"] = 1
    seg_ort = tmp["musteri_segmenti"].map(
        {s: SEG_FON_ORT[s] * 0.08 for s in SEG_FON_ORT}).values
    tmp["islem_tutari"] = lognorm_tutar(seg_ort, sigma=0.5)
    return tmp[["musteri_id","musteri_segmenti","islem_tarihi",
                "islem_yonu","urun_grubu","islem_turu","islem_tutari","islem_adeti"]]

sinyal_bloklar = []
for g in range(1, 8):
    b = sinyal_uret(alim_df,  -g, "Giriş", "Vadesiz", "EFT")
    if not b.empty: sinyal_bloklar.append(b)
for g in range(1, 8):
    b = sinyal_uret(satim_df,  g, "Çıkış", "Vadesiz", "EFT")
    if not b.empty: sinyal_bloklar.append(b)
if sinyal_bloklar:
    islem_df = (pd.concat([islem_df] + sinyal_bloklar, ignore_index=True)
                .sort_values(["musteri_id","islem_tarihi"]).reset_index(drop=True))

# ── Bakiye DF ────────────────────────────────────────────────────
net_fon = (alim_df.groupby("musteri_id")["alim_tutari"].sum()
           .sub(satim_df.groupby("musteri_id")["satim_tutari"].sum(), fill_value=0))
bakiye_kayitlar = []
for m_id in musteri_ids:
    seg = seg_map[m_id]
    baz = max(net_fon.get(m_id, 0) * 0.8, SEG_FON_ORT[seg] * 0.5)
    for i, donem in enumerate(DONEMLER):
        carpan     = (1 + 0.03 * i) * np.random.uniform(0.85, 1.20)
        bakiye     = round(baz * carpan / 100) * 100
        bakiye_mtd = round(bakiye * np.random.uniform(0.90, 1.10) / 100) * 100
        bakiye_kayitlar.append({
            "musteri_id": m_id, "tarih": pd.Timestamp(DONEM_ARALIK[donem][1]),
            "donem": donem, "fon_bakiye_tutari": max(bakiye, 0),
            "fon_bakiye_mtd_tutari": max(bakiye_mtd, 0),
        })
bakiye_df = (pd.DataFrame(bakiye_kayitlar)
             .assign(tarih=lambda d: pd.to_datetime(d["tarih"]))
             .sort_values(["musteri_id","tarih"]).reset_index(drop=True))

print(f"\n✅ § 2 Veri üretildi: {len(musteri_df):,} müşteri · {len(islem_df):,} işlem · "
      f"{len(alim_df):,} alım · {len(satim_df):,} satım")


# ══════════════════════════════════════════════════════════════════
# § 3 · VERİ KALİTE KONTROLÜ
# ══════════════════════════════════════════════════════════════════
_ns = globals()
alim_df    = _ns[DF_ALIM];   alim_df["tarih"]         = pd.to_datetime(alim_df["tarih"])
satim_df   = _ns[DF_SATIM];  satim_df["tarih"]        = pd.to_datetime(satim_df["tarih"])
islem_df   = _ns[DF_ISLEM];  islem_df["islem_tarihi"] = pd.to_datetime(islem_df["islem_tarihi"])
bakiye_df  = _ns[DF_BAKIYE]; bakiye_df["tarih"]       = pd.to_datetime(bakiye_df["tarih"])
musteri_df = _ns[DF_MUSTERI]

for _df, _col in [(islem_df,"musteri_segmenti"),(musteri_df,"musteri_segmenti")]:
    if _col in _df.columns:
        _df[_col] = pd.Categorical(_df[_col], categories=SEGMENT_SIRASI, ordered=True)

print(f"\n{'§ 3 Kalite Kontrolü':─<50}")
for isim, df in [("alim_df",alim_df),("satim_df",satim_df),
                  ("islem_df",islem_df),("bakiye_df",bakiye_df)]:
    n_null = df.isnull().sum().sum(); n_dup = df.duplicated().sum()
    print(f"  {'✅' if n_null==0 and n_dup==0 else '⚠️'}  {isim:<12} {len(df):>8,} satır  null={n_null}  dup={n_dup}")


# ══════════════════════════════════════════════════════════════════
# § 4 · ANALİZ ENGINE
# ══════════════════════════════════════════════════════════════════

# §4.1 Olay Tablosu
alim_events  = (alim_df.merge(musteri_df[["musteri_id","musteri_segmenti"]], on="musteri_id")
                .rename(columns={"tarih":"event_tarih","alim_tutari":"event_tutari"})
                .assign(event_type="Alım")
                [["musteri_id","musteri_segmenti","event_tarih","event_tutari","donem","event_type"]])
satim_events = (satim_df.merge(musteri_df[["musteri_id","musteri_segmenti"]], on="musteri_id")
                .rename(columns={"tarih":"event_tarih","satim_tutari":"event_tutari"})
                .assign(event_type="Satım")
                [["musteri_id","musteri_segmenti","event_tarih","event_tutari","donem","event_type"]])
events_df = (pd.concat([alim_events, satim_events]).reset_index(drop=True)
             .assign(event_id=lambda d: d.index))

# §4.2 Bakiye Dilimi
bakiye_donem = (bakiye_df[["musteri_id","donem","fon_bakiye_tutari"]]
                .merge(musteri_df[["musteri_id","musteri_segmenti"]], on="musteri_id")
                .rename(columns={"fon_bakiye_tutari":"ort_bakiye"}))

def dilim_ata(grp):
    thresholds = [grp["ort_bakiye"].quantile(p) for p in BAKIYE_PERCENTIL]
    bins = [-np.inf] + thresholds + [np.inf]
    return pd.cut(grp["ort_bakiye"], bins=bins, labels=BAKIYE_DILIM_ETIKETLERI)

bakiye_donem["bakiye_dilimi"] = (
    bakiye_donem.groupby(["musteri_segmenti","donem"], group_keys=False)
                .apply(dilim_ata).astype(str))
musteri_bakiye_dilim = bakiye_donem[["musteri_id","donem","bakiye_dilimi"]].copy()
DILIM_ESLIKLERI = (
    bakiye_donem.groupby(["musteri_segmenti","donem"], observed=True)["ort_bakiye"]
    .quantile(BAKIYE_PERCENTIL).unstack(level=-1)
    .rename(columns={p: f"p{int(p*100)}" for p in BAKIYE_PERCENTIL}))

# §4.3 Zenginleştirilmiş Olay
events_enriched_df = (
    events_df
    .merge(musteri_bakiye_dilim, on=["musteri_id","donem"], how="left")
    .merge(bakiye_df[["musteri_id","donem","fon_bakiye_tutari"]]
           .rename(columns={"fon_bakiye_tutari":"bakiye"}), on=["musteri_id","donem"], how="left"))
events_enriched_df["bakiye_dilimi"] = events_enriched_df["bakiye_dilimi"].fillna(BAKIYE_DILIM_ETIKETLERI[1])

# §4.4 İşlem Penceresi ±X_DAYS
ev_keys = events_df[["event_id","musteri_id","event_tarih","event_type","donem"]].copy()
islem_ev = islem_df.merge(ev_keys, on="musteri_id")
islem_ev["gun_fark"] = (islem_ev["islem_tarihi"] - islem_ev["event_tarih"]).dt.days
islem_window_df = (islem_ev[islem_ev["gun_fark"].between(-X_DAYS, X_DAYS) &
                             (islem_ev["gun_fark"] != 0)].copy())
islem_window_df["pencere"] = islem_window_df["gun_fark"].apply(lambda g: "Pre" if g < 0 else "Post")

pre_buy   = islem_window_df[(islem_window_df["event_type"]=="Alım")  & (islem_window_df["pencere"]=="Pre")]
post_buy  = islem_window_df[(islem_window_df["event_type"]=="Alım")  & (islem_window_df["pencere"]=="Post")]
pre_sell  = islem_window_df[(islem_window_df["event_type"]=="Satım") & (islem_window_df["pencere"]=="Pre")]
post_sell = islem_window_df[(islem_window_df["event_type"]=="Satım") & (islem_window_df["pencere"]=="Post")]

# §4.5 Genel Metrikler
METRIK_GENEL = (
    events_enriched_df.groupby(["musteri_segmenti","event_type"], observed=True)
    .agg(Olay_Adedi=("event_id","count"), Musteri_Sayisi=("musteri_id","nunique"),
         Toplam_Tutar_M=("event_tutari", lambda x: round(x.sum()/1e6,2)),
         Ort_Tutar_K=("event_tutari", lambda x: round(x.mean()/1e3,1)),
         Medyan_Tutar_K=("event_tutari", lambda x: round(x.median()/1e3,1)))
    .reset_index())

# §4.6 Ürün Dağılımı
def urun_pct(df):
    grp = (df.groupby(["musteri_segmenti","urun_grubu"], observed=True).size().reset_index(name="adet"))
    grp["toplam"] = grp.groupby("musteri_segmenti", observed=True)["adet"].transform("sum")
    grp["pct"]    = grp["adet"] / grp["toplam"] * 100
    return (grp.pivot(index="musteri_segmenti", columns="urun_grubu", values="pct").fillna(0)
              .reindex(SEGMENT_SIRASI).reindex(columns=URUNLER, fill_value=0))

URUN_PRE_BUY   = urun_pct(pre_buy)
URUN_POST_BUY  = urun_pct(post_buy)
URUN_PRE_SELL  = urun_pct(pre_sell)
URUN_POST_SELL = urun_pct(post_sell)

# §4.7 Giriş Oranı & Net Akış
def giris_pct(df):
    return (df.groupby("musteri_segmenti", observed=True)["islem_yonu"]
              .apply(lambda x: (x=="Giriş").mean()*100).round(1).reindex(SEGMENT_SIRASI))
def net_akis(df):
    gin = (df[df["islem_yonu"]=="Giriş"]
           .groupby(["musteri_segmenti","event_id"], observed=True)["islem_tutari"].sum())
    cik = (df[df["islem_yonu"]=="Çıkış"]
           .groupby(["musteri_segmenti","event_id"], observed=True)["islem_tutari"].sum())
    return (gin.sub(cik, fill_value=0).reset_index().rename(columns={"islem_tutari":"net"})
               .groupby("musteri_segmenti", observed=True)["net"].mean().reindex(SEGMENT_SIRASI))

GIRIS_PRE_BUY   = giris_pct(pre_buy);   GIRIS_POST_BUY  = giris_pct(post_buy)
GIRIS_PRE_SELL  = giris_pct(pre_sell);  GIRIS_POST_SELL = giris_pct(post_sell)
NET_PRE_BUY     = net_akis(pre_buy);    NET_POST_BUY    = net_akis(post_buy)
NET_PRE_SELL    = net_akis(pre_sell);   NET_POST_SELL   = net_akis(post_sell)

# §4.8 Bakiye Dilimi × Segment
ev_bd = (events_enriched_df[["event_id","bakiye_dilimi"]].drop_duplicates("event_id")
         .assign(bakiye_dilimi=lambda d: d["bakiye_dilimi"].astype(str)))

def bd_pivot(df, val="islem_tutari"):
    d2 = df.merge(ev_bd, on="event_id", how="left")
    d2["bakiye_dilimi"] = pd.Categorical(
        d2["bakiye_dilimi"].astype(str).fillna(BAKIYE_DILIM_ETIKETLERI[1]),
        categories=BAKIYE_DILIM_ETIKETLERI, ordered=True)
    if val == "count":
        pv = d2.groupby(["musteri_segmenti","bakiye_dilimi"], observed=True).size().unstack(fill_value=0)
    else:
        pv = d2.groupby(["musteri_segmenti","bakiye_dilimi"], observed=True)[val].mean().unstack(fill_value=0)
    return pv.reindex(SEGMENT_SIRASI).reindex(columns=BAKIYE_DILIM_ETIKETLERI, fill_value=0)

BD_ALIM_COUNT  = bd_pivot(pd.concat([pre_buy,  post_buy]),  "count")
BD_ALIM_TUTAR  = bd_pivot(pd.concat([pre_buy,  post_buy]))
BD_SATIM_COUNT = bd_pivot(pd.concat([pre_sell, post_sell]), "count")
BD_SATIM_TUTAR = bd_pivot(pd.concat([pre_sell, post_sell]))

# §4.9 Geçiş Matrisi
def gecis_mat(event_type):
    df_e  = islem_window_df[islem_window_df["event_type"]==event_type]
    pre_u = (df_e[df_e["pencere"]=="Pre"]
             .groupby(["event_id","urun_grubu"], observed=True).size()
             .reset_index(name="n").sort_values("n", ascending=False)
             .drop_duplicates("event_id")[["event_id","urun_grubu"]].rename(columns={"urun_grubu":"pre_urun"}))
    post_u = (df_e[df_e["pencere"]=="Post"]
              .groupby(["event_id","urun_grubu"], observed=True).size()
              .reset_index(name="n").sort_values("n", ascending=False)
              .drop_duplicates("event_id")[["event_id","urun_grubu"]].rename(columns={"urun_grubu":"post_urun"}))
    gecis = pre_u.merge(post_u, on="event_id", how="inner")
    mat   = (gecis.groupby(["pre_urun","post_urun"], observed=True).size()
                  .unstack(fill_value=0)
                  .reindex(index=URUNLER, columns=URUNLER, fill_value=0))
    return (mat.div(mat.sum(axis=1).clip(lower=1), axis=0)*100).round(1)

GECIS_ALIM  = gecis_mat("Alım")
GECIS_SATIM = gecis_mat("Satım")

# §4.10 Sadakat
musteri_donem_alim = alim_df.groupby("musteri_id")["donem"].nunique()
SADAKAT = (
    musteri_donem_alim.reset_index().rename(columns={"donem":"n_donem"})
    .merge(musteri_df, on="musteri_id")
    .groupby("musteri_segmenti", observed=True).agg(
        Ort_Donem_Alim=("n_donem","mean"),
        Cok_Donem_Pct=("n_donem", lambda x: (x > 1).mean()*100),
        Tek_Donem_Pct=("n_donem", lambda x: (x == 1).mean()*100))
    .round(1).reindex(SEGMENT_SIRASI))

# §4.11 Günlük Akış
def gunluk_ortalama(pre_df, post_df):
    pre_g  = pre_df.groupby("gun_fark",  observed=True)["islem_tutari"].mean()
    post_g = post_df.groupby("gun_fark", observed=True)["islem_tutari"].mean()
    return pre_g.combine_first(post_g).reindex(range(-X_DAYS, X_DAYS+1)).fillna(0)

GUNLUK_ALIM  = gunluk_ortalama(pre_buy,  post_buy)
GUNLUK_SATIM = gunluk_ortalama(pre_sell, post_sell)

# §4.12 Behavioral Scoring
freq_pre = (pre_buy.groupby(["musteri_segmenti","event_id"], observed=True).size()
            .reset_index(name="n").groupby("musteri_segmenti", observed=True)["n"].mean()
            .reindex(SEGMENT_SIRASI).fillna(0))
sadakat_pct = (
    musteri_donem_alim.reset_index().rename(columns={"donem":"n_donem"})
    .merge(musteri_df, on="musteri_id")
    .groupby("musteri_segmenti", observed=True)
    .apply(lambda x: (x["n_donem"] > 1).mean()*100).round(1)
    .reindex(SEGMENT_SIRASI).fillna(0))
DAVRANIS_SKOR = pd.DataFrame({
    "Pre_Buy_Giriş_%": GIRIS_PRE_BUY, "Post_Buy_Giriş_%": GIRIS_POST_BUY,
    "Pre_Sell_Giriş_%": GIRIS_PRE_SELL, "Post_Sell_Giriş_%": GIRIS_POST_SELL,
    "Pre_Buy_Frekans": freq_pre, "Sadakat_%": sadakat_pct,
}).round(1)
DAVRANIS_SKOR["Aktivite_Skoru"] = (
    DAVRANIS_SKOR["Pre_Buy_Giriş_%"]*0.25 + DAVRANIS_SKOR["Post_Buy_Giriş_%"]*0.20 +
    DAVRANIS_SKOR["Pre_Buy_Frekans"]*5.0  + DAVRANIS_SKOR["Sadakat_%"]*0.30
).clip(0, 100).round(1)

# §4.13 Penetrasyon
seg_toplam  = musteri_df.groupby("musteri_segmenti", observed=True).size()
alim_katil  = (alim_df.merge(musteri_df, on="musteri_id")
               .groupby("musteri_segmenti", observed=True)["musteri_id"].nunique())
satim_katil = (satim_df.merge(musteri_df, on="musteri_id")
               .groupby("musteri_segmenti", observed=True)["musteri_id"].nunique())
PENETRASYON = pd.DataFrame({
    "Toplam_Musteri": seg_toplam,
    "Alim_Yapan": alim_katil.reindex(SEGMENT_SIRASI),
    "Satim_Yapan": satim_katil.reindex(SEGMENT_SIRASI),
    "Alim_Pct": (alim_katil/seg_toplam*100).reindex(SEGMENT_SIRASI).round(1),
    "Satim_Pct": (satim_katil/seg_toplam*100).reindex(SEGMENT_SIRASI).round(1),
}).reindex(SEGMENT_SIRASI)

# §4.14 AUM
AUM = (alim_df.merge(musteri_df, on="musteri_id")
       .groupby("musteri_segmenti", observed=True)["alim_tutari"].sum()
       .sub(satim_df.merge(musteri_df, on="musteri_id")
            .groupby("musteri_segmenti", observed=True)["satim_tutari"].sum(), fill_value=0)
       .reindex(SEGMENT_SIRASI))

# §4.15 Sekansiyel Ağ
def _compute_pairs(df):
    d = (df[["musteri_id","islem_tarihi","urun_grubu","islem_tutari","musteri_segmenti"]]
         .sort_values(["musteri_id","islem_tarihi"]).reset_index(drop=True))
    d["next_urun"]    = d.groupby("musteri_id", sort=False)["urun_grubu"].shift(-1)
    d["next_musteri"] = d.groupby("musteri_id", sort=False)["musteri_id"].shift(-1)
    return (d.dropna(subset=["next_urun"]).pipe(lambda x: x[x["musteri_id"]==x["next_musteri"]])
             .reset_index(drop=True))

def build_seq_network(pairs_df, min_edge=None):
    if min_edge is None: min_edge = NET_SEQ_MIN_EDGE
    G = nx.DiGraph()
    for u in URUNLER:
        u_rows = pairs_df[pairs_df["urun_grubu"]==u] if not pairs_df.empty else pd.DataFrame()
        G.add_node(u, color=URUN_RENKLER[u], freq=len(u_rows),
                   volume=float(u_rows["islem_tutari"].sum()) if not u_rows.empty else 0.0)
    if pairs_df.empty: return G
    edge_stats = (pairs_df.groupby(["urun_grubu","next_urun"], observed=True)
                  .agg(count=("musteri_id","count"), volume=("islem_tutari","sum")).reset_index())
    edge_stats = edge_stats[edge_stats["count"] >= min_edge].copy()
    total_from = edge_stats.groupby("urun_grubu")["count"].sum()
    for _, row in edge_stats.iterrows():
        pct = row["count"] / max(total_from.get(row["urun_grubu"], 1), 1) * 100
        G.add_edge(row["urun_grubu"], row["next_urun"],
                   weight=int(row["count"]), pct=round(pct,1), volume=float(row["volume"]))
    return G

_pairs_all      = _compute_pairs(islem_df)
G_SEQ_ALL       = build_seq_network(_pairs_all, min_edge=NET_SEQ_MIN_EDGE)
G_SEQ_PRE_BUY   = build_seq_network(_compute_pairs(
    islem_window_df[(islem_window_df["event_type"]=="Alım") &(islem_window_df["pencere"]=="Pre")]) if len(islem_window_df)>=2 else pd.DataFrame(), min_edge=2)
G_SEQ_POST_BUY  = build_seq_network(_compute_pairs(
    islem_window_df[(islem_window_df["event_type"]=="Alım") &(islem_window_df["pencere"]=="Post")]) if len(islem_window_df)>=2 else pd.DataFrame(), min_edge=2)
G_SEQ_PRE_SELL  = build_seq_network(_compute_pairs(
    islem_window_df[(islem_window_df["event_type"]=="Satım")&(islem_window_df["pencere"]=="Pre")]) if len(islem_window_df)>=2 else pd.DataFrame(), min_edge=2)
G_SEQ_POST_SELL = build_seq_network(_compute_pairs(
    islem_window_df[(islem_window_df["event_type"]=="Satım")&(islem_window_df["pencere"]=="Post")]) if len(islem_window_df)>=2 else pd.DataFrame(), min_edge=2)
G_SEQ_SEG = {seg: build_seq_network(
    _pairs_all[_pairs_all["musteri_segmenti"]==seg].reset_index(drop=True), min_edge=3)
    for seg in SEGMENT_SIRASI}

def _net_metrics(G):
    base = {u: {"pagerank":0.0,"betweenness":0.0,"in_deg":0,"out_deg":0} for u in URUNLER}
    if G.number_of_edges()==0: return base
    pr = nx.pagerank(G, weight="weight", max_iter=500, tol=1e-6)
    bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
    return {u: {"pagerank":round(pr.get(u,0),4),"betweenness":round(bc.get(u,0),4),
                "in_deg":G.in_degree(u,weight="weight"),"out_deg":G.out_degree(u,weight="weight")}
            for u in URUNLER}

NET_METRICS    = _net_metrics(G_SEQ_ALL)
NET_METRICS_DF = pd.DataFrame(NET_METRICS).T.reset_index().rename(columns={"index":"urun"})

def _trans_mat(pairs_df):
    if pairs_df.empty:
        return pd.DataFrame(0.0, index=URUNLER, columns=URUNLER)
    mat = (pairs_df.groupby(["urun_grubu","next_urun"], observed=True)
                   .size().unstack(fill_value=0)
                   .reindex(index=URUNLER, columns=URUNLER, fill_value=0))
    return mat.div(mat.sum(axis=1).clip(lower=1), axis=0).round(3)

TRANS_MAT_ALL      = _trans_mat(_pairs_all)
TRANS_MAT_PRE_BUY  = _trans_mat(_compute_pairs(
    islem_window_df[(islem_window_df["event_type"]=="Alım")&(islem_window_df["pencere"]=="Pre")]) if len(islem_window_df)>=2 else pd.DataFrame())
TRANS_MAT_POST_BUY = _trans_mat(_compute_pairs(
    islem_window_df[(islem_window_df["event_type"]=="Alım")&(islem_window_df["pencere"]=="Post")]) if len(islem_window_df)>=2 else pd.DataFrame())

def _top_ngrams(n=3, top_k=15):
    d = (islem_df[["musteri_id","islem_tarihi","urun_grubu"]]
         .sort_values(["musteri_id","islem_tarihi"]).reset_index(drop=True))
    grams = []
    for _, grp in d.groupby("musteri_id", sort=False):
        seq = grp["urun_grubu"].tolist()
        for i in range(len(seq)-n+1):
            grams.append(" → ".join(seq[i:i+n]))
    return pd.Series(_Counter(grams)).sort_values(ascending=False).head(top_k)

TOP_TRIGRAMS = _top_ngrams(n=3, top_k=15)
TOP_BIGRAMS  = (_pairs_all.groupby(["urun_grubu","next_urun"], observed=True)
                .size().reset_index(name="count")
                .assign(label=lambda d: d["urun_grubu"]+" → "+d["next_urun"])
                .sort_values("count", ascending=False).head(15).reset_index(drop=True))

# §4.16 KPI
KPI = {
    "toplam_alim_m":    round(alim_df["alim_tutari"].sum()/1e6, 1),
    "toplam_satim_m":   round(satim_df["satim_tutari"].sum()/1e6, 1),
    "net_aum_m":        round(AUM.sum()/1e6, 1),
    "toplam_event":     len(events_df),
    "alim_musteri":     alim_df["musteri_id"].nunique(),
    "satim_musteri":    satim_df["musteri_id"].nunique(),
    "window_islem":     len(islem_window_df),
    "penetrasyon_ort":  PENETRASYON["Alim_Pct"].mean().round(1),
    "sadakat_ort":      SADAKAT["Cok_Donem_Pct"].mean().round(1),
    "aktivite_max_seg": DAVRANIS_SKOR["Aktivite_Skoru"].idxmax(),
}

print(f"\n{'═'*58}")
print(f"  ✅  §4 Analiz Engine tamamlandı")
print(f"  KPI: Alım {KPI['toplam_alim_m']:.1f}M TL · AUM {KPI['net_aum_m']:.1f}M TL")
print(f"{'═'*58}")


# ══════════════════════════════════════════════════════════════════
# § 4.17–4.20 · DÖNEMSELMETRİKLER · HEAVY · KOMPOZİT AĞ
# ══════════════════════════════════════════════════════════════════

# §4.17 Dönemsel Metrikler
DONEM_ALIM = (alim_df.merge(musteri_df, on="musteri_id")
              .groupby(["donem","musteri_segmenti"], observed=True)
              .agg(Alim_Adet=("alim_tutari","count"),
                   Alim_Tutar_M=("alim_tutari", lambda x: round(x.sum()/1e6,3)),
                   Alim_Musteri=("musteri_id","nunique")).reset_index())
DONEM_SATIM = (satim_df.merge(musteri_df, on="musteri_id")
               .groupby(["donem","musteri_segmenti"], observed=True)
               .agg(Satim_Adet=("satim_tutari","count"),
                    Satim_Tutar_M=("satim_tutari", lambda x: round(x.sum()/1e6,3)),
                    Satim_Musteri=("musteri_id","nunique")).reset_index())

DONEM_ALIM_PIVOT  = (DONEM_ALIM.pivot(index="musteri_segmenti", columns="donem", values="Alim_Tutar_M")
                     .reindex(SEGMENT_SIRASI)[DONEM_SIRASI].fillna(0))
DONEM_SATIM_PIVOT = (DONEM_SATIM.pivot(index="musteri_segmenti", columns="donem", values="Satim_Tutar_M")
                     .reindex(SEGMENT_SIRASI)[DONEM_SIRASI].fillna(0))
DONEM_NET_PIVOT   = DONEM_ALIM_PIVOT.sub(DONEM_SATIM_PIVOT, fill_value=0).round(3)
DONEM_ALIM_POC    = DONEM_ALIM_PIVOT.pct_change(axis=1).multiply(100).round(1)
DONEM_SATIM_POC   = DONEM_SATIM_PIVOT.pct_change(axis=1).multiply(100).round(1)

def _assign_donem_col(tarih_series):
    res = pd.Series(pd.NA, index=tarih_series.index, dtype=object)
    for _d in DONEM_SIRASI:
        _bas = pd.Timestamp(DONEM_ARALIK[_d][0]); _bit = pd.Timestamp(DONEM_ARALIK[_d][1])
        res[(tarih_series >= _bas) & (tarih_series <= _bit)] = _d
    return res

_islem_d = islem_df[["musteri_id","musteri_segmenti","islem_tarihi","islem_tutari","urun_grubu"]].copy()
_islem_d["donem"] = _assign_donem_col(_islem_d["islem_tarihi"])
DONEM_ISLEM_PIVOT = (_islem_d.dropna(subset=["donem"])
                     .groupby(["musteri_segmenti","donem"], observed=True)["islem_tutari"]
                     .sum().div(1e6).round(2).unstack(fill_value=0)
                     .reindex(SEGMENT_SIRASI).reindex(columns=DONEM_SIRASI, fill_value=0))

# §4.18 Heavy Buyer / Seller
alim_mst = (alim_df.merge(musteri_df, on="musteri_id")
            .groupby(["musteri_id","musteri_segmenti","donem"], observed=True)
            ["alim_tutari"].sum().reset_index(name="toplam_alim"))

def _heavy_flag(grp, col):
    grp = grp.copy(); grp["heavy"] = grp[col] >= grp[col].quantile(0.75); return grp

alim_mst  = (alim_mst.groupby(["musteri_segmenti","donem"], observed=True, group_keys=False)
             .apply(lambda g: _heavy_flag(g, "toplam_alim")))
satim_mst = (satim_df.merge(musteri_df, on="musteri_id")
             .groupby(["musteri_id","musteri_segmenti","donem"], observed=True)
             ["satim_tutari"].sum().reset_index(name="toplam_satim"))
satim_mst = (satim_mst.groupby(["musteri_segmenti","donem"], observed=True, group_keys=False)
             .apply(lambda g: _heavy_flag(g, "toplam_satim")))

HEAVY_BUYER_SEG  = (alim_mst.groupby(["musteri_segmenti","donem"], observed=True)["heavy"]
                    .mean().multiply(100).round(1).unstack(fill_value=0)
                    .reindex(SEGMENT_SIRASI).reindex(columns=DONEM_SIRASI, fill_value=0))
HEAVY_SELLER_SEG = (satim_mst.groupby(["musteri_segmenti","donem"], observed=True)["heavy"]
                    .mean().multiply(100).round(1).unstack(fill_value=0)
                    .reindex(SEGMENT_SIRASI).reindex(columns=DONEM_SIRASI, fill_value=0))
HEAVY_BUYER_TUTAR  = (alim_mst.groupby(["musteri_segmenti","heavy"], observed=True)["toplam_alim"]
                      .mean().div(1e3).round(1).unstack().reindex(SEGMENT_SIRASI)
                      .rename(columns={False:"Normal_K", True:"Heavy_K"}))
HEAVY_SELLER_TUTAR = (satim_mst.groupby(["musteri_segmenti","heavy"], observed=True)["toplam_satim"]
                      .mean().div(1e3).round(1).unstack().reindex(SEGMENT_SIRASI)
                      .rename(columns={False:"Normal_K", True:"Heavy_K"}))
_heavy_ids  = set(alim_mst.loc[alim_mst["heavy"]==True, "musteri_id"].tolist())
_normal_ids = set(alim_mst.loc[alim_mst["heavy"]==False, "musteri_id"].tolist())
HEAVY_ISLEM_DONEM = (_islem_d[_islem_d["musteri_id"].isin(_heavy_ids) & _islem_d["donem"].notna()]
                     .groupby(["musteri_segmenti","donem"], observed=True)["islem_tutari"]
                     .mean().div(1e3).round(1).unstack(fill_value=0)
                     .reindex(SEGMENT_SIRASI).reindex(columns=DONEM_SIRASI, fill_value=0))

# §4.19 Kompozit Boyut
islem_df["urun_tur_yon"] = (islem_df["urun_grubu"] + "|" +
                             islem_df["islem_turu"]  + "|" +
                             islem_df["islem_yonu"])
COMPOSITE_FREQ = (islem_df["urun_tur_yon"].value_counts().rename_axis("node")
                  .reset_index(name="count").head(20))

def _short_comp_label(n):
    parts = n.split("|")
    return f"{parts[0][:4]}/{parts[1][:5] if len(parts)>1 else ''}/{parts[2][0] if len(parts)>2 else ''}"

# §4.20 Kompozit Network
def build_composite_net(df, col="urun_tur_yon", min_edge=5):
    d = (df[["musteri_id","islem_tarihi",col,"islem_tutari"]]
         .sort_values(["musteri_id","islem_tarihi"]).reset_index(drop=True))
    d["next_node"] = d.groupby("musteri_id", sort=False)[col].shift(-1)
    d["next_id"]   = d.groupby("musteri_id", sort=False)["musteri_id"].shift(-1)
    pairs = (d.dropna(subset=["next_node"]).pipe(lambda x: x[x["musteri_id"]==x["next_id"]])
              .reset_index(drop=True))
    G = nx.DiGraph()
    if pairs.empty: return G, pairs
    edge_stats = (pairs.groupby([col,"next_node"], observed=True)
                  .agg(count=("musteri_id","count"), volume=("islem_tutari","sum")).reset_index())
    edge_stats = edge_stats[edge_stats["count"] >= min_edge].copy()
    nodes = set(edge_stats[col].tolist() + edge_stats["next_node"].tolist())
    _freq_map = islem_df[col].value_counts().to_dict()
    for n in nodes:
        urun = n.split("|")[0]
        G.add_node(n, color=URUN_RENKLER.get(urun,"#94A3B8"), urun=urun, freq=_freq_map.get(n,0))
    total_from = edge_stats.groupby(col)["count"].sum()
    for _, row in edge_stats.iterrows():
        pct = row["count"] / max(total_from.get(row[col], 1), 1) * 100
        G.add_edge(row[col], row["next_node"],
                   weight=int(row["count"]), pct=round(pct,1), volume=float(row["volume"]))
    return G, pairs

G_COMPOSITE, _composite_pairs = build_composite_net(islem_df, min_edge=8)
G_COMPOSITE_SEG = {}
for _cs in SEGMENT_SIRASI:
    G_COMPOSITE_SEG[_cs], _ = build_composite_net(
        islem_df[islem_df["musteri_segmenti"]==_cs].reset_index(drop=True), min_edge=3)

print(f"\n✅ §4.17-20 Dönemsel metrikler + Heavy + Kompozit ağ")
print(f"   Kompozit: {G_COMPOSITE.number_of_nodes()} node · {G_COMPOSITE.number_of_edges()} kenar")
print(f"   Heavy: {len(_heavy_ids):,} heavy buyer · {len(_normal_ids):,} normal")


# ══════════════════════════════════════════════════════════════════
# § 5 · t-1 BAKİYE HESAPLAMALARI
# ══════════════════════════════════════════════════════════════════
# 2 dönemde t-1: 2026.03 için önceki dönem 2025.09
_prev_donem_map = {d: DONEM_SIRASI[i-1]
                   for i, d in enumerate(DONEM_SIRASI) if i > 0}
_donems_with_prev = list(_prev_donem_map.keys())   # ["2026.03"]

_bak_t1_parts = []
for curr_d, prev_d in _prev_donem_map.items():
    _sub = (bakiye_df[bakiye_df["donem"]==prev_d]
            [["musteri_id","fon_bakiye_tutari"]].copy()
            .rename(columns={"fon_bakiye_tutari":"bakiye_t1"}))
    _sub["donem"] = curr_d
    _bak_t1_parts.append(_sub)

bakiye_t1_map = (pd.concat(_bak_t1_parts, ignore_index=True) if _bak_t1_parts
                 else pd.DataFrame(columns=["musteri_id","bakiye_t1","donem"]))

alim_d_mst = (alim_df.merge(musteri_df[["musteri_id","musteri_segmenti"]], on="musteri_id")
              .groupby(["musteri_id","musteri_segmenti","donem"], observed=True)
              ["alim_tutari"].sum().reset_index(name="alim_tutari"))
satim_d_mst = (satim_df.merge(musteri_df[["musteri_id","musteri_segmenti"]], on="musteri_id")
               .groupby(["musteri_id","musteri_segmenti","donem"], observed=True)
               ["satim_tutari"].sum().reset_index(name="satim_tutari"))

alim_t1 = (alim_d_mst[alim_d_mst["donem"].isin(_donems_with_prev)]
           .merge(bakiye_t1_map, on=["musteri_id","donem"], how="left")
           .dropna(subset=["bakiye_t1"]))
alim_t1["alim_bak_pct"] = (
    alim_t1["alim_tutari"] / alim_t1["bakiye_t1"].clip(lower=1) * 100).clip(0, 500).round(2)

satim_t1 = (satim_d_mst[satim_d_mst["donem"].isin(_donems_with_prev)]
            .merge(bakiye_t1_map, on=["musteri_id","donem"], how="left")
            .dropna(subset=["bakiye_t1"]))
satim_t1["satim_bak_pct"] = (
    satim_t1["satim_tutari"] / satim_t1["bakiye_t1"].clip(lower=1) * 100).clip(0, 500).round(2)

def _seg_donem_hm(df, val_col):
    return (df.groupby(["musteri_segmenti","donem"], observed=True)[val_col]
            .median().round(2).unstack(fill_value=0)
            .reindex(SEGMENT_SIRASI).reindex(columns=_donems_with_prev, fill_value=0))

ALIM_BAK_HM   = _seg_donem_hm(alim_t1,  "alim_bak_pct")
SATIM_BAK_HM  = _seg_donem_hm(satim_t1, "satim_bak_pct")
ALIM_BAK_DELTA  = ALIM_BAK_HM.diff(axis=1).fillna(0).round(2)
SATIM_BAK_DELTA = SATIM_BAK_HM.diff(axis=1).fillna(0).round(2)
print(f"\n✅ § 5 t-1 Bakiye hesaplandı  |  Dönem: {_donems_with_prev}")


# ══════════════════════════════════════════════════════════════════
# § 6 · FON-GEÇMEYEN İŞLEM SEKANS HESAPLAMALARI
# ══════════════════════════════════════════════════════════════════
FON_GECEN_TURLER    = {"EFT","Havale","ATM_Çekim","Fatura_Ödeme","Kredi_Ödemesi","Kredi_Kullanımı"}
FON_GECMEYEN_URUNLER = {"Vadeli","Yatırım","Döviz"}
FG_TURLER = ["Vadeli_Açılış","Vadeli_Kapanış","Hisse_Alım","Hisse_Satım",
             "TahvilBono_Alım","TahvilBono_Satım","Repo_Giriş","Döviz_Alım","Döviz_Satım"]
FG_TURLER_KISALT = {
    "Vadeli_Açılış":"Vdl.Açılış","Vadeli_Kapanış":"Vdl.Kapanış",
    "Hisse_Alım":"Hisse.Alım","Hisse_Satım":"Hisse.Satım",
    "TahvilBono_Alım":"TahvBono.A","TahvilBono_Satım":"TahvBono.S",
    "Repo_Giriş":"Repo.Giriş","Döviz_Alım":"Döviz.Alım","Döviz_Satım":"Döviz.Satım",
}

def _fg_filtre(df):
    return df[df["urun_grubu"].isin(FON_GECMEYEN_URUNLER) &
              (~df["islem_turu"].isin(FON_GECEN_TURLER))].copy()

fg_pre_buy   = _fg_filtre(pre_buy)
fg_post_buy  = _fg_filtre(post_buy)
fg_pre_sell  = _fg_filtre(pre_sell)
fg_post_sell = _fg_filtre(post_sell)

def _seg_tur_pct(df):
    if df.empty: return pd.DataFrame(0.0, index=SEGMENT_SIRASI, columns=FG_TURLER)
    sub = df[df["islem_turu"].isin(FG_TURLER)].copy()
    if sub.empty: return pd.DataFrame(0.0, index=SEGMENT_SIRASI, columns=FG_TURLER)
    grp = sub.groupby(["musteri_segmenti","islem_turu"], observed=True).size().reset_index(name="n")
    grp["toplam"] = grp.groupby("musteri_segmenti", observed=True)["n"].transform("sum")
    grp["pct"]    = (grp["n"] / grp["toplam"].clip(lower=1) * 100).round(1)
    return (grp.pivot(index="musteri_segmenti", columns="islem_turu", values="pct")
            .fillna(0).reindex(SEGMENT_SIRASI).reindex(columns=FG_TURLER, fill_value=0))

DIST_PRE_BUY   = _seg_tur_pct(fg_pre_buy)
DIST_POST_BUY  = _seg_tur_pct(fg_post_buy)
DIST_PRE_SELL  = _seg_tur_pct(fg_pre_sell)
DIST_POST_SELL = _seg_tur_pct(fg_post_sell)

def _seg_donem_tur(df):
    return {d: _seg_tur_pct(df[df["donem"]==d] if not df.empty else pd.DataFrame())
            for d in DONEM_SIRASI}

DIST_PRE_BUY_D   = _seg_donem_tur(fg_pre_buy)
DIST_POST_BUY_D  = _seg_donem_tur(fg_post_buy)
DIST_PRE_SELL_D  = _seg_donem_tur(fg_pre_sell)
DIST_POST_SELL_D = _seg_donem_tur(fg_post_sell)

print(f"\n✅ § 6 Fon-Geçmeyen Sekans")
print(f"   Pre-Buy: {len(fg_pre_buy):,}  Post-Buy: {len(fg_post_buy):,}  "
      f"Pre-Sell: {len(fg_pre_sell):,}  Post-Sell: {len(fg_post_sell):,}")


# ══════════════════════════════════════════════════════════════════
# ▶  PART 1 TAMAMLANDI
# ══════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print("  ✅  PART 1 TAMAMLANDI — Tüm analizler hazır")
print(f"{'═'*60}")
print(f"  Dönemler     : {DONEM_SIRASI}  (Eylül 2025 · Mart 2026)")
print(f"  Segmentler   : {len(SEGMENT_SIRASI)} segment · {N_MUSTERI:,} müşteri")
print(f"  Alım         : {KPI['toplam_alim_m']:.1f}M TL  |  {KPI['alim_musteri']:,} müşteri")
print(f"  AUM          : {KPI['net_aum_m']:.1f}M TL")
print(f"  t-1 Dönem    : {_donems_with_prev}")
print(f"  Kompozit Ağ  : {G_COMPOSITE.number_of_nodes()} node · {G_COMPOSITE.number_of_edges()} kenar")
print(f"\n  ▶  PART 2 hücresini çalıştırarak grafikleri üret.")
