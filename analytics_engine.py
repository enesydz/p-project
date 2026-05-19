from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import math
import networkx as nx
import numpy as np
import pandas as pd


SEGMENT_SIRASI = [
    "Bireysel_Standart",
    "Bireysel_Premium",
    "Bireysel_Elite",
    "KOBİ",
    "KOBİ_Orta",
    "KOBİ_Büyük",
    "Kurumsal",
    "Kurumsal_Premium",
    "Private_Banking",
    "Ultra_HNW",
]

DONEM_SIRASI = ["2025.06", "2025.09", "2025.12", "2026.03"]
DONEM_ARALIK = {
    "2025.06": ("2025-06-01", "2025-06-30"),
    "2025.09": ("2025-09-01", "2025-09-30"),
    "2025.12": ("2025-12-01", "2025-12-31"),
    "2026.03": ("2026-03-01", "2026-03-31"),
}

SEG_ADETLER = [300, 210, 120, 120, 90, 60, 50, 30, 15, 5]
SEG_FON_ORT = {
    "Bireysel_Standart": 15_000,
    "Bireysel_Premium": 75_000,
    "Bireysel_Elite": 300_000,
    "KOBİ": 250_000,
    "KOBİ_Orta": 1_000_000,
    "KOBİ_Büyük": 4_000_000,
    "Kurumsal": 2_000_000,
    "Kurumsal_Premium": 10_000_000,
    "Private_Banking": 25_000_000,
    "Ultra_HNW": 40_000_000,
}
SEG_GUN_ISLEM = {
    "Bireysel_Standart": 0.8,
    "Bireysel_Premium": 1.5,
    "Bireysel_Elite": 2.5,
    "KOBİ": 3.0,
    "KOBİ_Orta": 4.5,
    "KOBİ_Büyük": 6.0,
    "Kurumsal": 6.0,
    "Kurumsal_Premium": 8.0,
    "Private_Banking": 10.0,
    "Ultra_HNW": 12.0,
}
SEG_RENK = {
    "Bireysel_Standart": "#2563EB",
    "Bireysel_Premium": "#7C3AED",
    "Bireysel_Elite": "#059669",
    "KOBİ": "#DC2626",
    "KOBİ_Orta": "#D97706",
    "KOBİ_Büyük": "#0891B2",
    "Kurumsal": "#BE185D",
    "Kurumsal_Premium": "#7C2D12",
    "Private_Banking": "#1D4ED8",
    "Ultra_HNW": "#065F46",
}

URUNLER = ["Vadesiz", "Vadeli", "Yatırım", "Döviz", "Kredi"]
URUN_RENKLER = {
    "Vadesiz": "#3B82F6",
    "Vadeli": "#F59E0B",
    "Yatırım": "#10B981",
    "Döviz": "#8B5CF6",
    "Kredi": "#EF4444",
}
BAKIYE_PERCENTIL = [0.33, 0.67]
BAKIYE_DILIM_ETIKETLERI = ["Düşük", "Orta", "Yüksek"]
ISLEM_BASLANGIC = pd.Timestamp("2025-05-24")
ISLEM_BITIS = pd.Timestamp("2026-04-07")
X_DAYS = 7
NET_SEQ_MIN_EDGE = 10
NET_SEQ_MIN_PCT = 2.0
SINYAL_PROB = 0.35
RANDOM_SEED = 42
N_MUSTERI = 1000

URUN_ISLEM = pd.DataFrame(
    [
        ("Vadesiz", "EFT", "Giriş", 0.15),
        ("Vadesiz", "EFT", "Çıkış", 0.15),
        ("Vadesiz", "Havale", "Giriş", 0.08),
        ("Vadesiz", "Havale", "Çıkış", 0.08),
        ("Vadesiz", "ATM_Çekim", "Çıkış", 0.06),
        ("Vadesiz", "Fatura_Ödeme", "Çıkış", 0.05),
        ("Vadeli", "Vadeli_Açılış", "Çıkış", 0.04),
        ("Vadeli", "Vadeli_Kapanış", "Giriş", 0.04),
        ("Yatırım", "Hisse_Alım", "Çıkış", 0.05),
        ("Yatırım", "Hisse_Satım", "Giriş", 0.04),
        ("Yatırım", "TahvilBono_Alım", "Çıkış", 0.03),
        ("Yatırım", "TahvilBono_Satım", "Giriş", 0.03),
        ("Yatırım", "Repo_Giriş", "Giriş", 0.03),
        ("Döviz", "Döviz_Alım", "Çıkış", 0.06),
        ("Döviz", "Döviz_Satım", "Giriş", 0.05),
        ("Kredi", "Kredi_Ödemesi", "Çıkış", 0.06),
        ("Kredi", "Kredi_Kullanımı", "Giriş", 0.03),
    ],
    columns=["urun_grubu", "islem_turu", "islem_yonu", "agirlik"],
)


@dataclass(frozen=True)
class AnalysisBundle:
    data: dict[str, Any]


def _short_label(name: str, maxlen: int = 10) -> str:
    for old, new in [
        ("Bireysel_", "Bir."),
        ("_Standart", "_Std"),
        ("_Premium", "_Prm"),
        ("_Elite", "_Elt"),
        ("KOBİ_Orta", "KOBI.Ort"),
        ("KOBİ_Büyük", "KOBI.Byk"),
        ("Kurumsal", "Kur."),
        ("Private_Banking", "Priv.Bnk"),
        ("Ultra_HNW", "Ultra"),
    ]:
        name = name.replace(old, new)
    return name[:maxlen]


def short_comp_label(node: str) -> str:
    parts = str(node).split("|")
    urun = parts[0][:4] if len(parts) > 0 else "NA"
    tur = parts[1][:5] if len(parts) > 1 else "NA"
    yon = parts[2][:1] if len(parts) > 2 else "N"
    return f"{urun}/{tur}/{yon}"


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    den = float(den)
    if abs(den) < 1e-12:
        return float(default)
    return float(num) / den


def _bounded(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(np.clip(value, lo, hi))


def _sign_path(arr: np.ndarray) -> np.ndarray:
    signs = np.sign(np.asarray(arr, dtype=float))
    if np.all(signs == 0):
        return np.zeros_like(signs)
    for idx in range(1, len(signs)):
        if signs[idx] == 0:
            signs[idx] = signs[idx - 1]
    for idx in range(len(signs) - 2, -1, -1):
        if signs[idx] == 0:
            signs[idx] = signs[idx + 1]
    return signs


def _count_regime_shifts(arr: np.ndarray) -> int:
    signs = _sign_path(arr)
    return int(np.sum(signs[1:] != signs[:-1]))


def _regime_label(arr: np.ndarray) -> str:
    signs = _sign_path(arr)
    if len(signs) == 0:
        return "Belirsiz"
    first, last = signs[0], signs[-1]
    if first > 0 and last < 0:
        return "Net_Alici_to_Net_Satici"
    if first < 0 and last > 0:
        return "Net_Satici_to_Net_Alici"
    if np.all(signs > 0):
        return "Sürekli_Net_Alici"
    if np.all(signs < 0):
        return "Sürekli_Net_Satici"
    return "Dalgalı_Gecis"


def _series_slope(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(np.polyfit(np.arange(len(vals)), vals, 1)[0])


def _entropy_from_probs(probs: list[float] | np.ndarray) -> float:
    probs = np.asarray([p for p in probs if p > 0], dtype=float)
    if len(probs) <= 1:
        return 0.0
    probs = probs / probs.sum()
    ent = -np.sum(probs * np.log2(probs))
    return float(ent / np.log2(len(probs)))


def _assign_donem_col(tarih_series: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=tarih_series.index, dtype=object)
    for donem in DONEM_SIRASI:
        bas = pd.Timestamp(DONEM_ARALIK[donem][0])
        bit = pd.Timestamp(DONEM_ARALIK[donem][1])
        out[(tarih_series >= bas) & (tarih_series <= bit)] = donem
    return out


def _lognorm_tutar(rng: np.random.Generator, ort_dizi: np.ndarray, sigma: float = 0.55) -> np.ndarray:
    ort_dizi = np.maximum(np.asarray(ort_dizi, dtype=float), 1.0)
    vals = np.exp(np.log(ort_dizi) + rng.standard_normal(len(ort_dizi)) * sigma) / 100
    return np.round(vals) * 100


def _generate_synthetic_data(seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    musteri_ids = [f"MUS{str(i).zfill(5)}" for i in range(1, N_MUSTERI + 1)]
    segment_dizisi = np.repeat(SEGMENT_SIRASI, SEG_ADETLER)
    musteri_df = pd.DataFrame({"musteri_id": musteri_ids, "musteri_segmenti": segment_dizisi})
    seg_map = musteri_df.set_index("musteri_id")["musteri_segmenti"].to_dict()

    alim_kayitlar: list[dict[str, Any]] = []
    for donem in DONEM_SIRASI:
        bas = pd.Timestamp(DONEM_ARALIK[donem][0])
        bit = pd.Timestamp(DONEM_ARALIK[donem][1])
        gun_sayisi = (bit - bas).days + 1
        for seg in SEGMENT_SIRASI:
            seg_musteriler = musteri_df.loc[musteri_df["musteri_segmenti"] == seg, "musteri_id"].values
            n_katilan = int(len(seg_musteriler) * rng.uniform(0.55, 0.65))
            katilan = rng.choice(seg_musteriler, size=n_katilan, replace=False)
            for musteri_id in katilan:
                n_islem = int(rng.integers(1, 5))
                gun_ofset = rng.integers(0, gun_sayisi, size=n_islem)
                tarihler = bas + pd.to_timedelta(gun_ofset, unit="D")
                tutarlar = _lognorm_tutar(rng, np.full(n_islem, SEG_FON_ORT[seg]))
                for idx in range(n_islem):
                    alim_kayitlar.append(
                        {
                            "musteri_id": musteri_id,
                            "tarih": tarihler[idx],
                            "donem": donem,
                            "alim_tutari": tutarlar[idx],
                            "islem_adeti": 1,
                            "alim_flg": 1,
                        }
                    )
    alim_df = pd.DataFrame(alim_kayitlar)
    alim_df["tarih"] = pd.to_datetime(alim_df["tarih"])
    alim_df = alim_df.sort_values(["musteri_id", "tarih"]).reset_index(drop=True)

    satim_kayitlar: list[dict[str, Any]] = []
    for donem in DONEM_SIRASI:
        bit = pd.Timestamp(DONEM_ARALIK[donem][1])
        alim_uniq = (
            alim_df[alim_df["donem"] == donem]
            .sort_values("tarih")
            .drop_duplicates("musteri_id", keep="first")
        )
        if alim_uniq.empty:
            continue
        sample_size = max(1, int(len(alim_uniq) * 0.40))
        sampled_idx = rng.choice(alim_uniq.index.values, size=sample_size, replace=False)
        satim_yapanlar = alim_uniq.loc[sampled_idx]
        for _, row in satim_yapanlar.iterrows():
            musteri_id = row["musteri_id"]
            seg = seg_map[musteri_id]
            base_t = row["tarih"]
            kalan = int((bit - base_t).days)
            if kalan < 1:
                continue
            max_offset = min(kalan, 45)
            n_islem = int(rng.integers(1, 4))
            tutarlar = _lognorm_tutar(
                rng,
                np.full(n_islem, SEG_FON_ORT[seg] * rng.uniform(0.60, 0.90)),
            )
            for idx in range(n_islem):
                offset = int(rng.integers(1, max_offset + 1))
                tarih = min(base_t + pd.Timedelta(days=offset), bit)
                satim_kayitlar.append(
                    {
                        "musteri_id": musteri_id,
                        "tarih": tarih,
                        "donem": donem,
                        "satim_tutari": tutarlar[idx],
                        "islem_adeti": 1,
                        "satim_flg": 1,
                    }
                )
    satim_df = pd.DataFrame(satim_kayitlar)
    satim_df["tarih"] = pd.to_datetime(satim_df["tarih"])
    satim_df = satim_df.sort_values(["musteri_id", "tarih"]).reset_index(drop=True)

    agirliklar = (URUN_ISLEM["agirlik"] / URUN_ISLEM["agirlik"].sum()).values
    toplam_gun = (ISLEM_BITIS - ISLEM_BASLANGIC).days + 1
    tarih_dizisi = pd.date_range(ISLEM_BASLANGIC, ISLEM_BITIS, freq="D")
    lambda_dizisi = musteri_df["musteri_segmenti"].map(SEG_GUN_ISLEM).values
    toplam_islem = rng.poisson(lambda_dizisi * toplam_gun)
    musteri_rep = musteri_df.loc[musteri_df.index.repeat(toplam_islem)].reset_index(drop=True)
    n_islem = len(musteri_rep)
    rand_tarih_idx = rng.integers(0, toplam_gun, size=n_islem)
    rand_tarihler = tarih_dizisi[rand_tarih_idx]
    tx_idx = rng.choice(len(URUN_ISLEM), size=n_islem, p=agirliklar)
    seg_ort_arr = musteri_rep["musteri_segmenti"].map({s: SEG_FON_ORT[s] * 0.10 for s in SEG_FON_ORT}).values
    tutarlar_arr = _lognorm_tutar(rng, seg_ort_arr, sigma=0.6)
    islem_df = pd.DataFrame(
        {
            "musteri_id": musteri_rep["musteri_id"].values,
            "musteri_segmenti": musteri_rep["musteri_segmenti"].values,
            "islem_tarihi": rand_tarihler,
            "islem_yonu": URUN_ISLEM["islem_yonu"].values[tx_idx],
            "urun_grubu": URUN_ISLEM["urun_grubu"].values[tx_idx],
            "islem_turu": URUN_ISLEM["islem_turu"].values[tx_idx],
            "islem_tutari": tutarlar_arr,
            "islem_adeti": 1,
        }
    )

    def sinyal_uret(event_df: pd.DataFrame, gun_ofset: int, yon: str, urun: str, tur: str, tarih_kolon: str = "tarih") -> pd.DataFrame:
        tmp = event_df[["musteri_id", tarih_kolon]].copy()
        tmp["islem_tarihi"] = pd.to_datetime(tmp[tarih_kolon]) + pd.Timedelta(days=gun_ofset)
        tmp = tmp[(tmp["islem_tarihi"] >= ISLEM_BASLANGIC) & (tmp["islem_tarihi"] <= ISLEM_BITIS)].copy()
        if tmp.empty:
            return pd.DataFrame()
        mask = rng.random(len(tmp)) < SINYAL_PROB
        tmp = tmp[mask].copy()
        if tmp.empty:
            return pd.DataFrame()
        tmp["musteri_segmenti"] = tmp["musteri_id"].map(seg_map)
        tmp["islem_yonu"] = yon
        tmp["urun_grubu"] = urun
        tmp["islem_turu"] = tur
        tmp["islem_adeti"] = 1
        seg_ort = tmp["musteri_segmenti"].map({s: SEG_FON_ORT[s] * 0.08 for s in SEG_FON_ORT}).values
        tmp["islem_tutari"] = _lognorm_tutar(rng, seg_ort, sigma=0.5)
        return tmp[
            [
                "musteri_id",
                "musteri_segmenti",
                "islem_tarihi",
                "islem_yonu",
                "urun_grubu",
                "islem_turu",
                "islem_tutari",
                "islem_adeti",
            ]
        ]

    sinyal_bloklar = []
    for gun in range(1, 8):
        blok = sinyal_uret(alim_df, -gun, "Giriş", "Vadesiz", "EFT")
        if not blok.empty:
            sinyal_bloklar.append(blok)
    for gun in range(1, 8):
        blok = sinyal_uret(satim_df, gun, "Çıkış", "Vadesiz", "EFT")
        if not blok.empty:
            sinyal_bloklar.append(blok)
    if sinyal_bloklar:
        islem_df = pd.concat([islem_df] + sinyal_bloklar, ignore_index=True)
        islem_df = islem_df.sort_values(["musteri_id", "islem_tarihi"]).reset_index(drop=True)

    net_fon = alim_df.groupby("musteri_id")["alim_tutari"].sum().sub(
        satim_df.groupby("musteri_id")["satim_tutari"].sum(), fill_value=0
    )
    bakiye_kayitlar: list[dict[str, Any]] = []
    for musteri_id in musteri_ids:
        seg = seg_map[musteri_id]
        baz = max(net_fon.get(musteri_id, 0) * 0.8, SEG_FON_ORT[seg] * 0.5)
        for idx, donem in enumerate(DONEM_SIRASI):
            carpan = (1 + 0.03 * idx) * rng.uniform(0.85, 1.20)
            bakiye = round(baz * carpan / 100) * 100
            bakiye_mtd = round(bakiye * rng.uniform(0.90, 1.10) / 100) * 100
            bakiye_kayitlar.append(
                {
                    "musteri_id": musteri_id,
                    "tarih": pd.Timestamp(DONEM_ARALIK[donem][1]),
                    "donem": donem,
                    "fon_bakiye_tutari": max(bakiye, 0),
                    "fon_bakiye_mtd_tutari": max(bakiye_mtd, 0),
                }
            )
    bakiye_df = pd.DataFrame(bakiye_kayitlar)
    bakiye_df["tarih"] = pd.to_datetime(bakiye_df["tarih"])
    bakiye_df = bakiye_df.sort_values(["musteri_id", "tarih"]).reset_index(drop=True)

    for df, col in [(islem_df, "musteri_segmenti"), (musteri_df, "musteri_segmenti")]:
        df[col] = pd.Categorical(df[col], categories=SEGMENT_SIRASI, ordered=True)

    return {
        "musteri_df": musteri_df,
        "alim_df": alim_df,
        "satim_df": satim_df,
        "islem_df": islem_df,
        "bakiye_df": bakiye_df,
    }


def _urun_pct(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["musteri_segmenti", "urun_grubu"], observed=True).size().reset_index(name="adet")
    grp["toplam"] = grp.groupby("musteri_segmenti", observed=True)["adet"].transform("sum")
    grp["pct"] = grp["adet"] / grp["toplam"] * 100
    pv = grp.pivot(index="musteri_segmenti", columns="urun_grubu", values="pct").fillna(0)
    return pv.reindex(SEGMENT_SIRASI).reindex(columns=URUNLER, fill_value=0)


def _giris_pct(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby("musteri_segmenti", observed=True)["islem_yonu"]
        .apply(lambda x: (x == "Giriş").mean() * 100)
        .round(1)
        .reindex(SEGMENT_SIRASI)
    )


def _net_akis(df: pd.DataFrame) -> pd.Series:
    gin = df[df["islem_yonu"] == "Giriş"].groupby(["musteri_segmenti", "event_id"], observed=True)["islem_tutari"].sum()
    cik = df[df["islem_yonu"] == "Çıkış"].groupby(["musteri_segmenti", "event_id"], observed=True)["islem_tutari"].sum()
    return (
        gin.sub(cik, fill_value=0)
        .reset_index()
        .rename(columns={"islem_tutari": "net"})
        .groupby("musteri_segmenti", observed=True)["net"]
        .mean()
        .reindex(SEGMENT_SIRASI)
    )


def _compute_pairs(df: pd.DataFrame) -> pd.DataFrame:
    d = (
        df[["musteri_id", "islem_tarihi", "urun_grubu", "islem_tutari", "musteri_segmenti"]]
        .sort_values(["musteri_id", "islem_tarihi"])
        .reset_index(drop=True)
    )
    d["next_urun"] = d.groupby("musteri_id", sort=False)["urun_grubu"].shift(-1)
    d["next_musteri"] = d.groupby("musteri_id", sort=False)["musteri_id"].shift(-1)
    return d.dropna(subset=["next_urun"]).pipe(lambda x: x[x["musteri_id"] == x["next_musteri"]]).reset_index(drop=True)


def build_seq_network(pairs_df: pd.DataFrame, min_edge: int = NET_SEQ_MIN_EDGE) -> nx.DiGraph:
    graph = nx.DiGraph()
    for urun in URUNLER:
        u_rows = pairs_df[pairs_df["urun_grubu"] == urun] if not pairs_df.empty else pd.DataFrame()
        graph.add_node(
            urun,
            color=URUN_RENKLER[urun],
            freq=len(u_rows),
            volume=float(u_rows["islem_tutari"].sum()) if not u_rows.empty else 0.0,
        )
    if pairs_df.empty:
        return graph
    edge_stats = (
        pairs_df.groupby(["urun_grubu", "next_urun"], observed=True)
        .agg(count=("musteri_id", "count"), volume=("islem_tutari", "sum"))
        .reset_index()
    )
    edge_stats = edge_stats[edge_stats["count"] >= min_edge].copy()
    total_from = edge_stats.groupby("urun_grubu")["count"].sum()
    for _, row in edge_stats.iterrows():
        pct = row["count"] / max(total_from.get(row["urun_grubu"], 1), 1) * 100
        graph.add_edge(
            row["urun_grubu"],
            row["next_urun"],
            weight=int(row["count"]),
            pct=round(pct, 1),
            volume=float(row["volume"]),
        )
    return graph


def build_composite_net(df: pd.DataFrame, col: str = "urun_tur_yon", min_edge: int = 5) -> tuple[nx.DiGraph, pd.DataFrame]:
    d = (
        df[["musteri_id", "islem_tarihi", col, "islem_tutari"]]
        .sort_values(["musteri_id", "islem_tarihi"])
        .reset_index(drop=True)
    )
    d["next_node"] = d.groupby("musteri_id", sort=False)[col].shift(-1)
    d["next_id"] = d.groupby("musteri_id", sort=False)["musteri_id"].shift(-1)
    pairs = d.dropna(subset=["next_node"]).pipe(lambda x: x[x["musteri_id"] == x["next_id"]]).reset_index(drop=True)
    graph = nx.DiGraph()
    if pairs.empty:
        return graph, pairs
    edge_stats = (
        pairs.groupby([col, "next_node"], observed=True)
        .agg(count=("musteri_id", "count"), volume=("islem_tutari", "sum"))
        .reset_index()
    )
    edge_stats = edge_stats[edge_stats["count"] >= min_edge].copy()
    if edge_stats.empty:
        return graph, pairs
    nodes = set(edge_stats[col].tolist() + edge_stats["next_node"].tolist())
    freq_map = df[col].value_counts().to_dict()
    for node in nodes:
        urun = str(node).split("|")[0]
        graph.add_node(node, color=URUN_RENKLER.get(urun, "#94A3B8"), urun=urun, freq=freq_map.get(node, 0))
    total_from = edge_stats.groupby(col)["count"].sum()
    for _, row in edge_stats.iterrows():
        pct = row["count"] / max(total_from.get(row[col], 1), 1) * 100
        graph.add_edge(row[col], row["next_node"], weight=int(row["count"]), pct=round(pct, 1), volume=float(row["volume"]))
    return graph, pairs


def _trans_mat(pairs_df: pd.DataFrame) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame(0.0, index=URUNLER, columns=URUNLER)
    mat = (
        pairs_df.groupby(["urun_grubu", "next_urun"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=URUNLER, columns=URUNLER, fill_value=0)
    )
    return mat.div(mat.sum(axis=1).clip(lower=1), axis=0).round(3)


def _top_ngrams(islem_df: pd.DataFrame, n: int = 3, top_k: int = 15) -> pd.Series:
    d = islem_df[["musteri_id", "islem_tarihi", "urun_grubu"]].sort_values(["musteri_id", "islem_tarihi"]).reset_index(drop=True)
    grams: list[str] = []
    for _, grp in d.groupby("musteri_id", sort=False):
        seq = grp["urun_grubu"].tolist()
        for idx in range(len(seq) - n + 1):
            grams.append(" → ".join(seq[idx : idx + n]))
    return pd.Series(Counter(grams)).sort_values(ascending=False).head(top_k)


def _net_metrics(graph: nx.DiGraph) -> dict[str, dict[str, float]]:
    base = {u: {"pagerank": 0.0, "betweenness": 0.0, "in_deg": 0.0, "out_deg": 0.0} for u in URUNLER}
    if graph.number_of_edges() == 0:
        return base
    pr = nx.pagerank(graph, weight="weight", max_iter=500, tol=1e-6)
    bc = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    return {
        u: {
            "pagerank": round(pr.get(u, 0), 4),
            "betweenness": round(bc.get(u, 0), 4),
            "in_deg": float(graph.in_degree(u, weight="weight")),
            "out_deg": float(graph.out_degree(u, weight="weight")),
        }
        for u in URUNLER
    }


def _node_entropy(graph: nx.DiGraph, node: str) -> float:
    outs = [d.get("weight", 0) for _, _, d in graph.out_edges(node, data=True)]
    return _entropy_from_probs(outs)


def _graph_intel(graph: nx.DiGraph) -> dict[str, Any]:
    if graph is None or graph.number_of_nodes() == 0:
        return {
            "graph": {"nodes": 0, "edges": 0, "density": 0.0, "avg_entropy": 0.0, "route_concentration": 0.0, "complexity": 0.0, "top_route": "NA", "hub_node": "NA", "bridge_node": "NA", "source_node": "NA", "sink_node": "NA"},
            "nodes": pd.DataFrame(columns=["node", "label", "pagerank", "betweenness", "hub", "authority", "in_weight", "out_weight", "imbalance", "out_entropy", "role"]),
            "paths": pd.DataFrame(columns=["source", "target", "path", "cost"]),
        }
    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {n: 0 for n in graph.nodes}
    betw = nx.betweenness_centrality(graph, weight=None, normalized=True) if graph.number_of_edges() else {n: 0 for n in graph.nodes}
    try:
        hubs, auth = nx.hits(graph, max_iter=500, normalized=True)
    except Exception:
        hubs = {n: 0 for n in graph.nodes}
        auth = {n: 0 for n in graph.nodes}
    rows = []
    edge_weights = [d.get("weight", 0) for _, _, d in graph.edges(data=True)]
    total_edge_weight = max(sum(edge_weights), 1)
    for node in graph.nodes:
        in_w = sum(d.get("weight", 0) for _, _, d in graph.in_edges(node, data=True))
        out_w = sum(d.get("weight", 0) for _, _, d in graph.out_edges(node, data=True))
        role = "Core"
        if out_w > 0 and in_w == 0:
            role = "Source"
        elif in_w > 0 and out_w == 0:
            role = "Sink"
        rows.append(
            {
                "node": node,
                "label": short_comp_label(node),
                "pagerank": pagerank.get(node, 0),
                "betweenness": betw.get(node, 0),
                "hub": hubs.get(node, 0),
                "authority": auth.get(node, 0),
                "in_weight": in_w,
                "out_weight": out_w,
                "imbalance": _safe_div(out_w - in_w, out_w + in_w, default=0),
                "out_entropy": _node_entropy(graph, node),
                "role": role,
            }
        )
    node_df = pd.DataFrame(rows).sort_values(["pagerank", "betweenness"], ascending=False).reset_index(drop=True)
    if not node_df.empty:
        top_b = node_df["betweenness"].quantile(0.80)
        node_df.loc[node_df["betweenness"] >= top_b, "role"] = "Bridge"
    top_route = "NA"
    route_concentration = 0.0
    if graph.number_of_edges() > 0:
        edge_df = pd.DataFrame(
            [{"u": u, "v": v, "weight": d.get("weight", 0), "pct": d.get("pct", 0)} for u, v, d in graph.edges(data=True)]
        ).sort_values(["weight", "pct"], ascending=False)
        if not edge_df.empty:
            top_route = f"{short_comp_label(edge_df.iloc[0]['u'])} → {short_comp_label(edge_df.iloc[0]['v'])}"
            route_concentration = _safe_div(edge_df.iloc[0]["weight"], total_edge_weight, default=0)
    avg_entropy = float(node_df["out_entropy"].mean()) if not node_df.empty else 0.0
    density = float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0
    complexity = _safe_div(graph.number_of_edges(), graph.number_of_nodes(), default=0)

    path_rows = []
    if graph.number_of_edges() > 0 and not node_df.empty:
        g_cost = graph.copy()
        for u, v, d in g_cost.edges(data=True):
            d["cost"] = 1 / max(d.get("pct", 0.1), 0.1)
        top_nodes = node_df.head(min(6, len(node_df)))["node"].tolist()
        for idx, src in enumerate(top_nodes):
            for tgt in top_nodes[idx + 1 :]:
                try:
                    path = nx.shortest_path(g_cost, source=src, target=tgt, weight="cost")
                    path_rows.append({
                        "source": src,
                        "target": tgt,
                        "path": " → ".join(short_comp_label(p) for p in path),
                        "cost": round(nx.path_weight(g_cost, path, weight="cost"), 3),
                    })
                except Exception:
                    pass
    path_df = pd.DataFrame(path_rows).sort_values("cost").reset_index(drop=True) if path_rows else pd.DataFrame(columns=["source", "target", "path", "cost"])
    graph_info = {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "density": density,
        "avg_entropy": avg_entropy,
        "route_concentration": route_concentration,
        "complexity": complexity,
        "top_route": top_route,
        "hub_node": node_df.sort_values("hub", ascending=False).iloc[0]["label"] if not node_df.empty else "NA",
        "bridge_node": node_df.sort_values("betweenness", ascending=False).iloc[0]["label"] if not node_df.empty else "NA",
        "source_node": node_df.sort_values("imbalance", ascending=False).iloc[0]["label"] if not node_df.empty else "NA",
        "sink_node": node_df.sort_values("imbalance", ascending=True).iloc[0]["label"] if not node_df.empty else "NA",
    }
    return {"graph": graph_info, "nodes": node_df, "paths": path_df}


def _bd_pivot(df: pd.DataFrame, ev_bd: pd.DataFrame, val: str = "islem_tutari") -> pd.DataFrame:
    d2 = df.merge(ev_bd, on="event_id", how="left")
    d2["bakiye_dilimi"] = pd.Categorical(d2["bakiye_dilimi"].astype(str).fillna(BAKIYE_DILIM_ETIKETLERI[1]), categories=BAKIYE_DILIM_ETIKETLERI, ordered=True)
    if val == "count":
        pv = d2.groupby(["musteri_segmenti", "bakiye_dilimi"], observed=True).size().unstack(fill_value=0)
    else:
        pv = d2.groupby(["musteri_segmenti", "bakiye_dilimi"], observed=True)[val].mean().unstack(fill_value=0)
    return pv.reindex(SEGMENT_SIRASI).reindex(columns=BAKIYE_DILIM_ETIKETLERI, fill_value=0)


def _compute_base_metrics(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    musteri_df = data["musteri_df"].copy()
    alim_df = data["alim_df"].copy()
    satim_df = data["satim_df"].copy()
    islem_df = data["islem_df"].copy()
    bakiye_df = data["bakiye_df"].copy()

    alim_events = (
        alim_df.merge(musteri_df[["musteri_id", "musteri_segmenti"]], on="musteri_id")
        .rename(columns={"tarih": "event_tarih", "alim_tutari": "event_tutari"})
        .assign(event_type="Alım")[["musteri_id", "musteri_segmenti", "event_tarih", "event_tutari", "donem", "event_type"]]
    )
    satim_events = (
        satim_df.merge(musteri_df[["musteri_id", "musteri_segmenti"]], on="musteri_id")
        .rename(columns={"tarih": "event_tarih", "satim_tutari": "event_tutari"})
        .assign(event_type="Satım")[["musteri_id", "musteri_segmenti", "event_tarih", "event_tutari", "donem", "event_type"]]
    )
    events_df = pd.concat([alim_events, satim_events]).reset_index(drop=True).assign(event_id=lambda d: d.index)

    bakiye_donem = (
        bakiye_df[["musteri_id", "donem", "fon_bakiye_tutari"]]
        .merge(musteri_df[["musteri_id", "musteri_segmenti"]], on="musteri_id")
        .rename(columns={"fon_bakiye_tutari": "ort_bakiye"})
    )

    def dilim_ata(grp: pd.DataFrame) -> pd.Series:
        thresholds = [grp["ort_bakiye"].quantile(p) for p in BAKIYE_PERCENTIL]
        bins = [-np.inf] + thresholds + [np.inf]
        return pd.cut(grp["ort_bakiye"], bins=bins, labels=BAKIYE_DILIM_ETIKETLERI)

    bakiye_donem["bakiye_dilimi"] = bakiye_donem.groupby(["musteri_segmenti", "donem"], group_keys=False).apply(dilim_ata).astype(str)
    musteri_bakiye_dilim = bakiye_donem[["musteri_id", "donem", "bakiye_dilimi"]].copy()
    dilim_esikleri = (
        bakiye_donem.groupby(["musteri_segmenti", "donem"], observed=True)["ort_bakiye"]
        .quantile(BAKIYE_PERCENTIL)
        .unstack(level=-1)
        .rename(columns={p: f"p{int(p * 100)}" for p in BAKIYE_PERCENTIL})
    )

    events_enriched_df = (
        events_df.merge(musteri_bakiye_dilim, on=["musteri_id", "donem"], how="left")
        .merge(
            bakiye_df[["musteri_id", "donem", "fon_bakiye_tutari"]].rename(columns={"fon_bakiye_tutari": "bakiye"}),
            on=["musteri_id", "donem"],
            how="left",
        )
    )
    events_enriched_df["bakiye_dilimi"] = events_enriched_df["bakiye_dilimi"].fillna(BAKIYE_DILIM_ETIKETLERI[1])

    ev_keys = events_df[["event_id", "musteri_id", "event_tarih", "event_type", "donem"]].copy()
    islem_ev = islem_df.merge(ev_keys, on="musteri_id")
    islem_ev["gun_fark"] = (islem_ev["islem_tarihi"] - islem_ev["event_tarih"]).dt.days
    islem_window_df = islem_ev[islem_ev["gun_fark"].between(-X_DAYS, X_DAYS) & (islem_ev["gun_fark"] != 0)].copy()
    islem_window_df["pencere"] = islem_window_df["gun_fark"].apply(lambda gun: "Pre" if gun < 0 else "Post")
    pre_buy = islem_window_df[(islem_window_df["event_type"] == "Alım") & (islem_window_df["pencere"] == "Pre")]
    post_buy = islem_window_df[(islem_window_df["event_type"] == "Alım") & (islem_window_df["pencere"] == "Post")]
    pre_sell = islem_window_df[(islem_window_df["event_type"] == "Satım") & (islem_window_df["pencere"] == "Pre")]
    post_sell = islem_window_df[(islem_window_df["event_type"] == "Satım") & (islem_window_df["pencere"] == "Post")]

    metrik_genel = (
        events_enriched_df.groupby(["musteri_segmenti", "event_type"], observed=True)
        .agg(
            Olay_Adedi=("event_id", "count"),
            Musteri_Sayisi=("musteri_id", "nunique"),
            Toplam_Tutar_M=("event_tutari", lambda x: round(x.sum() / 1e6, 2)),
            Ort_Tutar_K=("event_tutari", lambda x: round(x.mean() / 1e3, 1)),
            Medyan_Tutar_K=("event_tutari", lambda x: round(x.median() / 1e3, 1)),
        )
        .reset_index()
    )

    urun_pre_buy = _urun_pct(pre_buy)
    urun_post_buy = _urun_pct(post_buy)
    urun_pre_sell = _urun_pct(pre_sell)
    urun_post_sell = _urun_pct(post_sell)
    giris_pre_buy = _giris_pct(pre_buy)
    giris_post_buy = _giris_pct(post_buy)
    giris_pre_sell = _giris_pct(pre_sell)
    giris_post_sell = _giris_pct(post_sell)
    net_pre_buy = _net_akis(pre_buy)
    net_post_buy = _net_akis(post_buy)
    net_pre_sell = _net_akis(pre_sell)
    net_post_sell = _net_akis(post_sell)

    ev_bd = events_enriched_df[["event_id", "bakiye_dilimi"]].drop_duplicates("event_id").assign(bakiye_dilimi=lambda d: d["bakiye_dilimi"].astype(str))
    bd_alim_count = _bd_pivot(pd.concat([pre_buy, post_buy]), ev_bd, "count")
    bd_alim_tutar = _bd_pivot(pd.concat([pre_buy, post_buy]), ev_bd)
    bd_satim_count = _bd_pivot(pd.concat([pre_sell, post_sell]), ev_bd, "count")
    bd_satim_tutar = _bd_pivot(pd.concat([pre_sell, post_sell]), ev_bd)

    def gecis_mat(event_type: str) -> pd.DataFrame:
        df_e = islem_window_df[islem_window_df["event_type"] == event_type]
        pre_u = (
            df_e[df_e["pencere"] == "Pre"].groupby(["event_id", "urun_grubu"], observed=True).size().reset_index(name="n").sort_values("n", ascending=False).drop_duplicates("event_id")[["event_id", "urun_grubu"]].rename(columns={"urun_grubu": "pre_urun"})
        )
        post_u = (
            df_e[df_e["pencere"] == "Post"].groupby(["event_id", "urun_grubu"], observed=True).size().reset_index(name="n").sort_values("n", ascending=False).drop_duplicates("event_id")[["event_id", "urun_grubu"]].rename(columns={"urun_grubu": "post_urun"})
        )
        gecis = pre_u.merge(post_u, on="event_id", how="inner")
        mat = gecis.groupby(["pre_urun", "post_urun"], observed=True).size().unstack(fill_value=0).reindex(index=URUNLER, columns=URUNLER, fill_value=0)
        return (mat.div(mat.sum(axis=1).clip(lower=1), axis=0) * 100).round(1)

    gecis_alim = gecis_mat("Alım")
    gecis_satim = gecis_mat("Satım")

    musteri_donem_alim = alim_df.groupby("musteri_id")["donem"].nunique()
    sadakat = (
        musteri_donem_alim.reset_index().rename(columns={"donem": "n_donem"})
        .merge(musteri_df, on="musteri_id")
        .groupby("musteri_segmenti", observed=True)
        .agg(
            Ort_Donem_Alim=("n_donem", "mean"),
            Cok_Donem_Pct=("n_donem", lambda x: (x > 1).mean() * 100),
            Tek_Donem_Pct=("n_donem", lambda x: (x == 1).mean() * 100),
        )
        .round(1)
        .reindex(SEGMENT_SIRASI)
    )

    def gunluk_ortalama(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.Series:
        pre_g = pre_df.groupby("gun_fark", observed=True)["islem_tutari"].mean()
        post_g = post_df.groupby("gun_fark", observed=True)["islem_tutari"].mean()
        return pre_g.combine_first(post_g).reindex(range(-X_DAYS, X_DAYS + 1)).fillna(0)

    gunluk_alim = gunluk_ortalama(pre_buy, post_buy)
    gunluk_satim = gunluk_ortalama(pre_sell, post_sell)

    freq_pre = (
        pre_buy.groupby(["musteri_segmenti", "event_id"], observed=True)
        .size()
        .reset_index(name="n")
        .groupby("musteri_segmenti", observed=True)["n"]
        .mean()
        .reindex(SEGMENT_SIRASI)
        .fillna(0)
    )
    sadakat_pct = (
        musteri_donem_alim.reset_index().rename(columns={"donem": "n_donem"})
        .merge(musteri_df, on="musteri_id")
        .groupby("musteri_segmenti", observed=True)
        .apply(lambda x: (x["n_donem"] > 1).mean() * 100)
        .round(1)
        .reindex(SEGMENT_SIRASI)
        .fillna(0)
    )
    davranis_skor = pd.DataFrame(
        {
            "Pre_Buy_Giriş_%": giris_pre_buy,
            "Post_Buy_Giriş_%": giris_post_buy,
            "Pre_Sell_Giriş_%": giris_pre_sell,
            "Post_Sell_Giriş_%": giris_post_sell,
            "Pre_Buy_Frekans": freq_pre,
            "Sadakat_%": sadakat_pct,
        }
    ).round(1)
    davranis_skor["Aktivite_Skoru"] = (
        davranis_skor["Pre_Buy_Giriş_%"] * 0.25
        + davranis_skor["Post_Buy_Giriş_%"] * 0.20
        + davranis_skor["Pre_Buy_Frekans"] * 5.0
        + davranis_skor["Sadakat_%"] * 0.30
    ).clip(0, 100).round(1)

    seg_toplam = musteri_df.groupby("musteri_segmenti", observed=True).size()
    alim_katil = alim_df.merge(musteri_df, on="musteri_id").groupby("musteri_segmenti", observed=True)["musteri_id"].nunique()
    satim_katil = satim_df.merge(musteri_df, on="musteri_id").groupby("musteri_segmenti", observed=True)["musteri_id"].nunique()
    penetrasyon = pd.DataFrame(
        {
            "Toplam_Musteri": seg_toplam,
            "Alim_Yapan": alim_katil.reindex(SEGMENT_SIRASI),
            "Satim_Yapan": satim_katil.reindex(SEGMENT_SIRASI),
            "Alim_Pct": (alim_katil / seg_toplam * 100).reindex(SEGMENT_SIRASI).round(1),
            "Satim_Pct": (satim_katil / seg_toplam * 100).reindex(SEGMENT_SIRASI).round(1),
        }
    ).reindex(SEGMENT_SIRASI)

    aum = (
        alim_df.merge(musteri_df, on="musteri_id")
        .groupby("musteri_segmenti", observed=True)["alim_tutari"]
        .sum()
        .sub(
            satim_df.merge(musteri_df, on="musteri_id").groupby("musteri_segmenti", observed=True)["satim_tutari"].sum(),
            fill_value=0,
        )
        .reindex(SEGMENT_SIRASI)
    )

    pairs_all = _compute_pairs(islem_df)
    g_seq_all = build_seq_network(pairs_all, min_edge=NET_SEQ_MIN_EDGE)

    def win_pairs(ev_type: str, win_label: str) -> pd.DataFrame:
        sub = islem_window_df[(islem_window_df["event_type"] == ev_type) & (islem_window_df["pencere"] == win_label)].copy()
        return _compute_pairs(sub) if len(sub) >= 2 else pd.DataFrame()

    g_seq_pre_buy = build_seq_network(win_pairs("Alım", "Pre"), min_edge=2)
    g_seq_post_buy = build_seq_network(win_pairs("Alım", "Post"), min_edge=2)
    g_seq_pre_sell = build_seq_network(win_pairs("Satım", "Pre"), min_edge=2)
    g_seq_post_sell = build_seq_network(win_pairs("Satım", "Post"), min_edge=2)
    g_seq_seg = {seg: build_seq_network(pairs_all[pairs_all["musteri_segmenti"] == seg].reset_index(drop=True), min_edge=3) for seg in SEGMENT_SIRASI}
    net_metrics = _net_metrics(g_seq_all)
    net_metrics_df = pd.DataFrame(net_metrics).T.reset_index().rename(columns={"index": "urun"})
    trans_mat_all = _trans_mat(pairs_all)
    trans_mat_pre_buy = _trans_mat(win_pairs("Alım", "Pre"))
    trans_mat_post_buy = _trans_mat(win_pairs("Alım", "Post"))
    top_trigrams = _top_ngrams(islem_df, n=3, top_k=15)
    top_bigrams = (
        pairs_all.groupby(["urun_grubu", "next_urun"], observed=True)
        .size()
        .reset_index(name="count")
        .assign(label=lambda d: d["urun_grubu"] + " → " + d["next_urun"])
        .sort_values("count", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    kpi = {
        "toplam_alim_m": round(alim_df["alim_tutari"].sum() / 1e6, 1),
        "toplam_satim_m": round(satim_df["satim_tutari"].sum() / 1e6, 1),
        "net_aum_m": round(aum.sum() / 1e6, 1),
        "toplam_event": len(events_df),
        "alim_musteri": int(alim_df["musteri_id"].nunique()),
        "satim_musteri": int(satim_df["musteri_id"].nunique()),
        "window_islem": len(islem_window_df),
        "penetrasyon_ort": float(penetrasyon["Alim_Pct"].mean().round(1)),
        "sadakat_ort": float(sadakat["Cok_Donem_Pct"].mean().round(1)),
        "aktivite_max_seg": davranis_skor["Aktivite_Skoru"].idxmax(),
    }

    return {
        **data,
        "events_df": events_df,
        "events_enriched_df": events_enriched_df,
        "islem_window_df": islem_window_df,
        "pre_buy": pre_buy,
        "post_buy": post_buy,
        "pre_sell": pre_sell,
        "post_sell": post_sell,
        "METRIK_GENEL": metrik_genel,
        "DILIM_ESLIKLERI": dilim_esikleri,
        "URUN_PRE_BUY": urun_pre_buy,
        "URUN_POST_BUY": urun_post_buy,
        "URUN_PRE_SELL": urun_pre_sell,
        "URUN_POST_SELL": urun_post_sell,
        "GIRIS_PRE_BUY": giris_pre_buy,
        "GIRIS_POST_BUY": giris_post_buy,
        "GIRIS_PRE_SELL": giris_pre_sell,
        "GIRIS_POST_SELL": giris_post_sell,
        "NET_PRE_BUY": net_pre_buy,
        "NET_POST_BUY": net_post_buy,
        "NET_PRE_SELL": net_pre_sell,
        "NET_POST_SELL": net_post_sell,
        "BD_ALIM_COUNT": bd_alim_count,
        "BD_ALIM_TUTAR": bd_alim_tutar,
        "BD_SATIM_COUNT": bd_satim_count,
        "BD_SATIM_TUTAR": bd_satim_tutar,
        "GECIS_ALIM": gecis_alim,
        "GECIS_SATIM": gecis_satim,
        "SADAKAT": sadakat,
        "GUNLUK_ALIM": gunluk_alim,
        "GUNLUK_SATIM": gunluk_satim,
        "DAVRANIS_SKOR": davranis_skor,
        "PENETRASYON": penetrasyon,
        "AUM": aum,
        "G_SEQ_ALL": g_seq_all,
        "G_SEQ_PRE_BUY": g_seq_pre_buy,
        "G_SEQ_POST_BUY": g_seq_post_buy,
        "G_SEQ_PRE_SELL": g_seq_pre_sell,
        "G_SEQ_POST_SELL": g_seq_post_sell,
        "G_SEQ_SEG": g_seq_seg,
        "NET_METRICS": net_metrics,
        "NET_METRICS_DF": net_metrics_df,
        "TRANS_MAT_ALL": trans_mat_all,
        "TRANS_MAT_PRE_BUY": trans_mat_pre_buy,
        "TRANS_MAT_POST_BUY": trans_mat_post_buy,
        "TOP_TRIGRAMS": top_trigrams,
        "TOP_BIGRAMS": top_bigrams,
        "KPI": kpi,
        "_pairs_all": pairs_all,
    }


def _compute_temporal_and_heavy(bundle: dict[str, Any]) -> dict[str, Any]:
    alim_df = bundle["alim_df"]
    satim_df = bundle["satim_df"]
    islem_df = bundle["islem_df"].copy()
    musteri_df = bundle["musteri_df"]

    donem_alim = (
        alim_df.merge(musteri_df, on="musteri_id")
        .groupby(["donem", "musteri_segmenti"], observed=True)
        .agg(Alim_Adet=("alim_tutari", "count"), Alim_Tutar_M=("alim_tutari", lambda x: round(x.sum() / 1e6, 3)), Alim_Musteri=("musteri_id", "nunique"))
        .reset_index()
    )
    donem_satim = (
        satim_df.merge(musteri_df, on="musteri_id")
        .groupby(["donem", "musteri_segmenti"], observed=True)
        .agg(Satim_Adet=("satim_tutari", "count"), Satim_Tutar_M=("satim_tutari", lambda x: round(x.sum() / 1e6, 3)), Satim_Musteri=("musteri_id", "nunique"))
        .reset_index()
    )
    donem_alim_pivot = donem_alim.pivot(index="musteri_segmenti", columns="donem", values="Alim_Tutar_M").reindex(SEGMENT_SIRASI)[DONEM_SIRASI].fillna(0)
    donem_satim_pivot = donem_satim.pivot(index="musteri_segmenti", columns="donem", values="Satim_Tutar_M").reindex(SEGMENT_SIRASI)[DONEM_SIRASI].fillna(0)
    donem_net_pivot = donem_alim_pivot.sub(donem_satim_pivot, fill_value=0).round(3)
    donem_alim_poc = donem_alim_pivot.pct_change(axis=1).multiply(100).round(1)
    donem_satim_poc = donem_satim_pivot.pct_change(axis=1).multiply(100).round(1)

    islem_d = islem_df[["musteri_id", "musteri_segmenti", "islem_tarihi", "islem_tutari", "urun_grubu", "islem_turu", "islem_yonu"]].copy()
    islem_d["donem"] = _assign_donem_col(islem_d["islem_tarihi"])
    donem_islem_pivot = (
        islem_d.dropna(subset=["donem"])
        .groupby(["musteri_segmenti", "donem"], observed=True)["islem_tutari"]
        .sum()
        .div(1e6)
        .round(2)
        .unstack(fill_value=0)
        .reindex(SEGMENT_SIRASI)
        .reindex(columns=DONEM_SIRASI, fill_value=0)
    )

    alim_mst = (
        alim_df.merge(musteri_df, on="musteri_id")
        .groupby(["musteri_id", "musteri_segmenti", "donem"], observed=True)["alim_tutari"]
        .sum()
        .reset_index(name="toplam_alim")
    )
    satim_mst = (
        satim_df.merge(musteri_df, on="musteri_id")
        .groupby(["musteri_id", "musteri_segmenti", "donem"], observed=True)["satim_tutari"]
        .sum()
        .reset_index(name="toplam_satim")
    )

    def heavy_flag(grp: pd.DataFrame, col: str) -> pd.DataFrame:
        grp = grp.copy()
        grp["heavy"] = grp[col] >= grp[col].quantile(0.75)
        return grp

    alim_mst = alim_mst.groupby(["musteri_segmenti", "donem"], observed=True, group_keys=False).apply(lambda g: heavy_flag(g, "toplam_alim"))
    satim_mst = satim_mst.groupby(["musteri_segmenti", "donem"], observed=True, group_keys=False).apply(lambda g: heavy_flag(g, "toplam_satim"))
    heavy_buyer_seg = (
        alim_mst.groupby(["musteri_segmenti", "donem"], observed=True)["heavy"]
        .mean()
        .multiply(100)
        .round(1)
        .unstack(fill_value=0)
        .reindex(SEGMENT_SIRASI)
        .reindex(columns=DONEM_SIRASI, fill_value=0)
    )
    heavy_seller_seg = (
        satim_mst.groupby(["musteri_segmenti", "donem"], observed=True)["heavy"]
        .mean()
        .multiply(100)
        .round(1)
        .unstack(fill_value=0)
        .reindex(SEGMENT_SIRASI)
        .reindex(columns=DONEM_SIRASI, fill_value=0)
    )
    heavy_buyer_tutar = (
        alim_mst.groupby(["musteri_segmenti", "heavy"], observed=True)["toplam_alim"]
        .mean()
        .div(1e3)
        .round(1)
        .unstack()
        .reindex(SEGMENT_SIRASI)
        .rename(columns={False: "Normal_K", True: "Heavy_K"})
    )
    heavy_seller_tutar = (
        satim_mst.groupby(["musteri_segmenti", "heavy"], observed=True)["toplam_satim"]
        .mean()
        .div(1e3)
        .round(1)
        .unstack()
        .reindex(SEGMENT_SIRASI)
        .rename(columns={False: "Normal_K", True: "Heavy_K"})
    )
    heavy_ids = set(alim_mst.loc[alim_mst["heavy"] == True, "musteri_id"].tolist())
    normal_ids = set(alim_mst.loc[alim_mst["heavy"] == False, "musteri_id"].tolist())
    heavy_islem_donem = (
        islem_d[islem_d["musteri_id"].isin(heavy_ids) & islem_d["donem"].notna()]
        .groupby(["musteri_segmenti", "donem"], observed=True)["islem_tutari"]
        .mean()
        .div(1e3)
        .round(1)
        .unstack(fill_value=0)
        .reindex(SEGMENT_SIRASI)
        .reindex(columns=DONEM_SIRASI, fill_value=0)
    )

    islem_df["urun_tur_yon"] = islem_df["urun_grubu"] + "|" + islem_df["islem_turu"] + "|" + islem_df["islem_yonu"]
    composite_freq = islem_df["urun_tur_yon"].value_counts().rename_axis("node").reset_index(name="count").head(20)
    g_composite, composite_pairs = build_composite_net(islem_df, min_edge=8)
    g_composite_seg = {}
    for seg in SEGMENT_SIRASI:
        df_seg = islem_df[islem_df["musteri_segmenti"] == seg].reset_index(drop=True)
        g_composite_seg[seg], _ = build_composite_net(df_seg, min_edge=3)

    return {
        **bundle,
        "islem_df": islem_df,
        "DONEM_ALIM": donem_alim,
        "DONEM_SATIM": donem_satim,
        "DONEM_ALIM_PIVOT": donem_alim_pivot,
        "DONEM_SATIM_PIVOT": donem_satim_pivot,
        "DONEM_NET_PIVOT": donem_net_pivot,
        "DONEM_ALIM_POC": donem_alim_poc,
        "DONEM_SATIM_POC": donem_satim_poc,
        "DONEM_ISLEM_PIVOT": donem_islem_pivot,
        "alim_mst": alim_mst,
        "satim_mst": satim_mst,
        "HEAVY_BUYER_SEG": heavy_buyer_seg,
        "HEAVY_SELLER_SEG": heavy_seller_seg,
        "HEAVY_BUYER_TUTAR": heavy_buyer_tutar,
        "HEAVY_SELLER_TUTAR": heavy_seller_tutar,
        "HEAVY_ISLEM_DONEM": heavy_islem_donem,
        "_heavy_ids": heavy_ids,
        "_normal_ids": normal_ids,
        "COMPOSITE_FREQ": composite_freq,
        "G_COMPOSITE": g_composite,
        "G_COMPOSITE_SEG": g_composite_seg,
        "_composite_pairs": composite_pairs,
    }


def _compute_behavioral_intel(bundle: dict[str, Any]) -> dict[str, Any]:
    donem_net_pivot = bundle["DONEM_NET_PIVOT"]
    donem_alim_pivot = bundle["DONEM_ALIM_PIVOT"]
    donem_satim_pivot = bundle["DONEM_SATIM_PIVOT"]
    alim_mst = bundle["alim_mst"]
    satim_mst = bundle["satim_mst"]
    heavy_buyer_tutar = bundle["HEAVY_BUYER_TUTAR"]
    heavy_islem_donem = bundle["HEAVY_ISLEM_DONEM"]
    islem_df = bundle["islem_df"]
    g_composite = bundle["G_COMPOSITE"]
    g_composite_seg = bundle["G_COMPOSITE_SEG"]

    net = donem_net_pivot.reindex(SEGMENT_SIRASI)[DONEM_SIRASI].fillna(0)
    buy = donem_alim_pivot.reindex(SEGMENT_SIRASI)[DONEM_SIRASI].fillna(0)
    sell = donem_satim_pivot.reindex(SEGMENT_SIRASI)[DONEM_SIRASI].fillna(0)

    segment_rows = []
    for seg in SEGMENT_SIRASI:
        net_vals = net.loc[seg].values.astype(float)
        buy_vals = buy.loc[seg].values.astype(float)
        sell_vals = sell.loc[seg].values.astype(float)
        abs_mean = np.mean(np.abs(net_vals)) + 1e-9
        vol = 100 * np.std(net_vals) / abs_mean
        stability = 100 / (1 + vol / 100)
        persistence = 100 * np.mean(net_vals > 0)
        momentum = 100 * _safe_div(_series_slope(net_vals), abs_mean, default=0)
        buy_share = buy_vals / np.maximum(buy_vals + sell_vals, 1e-9)
        sell_share = sell_vals / np.maximum(buy_vals + sell_vals, 1e-9)
        risk_score = 100 * max(_series_slope(buy_share), 0)
        defensive_score = 100 * max(_series_slope(sell_share), 0)
        regime_shifts = _count_regime_shifts(net_vals)
        cyclical = _bounded(regime_shifts * 35 + min(vol, 100) * 0.25, 0, 100)
        regime = _regime_label(net_vals)
        if persistence == 100 and stability >= 55:
            state = "Stable_Accumulator"
        elif regime_shifts >= 2:
            state = "Cyclical_Rotator"
        elif defensive_score > risk_score and momentum < 0:
            state = "Defensive_Shift"
        elif risk_score >= defensive_score and momentum > 0:
            state = "Risk_Acceleration"
        elif persistence <= 25 and np.mean(net_vals) < 0:
            state = "Net_Distributor"
        else:
            state = "Adaptive_Mix"
        segment_rows.append(
            {
                "segment": seg,
                "Net_Flow_Stability_Score": round(stability, 1),
                "Buy_Sell_Regime_Shift_Indicator": regime_shifts,
                "Temporal_Volatility_Score": round(vol, 1),
                "Investment_Persistence_Score": round(persistence, 1),
                "Flow_Momentum_Score": round(momentum, 1),
                "Defensive_Score": round(defensive_score, 1),
                "Risk_Taking_Score": round(risk_score, 1),
                "Cyclical_Behavior_Score": round(cyclical, 1),
                "Regime_Label": regime,
                "Behavior_State": state,
                "Net_Mean_M": round(float(np.mean(net_vals)), 3),
                "Net_Last_M": round(float(net_vals[-1]), 3),
            }
        )
    segment_evolution_df = pd.DataFrame(segment_rows).set_index("segment").reindex(SEGMENT_SIRASI)

    buyer_heavy_period_rows = []
    for (seg, donem), grp in alim_mst.groupby(["musteri_segmenti", "donem"], observed=True):
        total = grp["toplam_alim"].sum()
        heavy_grp = grp[grp["heavy"] == True]
        buyer_heavy_period_rows.append(
            {
                "segment": seg,
                "donem": donem,
                "heavy_ratio": 100 * grp["heavy"].mean(),
                "heavy_volume_share": 100 * _safe_div(heavy_grp["toplam_alim"].sum(), total, default=0),
                "top1_share": 100 * _safe_div(grp["toplam_alim"].max(), total, default=0),
                "top3_share": 100 * _safe_div(grp["toplam_alim"].nlargest(min(3, len(grp))).sum(), total, default=0),
                "heavy_count": int(heavy_grp.shape[0]),
            }
        )
    buyer_heavy_period_df = pd.DataFrame(buyer_heavy_period_rows)
    seller_heavy_period_rows = []
    for (seg, donem), grp in satim_mst.groupby(["musteri_segmenti", "donem"], observed=True):
        total = grp["toplam_satim"].sum()
        heavy_grp = grp[grp["heavy"] == True]
        seller_heavy_period_rows.append(
            {
                "segment": seg,
                "donem": donem,
                "heavy_ratio": 100 * grp["heavy"].mean(),
                "heavy_volume_share": 100 * _safe_div(heavy_grp["toplam_satim"].sum(), total, default=0),
                "top1_share": 100 * _safe_div(grp["toplam_satim"].max(), total, default=0),
                "top3_share": 100 * _safe_div(grp["toplam_satim"].nlargest(min(3, len(grp))).sum(), total, default=0),
                "heavy_count": int(heavy_grp.shape[0]),
            }
        )
    seller_heavy_period_df = pd.DataFrame(seller_heavy_period_rows)

    heavy_rows = []
    for seg in SEGMENT_SIRASI:
        bseg = buyer_heavy_period_df[buyer_heavy_period_df["segment"] == seg].copy()
        sseg = seller_heavy_period_df[seller_heavy_period_df["segment"] == seg].copy()
        heavy_ids = alim_mst[(alim_mst["musteri_segmenti"] == seg) & (alim_mst["heavy"] == True)]
        recurring_ratio = 0.0
        if not heavy_ids.empty:
            counts = heavy_ids.groupby("musteri_id")["donem"].nunique()
            recurring_ratio = 100 * np.mean(counts >= 2)
        flip_ratio = 0.0
        buy_heavy = alim_mst[(alim_mst["musteri_segmenti"] == seg) & (alim_mst["heavy"] == True)][["musteri_id", "donem"]].copy()
        sell_heavy = satim_mst[(satim_mst["musteri_segmenti"] == seg) & (satim_mst["heavy"] == True)][["musteri_id", "donem"]].copy()
        if not buy_heavy.empty and not sell_heavy.empty:
            flip_ids = set()
            for _, row in buy_heavy.iterrows():
                later = sell_heavy[(sell_heavy["musteri_id"] == row["musteri_id"]) & (sell_heavy["donem"] >= row["donem"])]
                if not later.empty:
                    flip_ids.add(row["musteri_id"])
            flip_ratio = 100 * _safe_div(len(flip_ids), buy_heavy["musteri_id"].nunique(), default=0)
        heavy_gap = _safe_div(
            heavy_buyer_tutar.loc[seg, "Heavy_K"] if seg in heavy_buyer_tutar.index and "Heavy_K" in heavy_buyer_tutar.columns else 0,
            heavy_buyer_tutar.loc[seg, "Normal_K"] if seg in heavy_buyer_tutar.index and "Normal_K" in heavy_buyer_tutar.columns else 1,
            default=0,
        )
        intensity = float(heavy_islem_donem.loc[seg].mean()) if seg in heavy_islem_donem.index else 0.0
        dominance = 0.60 * (bseg["heavy_volume_share"].mean() if not bseg.empty else 0) + 0.40 * (bseg["top3_share"].mean() if not bseg.empty else 0)
        liquidity_dep = ((bseg["heavy_volume_share"].mean() if not bseg.empty else 0) + (sseg["heavy_volume_share"].mean() if not sseg.empty else 0)) / 2
        instability = (bseg["heavy_volume_share"].std(ddof=0) if len(bseg) > 1 else 0) + (sseg["heavy_volume_share"].std(ddof=0) if len(sseg) > 1 else 0)
        if dominance >= 70 and recurring_ratio >= 45:
            h_state = "Institutional_Heavy_Core"
        elif instability >= 18:
            h_state = "Unstable_Whale_Dependence"
        elif flip_ratio >= 35:
            h_state = "Roundtrip_Heavy_Flow"
        else:
            h_state = "Broad_Heavy_Base"
        heavy_rows.append(
            {
                "segment": seg,
                "Heavy_Dominance_Score": round(dominance, 1),
                "Whale_Dependency_Ratio": round(bseg["top1_share"].mean() if not bseg.empty else 0, 1),
                "Segment_Liquidity_Dependency": round(liquidity_dep, 1),
                "Heavy_Persistence_Ratio": round(recurring_ratio, 1),
                "Heavy_to_Seller_Flip_Ratio": round(flip_ratio, 1),
                "Heavy_Normal_Divergence": round(heavy_gap, 2),
                "Heavy_Transaction_Intensity_K": round(intensity, 1),
                "Heavy_Instability_Score": round(instability, 1),
                "Heavy_Behavior_State": h_state,
            }
        )
    heavy_intel_df = pd.DataFrame(heavy_rows).set_index("segment").reindex(SEGMENT_SIRASI)

    global_graph_intel = _graph_intel(g_composite)
    composite_node_intel_df = global_graph_intel["nodes"].copy()
    composite_shortest_paths_df = global_graph_intel["paths"].copy()
    segment_graph_rows = []
    for seg in SEGMENT_SIRASI:
        gseg = g_composite_seg.get(seg)
        gint = _graph_intel(gseg)
        seg_tx = islem_df[islem_df["musteri_segmenti"] == seg].copy()
        node_share = seg_tx["urun_tur_yon"].value_counts(normalize=True) if not seg_tx.empty else pd.Series(dtype=float)
        liquidity_ratio = 100 * node_share[[n for n in node_share.index if str(n).startswith("Vadesiz|") or str(n).startswith("Vadeli|")]].sum()
        aggressive_ratio = 100 * node_share[[n for n in node_share.index if str(n).startswith("Yatırım|") or str(n).startswith("Döviz|")]].sum()
        diversity = 100 * _safe_div(seg_tx["urun_tur_yon"].nunique(), max(islem_df["urun_tur_yon"].nunique(), 1), default=0)
        segment_graph_rows.append(
            {
                "segment": seg,
                "Node_Count": gint["graph"]["nodes"],
                "Edge_Count": gint["graph"]["edges"],
                "Graph_Density": round(gint["graph"]["density"], 4),
                "Transition_Entropy": round(gint["graph"]["avg_entropy"], 3),
                "Route_Concentration": round(100 * gint["graph"]["route_concentration"], 1),
                "Complexity_Score": round(100 * (0.55 * gint["graph"]["density"] + 0.45 * min(gint["graph"]["avg_entropy"], 1)), 1),
                "Dominant_Route": gint["graph"]["top_route"],
                "Hub_Node": gint["graph"]["hub_node"],
                "Bridge_Node": gint["graph"]["bridge_node"],
                "Source_Node": gint["graph"]["source_node"],
                "Sink_Node": gint["graph"]["sink_node"],
                "Liquidity_Parking_Ratio": round(liquidity_ratio, 1),
                "Aggressive_Routing_Ratio": round(aggressive_ratio, 1),
                "Transaction_Diversity": round(diversity, 1),
            }
        )
    segment_graph_fingerprint_df = pd.DataFrame(segment_graph_rows).set_index("segment").reindex(SEGMENT_SIRASI)

    affluent = ["Kurumsal", "Kurumsal_Premium", "Private_Banking", "Ultra_HNW"]
    retail = [seg for seg in SEGMENT_SIRASI if seg not in affluent]
    affluent_node = islem_df[islem_df["musteri_segmenti"].isin(affluent)]["urun_tur_yon"].value_counts(normalize=True)
    retail_node = islem_df[islem_df["musteri_segmenti"].isin(retail)]["urun_tur_yon"].value_counts(normalize=True)
    affluent_node_dominance_df = (
        pd.DataFrame({"Affluent_Share": affluent_node, "Retail_Share": retail_node})
        .fillna(0)
        .assign(Dominance_Ratio=lambda x: (x["Affluent_Share"] + 1e-9) / (x["Retail_Share"] + 1e-9))
        .sort_values("Dominance_Ratio", ascending=False)
        .reset_index()
        .rename(columns={"index": "node"})
    )

    islem_temporal = islem_df[["musteri_id", "musteri_segmenti", "islem_tarihi", "islem_tutari", "urun_tur_yon"]].copy()
    islem_temporal["donem"] = _assign_donem_col(islem_temporal["islem_tarihi"])
    temporal_composite_graphs = {}
    temporal_graph_rows = []
    temporal_edgesets = {}
    temporal_pr = {}
    for donem in DONEM_SIRASI:
        df_d = islem_temporal[islem_temporal["donem"] == donem].copy()
        g_d, _ = build_composite_net(df_d, min_edge=4)
        temporal_composite_graphs[donem] = g_d
        intel_d = _graph_intel(g_d)
        temporal_edgesets[donem] = set((u, v) for u, v in g_d.edges())
        temporal_pr[donem] = intel_d["nodes"][["node", "pagerank"]].copy() if not intel_d["nodes"].empty else pd.DataFrame(columns=["node", "pagerank"])
        temporal_graph_rows.append(
            {
                "donem": donem,
                "Node_Count": intel_d["graph"]["nodes"],
                "Edge_Count": intel_d["graph"]["edges"],
                "Graph_Density": round(intel_d["graph"]["density"], 4),
                "Transition_Entropy": round(intel_d["graph"]["avg_entropy"], 3),
                "Route_Concentration": round(100 * intel_d["graph"]["route_concentration"], 1),
                "Complexity_Score": round(100 * (0.55 * intel_d["graph"]["density"] + 0.45 * min(intel_d["graph"]["avg_entropy"], 1)), 1),
                "Dominant_Route": intel_d["graph"]["top_route"],
                "Top_Bridge": intel_d["graph"]["bridge_node"],
            }
        )
    temporal_graph_evolution_df = pd.DataFrame(temporal_graph_rows).set_index("donem").reindex(DONEM_SIRASI)

    stability_rows = []
    for idx, d1 in enumerate(DONEM_SIRASI):
        for d2 in DONEM_SIRASI[idx + 1 :]:
            e1 = temporal_edgesets.get(d1, set())
            e2 = temporal_edgesets.get(d2, set())
            inter = len(e1.intersection(e2))
            union = max(len(e1.union(e2)), 1)
            stability_rows.append(
                {
                    "from": d1,
                    "to": d2,
                    "Transition_Stability": round(100 * inter / union, 1),
                    "Emerging_Edges": max(len(e2 - e1), 0),
                    "Disappearing_Edges": max(len(e1 - e2), 0),
                }
            )
    temporal_stability_df = pd.DataFrame(stability_rows)

    node_survival = Counter()
    for donem in DONEM_SIRASI:
        for node in temporal_composite_graphs[donem].nodes():
            node_survival[node] += 1
    node_survival_df = (
        pd.DataFrame([
            {"node": node, "Node_Survival_Score": round(100 * cnt / len(DONEM_SIRASI), 1), "label": short_comp_label(node)}
            for node, cnt in node_survival.items()
        ])
        .sort_values(["Node_Survival_Score", "node"], ascending=[False, True])
        .reset_index(drop=True)
        if node_survival
        else pd.DataFrame(columns=["node", "Node_Survival_Score", "label"])
    )

    temporal_pr_rows = []
    for donem, pr_df in temporal_pr.items():
        if pr_df.empty:
            continue
        tmp = pr_df.copy()
        tmp["donem"] = donem
        temporal_pr_rows.append(tmp)
    temporal_centrality_shift_df = (
        pd.concat(temporal_pr_rows, ignore_index=True).pivot(index="node", columns="donem", values="pagerank").reindex(columns=DONEM_SIRASI).fillna(0)
        if temporal_pr_rows
        else pd.DataFrame()
    )

    period_shocks = []
    for idx, donem in enumerate(DONEM_SIRASI):
        if idx == 0:
            period_shocks.append({"donem": donem, "Transition_Shock_Score": 0.0})
            continue
        prev = DONEM_SIRASI[idx - 1]
        stab = temporal_stability_df[(temporal_stability_df["from"] == prev) & (temporal_stability_df["to"] == donem)]
        stab_score = 100 - (float(stab["Transition_Stability"].iloc[0]) if not stab.empty else 0)
        dens_delta = abs(temporal_graph_evolution_df.loc[donem, "Graph_Density"] - temporal_graph_evolution_df.loc[prev, "Graph_Density"]) * 400
        ent_delta = abs(temporal_graph_evolution_df.loc[donem, "Transition_Entropy"] - temporal_graph_evolution_df.loc[prev, "Transition_Entropy"]) * 100
        period_shocks.append({"donem": donem, "Transition_Shock_Score": round(stab_score * 0.55 + dens_delta * 0.20 + ent_delta * 0.25, 1)})
    period_shock_df = pd.DataFrame(period_shocks).set_index("donem").reindex(DONEM_SIRASI)

    edge_period_rows = []
    for donem in DONEM_SIRASI:
        graph = temporal_composite_graphs[donem]
        total_w = max(sum(d.get("weight", 0) for _, _, d in graph.edges(data=True)), 1)
        for u, v, d in graph.edges(data=True):
            edge_period_rows.append({"donem": donem, "edge": f"{u} -> {v}", "share": 100 * d.get("weight", 0) / total_w, "weight": d.get("weight", 0)})
    temporal_edge_share_df = pd.DataFrame(edge_period_rows)
    rare_transition_spikes_df = pd.DataFrame(columns=["donem", "edge", "Anomaly_Score", "weight"])
    if not temporal_edge_share_df.empty:
        global_edge_share = temporal_edge_share_df.groupby("edge")["share"].mean().rename("global_share")
        rare = temporal_edge_share_df.merge(global_edge_share, on="edge", how="left")
        rare["Anomaly_Score"] = rare["share"] - rare["global_share"]
        rare_transition_spikes_df = rare[rare["Anomaly_Score"] > rare["Anomaly_Score"].quantile(0.85)].sort_values(["Anomaly_Score", "weight"], ascending=False).reset_index(drop=True)

    archetype_rows = []
    for seg in SEGMENT_SIRASI:
        evo = segment_evolution_df.loc[seg]
        hvy = heavy_intel_df.loc[seg]
        gfp = segment_graph_fingerprint_df.loc[seg]
        scores = {
            "Liquidity_Parker": gfp["Liquidity_Parking_Ratio"] * 1.1 + (100 - evo["Risk_Taking_Score"]) * 0.2,
            "Tactical_Allocator": evo["Cyclical_Behavior_Score"] * 0.8 + gfp["Transition_Entropy"] * 45,
            "Momentum_Chaser": max(evo["Flow_Momentum_Score"], 0) * 1.2 + evo["Temporal_Volatility_Score"] * 0.35,
            "Defensive_Investor": evo["Defensive_Score"] * 1.4 + max(-evo["Flow_Momentum_Score"], 0) * 0.7,
            "Rotational_Trader": evo["Cyclical_Behavior_Score"] * 1.1 + gfp["Complexity_Score"] * 0.45,
            "Institutional_Rebalancer": gfp["Complexity_Score"] * 0.85 + hvy["Heavy_Persistence_Ratio"] * 0.55 + (20 if seg in affluent else 0),
            "Opportunistic_Buyer": max(evo["Flow_Momentum_Score"], 0) * 1.0 + (35 if evo["Regime_Label"] == "Net_Satici_to_Net_Alici" else 0),
            "Panic_Seller": max(-evo["Flow_Momentum_Score"], 0) * 1.0 + hvy["Heavy_to_Seller_Flip_Ratio"] * 0.55,
            "Stable_Accumulator": evo["Investment_Persistence_Score"] * 0.8 + evo["Net_Flow_Stability_Score"] * 0.7,
        }
        rank = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        archetype_rows.append({
            "segment": seg,
            "Primary_Archetype": rank[0][0],
            "Secondary_Archetype": rank[1][0],
            "Primary_Score": round(rank[0][1], 1),
            "Secondary_Score": round(rank[1][1], 1),
            **{f"score_{k}": round(v, 1) for k, v in scores.items()},
        })
    behavioral_archetype_df = pd.DataFrame(archetype_rows).set_index("segment").reindex(SEGMENT_SIRASI)

    alert_rows = []
    for seg in SEGMENT_SIRASI:
        evo = segment_evolution_df.loc[seg]
        hvy = heavy_intel_df.loc[seg]
        gfp = segment_graph_fingerprint_df.loc[seg]
        instability = evo["Temporal_Volatility_Score"] * 0.30 + (100 - evo["Net_Flow_Stability_Score"]) * 0.25 + hvy["Heavy_Instability_Score"] * 0.20 + hvy["Whale_Dependency_Ratio"] * 0.15 + gfp["Route_Concentration"] * 0.10
        anomaly = instability + abs(evo["Flow_Momentum_Score"]) * 0.18 + evo["Buy_Sell_Regime_Shift_Indicator"] * 8
        alert_rows.append({"segment": seg, "Segment_Instability_Index": round(instability, 1), "Anomaly_Score": round(anomaly, 1), "Alert_Label": "Yüksek" if anomaly >= 65 else ("Orta" if anomaly >= 40 else "Düşük")})
    segment_alert_df = pd.DataFrame(alert_rows).set_index("segment").reindex(SEGMENT_SIRASI)

    cmp_rows = []
    for grp_name, grp_segs in {"Affluent_Institutions": affluent, "Retail_Base": retail}.items():
        cmp_rows.append(
            {
                "group": grp_name,
                "Avg_Stability": round(segment_evolution_df.loc[grp_segs, "Net_Flow_Stability_Score"].mean(), 1),
                "Avg_Momentum": round(segment_evolution_df.loc[grp_segs, "Flow_Momentum_Score"].mean(), 1),
                "Avg_Heavy_Persistence": round(heavy_intel_df.loc[grp_segs, "Heavy_Persistence_Ratio"].mean(), 1),
                "Avg_Complexity": round(segment_graph_fingerprint_df.loc[grp_segs, "Complexity_Score"].mean(), 1),
                "Avg_Liquidity_Parking": round(segment_graph_fingerprint_df.loc[grp_segs, "Liquidity_Parking_Ratio"].mean(), 1),
                "Avg_Aggressive_Routing": round(segment_graph_fingerprint_df.loc[grp_segs, "Aggressive_Routing_Ratio"].mean(), 1),
            }
        )
    executive_comparison_df = pd.DataFrame(cmp_rows).set_index("group")

    return {
        **bundle,
        "SEGMENT_EVOLUTION_DF": segment_evolution_df,
        "BUYER_HEAVY_PERIOD_DF": buyer_heavy_period_df,
        "SELLER_HEAVY_PERIOD_DF": seller_heavy_period_df,
        "HEAVY_INTEL_DF": heavy_intel_df,
        "GLOBAL_GRAPH_INTEL": global_graph_intel,
        "COMPOSITE_NODE_INTEL_DF": composite_node_intel_df,
        "COMPOSITE_SHORTEST_PATHS_DF": composite_shortest_paths_df,
        "SEGMENT_GRAPH_FINGERPRINT_DF": segment_graph_fingerprint_df,
        "AFFLUENT_NODE_DOMINANCE_DF": affluent_node_dominance_df,
        "TEMPORAL_COMPOSITE_GRAPHS": temporal_composite_graphs,
        "TEMPORAL_GRAPH_EVOLUTION_DF": temporal_graph_evolution_df,
        "TEMPORAL_STABILITY_DF": temporal_stability_df,
        "NODE_SURVIVAL_DF": node_survival_df,
        "TEMPORAL_CENTRALITY_SHIFT_DF": temporal_centrality_shift_df,
        "PERIOD_SHOCK_DF": period_shock_df,
        "TEMPORAL_EDGE_SHARE_DF": temporal_edge_share_df,
        "RARE_TRANSITION_SPIKES_DF": rare_transition_spikes_df,
        "BEHAVIORAL_ARCHETYPE_DF": behavioral_archetype_df,
        "SEGMENT_ALERT_DF": segment_alert_df,
        "EXECUTIVE_COMPARISON_DF": executive_comparison_df,
    }


def _generate_executive_report(bundle: dict[str, Any]) -> str:
    segment_evolution_df = bundle["SEGMENT_EVOLUTION_DF"]
    heavy_intel_df = bundle["HEAVY_INTEL_DF"]
    segment_graph_fingerprint_df = bundle["SEGMENT_GRAPH_FINGERPRINT_DF"]
    composite_node_intel_df = bundle["COMPOSITE_NODE_INTEL_DF"]
    affluent_node_dominance_df = bundle["AFFLUENT_NODE_DOMINANCE_DF"]
    period_shock_df = bundle["PERIOD_SHOCK_DF"]
    temporal_graph_evolution_df = bundle["TEMPORAL_GRAPH_EVOLUTION_DF"]
    behavioral_archetype_df = bundle["BEHAVIORAL_ARCHETYPE_DF"]
    executive_comparison_df = bundle["EXECUTIVE_COMPARISON_DF"]
    segment_alert_df = bundle["SEGMENT_ALERT_DF"]

    def fmt_seg_list(items: list[str]) -> str:
        return ", ".join(str(x).replace("_", " ") for x in items) if len(items) else "bulgu yok"

    def top_segments(df: pd.DataFrame, col: str, n: int = 3, ascending: bool = False) -> list[str]:
        return df.sort_values(col, ascending=ascending).head(n).index.tolist()

    regime_turners = segment_evolution_df[segment_evolution_df["Regime_Label"].isin(["Net_Alici_to_Net_Satici", "Net_Satici_to_Net_Alici"])].index.tolist()
    stable_acc = top_segments(segment_evolution_df.assign(_rank=segment_evolution_df["Investment_Persistence_Score"] + segment_evolution_df["Net_Flow_Stability_Score"]), "_rank", n=3)
    defensive = top_segments(segment_evolution_df.assign(_rank=segment_evolution_df["Defensive_Score"] - segment_evolution_df["Flow_Momentum_Score"]), "_rank", n=3)
    risk_on = top_segments(segment_evolution_df.assign(_rank=segment_evolution_df["Risk_Taking_Score"] + segment_evolution_df["Flow_Momentum_Score"].clip(lower=0)), "_rank", n=3)
    cyclical = top_segments(segment_evolution_df, "Cyclical_Behavior_Score", n=3)
    heavy_dep = top_segments(heavy_intel_df, "Heavy_Dominance_Score", n=3)
    heavy_unstable = top_segments(heavy_intel_df, "Heavy_Instability_Score", n=3)
    heavy_flip = top_segments(heavy_intel_df, "Heavy_to_Seller_Flip_Ratio", n=3)
    inst_like = top_segments(
        heavy_intel_df[["Heavy_Persistence_Ratio"]]
        .join(segment_graph_fingerprint_df[["Complexity_Score"]])
        .assign(Composite=lambda x: x["Heavy_Persistence_Ratio"] * 0.45 + x["Complexity_Score"] * 0.55),
        "Composite",
        n=3,
    )
    hubs = composite_node_intel_df.sort_values("pagerank", ascending=False).head(5)["label"].tolist() if not composite_node_intel_df.empty else []
    bridges = composite_node_intel_df.sort_values("betweenness", ascending=False).head(5)["label"].tolist() if not composite_node_intel_df.empty else []
    node_col = "node" if "node" in affluent_node_dominance_df.columns else affluent_node_dominance_df.columns[0]
    affluent_nodes = [short_comp_label(n) for n in affluent_node_dominance_df.head(5)[node_col].tolist()] if not affluent_node_dominance_df.empty else []
    shock_period = period_shock_df["Transition_Shock_Score"].idxmax()
    shock_val = period_shock_df.loc[shock_period, "Transition_Shock_Score"]
    graph_first = temporal_graph_evolution_df.iloc[0]
    graph_last = temporal_graph_evolution_df.iloc[-1]
    complexity_delta = graph_last["Complexity_Score"] - graph_first["Complexity_Score"]
    entropy_delta = graph_last["Transition_Entropy"] - graph_first["Transition_Entropy"]

    affluent_lines = []
    for seg in ["Kurumsal", "Kurumsal_Premium", "Private_Banking", "Ultra_HNW"]:
        evo = segment_evolution_df.loc[seg]
        hvy = heavy_intel_df.loc[seg]
        gfp = segment_graph_fingerprint_df.loc[seg]
        arc = behavioral_archetype_df.loc[seg]
        affluent_lines.append(
            f"- {seg.replace('_', ' ')}: {arc['Primary_Archetype']} baskın; complexity={gfp['Complexity_Score']:.1f}, route concentration={gfp['Route_Concentration']:.1f}%, heavy persistence={hvy['Heavy_Persistence_Ratio']:.1f}%, stability={evo['Net_Flow_Stability_Score']:.1f}, momentum={evo['Flow_Momentum_Score']:.1f}."
        )

    return "\n".join(
        [
            "=" * 88,
            "  BANKACILIK DAVRANIŞSAL ZEKA PLATFORMU — YÖNETİCİ İÇGÖRÜ RAPORU",
            "=" * 88,
            "",
            "1) Segment Evolution",
            f"- Net rejim dönüşü yaşayan segmentler: {fmt_seg_list(regime_turners)}.",
            f"- En istikrarlı birikim davranışı gösterenler: {fmt_seg_list(stable_acc)}.",
            f"- Defansifleşen segmentler: {fmt_seg_list(defensive)}.",
            f"- Risk alma iştahı hızlanan segmentler: {fmt_seg_list(risk_on)}.",
            f"- Döngüsel/rotasyonel davranış sergileyenler: {fmt_seg_list(cyclical)}.",
            "",
            "2) Heavy Buyer / Seller Intelligence",
            f"- Heavy yatırımcı bağımlılığı en yüksek segmentler: {fmt_seg_list(heavy_dep)}.",
            f"- Heavy davranışı en oynak segmentler: {fmt_seg_list(heavy_unstable)}.",
            f"- Heavy buyer sonrası heavy seller dönüşü en belirgin segmentler: {fmt_seg_list(heavy_flip)}.",
            f"- Kurumsal benzeri tekrar eden heavy tabanı üreten segmentler: {fmt_seg_list(inst_like)}.",
            "",
            "3) Composite Graph Intelligence",
            f"- Global ağın ana transition hub node'ları: {', '.join(hubs)}.",
            f"- Köprü/gateway rolü en güçlü node'lar: {', '.join(bridges)}.",
            f"- Affluent segmentlerde retail'e göre daha baskın node'lar: {', '.join(affluent_nodes)}.",
            f"- Dönemsel en yüksek rejim şoku: {shock_period} döneminde {float(shock_val):.1f} skor.",
            f"- İlk dönemden son döneme network complexity değişimi: {float(complexity_delta):+.1f}; entropy değişimi: {float(entropy_delta):+.2f}.",
            "",
            "4) Affluent / Institutional Deep Dive",
            *affluent_lines,
            "",
            "5) Board-Level Reading",
            f"- Affluent/kurumsal havuzun ortalama graph complexity skoru {executive_comparison_df.loc['Affluent_Institutions', 'Avg_Complexity']:.1f}; retail bazında bu seviye {executive_comparison_df.loc['Retail_Base', 'Avg_Complexity']:.1f}.",
            f"- Affluent/kurumsal müşteri gruplarında heavy persistence {executive_comparison_df.loc['Affluent_Institutions', 'Avg_Heavy_Persistence']:.1f}%; retail bazında {executive_comparison_df.loc['Retail_Base', 'Avg_Heavy_Persistence']:.1f}%.",
            f"- Liquidity parking oranı affluent/kurumsal tarafta {executive_comparison_df.loc['Affluent_Institutions', 'Avg_Liquidity_Parking']:.1f}%, retail bazda {executive_comparison_df.loc['Retail_Base', 'Avg_Liquidity_Parking']:.1f}%.",
            f"- Agresif routing affluent/kurumsal grupta {executive_comparison_df.loc['Affluent_Institutions', 'Avg_Aggressive_Routing']:.1f}%, retail bazda {executive_comparison_df.loc['Retail_Base', 'Avg_Aggressive_Routing']:.1f}% seviyesinde.",
            f"- En yüksek segment anomalisi: {segment_alert_df['Anomaly_Score'].idxmax().replace('_', ' ')} ({segment_alert_df['Anomaly_Score'].max():.1f}).",
            "",
            "6) Strategic Interpretation",
            "- Platform, segment bazlı betimleyici görünümün ötesine geçerek dönemsel rejim değişimi, heavy yatırımcı yoğunlaşması, kompozit geçiş rotaları ve davranış arketipleri üzerinden yatırım eğilimi okuyan bir davranışsal zeka katmanı sunar.",
            "- Persistence düşük ama heavy dominance yüksek segmentler likidite kırılganlığı; complexity ve heavy persistence birlikte yüksek segmentler ise daha sofistike, kurumsal benzeri davranış kalıbı olarak yorumlanmalıdır.",
            "- Transition shock skoru yükselen dönemler, ürün yönelimi ve kompozit rota yapısında rejim değişimi barındırdığı için bilanço davranışı, kampanya etkisi ya da sentetik makro stres ile birlikte okunmalıdır.",
        ]
    )


def build_analysis_bundle(seed: int = RANDOM_SEED) -> AnalysisBundle:
    data = _generate_synthetic_data(seed=seed)
    bundle = _compute_base_metrics(data)
    bundle = _compute_temporal_and_heavy(bundle)
    bundle = _compute_behavioral_intel(bundle)
    bundle["SEG_SHORT"] = [_short_label(seg) for seg in SEGMENT_SIRASI]
    bundle["EXECUTIVE_REPORT_TEXT"] = _generate_executive_report(bundle)
    bundle["seed"] = seed
    return AnalysisBundle(data=bundle)
