# Banka Fon Müşteri Davranış Analizi — 2025.09 & 2026.03

2 dönemlik (Eylül 2025 · Mart 2026) banka fon yatırım müşteri analiz platformu.

## İçerik

| Dosya | Açıklama |
|---|---|
| `fund_analysis.ipynb` | Ana notebook (2 hücre) |
| `fund_analysis_part1.py` | Hücre 1: Konfigürasyon + Veri + Analiz |
| `fund_analysis_part2.py` | Hücre 2: Chartlar + Profesyonel Network |
| `make_notebook.py` | .ipynb oluşturma scripti |
| `analytics_engine.py` | Streamlit analitik motoru |
| `streamlit_app.py` | Streamlit dashboard (5 sekme) |

## Analizler

- **A — Genel Bakış**: Penetrasyon, AUM, Sadakat, Bakiye Dilimi (2 dönem)
- **B — Alım Davranışı**: Pre/Post-Buy ürün mix, giriş oranı, net akış
- **C — Satım Davranışı**: Pre/Post-Sell analizi, geçiş matrisi
- **D — Dönemsel Karşılaştırma**: Heatmaplar, heavy buyer/seller
- **E — Sekansiyel Ağlar**: İşlem geçiş ağları (tüm + pencere bazlı)
- **G — t-1 Bakiye Analizi**: Önceki dönem bakiyesine göre alım/satım oranı
- **H — Fon-Geçmeyen Sekans**: Vadeli/Yatırım/Döviz işlem pattern analizi
- **I — Profesyonel Network**: Yapılandırılmış hiyerarşik kompozit ağ (HTML interaktif)

## Dönemler

| Dönem | Başlangıç | Bitiş |
|---|---|---|
| 2025.09 | 2025-09-01 | 2025-09-30 |
| 2026.03 | 2026-03-01 | 2026-03-31 |

## Kurulum

```bash
pip install numpy pandas matplotlib seaborn networkx plotly nbformat
jupyter notebook fund_analysis.ipynb
```

## Segmentler (10 adet)

Bireysel_Standart · Bireysel_Premium · Bireysel_Elite ·
KOBİ · KOBİ_Orta · KOBİ_Büyük ·
Kurumsal · Kurumsal_Premium · Private_Banking · Ultra_HNW
