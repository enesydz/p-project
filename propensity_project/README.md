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

## Newsell / Upsell Propensity ML

Yeni [fund_propensity_ml.ipynb](fund_propensity_ml.ipynb) notebook'u mevcut betimleyici fon analizinden bağımsız müşteri-ay modelleme akışıdır. Çekirdek fonksiyonlar [fund_propensity.py](fund_propensity.py) içinde tutulur; notebook [make_propensity_notebook.py](make_propensity_notebook.py) ile yeniden üretilebilir.

Notebook yalnızca şu beş uygulama bölümünü içerir: `Data Load`, `Preprocess`, `Feature Engineering`, `Modelling` ve `Reporting`. Ayrı EDA veya görsel analiz hücreleri bulunmaz. Modelleme aşamasında LightGBM parametreleri Optuna TPE ile aranır; `MedianPruner` ve iterasyon callback'i zayıf denemeleri erken keser.

### Model kapsamı

- 2 ürün sınıfı: `para_piyasasi` ve `nitelikli`
- 2 davranış: `newsell` ve `upsell`
- Toplam 4 model ailesi
- Segment ID `1..5`: her segment ve her grid konfigürasyonu için bağımsız model
- Parametre grid'i: `x_grid`, `y_grid`, `r_grid`
- Zaman bölme: kronolojik train, test ve OOT; random split kullanılmaz
- Her Optuna trial ve seçilen model için train/test/OOT PR-AUC, ROC-AUC, Brier, precision@k, recall@k ve lift@k
- Rare-event desteği: yüzde `0.1` ve daha düşük hedef oranlarında split başına minimum pozitif kontrolü, çok aylı test/OOT penceresi, prevalence ve pozitif adet raporu
- Sürekli feature standardizasyonu: train, test ve OOT feature değerleri enflasyon tablosundan hesaplanan ortak OOT referans tarihine taşınır; target üretimi nominal iş kuralı üzerinde korunur
- Chart çıktıları: notebook output'unda teknik/genel performans chart'ları, output klasöründe PNG dosyaları ve Excel'de gömülü chart'lar

### Manuel aylık DataFrame sözleşmesi

Notebook örneği veri üretimini beş DataFrame olarak bellekte yapar; CSV round-trip kullanmaz. Gerçek uygulamada bu DataFrame'ler banka kaynaklarından manuel olarak hazırlanıp doğrudan `build_source_bundle_from_dataframes` fonksiyonuna verilir. Tüm tablolar müşteri ID + ay anahtarıyla bağlanır; beklenen dönem `2025-09` ile `2026-06` arasındadır.

| DataFrame | Zorunlu alanlar | Kullanım |
|---|---|---|
| `input_df` | `musteri_id`, `tarih`, `segment_id`, `tutar_amount`, aylık değişkenler | Müşteri-ay ana tablosu, ana tutar ve feature adayları |
| `activity_df` | `musteri_id`, `tarih`, `ppf_aktif`, `nf_aktif` | PPF/NF aktiflik flag'leri; yalnızca `0/1` |
| `fund_df` | `musteri_id`, `tarih`, `para_flg`, `fon_amount` | `para_flg=1` PPF, `para_flg=0` NF fon tutarı |
| `transaction_df` | `musteri_id`, `tarih`, `para_flg`, `fon_alim_amount`, `fon_satim_amount` | PPF/NF işlemleri; net alım = alım - satım |
| `inflation_df` | `tarih`, `aylik_enflasyon` | Sürekli tutarları ortak OOT ayına taşımak için aylık enflasyon |

Doğrudan kullanım:

```python
from fund_propensity import PropensityConfig, build_source_bundle_from_dataframes

bundle = build_source_bundle_from_dataframes(
	input_df=df_input,
	activity_df=df_aktiflik,
	fund_df=df_fon_tutar,
	transaction_df=df_fon_alim_satim,
	inflation_df=df_enflasyon,
	config=PropensityConfig(),
)
panel = build_canonical_panel(bundle, PropensityConfig())
```

Adapter `tarih` kolonunu dahili `month` adına, `segment_id` kolonunu `segment` adına ve tutar/işlem kolonlarını canonical isimlere çevirir. Ana input'taki `tutar_amount` feature tablosuna taşınır ve OOT tutar gruplamasının sıralama kolonu olarak kullanılır. Kaynaklarda aynı müşteri-ay-ürün birden fazla kez bulunursa tutar ve işlem akışları toplanır; aktiflik ve enflasyon değerleri doğrulanır.

Dosya adları CSV, TXT veya Parquet olabilir. Farklı kolon adları varsa Data Load hücresindeki kolon config değerleri değiştirilir. Input tablosundaki aylık değişkenler `INPUT_TABLE_FEATURE_COLUMNS` ile açıkça seçilebilir.

### Normalize veri sözleşmesi

Notebook gerçek kaynak tablolarını aşağıdaki dosya adlarıyla CSV veya Parquet olarak bekler. `PROPENSITY_DATA_ROOT` klasöründe Parquet varsa CSV'ye göre önceliklidir.

| Dosya | Zorunlu kolonlar |
|---|---|
| `customers` | `musteri_id`, `segment` veya `segment_id` (`1..5`) |
| `activity` | `musteri_id`, `month`, `ppf_aktif`, `nf_aktif` veya canonical `product_class`, `active_flag` |
| `flows` | `musteri_id`, `month`, `product_class`, `buy_amount`, `sell_amount`, `fund_value` |
| `monthly_features` | `musteri_id`, `month`, aylık değişkenler; opsiyonel |
| `inflation` | `month`, `inflation_rate`; opsiyonel |

Aktiflik kaynağı bankadaki aylık geniş formatta `ppf_aktif` ve `nf_aktif` kolonlarını içeriyorsa notebook bunları sırasıyla `para_piyasasi` ve `nitelikli` ürün satırlarına açar. Flag değerleri yalnızca `0` veya `1` olmalıdır. Canonical uzun format (`product_class`, `active_flag`) da desteklenir. `product_class` yalnızca `para_piyasasi` veya `nitelikli` olmalıdır. Upsell net alımı varsayılan olarak `buy_amount - sell_amount` hesaplanır.

### Çalıştırma

```powershell
$env:PROPENSITY_DATA_ROOT = "C:\data\fund_propensity_normalized"
python make_propensity_notebook.py
jupyter notebook fund_propensity_ml.ipynb
```

`PROPENSITY_DATA_ROOT` kullanıcı tarafından belirtilen manuel tablo klasörüdür. Klasör veya tablo config'i eksikse notebook açık hata verir; sentetik fallback yoktur.

Gerçek manuel tablolarla beş segmentte dört model ailesi (`newsell/upsell × para_piyasasi/nitelikli`) çalıştırılır. `segment` model girdisi değildir; model partition anahtarıdır.

### Audit ve Excel çıktısı

Reporting aşaması varsayılan olarak `propensity_outputs/fund_propensity_pipeline_audit_segmented.xlsx` dosyasını üretir. Workbook'un ilk sheet'i `00_Runtime_Summary` çalışma süresini, veri hacmini, uygulanan preprocessing adımlarını, rare-event PR-AUC kontrolünü ve overfit kararını genel bilgilendirme tablosu olarak gösterir. `Stage_Summary` ve `Operation_Log` sheet'leri her aşamanın satır/kolon akışını, uygulanan işlemi ve durumunu gösterir. `Variable_Quality` kaynak, panel ve feature tablolarındaki her kolon için dtype, null oranı, unique sayısı, sıfır/negatif oranı, sonsuz değer sayısı, min/max/ortalama/standart sapma ve kalite durumunu içerir. `Inflation_Audit`, her ayın enflasyon index'ini, ortak OOT referans tarihine taşıma katsayısını ve dönüştürülen sürekli kolonları gösterir. `Performance_Summary` teknik chart verisini, `General_Summary` genel pipeline hacim chart verisini, `Overfit_Audit` ise model bazında train-test ve test-OOT PR-AUC gap/lift kontrollerini taşır; chart nesneleri aynı Excel sheet'lerine gömülür.

Çıktı adı `PROPENSITY_REPORT_NAME` ortam değişkeniyle değiştirilebilir; Excel dosyası açıkken yeni raporu örneğin `fund_propensity_pipeline_audit_advanced.xlsx` adıyla yazabilirsiniz.

`Eliminated_Vars` ve `Feature_Quality` sheet'lerinde key kolonlarının, constant kolonların, dönüştürülen aktivite kolonlarının ve modele alınmayan diğer değişkenlerin açık karar gerekçesi bulunur. `Feature_Engineering` her segment için değişkenin kaynağını, dönüşümünü, lookback/anchor dönemini ve dağılım bazlı outlier analizini taşır. `Model_Feature_Audit` her segment-model splitinde train verisinden öğrenilen missing, outlier, high-cardinality, correlation ve encoding kararlarını; train dönemi kapsamını ve fit scope'u gösterir. Missingness, constant, kategorik kardinalite/oran, outlier clipping, missing indicator ve correlation kontrollerinin her biri notebook config'inde ayrı ayrı açılıp kapatılabilir; kapalı kontroller audit'te `disabled` olarak görünür. Numeric değişkenler ilgili segmentin train quantile sınırlarıyla clip edilir; kategorik değişkenler yalnız train kategorileriyle one-hot encode edilir ve validation/OOT'ta görülmeyen kategoriler güvenli biçimde sıfır vektörüne dönüşür. Yalnız quality kontrolünü geçen ve `musteri_id`, `anchor_month`, `product_class`, `segment` olmayan kolonlar model girdisi olarak Optuna-LightGBM'e aktarılır. `Target_Quality` segment x grid bazında tüm uygunluk kararlarını içerir. `Model_Metrics` içinde ROC-AUC, Gini, PR-AUC, Brier ve lift metrikleri; `best_trial_model` train-only performansı ile `final_selected_model` train+test refit performansı ayrı etiketlenir. `Amount_Group_Performance` OOT skorlarını her segment içinde ana input tablosundan gelen `tutar_amount` sıralamasına göre eşit adetli 10 gruba ayırır ve her grup için örneklem, pozitif oranı, ROC-AUC, Gini, PR-AUC ve TOP-K lift değerlerini raporlar. `Optuna_Trials` her trial için train/test/OOT metriklerini ve durumunu taşır; `Target_Audit`, `OOT_Scores` ve `Campaign_Scores` tamamlayıcı model ve aksiyon çıktılarıdır. Excel yazımı için `openpyxl` gereklidir.
Enflasyon referans ayı notebook config'inde `PROPENSITY_INFLATION_REFERENCE_MONTH` ile verilebilir. Boş bırakılırsa tüm grid için ortak referans, panelin son ayından en küçük `y_window` çıkarılarak seçilen en güncel tamamlanmış OOT anchor ayıdır. `fund_value`, `buy_amount`, `sell_amount`, `net_buy`, `monthly_income` gibi sürekli değişkenler bu tarihe taşınır; flag, rate, ratio, count ve index kolonları taşınmaz. `fund_value_real` ve rolling monetary feature'lar bu ortak bazda üretilir.

Notebook reporting hücresi eleme/trimming özetlerini, `Runtime_Summary` ve `Overfit_Audit` tablolarını doğrudan output'a basar; ayrıca `propensity_performance_technical.png` ve `propensity_performance_general.png` dosyalarını output klasörüne yazar. Teknik chart rare-event için ana metrik olan PR-AUC ile ROC-AUC'yi ayrı panellerde ve ayrı Excel chart'larında gösterir. `Overfit_Audit`, train-test veya test-OOT PR-AUC farkı `0.20` değerini aşarsa `CHECK` üretir; test ve OOT PR-AUC prevalence baseline'ının altında kalırsa performans kontrolü de `CHECK` olur. Böylece düşük target oranında yalnızca yüksek görünen ROC-AUC'ye dayanılmaz.

### Target ve veri sızıntısı kuralları

### Opsiyonel kontroller ve tutar raporu

Missingness, constant, kategorik kardinalite/oran, outlier clipping, missing indicator ve correlation kontrolleri notebook config'inde ayrı ayrı açılıp kapatılabilir: `ENABLE_MISSING_FILTER`, `ENABLE_CONSTANT_FILTER`, `ENABLE_CATEGORICAL_FILTER`, `ENABLE_OUTLIER_CLIPPING`, `ENABLE_CORRELATION_FILTER` ve `ADD_MISSING_INDICATORS`. Kapalı kontroller model feature audit'inde `disabled` olarak görünür; tüm kararlar yalnızca ilgili segmentin train verisinde fit edilir. `CORRELATION_SAMPLE_SIZE` yalnızca correlation matrisi için kullanılır.

`AMOUNT_GROUP_COUNT = 10` ve `AMOUNT_GROUP_COLUMN = "tutar_amount"` ile ana input tablosundan gelen tutar, her segment ve model/config içinde OOT skorlarına göre eşit adetli 10 gruba ayrılır. `Amount_Group_Performance` Excel sheet'i her segment/model/tutar grubu için örneklem, pozitif sayısı, prevalence, PR-AUC, ROC-AUC, Gini ve `TOP_K` lift değerlerini içerir. Ana input'ta `tutar_amount` bulunamazsa açıkça yapılandırılmış kolon veya `tutar amount`, `amount`, `fund_value_real`, `fund_value` fallback sırası kullanılır.

### Nihai model ve elbow analizi

`ELBOW_ENABLED`, `ELBOW_MIN_FEATURES` ve `ELBOW_MAX_FEATURES` ile segment bazlı en iyi OOT modelinin feature importance eğrisi analiz edilir. Varsayılan üst sınır 50'dir. Her segment için seçilen modelin `feature_count`, cumulative importance ve elbow mesafesi `Elbow_Analysis` sheet'inde; importance sıralaması `Feature_Importance` sheet'inde; nihai modelin train/test/OOT ROC-AUC, Gini, PR-AUC ve lift değerleri `Final_Model_Performance` sheet'inde yer alır. Notebook ayrıca `propensity_feature_elbow.png` ve `propensity_final_model_performance.png` grafiklerini üretir; ilgili Excel sheet'lerine de grafik gömülür.

- Newsell: anchor dahil son `x` ayda pasif olup sonraki `y` ayda alım yapan müşteri.
- Upsell: anchor ayında aktif olup sonraki `y` ay net alımının anchor fon değerine oranı en az `r` olan müşteri.
- Future penceresi tamamlanmayan anchor ayları otomatik dışarıda kalır.
- Feature'lar yalnızca anchor ayı ve geçmişinden hesaplanır; feature lineage notebook içinde denetlenir.
- Yeterli eligible veya pozitif gözlem yoksa ilgili grid kombinasyonu eğitilmez.

### Parametre girişi

Notebook içindeki `Data Load` kod hücresinde `KULLANICI CONFIG` bölümü bulunur. `X_GRID` son `x` aylık pasiflik penceresini, `Y_GRID` gelecek `y` aylık hedef penceresini, `R_GRID` ise Upsell oran eşiklerini belirler. Örneğin `X_GRID = (1, 3, 6)`, `Y_GRID = (1, 3)` ve `R_GRID = (0.10, 0.25, 0.50)` bütün kombinasyonları çalıştırır. `PRODUCT_CLASSES` ürün sınıflarını, `MIN_ELIGIBLE` ve `MIN_POSITIVE` veri yeterlilik filtresini belirler; bu eşiği geçmeyen kombinasyonlar `Target_Quality` içinde görünür ancak model eğitimi yapılmaz.

Aynı bölümde `SEGMENT_COLUMN`, `SEGMENT_VALUES`, `SYNTHETIC_CUSTOMER_COUNT`, `SYNTHETIC_TARGET_RATE`, `MIN_POSITIVE_TRAIN`, `MIN_POSITIVE_TEST`, `MIN_POSITIVE_OOT`, `TEST_MONTHS`, `OOT_MONTHS`, `TOP_K`, `RARE_EVENT_RATE_THRESHOLD`, `MISSING_THRESHOLD`, `CORRELATION_THRESHOLD`, `MAX_CATEGORICAL_LEVELS`, `MAX_CATEGORICAL_RATIO`, `OUTLIER_LOWER_QUANTILE`, `OUTLIER_UPPER_QUANTILE` ve `ADD_MISSING_INDICATORS` model öncesi adaptive preprocessing ve rare-event kurallarıdır. `SEGMENT_VALUES = (1, 2, 3, 4, 5)` her segment için bağımsız model kurulacağını belirler. `TEST_MONTHS` ve `OOT_MONTHS` düşük oranlı hedeflerde tek ay yerine daha geniş değerlendirme penceresi açar; `MIN_POSITIVE_*` değerleri pozitif örneği yetersiz splitleri model performansı gibi raporlamadan `skipped` olarak işaretler. Diğer kurallar her segment/configuration'ın train splitinde ayrı fit edilir; validation/OOT dağılımı karar vermek için kullanılmaz. `TOP_K` içinde `0.001` gibi oranlar desteklenir ve kolon adları çakışmayacak biçimde üretilir.

`CORRELATION_SAMPLE_SIZE` yalnızca train verisinden korelasyon matrisi hesaplanırken kullanılan satır örneklem büyüklüğüdür; diğer preprocessing adımlarının sample size'ı değildir. `None` verilirse tüm train satırları kullanılır. Notebook'u yeniden üretmek için `python make_propensity_notebook.py` çalıştırın; ardından `fund_propensity_ml.ipynb` içindeki manuel config hücresini düzenleyerek çalıştırın.

Gerçek veri kolonları bu sözleşmeyle aynı değilse önce normalize edilmiş Parquet katmanı üretilmelidir. Kaynak kolon eşlemesini bu katmana taşımak, notebook içinde sessiz ve denetlenemez dönüşümler yapmaktan daha güvenlidir.

### Tek geniş dış tablo sözleşmesi

10 milyon satır ve çok sayıda değişken içeren tek bir aylık tablo için `PROPENSITY_DATA_ROOT` ile birlikte `PROPENSITY_INPUT_TABLE_FILE` verilebilir. Notebook config'inde veya ortam değişkenleriyle `musteri_id`, tarih, `product_class`, PPF/NF aktiflik kolonları ve `buy_amount`, `sell_amount`, `fund_value` kolonları eşlenir. Ürün kolonu yoksa aynı müşteri-ay kaydı her ürün sınıfına uygulanır; ürün kolonu varsa flow kayıtları ürün bazında ayrıştırılır. `PROPENSITY_FEATURE_COLUMNS` boş bırakılırsa anahtar, aktiflik ve parasal kolonlar dışındaki aylık kolonlar feature adayı olarak alınır.

Enflasyon tablosu `PROPENSITY_INFLATION_TABLE_FILE`, tarih kolonu `PROPENSITY_INFLATION_DATE_COLUMN` ve aylık enflasyon kolonu `PROPENSITY_INFLATION_VALUE_COLUMN` ile bağlanır. Aylık feature'lar müşteri-ay düzeyinde temizlenir; modelde kullanılmadan önce yalnızca anchor ayı ve geçmişinden trend/delta, rolling mean oranı ve `FEATURE_WINDOWS` içinde 12 varsa yıllık değişim feature'ları üretilir. Gelecek dönem bilgisi feature üretimine girmez. Tüm bu kararlar `Feature_Engineering`, `Model_Feature_Audit`, `Inflation_Audit`, `Eliminated_Vars`, `00_Runtime_Summary` ve `19_Overfit_Audit` sheet'lerinde görünür.
