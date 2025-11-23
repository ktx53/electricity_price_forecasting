# CAISO Day-Ahead Elektrik Fiyat Tahmini

Bu proje, Kaliforniya ISO (CAISO) Day-Ahead piyasası için saatlik elektrik fiyatlarını tahmin etmeye yönelik uçtan uca bir çalışma içerir. 

2016–2022 dönemi boyunca:

- SCE bölgesi saatlik fiyat verileri
- CAISO day-ahead yük verisi
- Kaliforniya bölgesi saatlik güneş üretimi verisi

kullanılarak zengin bir özellik seti oluşturulmuş, farklı zaman serisi ve makine öğrenmesi modelleri eğitilmiş ve sonuçlar bir Streamlit arayüzü ile görselleştirilmiştir.

---

## İçindekiler

1. [Proje Özeti](#proje-özeti)  
2. [Veri Kaynakları](#veri-kaynakları)  
3. [Dizin Yapısı](#dizin-yapısı)  
4. [Kurulum](#kurulum)  
5. [Streamlit Uygulamasını Çalıştırma](#streamlit-uygulamasını-çalıştırma)  
6. [Notebook: electricity_price_forecasting_ENHANCED.ipynb](#notebook-electricity_price_forecasting_enhancedipynb)  
   - [Keşifsel Veri Analizi (EDA)](#keşifsel-veri-analizi-eda)  
   - [Özellik Mühendisliği](#özellik-mühendisliği)  
   - [Modelleme Yaklaşımı](#modelleme-yaklaşımı)  
   - [Model Değerlendirme ve Karşılaştırma](#model-değerlendirme-ve-karşılaştırma)  
   - [Model ve Scaler Çıktılarının Kaydedilmesi](#model-ve-scaler-çıktılarının-kaydedilmesi)  
7. [Streamlit Arayüzü Sayfaları](#streamlit-arayüzü-sayfaları)  
8. [Tekrar Çalıştırılabilirlik ve Notlar](#tekrar-çalıştırılabilirlik-ve-notlar)

---

## Proje Özeti

Amaç, CAISO Day-Ahead piyasasında belirli bir bölge (SCE) için saatlik elektrik fiyatlarını kısa vadede tahmin etmektir. Fiyat tahmini yapılırken:

- Fiyat serisinin kendi gecikmeleri,
- Güneş üretim seviyeleri,
- Sistem yükü ve net yük

gibi dışsal değişkenler bir arada kullanılır. Bu sayede hem mevsimsellik ve saatlik döngüler hem de talep/arz tarafındaki dinamikler modele yansıtılır.

Projede:

- CatBoost, XGBoost ve LightGBM ile regresyon modelleri eğitilmiş,
- Zaman serisi tabanlı modeller (LSTM, Prophet, SARIMAX) referans olarak kullanılmış,
- En iyi performans veren modeller üretim ortamına uygun şekilde kaydedilerek Streamlit arayüzü üzerinden gerçek zamanlı tahmin için kullanıma açılmıştır.

---

## Veri Kaynakları

`data/raw` dizini altında kullanılan temel veri kaynakları:

- `2016–2022 CAISO Day-Ahead Price.csv`  
  - Saatlik day-ahead fiyatları içerir.  
  - SCE bölgesi filtrelenerek modelde kullanılmıştır.

- `CAISO Day-Ahead Load Data.xlsx`  
  - CAISO bölgesi day-ahead yük tahminleri.  
  - Gerekirse bölge bazlı ayrıştırma yapılabilir, projede toplam sistem yükü kullanılmıştır.

- `Net_generation_from_solar_for_California_(region)_hourly_-_UTC_time.csv`  
  - Kaliforniya bölgesi için saatlik net güneş üretimi.  
  - UTC zaman damgası yerel saate (UTC−8) çevrilerek zaman ekseni fiyat serisi ile hizalanmıştır.

Bu ham veriler, not defteri içerisinde birleştirilerek tek bir zaman serisi üzerinde:

- `Price (cents/kWh)`
- `Solar_Generation_MWh`
- `Load (MW)`
- `Net_Load_MW`

kolonlarına sahip zengin bir veri setine dönüştürülür.

---

## Dizin Yapısı

```text
.
├── app
│   └── streamlit_app.py          # Streamlit arayüzü
│
├── data
│   └── raw                       # Ham veri dosyaları
│       ├── 2016 CAISO Day-Ahead Price.csv
│       ├── 2017 CAISO Day-Ahead Price.csv
│       ├── 2018 CAISO Day-Ahead Price.csv
│       ├── 2019 CAISO Day-Ahead Price.csv
│       ├── 2020 CAISO Day-Ahead Price.csv
│       ├── 2021 CAISO Day-Ahead Price.csv
│       ├── 2022 CAISO Day-Ahead Price.csv
│       ├── CAISO Day-Ahead Load Data.xlsx
│       └── Net_generation_from_solar_for_California_(region)_hourly_-_UTC_time.csv
│
├── model
│   ├── feature_columns.pkl       # Eğitimde kullanılan feature listesini içerir
│   ├── model_catboost.pkl        # CatBoost fiyat tahmin modeli
│   ├── model_lightgbm.pkl        # LightGBM fiyat tahmin modeli
│   ├── model_lstm.h5             # LSTM modeli (karşılaştırma)
│   ├── model_prophet.pkl         # Prophet modeli (karşılaştırma)
│   ├── model_sarimax.pkl         # SARIMAX modeli (karşılaştırma)
│   ├── model_xgboost.pkl         # XGBoost fiyat tahmin modeli
│   ├── scaler_X.pkl              # Girdi özellikleri için scaler
│   └── scaler_y.pkl              # Hedef değişken için scaler (gerekirse)
│
├── notebooks
│   ├── electricity_price_forecasting_ENHANCED.ipynb  # Eğitim ve analiz not defteri
│   └── catboost_info                                 # CatBoost eğitim logları
│
├── outputs
│   ├── acf_pacf.png
│   ├── correlation_matrix.png
│   ├── decomposition.png
│   ├── eda_basic.png
│   ├── external_variables.png
│   ├── model_comparison.png
│   └── predictions.png
│
└── requirements.txt
```

---

## Kurulum

### 1. Depoyu klonlama

```bash
git clone https://github.com/ktx53/electricity_price_forecasting.git
cd [<repo-klasörü>]
```

### 2. Sanal ortam

Python 3.10+ önerilir.

```bash
# Windows (PowerShell)
python -m venv venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkların kurulumu

```bash
pip install -r requirements.txt
```

Bağımlılıklar arasında temel olarak şunlar bulunur:

- pandas, numpy
- matplotlib, seaborn, plotly
- scikit-learn
- xgboost, lightgbm, catboost
- statsmodels, prophet
- streamlit

---

## Streamlit Uygulamasını Çalıştırma

Uygulama dosyası `app/streamlit_app.py` içinde yer alır.

Proje kök dizininden:

```bash
cd app
streamlit run streamlit_app.py
```

Çıktıda aşağıdakine benzer bir satır göründükten sonra:

```text
Local URL: http://localhost:8501
```

tarayıcıdan `http://localhost:8501` adresine giderek arayüze erişebilirsiniz.

---

## Notebook: electricity_price_forecasting_ENHANCED.ipynb

Bu not defteri, veri işleme, özellik mühendisliği, modelleme ve değerlendirme adımlarının tamamını içerir. Aynı zamanda `model` klasöründe kullanılan `pkl` ve scaler dosyaları da bu not defteri üzerinden üretilmiştir.

Not defterinin ana bölümleri şöyle özetlenebilir:

### Keşifsel Veri Analizi (EDA)

1. **Veri yükleme ve birleştirme**  
   - Fiyat, yük ve güneş verileri ayrı ayrı okunur.
   - Zaman damgaları saatlik düzeye getirilir ve tek bir `DateTime` ekseninde birleştirilir.
   - Fiyat serisi için SCE bölgesi seçilir ve eksik saatler reindex ile tamamlanır.
   - Güneş verisi UTC’den yerel saate çevrilerek fiyat serisi ile hizalanır.
   - Yük verisi gerekirse CA ISO toplamı olarak gruplanır.

2. **Temel istatistikler ve görseller**  
   - Ortalama, medyan, min, max, standart sapma değerleri hesaplanır.
   - Zaman serisinin genel trendi görselleştirilir.  
   - Mevsimsellik ve trend bileşenleri `decomposition.png` ile incelenir.
   - ACF/PACF grafikleri `acf_pacf.png` dosyasında yer alır.
   - Fiyat ile solar/yük/net yük arasındaki ilişkiler `external_variables.png` ve `correlation_matrix.png` görselleriyle analiz edilir.

3. **Aykırı değer analizi**  
   - Fiyat serisi için IQR (Interquartile Range) yöntemiyle aykırı değer sınırları belirlenir.
   - Belirlenen eşiklerin altında veya üstünde kalan değerler sınır değerlerle kırpılır.
   - Bu sayede model eğitiminde aşırı uçların etkisi azaltılır.

### Özellik Mühendisliği

Not defterinde, kod tarafında da kullanılan birkaç ana fonksiyon bulunur:

1. **Zaman özellikleri** (`create_time_features`)  
   - Saat, gün, ay, yıl, hafta günü, çeyrek, yılın günü, haftanın numarası
   - Hafta içi / hafta sonu, ay başı / ay sonu bayrakları
   - Uhr bazlı dört segment: gece, sabah, öğlen, akşam
   - Sirküler değişkenler:
     - `hour_sin`, `hour_cos`
     - `month_sin`, `month_cos`
     - `dayofweek_sin`, `dayofweek_cos`
   - Mevsim bilgisi (kış, ilkbahar, yaz, sonbahar)

2. **Fiyat için gecikme ve rolling özellikleri** (`create_lag_features`)  
   - Lag’ler:
     - `price_lag_1`, `price_lag_2`, `price_lag_3`
     - `price_lag_24`, `price_lag_48`, `price_lag_168`
   - Rolling istatistikler:
     - 3, 6, 12, 24, 48, 168 saatlik pencerelerde ortalama, standart sapma, min, max
   - EMA ve farklar:
     - `price_ema_24`, `price_ema_168`
     - `price_change_1h`, `price_change_24h`
     - `price_change_pct_1h`, `price_change_pct_24h`

3. **Dışsal değişken özellikleri** (`create_external_features`)  
   Güneş üretimi, sistem yükü ve net yük üzerinden türetilen değişkenler:

   - Solar:
     - `solar_lag_1`, `solar_lag_24`, `solar_lag_168`
     - `solar_rolling_mean_24`, `solar_rolling_max_24`
     - `solar_rolling_mean_168`, `solar_rolling_max_168`
     - `solar_penetration` (solar üretimin ortalama seviyeye göre normalize edilmesi)

   - Load:
     - `load_lag_1`, `load_lag_24`, `load_lag_168`
     - `load_rolling_mean_24`, `load_rolling_std_24`
     - `load_rolling_mean_168`, `load_rolling_std_168`
     - `load_change_24h`, `load_change_pct_24h`

   - Net load:
     - `netload_lag_1`, `netload_lag_24`, `netload_lag_168`
     - `netload_rolling_mean_24`, `netload_rolling_std_24`
     - `netload_rolling_mean_168`, `netload_rolling_std_168`

   - Solar / yük oranı:
     - `solar_load_ratio` = `Solar_Generation_MWh / (Load (MW) + 1)`

Tüm bu özellikler üretildikten sonra:

- Sonsuz değerler NaN’e çevrilir,
- Eksik değerler ileri/geri doldurma ve gerektiğinde sıfır ile tamamlanır,
- Sonuçta eğitim için hazır bir feature matrisi elde edilir.

### Modelleme Yaklaşımı

Not defterinde, farklı model aileleri denenir:

1. **Ağaç tabanlı gradient boosting modelleri**
   - XGBoost
   - LightGBM
   - CatBoost

Bu modellerde tipik olarak şu hiperparametrelerle oynanır:

- Maksimum derinlik / num_leaves
- Öğrenme oranı
- Ağaç sayısı (iterations / n_estimators)
- Satır ve sütun örnekleme oranları (subsample, colsample_bytree, feature_fraction)
- L2 regulizasyon parametreleri

Amaç, hem eğitim hem de zaman bazlı validasyon setinde düşük MAE/RMSE ve yüksek R² sağlamaktır.

2. **Zaman serisi modelleri**
   - Prophet
   - SARIMAX
   - LSTM

Bu modeller, referans olarak kullanılır ve genel olarak ağaç tabanlı yöntemlerle karşılaştırılır. Özellikle karmaşık mevsimsellik ve dışsal değişken etkileşimlerinde gradient boosting modellerinin daha iyi performans verdiği gözlemlenir.

Not defterinde, zaman bazlı train/validation/test ayrımı kullanılır. Böylece model:

- Gelecekteki saatleri tahmin ederken
- Sadece geçmiş bilgiyi görecek şekilde değerlendirilir.

### Model Değerlendirme ve Karşılaştırma

Her model için tipik olarak şu metrikler hesaplanır:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

Sonuçlar hem tablo hem de grafik ile özetlenir:

- `model_comparison.png` dosyasında modellerin hata metrikleri yan yana gösterilir.
- `predictions.png` dosyasında belirli bir dönem için gerçek vs tahmin eğrileri çizilir.
- Hangi modelin:
  - Ortalama hatayı en düşük tuttuğu,
  - Pik seviyeleri daha iyi yakaladığı,
  - Hesaplama maliyeti ve eğitim süresi açısından daha uygun olduğu  
  not defterinde tartışılır.

Genel olarak CatBoost, XGBoost ve LightGBM çok yakın performans sergiler; CatBoost hafifçe en yüksek R² değerine sahip olduğu için üretim modeli olarak seçilmiştir. XGBoost ve LightGBM ise ensemble ve kıyaslama amaçlı tutulmuştur.

### Model ve Scaler Çıktılarının Kaydedilmesi

Eğitim tamamlandıktan sonra:

- Seçilen modeller uygun formatlarda diske kaydedilir:
  - `model/model_catboost.pkl`
  - `model/model_xgboost.pkl`
  - `model/model_lightgbm.pkl`
- Girdi özellikleri için kullanılan scaler:
  - `model/scaler_X.pkl`
- Hedef değişken için scaler (kullanıldıysa):
  - `model/scaler_y.pkl`
- Modellerin beklediği feature kolonlarının listesi:
  - `model/feature_columns.pkl`

Streamlit uygulaması, tahmin yaparken doğrudan bu dosyaları kullanır. Böylece eğitim ortamı ile üretim arayüzü arasında tutarlılık sağlanır.

---

## Streamlit Arayüzü Sayfaları

`streamlit_app.py` dosyasında başlıca şu sayfalar bulunur:

1. **Ana Sayfa**
   - Veri kapsamı, gözlem sayısı ve tipik hata seviyelerini gösteren metrik kartları
   - Son 7 gün için SCE bölgesi saatlik fiyat grafiği
   - Modelleme yaklaşımı ve kullanım senaryolarının kısa özeti

2. **24 Saatlik Tahmin**
   - Son gözlem noktasından itibaren 24 saat ileriye dönük fiyat tahmini üretir.
   - Her saat için CatBoost, XGBoost ve LightGBM modellerinden çıkan tahminler kullanılır.
   - Üç modelin tahminlerinden P10, P50 ve P90 yüzdelik değerleri hesaplanarak olasılıksal bir tahmin bandı oluşturulur.
   - Geçmişte gösterilecek pencere (24–240 saat arası) kullanıcı tarafından ayarlanabilir.
   - Tahmin tablosu CSV olarak indirilebilir.

   Gelecek 24 saat için güneş ve yük değerleri, geçmişteki t−24 saatlik değerler üzerinden türetilir. Bu, dışsal değişkenler için basit ve deterministik bir forecast yöntemidir. Gerektiğinde, bu kısım daha sofistike ayrı bir yük/güneş tahmin modeli ile değiştirilebilir.

3. **Model Karşılaştırma**
   - CatBoost, XGBoost ve LightGBM için:
     - Test MAE
     - Test RMSE
     - Test R²
     - Eğitim süreleri
   - MAE ve R² bar grafikleri
   - Her modelin güçlü yönlerini açıklayan kısa metinler

   Bu sayfadaki metrikler, `electricity_price_forecasting_ENHANCED.ipynb` not defterinde elde edilen sonuçlara dayanmaktadır.

4. **Geçmiş Veri Analizi**
   - Kullanıcı, tarih aralığı seçerek veriyi filtreleyebilir.
   - Seçilen aralık için:
     - Ortalama, minimum, maksimum ve standart sapma değerleri
     - Zaman serisi grafiği
     - Saatlik ortalama fiyat bar grafiği
     - Fiyat dağılımı histogramı

5. **Hata Analizi**
   - Son N saat (24–240 arası seçilebilir) için:
     - Her modelin MAE ve MAPE değerleri hesaplanır.
     - Gerçek ve tahmin serileri aynı grafikte gösterilir.
   - Bu sayfa, özellikle yakın dönemde model performansını izlemek ve olası rejim değişikliklerini yakalamak için kullanılabilir.

---

## Tekrar Çalıştırılabilirlik ve Notlar

- Not defteri, `data/raw` altındaki dosyaların var olduğu ve isimlerin korunmuş olduğu varsayımıyla çalışır.
- Aynı veriler ve aynı hiperparametreler kullanıldığında, not defteri yeniden çalıştırılarak:
  - Model çıktı dosyaları (`model/*.pkl`)
  - Görsel çıktılar (`outputs/*.png`)
  tekrar üretilebilir.
- Streamlit arayüzünün, not defterinde üretilen modellerle uyumlu çalışması için:
  - `feature_columns.pkl` dosyasının güncel olması,
  - `scaler_X.pkl` dosyasının eğitimde kullanılan scaler ile aynı olması gerekir.

Bu koşullar sağlandığı sürece:

- Not defteri, veri bilimi tarafında keşif, eğitim ve tuning için,
- Streamlit uygulaması ise tahminlerin ve model performansının son kullanıcıya sunumu için kullanılabilir.

Ekran Görüntüleri

<img width="1600" height="1812" alt="FireShot Capture 002 - CAISO Elektrik Fiyat Tahmini -  localhost" src="https://github.com/user-attachments/assets/7be3c7f8-35a2-438d-b8a2-700d8bb53985" />

<img width="1600" height="3752" alt="FireShot Capture 003 - CAISO Elektrik Fiyat Tahmini -  localhost" src="https://github.com/user-attachments/assets/07525669-662e-4d49-9a74-c6cf1db247f9" />

<img width="1600" height="1741" alt="FireShot Capture 004 - CAISO Elektrik Fiyat Tahmini -  localhost" src="https://github.com/user-attachments/assets/3b25562c-4e87-47ec-86d1-48a99b92d0a4" />

<img width="1600" height="2749" alt="FireShot Capture 005 - CAISO Elektrik Fiyat Tahmini -  localhost" src="https://github.com/user-attachments/assets/4d347831-11ac-4662-b592-30e87376e21d" />



