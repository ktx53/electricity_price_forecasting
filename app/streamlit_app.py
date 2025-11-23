import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# GENEL AYARLAR
# =========================================================
st.set_page_config(
    page_title="CAISO Elektrik Fiyat Tahmini",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Basit mavi-tonlu tema
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fb;
    }
    .main-header {
        font-size: 40px;
        font-weight: 600;
        color: #0f4c81;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #e2ecff 0%, #f5f7fb 100%);
        border-radius: 12px;
        margin-bottom: 30px;
        border: 1px solid #d0dcf0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(15, 76, 129, 0.07);
        border: 1px solid #e1e6f0;
    }
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# PATH YAPISI
# =========================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_DIR = PROJECT_DIR / "model"
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"

# =========================================================
# MODELLERİ YÜKLE
# =========================================================
@st.cache_resource
def load_models_and_scalers():
    """
    model klasöründeki pkl modelleri ve scaler'ları yükler.
    Beklenen dosyalar:
      - model_xgboost.pkl
      - model_lightgbm.pkl
      - model_catboost.pkl
      - scaler_X.pkl
      - scaler_y.pkl
      - feature_columns.pkl
    """
    try:
        xgb_path = MODEL_DIR / "model_xgboost.pkl"
        lgb_path = MODEL_DIR / "model_lightgbm.pkl"
        cat_path = MODEL_DIR / "model_catboost.pkl"
        scaler_X_path = MODEL_DIR / "scaler_X.pkl"
        scaler_y_path = MODEL_DIR / "scaler_y.pkl"
        feature_cols_path = MODEL_DIR / "feature_columns.pkl"

        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
        with open(lgb_path, "rb") as f:
            lgb_model = pickle.load(f)
        with open(cat_path, "rb") as f:
            catboost_model = pickle.load(f)
        with open(scaler_X_path, "rb") as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, "rb") as f:
            scaler_y = pickle.load(f)
        with open(feature_cols_path, "rb") as f:
            feature_cols = pickle.load(f)

        models = {
            "XGBoost": xgb_model,
            "LightGBM": lgb_model,
            "CatBoost": catboost_model,
        }

        return models, scaler_X, scaler_y, feature_cols

    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None, None, None, None


# =========================================================
# VERİYİ YÜKLE (FİYAT + SOLAR + LOAD + NET LOAD)
# =========================================================
@st.cache_data
def load_full_dataset():
    """
    Notebook'taki pipeline'a uygun şekilde:
      - CAISO Day-Ahead Price (2016-2022, SCE)
      - Net solar generation (hourly, UTC -> local)
      - CAISO Day-Ahead Load Data (CA ISO total)
    hepsini tek bir df_sce DataFrame'inde birleştirir.

    Dönen kolonlar:
      - DateTime
      - Price (cents/kWh)
      - Solar_Generation_MWh
      - Load (MW)
      - Net_Load_MW
    """

    # -----------------------------
    # 1) Fiyat verisi (price_dfs)
    # -----------------------------
    price_files = {
        2016: RAW_DATA_DIR / "2016 CAISO Day-Ahead Price.csv",
        2017: RAW_DATA_DIR / "2017 CAISO Day-Ahead Price.csv",
        2018: RAW_DATA_DIR / "2018 CAISO Day-Ahead Price.csv",
        2019: RAW_DATA_DIR / "2019 CAISO Day-Ahead Price.csv",
        2020: RAW_DATA_DIR / "2020 CAISO Day-Ahead Price.csv",
        2021: RAW_DATA_DIR / "2021 CAISO Day-Ahead Price.csv",
        2022: RAW_DATA_DIR / "2022 CAISO Day-Ahead Price.csv",
    }

    price_dfs = []
    for year, path in price_files.items():
        try:
            df_temp = pd.read_csv(path)
            price_dfs.append(df_temp)
        except Exception as e:
            st.warning(f"{year} fiyat verisi yüklenemedi: {e}")

    if len(price_dfs) == 0:
        st.error("Hiçbir fiyat verisi yüklenemedi. data/raw klasörünü kontrol et.")
        return None

    df_price = pd.concat(price_dfs, axis=0, ignore_index=True)

    # -----------------------------
    # 2) SCE bölgesi + saatlik reindex + outlier clipping
    # -----------------------------
    df_price["DateTime"] = pd.to_datetime(df_price["Date"])
    df_price = df_price.sort_values("DateTime").reset_index(drop=True)

    df_sce = df_price[df_price["Zone"] == "SCE"].copy()
    df_sce = df_sce.drop("Zone", axis=1)
    df_sce = df_sce.sort_values("DateTime").reset_index(drop=True)

    df_sce = df_sce.drop_duplicates(subset="DateTime", keep="first")

    df_sce = df_sce.set_index("DateTime")
    full_range = pd.date_range(start=df_sce.index.min(), end=df_sce.index.max(), freq="H")
    df_sce = df_sce.reindex(full_range)
    df_sce["Price (cents/kWh)"] = df_sce["Price (cents/kWh)"].fillna(method="ffill")
    df_sce = df_sce.reset_index().rename(columns={"index": "DateTime"})

    # IQR clipping (notebook ile aynı mantık)
    Q1 = df_sce["Price (cents/kWh)"].quantile(0.25)
    Q3 = df_sce["Price (cents/kWh)"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    df_sce.loc[df_sce["Price (cents/kWh)"] < lower_bound, "Price (cents/kWh)"] = lower_bound
    df_sce.loc[df_sce["Price (cents/kWh)"] > upper_bound, "Price (cents/kWh)"] = upper_bound

    # -----------------------------
    # 3) Solar verisi
    # -----------------------------
    solar_path = RAW_DATA_DIR / "Net_generation_from_solar_for_California_(region)_hourly_-_UTC_time.csv"
    try:
        df_solar = pd.read_csv(solar_path, skiprows=5)
        # Notebook'ta: df_solar.columns = ['DateTime_UTC', 'Solar_Generation_MWh']
        df_solar.columns = ["DateTime_UTC", "Solar_Generation_MWh"]
        df_solar["DateTime_UTC"] = pd.to_datetime(
            df_solar["DateTime_UTC"],
            format="%m/%d/%Y %HH",
            errors="coerce",
        )
        df_solar["DateTime"] = df_solar["DateTime_UTC"] - pd.Timedelta(hours=8)
        df_solar = df_solar[["DateTime", "Solar_Generation_MWh"]].dropna()
        df_solar = df_solar.groupby("DateTime")["Solar_Generation_MWh"].mean().reset_index()

        df_sce = df_sce.merge(df_solar, on="DateTime", how="left")
        df_sce["Solar_Generation_MWh"] = df_sce["Solar_Generation_MWh"].fillna(0.0)
    except Exception as e:
        st.warning(f"Güneş verisi yüklenemedi veya entegre edilemedi: {e}")
        df_sce["Solar_Generation_MWh"] = 0.0

    # -----------------------------
    # 4) Load verisi (CAISO Day-Ahead Load Data.xlsx)
    # -----------------------------
    load_path = RAW_DATA_DIR / "CAISO Day-Ahead Load Data.xlsx"
    try:
        df_load = pd.read_excel(load_path)

        df_load["DateTime"] = pd.to_datetime(df_load["Date"])
        if "Zone" in df_load.columns:
            df_load_total = df_load[df_load["Zone"] == "CA ISO"].copy()
        else:
            df_load_total = df_load.copy()

        df_load_total = df_load_total[["DateTime", "Load (MW)"]].dropna()
        df_load_total = df_load_total.groupby("DateTime")["Load (MW)"].mean().reset_index()

        df_sce = df_sce.merge(df_load_total, on="DateTime", how="left")
        df_sce["Load (MW)"] = df_sce["Load (MW)"].interpolate(method="linear")
    except Exception as e:
        st.warning(f"Yük verisi yüklenemedi veya entegre edilemedi: {e}")
        df_sce["Load (MW)"] = 0.0

    # -----------------------------
    # 5) Net yük (Load - Solar) - notebook logic
    # -----------------------------
    if "Solar_Generation_MWh" in df_sce.columns and "Load (MW)" in df_sce.columns:
        df_sce["Net_Load_MW"] = df_sce["Load (MW)"] - df_sce["Solar_Generation_MWh"]
    else:
        df_sce["Net_Load_MW"] = 0.0

    df_sce = df_sce.sort_values("DateTime").reset_index(drop=True)
    return df_sce


# =========================================================
# FEATURE OLUŞTURMA
# =========================================================
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["DateTime"]

    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["dayofweek"] = dt.dt.dayofweek
    df["quarter"] = dt.dt.quarter
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)

    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["season"] = df["month"].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )

    df["is_peak_hour"] = df["hour"].apply(lambda x: 1 if 17 <= x <= 21 else 0)
    df["is_night"] = df["hour"].apply(lambda x: 1 if x < 6 or x >= 22 else 0)
    df["is_morning"] = df["hour"].apply(lambda x: 1 if 6 <= x < 12 else 0)
    df["is_afternoon"] = df["hour"].apply(lambda x: 1 if 12 <= x < 17 else 0)
    df["is_evening"] = df["hour"].apply(lambda x: 1 if 17 <= x < 22 else 0)

    return df


def create_lag_features(df: pd.DataFrame, target_col: str = "Price (cents/kWh)") -> pd.DataFrame:
    df = df.copy()

    # Fiyat lag'leri
    lags = [1, 2, 3, 24, 48, 168]
    for lag in lags:
        df[f"price_lag_{lag}"] = df[target_col].shift(lag)

    # Rolling istatistikler
    windows = [3, 6, 12, 24, 48, 168]
    shifted = df[target_col].shift(1)
    for window in windows:
        roll = shifted.rolling(window=window)
        df[f"price_rolling_mean_{window}"] = roll.mean()
        df[f"price_rolling_std_{window}"] = roll.std()
        df[f"price_rolling_min_{window}"] = roll.min()
        df[f"price_rolling_max_{window}"] = roll.max()

    # EMA & farklar
    df["price_ema_24"] = shifted.ewm(span=24, adjust=False).mean()
    df["price_ema_168"] = shifted.ewm(span=168, adjust=False).mean()
    df["price_change_1h"] = df[target_col].diff(1)
    df["price_change_24h"] = df[target_col].diff(24)
    df["price_change_pct_1h"] = df[target_col].pct_change(1)
    df["price_change_pct_24h"] = df[target_col].pct_change(24)

    return df


def create_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Notebook'taki create_external_features fonksiyonunun aynısı.
    Solar, load ve net load üzerinden lag & rolling feature'lar üretir.
    """
    df = df.copy()

    # Solar
    if "Solar_Generation_MWh" in df.columns:
        for lag in [1, 24, 168]:
            df[f"solar_lag_{lag}"] = df["Solar_Generation_MWh"].shift(lag)
        for window in [24, 168]:
            df[f"solar_rolling_mean_{window}"] = (
                df["Solar_Generation_MWh"].shift(1).rolling(window=window).mean()
            )
            df[f"solar_rolling_max_{window}"] = (
                df["Solar_Generation_MWh"].shift(1).rolling(window=window).max()
            )
        df["solar_penetration"] = df["Solar_Generation_MWh"] / (df["Solar_Generation_MWh"].mean() + 1)

    # Load
    if "Load (MW)" in df.columns:
        for lag in [1, 24, 168]:
            df[f"load_lag_{lag}"] = df["Load (MW)"].shift(lag)
        for window in [24, 168]:
            df[f"load_rolling_mean_{window}"] = (
                df["Load (MW)"].shift(1).rolling(window=window).mean()
            )
            df[f"load_rolling_std_{window}"] = (
                df["Load (MW)"].shift(1).rolling(window=window).std()
            )
        df["load_change_24h"] = df["Load (MW)"].diff(24)
        df["load_change_pct_24h"] = df["Load (MW)"].pct_change(24)

    # Net load
    if "Net_Load_MW" in df.columns:
        for lag in [1, 24, 168]:
            df[f"netload_lag_{lag}"] = df["Net_Load_MW"].shift(lag)
        for window in [24, 168]:
            df[f"netload_rolling_mean_{window}"] = (
                df["Net_Load_MW"].shift(1).rolling(window=window).mean()
            )
            df[f"netload_rolling_std_{window}"] = (
                df["Net_Load_MW"].shift(1).rolling(window=window).std()
            )

    # Solar / Load oranı
    if "Solar_Generation_MWh" in df.columns and "Load (MW)" in df.columns:
        df["solar_load_ratio"] = df["Solar_Generation_MWh"] / (df["Load (MW)"] + 1)

    return df


def ensure_feature_columns(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Modellerin beklediği tüm feature kolonlarını DataFrame'e ekler.
    Eksik olanları 0.0 ile doldurur (KeyError riskini engeller).
    """
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df


def build_feature_frame(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Eğitim pipeline'ına paralel bir feature DataFrame üretir.
    """
    df_features = create_time_features(df_base)
    df_features = create_lag_features(df_features, target_col="Price (cents/kWh)")
    df_features = create_external_features(df_features)

    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(method="ffill").fillna(method="bfill").fillna(0)
    df_features = df_features.dropna().reset_index(drop=True)
    return df_features


# =========================================================
# 24 SAATLİK TAHMİN (P10-P50-P90, SOLAR/LOAD DAHİL)
# =========================================================
def predict_next_24h(models, df_base, feature_cols, scaler_X):
    """
    Son gözlemden itibaren 24 saat ileri forecast.
    Her model için ayrı tahmin, external (solar/load/netload) feature'ları ile birlikte.

    Solar / Load / Net_Load_MW için gelecek 24 saatte:
      - Bir önceki günün aynı saatindeki (t-24h) değerleri kullanılır.
      - Eğer yoksa, en son bilinen değer kullanılır.
    """
    df_base = df_base.sort_values("DateTime").reset_index(drop=True)

    last_time = df_base["DateTime"].iloc[-1]
    future_dates = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=24, freq="H")

    predictions = {}

    # Orijinal geçmiş (exogenous için referans)
    original_history = df_base.copy()

    for model_name, model in models.items():
        model_preds = []
        current_data = df_base.copy()

        for future_time in future_dates:
            # Yeni satır oluştur
            new_row = {
                "DateTime": future_time,
            }

            # Exogenous (solar, load, net load) için t-24h'den değer çek
            prev_time = future_time - pd.Timedelta(hours=24)
            prev_row = original_history[original_history["DateTime"] == prev_time]

            if not prev_row.empty:
                new_row["Solar_Generation_MWh"] = float(prev_row["Solar_Generation_MWh"].iloc[0])
                new_row["Load (MW)"] = float(prev_row["Load (MW)"].iloc[0])
                new_row["Net_Load_MW"] = float(prev_row["Net_Load_MW"].iloc[0])
            else:
                last_known = current_data.iloc[-1]
                new_row["Solar_Generation_MWh"] = float(last_known.get("Solar_Generation_MWh", 0.0))
                new_row["Load (MW)"] = float(last_known.get("Load (MW)", 0.0))
                new_row["Net_Load_MW"] = float(last_known.get("Net_Load_MW", 0.0))

            # Fiyatı ilk etapta son fiyattan başlat; birazdan model tahminiyle güncellenecek
            new_row["Price (cents/kWh)"] = float(
                current_data["Price (cents/kWh)"].iloc[-1]
            )

            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)

            # Feature setini güncel data üzerinden üret
            features = create_time_features(current_data)
            features = create_lag_features(features)
            features = create_external_features(features)
            features = ensure_feature_columns(features, feature_cols)

            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method="ffill").fillna(method="bfill").fillna(0)

            last_feat = features.iloc[-1:][feature_cols]
            X_scaled = scaler_X.transform(last_feat)

            pred = float(model.predict(X_scaled)[0])
            model_preds.append(pred)

            # Bir sonraki adım için tahmini fiyat olarak kullan
            current_data.loc[current_data.index[-1], "Price (cents/kWh)"] = pred

        predictions[model_name] = model_preds

    return predictions, future_dates


def calculate_prediction_intervals(predictions):
    preds_array = np.array(list(predictions.values()))  # shape: (n_models, 24)
    p10 = np.percentile(preds_array, 10, axis=0)
    p50 = np.percentile(preds_array, 50, axis=0)
    p90 = np.percentile(preds_array, 90, axis=0)
    return p10, p50, p90


# =========================================================
# HATA ANALİZİ (SON N SAAT)
# =========================================================
def evaluate_last_n_hours(models, df_base, feature_cols, scaler_X, n_hours=24):
    """
    Son n saat için gerçek vs tahmin hata analizi.
    Eğitim pipeline'ı ile aynı feature set kullanılır.
    """
    df_features = build_feature_frame(df_base)
    # Zaman kolonunu saklayalım
    date_col = df_features["DateTime"]
    target = df_features["Price (cents/kWh)"]

    df_features = ensure_feature_columns(df_features, feature_cols)
    X_all = df_features[feature_cols]

    X_all_clean = (
        X_all.replace([np.inf, -np.inf], np.nan)
        .fillna(method="ffill")
        .fillna(method="bfill")
        .fillna(0)
    )

    X_scaled = scaler_X.transform(X_all_clean)

    # Son n saat: zaten df_features zaman serisine göre sıralı
    last_indices = np.arange(len(df_features) - n_hours, len(df_features))
    results = {
        "DateTime": date_col.iloc[last_indices].values,
        "Actual": target.iloc[last_indices].values,
    }

    for model_name, model in models.items():
        preds_all = model.predict(X_scaled)
        preds_last = preds_all[last_indices]
        results[model_name] = preds_last

        actual = results["Actual"]
        mae = float(np.mean(np.abs(preds_last - actual)))
        mape = float(np.mean(np.abs((preds_last - actual) / (actual + 1e-6))) * 100.0)

        st.write(f"{model_name} - Son {n_hours} saat hata analizi")
        st.write(f"- MAE: {mae:.4f} cents/kWh")
        st.write(f"- MAPE: {mape:.2f}%")
        st.markdown("---")

    return pd.DataFrame(results)


# =========================================================
# ANA UYGULAMA LAYOUT
# =========================================================
st.markdown(
    '<div class="main-header">CAISO Day-Ahead Elektrik Fiyat Tahmin Sistemi</div>',
    unsafe_allow_html=True,
)

models, scaler_X, scaler_y, feature_cols = load_models_and_scalers()
data_full = load_full_dataset()

if models is None or data_full is None:
    st.error("Modeller veya veri yüklenemedi. Dosya yapısını kontrol et.")
    st.stop()

# Basit fiyat serisi (grafikler için)
historical_price = data_full[["DateTime", "Price (cents/kWh)"]].copy()

# Sidebar
st.sidebar.title("Kontrol Paneli")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Sayfa Seçin",
    ["Ana Sayfa", "24 Saatlik Tahmin", "Model Karşılaştırma", "Geçmiş Veri Analizi", "Hata Analizi"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Özeti")
st.sidebar.info(
    """
XGBoost, LightGBM ve CatBoost ile saatlik fiyat tahmini yapılmaktadır.

Özellik sayısı: 80+  
Eğitim verisi: 2016 - 2022 (SCE bölgesi, CAISO)
"""
)

# =========================================================
# SAYFA: ANA SAYFA
# =========================================================
if page == "Ana Sayfa":
    st.header("Genel Bakış")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="Öne Çıkan Model", value="CatBoost", delta="R² ≈ 0.997")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="Tipik MAE", value="≈ 0.7 cents/kWh")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(
            label="Veri Aralığı",
            value="2016 - 2022",
            delta=f"{len(historical_price):,} saat",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Son 7 Günlük Fiyat Trendi")

    recent_data = historical_price.tail(168)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent_data["DateTime"],
            y=recent_data["Price (cents/kWh)"],
            mode="lines",
            name="Gerçek Fiyat",
            line=dict(color="#0f4c81", width=2),
        )
    )

    fig.update_layout(
        title="Son 7 Gün Elektrik Fiyatı (SCE)",
        xaxis_title="Tarih",
        yaxis_title="Fiyat (cents/kWh)",
        hovermode="x unified",
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Modelleme Özeti")
        st.markdown(
            """
        - Gradient boosting tabanlı üç model (CatBoost, XGBoost, LightGBM)  
        - Zaman özellikleri (saat, gün, ay, mevsim, haftasonu vb.)  
        - Fiyat için lag ve rolling istatistikler  
        - Güneş üretimi ve yük üzerinden external feature set  
        - Ensemble ile P10 - P50 - P90 olasılıksal tahmin
        """
        )

    with col2:
        st.subheader("Kullanım Senaryoları")
        st.markdown(
            """
        - Risk yönetimi ve hedge stratejileri  
        - Kısa vadeli ticaret ve arbitraj  
        - Gün öncesi üretim / tüketim planlama  
        - Fiyat duyarlı tüketim optimizasyonu  
        - Karar destek ve senaryo analizi  
        """
        )

# =========================================================
# SAYFA: 24 SAATLİK TAHMİN
# =========================================================
elif page == "24 Saatlik Tahmin":
    st.header("24 Saatlik İleriye Dönük Tahmin")

    st.markdown(
        """
        Son gözlem noktasından itibaren 24 saat için fiyat tahmini üretilir.  
        Solar ve yük verisi kullanılarak external feature'lar hesaplanır,  
        üç modelin tahminlerinden P10 - P50 - P90 bandı oluşturulur.
        """
    )

    n_history = st.slider(
        "Grafikte gösterilecek geçmiş saat sayısı",
        min_value=24,
        max_value=240,
        value=48,
        step=24,
    )

    if st.button("Tahmin Oluştur"):
        with st.spinner("Tahminler hesaplanıyor..."):
            predictions, future_dates = predict_next_24h(
                models, data_full, feature_cols, scaler_X
            )
            p10, p50, p90 = calculate_prediction_intervals(predictions)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ortalama Tahmin (P50)", f"{p50.mean():.2f} cents/kWh")
            with col2:
                st.metric("Minimum (P10)", f"{p10.min():.2f} cents/kWh")
            with col3:
                st.metric("Maksimum (P90)", f"{p90.max():.2f} cents/kWh")

            st.markdown("---")

            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    "24 Saatlik Fiyat Tahmini (P10 - P50 - P90)",
                    "Modellere Göre Tahminler",
                ),
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4],
            )

            # Geçmiş veri
            recent_history = historical_price.tail(n_history)
            fig.add_trace(
                go.Scatter(
                    x=recent_history["DateTime"],
                    y=recent_history["Price (cents/kWh)"],
                    mode="lines",
                    name="Geçmiş Veri",
                    line=dict(color="#7a869a", width=2),
                ),
                row=1,
                col=1,
            )

            # P10 - P90 bandı
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=p90,
                    mode="lines",
                    name="P90",
                    line=dict(width=0),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=p10,
                    mode="lines",
                    name="P10 - P90 bandı",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(15, 76, 129, 0.18)",
                ),
                row=1,
                col=1,
            )

            # P50
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=p50,
                    mode="lines+markers",
                    name="P50 (Medyan)",
                    line=dict(color="#d62728", width=3),
                    marker=dict(size=7),
                ),
                row=1,
                col=1,
            )

            # Model bazında tahminler
            colors = {"XGBoost": "#ff7f0e", "LightGBM": "#2ca02c", "CatBoost": "#1f77b4"}
            for model_name, preds in predictions.items():
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=preds,
                        mode="lines+markers",
                        name=model_name,
                        line=dict(color=colors.get(model_name, "#555"), width=2, dash="dot"),
                    ),
                    row=2,
                    col=1,
                )

            fig.update_xaxes(title_text="Tarih", row=2, col=1)
            fig.update_yaxes(title_text="Fiyat (cents/kWh)", row=1, col=1)
            fig.update_yaxes(title_text="Fiyat (cents/kWh)", row=2, col=1)
            fig.update_layout(
                height=800,
                hovermode="x unified",
                showlegend=True,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            st.plotly_chart(fig, width="stretch")

            st.markdown("---")
            st.subheader("Tahmin Tablosu")

            df_predictions = pd.DataFrame(
                {
                    "Tarih": future_dates,
                    "Saat": [d.hour for d in future_dates],
                    "P10": p10,
                    "P50 (Medyan)": p50,
                    "P90": p90,
                    "XGBoost": predictions["XGBoost"],
                    "LightGBM": predictions["LightGBM"],
                    "CatBoost": predictions["CatBoost"],
                }
            )

            st.dataframe(
                df_predictions.style.format(
                    {
                        "P10": "{:.2f}",
                        "P50 (Medyan)": "{:.2f}",
                        "P90": "{:.2f}",
                        "XGBoost": "{:.2f}",
                        "LightGBM": "{:.2f}",
                        "CatBoost": "{:.2f}",
                    }
                ),
                width="stretch",
            )

            st.download_button(
                label="Tahminleri CSV Olarak İndir",
                data=df_predictions.to_csv(index=False).encode("utf-8"),
                file_name=f"tahmin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

# =========================================================
# SAYFA: MODEL KARŞILAŞTIRMA
# =========================================================
elif page == "Model Karşılaştırma":
    st.header("Model Performans Karşılaştırması")

    model_performance = pd.DataFrame(
        {
            "Model": ["CatBoost", "XGBoost", "LightGBM"],
            "Test MAE": [0.8141, 0.8033, 0.8583],
            "Test RMSE": [1.00, 1.05, 1.10],
            "Test R²": [0.9972, 0.9968, 0.9965],
            "Training Time": ["45s", "38s", "32s"],
        }
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=model_performance["Model"],
                    y=model_performance["Test MAE"],
                    marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                    text=model_performance["Test MAE"],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Test MAE Karşılaştırması",
            yaxis_title="MAE (cents/kWh)",
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=model_performance["Model"],
                    y=model_performance["Test R²"],
                    marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                    text=model_performance["Test R²"],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Test R² Karşılaştırması",
            yaxis_title="R² Skoru",
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    st.subheader("Model Performans Tablosu")
    st.dataframe(
        model_performance.style.highlight_max(subset=["Test R²"], color="#c7f2c4")
        .highlight_min(subset=["Test MAE"], color="#c7f2c4"),
        width="stretch",
    )

    st.markdown("---")

    st.subheader("Model Notları")

    tab1, tab2, tab3 = st.tabs(["CatBoost", "XGBoost", "LightGBM"])

    with tab1:
        st.markdown(
            """
        CatBoost:
        - R² skoru en yüksek model  
        - Ağaç tabanlı gradient boosting  
        - Dengesiz ve karmaşık feature set için güçlü  
        - Üretim ortamı için stabil
        """
        )

    with tab2:
        st.markdown(
            """
        XGBoost:
        - Çok hızlı eğitim  
        - Güçlü topluluk ve dokümantasyon  
        - Feature importance analizi kolay  
        - Prototip ve benchmark için ideal
        """
        )

    with tab3:
        st.markdown(
            """
        LightGBM:
        - Çok hızlı ve hafif  
        - Büyük veri setlerinde avantajlı  
        - Histogram tabanlı ağaçlar  
        - Düşük gecikmeli tahminler için uygun
        """
        )

# =========================================================
# SAYFA: GEÇMİŞ VERİ ANALİZİ
# =========================================================
elif page == "Geçmiş Veri Analizi":
    st.header("Geçmiş Veri Analizi")

    date_range = st.date_input(
        "Tarih aralığı seçin",
        value=(
            historical_price["DateTime"].min().date(),
            historical_price["DateTime"].max().date(),
        ),
    )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        mask = (historical_price["DateTime"].dt.date >= date_range[0]) & (
            historical_price["DateTime"].dt.date <= date_range[1]
        )
        filtered_data = historical_price[mask].copy()

        if filtered_data.empty:
            st.warning("Seçilen aralıkta veri bulunamadı.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Ortalama Fiyat", f"{filtered_data['Price (cents/kWh)'].mean():.2f}")
            with col2:
                st.metric("Minimum Fiyat", f"{filtered_data['Price (cents/kWh)'].min():.2f}")
            with col3:
                st.metric("Maksimum Fiyat", f"{filtered_data['Price (cents/kWh)'].max():.2f}")
            with col4:
                st.metric("Standart Sapma", f"{filtered_data['Price (cents/kWh)'].std():.2f}")

            st.markdown("---")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=filtered_data["DateTime"],
                    y=filtered_data["Price (cents/kWh)"],
                    mode="lines",
                    name="Fiyat",
                    line=dict(color="#0f4c81", width=1.5),
                )
            )

            fig.update_layout(
                title="Fiyat Trendi",
                xaxis_title="Tarih",
                yaxis_title="Fiyat (cents/kWh)",
                hovermode="x unified",
                height=480,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            st.plotly_chart(fig, width="stretch")

            col1, col2 = st.columns(2)

            with col1:
                filtered_data["hour"] = filtered_data["DateTime"].dt.hour
                hourly_avg = filtered_data.groupby("hour")["Price (cents/kWh)"].mean()

                fig_bar = go.Figure(
                    data=[go.Bar(x=hourly_avg.index, y=hourly_avg.values, marker_color="#1f77b4")]
                )
                fig_bar.update_layout(
                    title="Saatlik Ortalama Fiyat",
                    xaxis_title="Saat",
                    yaxis_title="Ortalama Fiyat (cents/kWh)",
                    height=400,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig_bar, width="stretch")

            with col2:
                fig_hist = go.Figure(
                    data=[go.Histogram(x=filtered_data["Price (cents/kWh)"], nbinsx=50)]
                )
                fig_hist.update_layout(
                    title="Fiyat Dağılımı",
                    xaxis_title="Fiyat (cents/kWh)",
                    yaxis_title="Frekans",
                    height=400,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig_hist, width="stretch")

# =========================================================
# SAYFA: HATA ANALİZİ
# =========================================================
elif page == "Hata Analizi":
    st.header("Model Hata Analizi")

    st.markdown(
        """
        Modellerin geçmiş verideki performansını inceleyin.  
        Son N saat için gerçek fiyatlar ile model tahminleri karşılaştırılır  
        ve MAE / MAPE metrikleri hesaplanır.
        """
    )

    n_hours = st.slider(
        "Analiz edilecek son saat sayısı",
        min_value=24,
        max_value=240,
        value=48,
        step=24,
    )

    if st.button("Hata Analizini Çalıştır"):
        with st.spinner("Hata analizi çalışıyor..."):
            df_eval = evaluate_last_n_hours(
                models, data_full, feature_cols, scaler_X, n_hours=n_hours
            )

            st.subheader("Gerçek vs Tahmin Tablosu")
            st.dataframe(df_eval, width="stretch")

            for model_name in models.keys():
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df_eval["DateTime"],
                        y=df_eval["Actual"],
                        mode="lines+markers",
                        name="Gerçek",
                        line=dict(width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_eval["DateTime"],
                        y=df_eval[model_name],
                        mode="lines+markers",
                        name=f"Tahmin - {model_name}",
                        line=dict(width=2, dash="dot"),
                    )
                )

                fig.update_layout(
                    title=f"Gerçek vs Tahmin - {model_name}",
                    xaxis_title="Tarih",
                    yaxis_title="Fiyat (cents/kWh)",
                    hovermode="x unified",
                    height=420,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig, width="stretch")
