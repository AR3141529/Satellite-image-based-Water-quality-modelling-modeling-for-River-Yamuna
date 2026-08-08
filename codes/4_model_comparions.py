import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings("ignore")

CONFIG = {
    # data
    "DATA_PATH": r"D:\Downloads\Final Dataset.xlsx",   # .xml or .xlsx both supported
    "OUT_DIR": r"D:\Downloads\Model_Comparison_Results",
    "DATE_COL": "Date",
    "TARGET_COL": "NDCI",
    "EXCLUDE_COLS": ["Unnamed: 0"],

    # Train/test split
    "TEST_SIZE_RATIO": 0.2,   # 80:20 split
    
 
    "ARIMAX_ORDER": (2, 1, 1),          # (p, d, q)
    "ARIMAX_TREND": "n",                # with d=1, a constant is redundant/over-identified; 'n' had lower AIC/BIC than 'c'
    "SARIMAX_ORDER": (2, 1, 1),         # (p, d, q) - same identification as ARIMAX
    "SARIMAX_SEASONAL_ORDER": (1, 0, 0, 30),  # (P, D, Q, s) - kept at s=30 (monthly-scale cycle, as in original design)
    "SARIMAX_TREND": "n",

    # XGBoost
    "XGB_PARAMS": {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 100,
    },

    # LSTM
    "LSTM_LOOKBACK": 14,        # features used for each sample
    "LSTM_UNITS": [64, 32],     
    "LSTM_DROPOUT": 0.2,
    "LSTM_EPOCHS": 150,
    "LSTM_BATCH_SIZE": 16,
    "LSTM_PATIENCE": 15,        # early stopping 
    "LSTM_LEARNING_RATE": 1e-3,
    "RANDOM_SEED": 42,
    "SEASONAL_DECOMPOSE_PERIOD": 365,
    "DPI": 150,
    "RUN_LEARNING_CURVE": True,  
    # SARIMAX_SEASONAL_ORDER=(1,0,0,30), each SARIMAX fit takes
}

np.random.seed(CONFIG["RANDOM_SEED"])
os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)

def load_data(cfg):    # data
    path = cfg["DATA_PATH"]
    if str(path).lower().endswith(".xml"):
        raw = pd.read_xml(path, xpath=".//row", parser="etree")
        header = raw.iloc[0].tolist()
        df = raw.iloc[1:].copy()
        df.columns = header
        df = df.reset_index(drop=True)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Column B held Excel serial date numbers (origin 1899-12-30)
        df[cfg["DATE_COL"]] = pd.to_datetime(df[cfg["DATE_COL"]], unit="D", origin="1899-12-30")
    else:
        df = pd.read_excel(path)
        df[cfg["DATE_COL"]] = pd.to_datetime(df[cfg["DATE_COL"]])
    df = df.set_index(cfg["DATE_COL"]).asfreq("D")
    drop_cols = [c for c in cfg["EXCLUDE_COLS"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    feature_cols = [c for c in df.columns if c != cfg["TARGET_COL"]]
    df[feature_cols + [cfg["TARGET_COL"]]] = df[feature_cols + [cfg["TARGET_COL"]]].replace(
        [np.inf, -np.inf], np.nan
    )
    df[feature_cols + [cfg["TARGET_COL"]]] = df[feature_cols + [cfg["TARGET_COL"]]].ffill().bfill()
    return df, feature_cols

# comparisions

def regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)  # standard formula: 1 - SS_res/SS_tot
    nonzero = np.abs(y_true) > 1e-8
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        if nonzero.any()
        else np.nan
    )
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > 1e-8
    smape = (
        np.mean(2.0 * np.abs(y_pred[valid] - y_true[valid]) / denom[valid]) * 100
        if valid.any()
        else np.nan
    )
    if len(y_true) > 1:
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        dir_acc = np.mean(true_dir == pred_dir) * 100
    else:
        dir_acc = np.nan
    return {
        "RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "SMAPE (%)": smape,
        "R2": r2, "Directional Accuracy (%)": dir_acc,
    }


def pseudo_aic_bic(rss, n, k):     # aic and bic values
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return aic, bic

# XGBOOST 

def xgb_effective_k(model, fallback_k):
    try:
        tree_df = model.get_booster().trees_to_dataframe()
        leaf_count = int((tree_df["Feature"] == "Leaf").sum())
        return max(leaf_count, 1)
    except Exception:
        return fallback_k

def run_xgboost(train_y, test_y, train_X, test_X, cfg):
    from xgboost import XGBRegressor
    from sklearn.linear_model import LinearRegression
    origin = train_y.index[0]
    t_train = (train_y.index - origin).days.values.reshape(-1, 1).astype(float)
    t_test = (test_y.index - origin).days.values.reshape(-1, 1).astype(float)

    trend_model = LinearRegression().fit(t_train, train_y.values)
    train_trend = trend_model.predict(t_train)
    test_trend = trend_model.predict(t_test)
    train_y_detrended = train_y.values - train_trend
    n_val = max(int(len(train_X) * 0.15), 1)
    fit_X, val_X = train_X.iloc[:-n_val], train_X.iloc[-n_val:]
    fit_y, val_y = train_y_detrended[:-n_val], train_y_detrended[-n_val:]

    model = XGBRegressor(**cfg["XGB_PARAMS"])
    model.fit(
        fit_X, fit_y,
        eval_set=[(val_X, val_y)],
        verbose=False,)

    # Add the extrapolated trend back on to get real-scale NDCI predictions
    train_pred = model.predict(train_X) + train_trend
    test_pred = model.predict(test_X) + test_trend

    # Diagnostic: with detrending, predicted range should now track the
    print(f"  Train target range : [{train_y.min():.4f}, {train_y.max():.4f}]")
    print(f"  Test  target range : [{test_y.min():.4f}, {test_y.max():.4f}]")
    print(f"  XGB  predicted range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    rss_train = np.sum((train_y.values - train_pred) ** 2)
    k = xgb_effective_k(model, fallback_k=train_X.shape[1] + 1) + 1 
    aic, bic = pseudo_aic_bic(rss_train, len(train_y), k)
    metrics = regression_metrics(test_y, test_pred)
    metrics.update({"Model": "XGBoost", "AIC": aic, "BIC": bic})
    return metrics, test_pred, model, trend_model, origin

# ARIMAX / SARIMAX 

def run_sarimax_family(train_y, test_y, train_exog, test_exog, order, seasonal_order, trend, name, cfg):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(
        train_y, exog=train_exog, order=order, trend=trend,
        seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,)
    fit = model.fit(disp=False, maxiter=1000, method="lbfgs")
    if not fit.mle_retvals.get("converged", True):
        fit = model.fit(disp=False, maxiter=5000, method="powell")

    # One-step-ahead walk-forward evaluation: coefficients are estimated ONLY
    # on train_y (no leakage), but the Kalman filter state is then updated
    # with the true observed NDCI history as we move through the test period
    # (exactly how a daily monitoring pipeline would run: forecast today
    # using yesterday's actual reading). This is done via .apply(refit=False),
    # which re-runs the filter over the full series with the train-fitted
    # parameters frozen, then get_prediction() over the test window returns
    # genuine one-step-ahead in-sample-filtered predictions.
    full_y = pd.concat([train_y, test_y])
    full_exog = pd.concat([train_exog, test_exog])
    applied = fit.apply(full_y, exog=full_exog, refit=False)
    test_pred = applied.get_prediction(start=test_y.index[0], end=test_y.index[-1]).predicted_mean

    metrics = regression_metrics(test_y, test_pred)
    metrics.update({"Model": name, "AIC": fit.aic, "BIC": fit.bic})
    return metrics, test_pred, fit

# LSTM 

def make_lstm_windows(X_scaled, y_scaled, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X_scaled)):
        Xs.append(X_scaled[i - lookback:i])
        ys.append(y_scaled[i])
    return np.array(Xs), np.array(ys)

def run_lstm(df, feature_cols, target_col, n_test, cfg):
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.random.set_seed(cfg["RANDOM_SEED"])
    lookback = cfg["LSTM_LOOKBACK"]
    X_all = df[feature_cols].values
    y_all = df[[target_col]].values

    n_total = len(df)
    train_end = n_total - n_test
    x_scaler = MinMaxScaler().fit(X_all[:train_end])
    y_scaler = MinMaxScaler().fit(y_all[:train_end])
    X_scaled = x_scaler.transform(X_all)
    y_scaled = y_scaler.transform(y_all).flatten()
    X_seq, y_seq = make_lstm_windows(X_scaled, y_scaled, lookback)
    seq_dates = df.index[lookback:]
    split_point = train_end - lookback 
    X_trainval, X_test = X_seq[:split_point], X_seq[split_point:]
    y_trainval, y_test = y_seq[:split_point], y_seq[split_point:]
    test_dates = seq_dates[split_point:]

    n_val = max(int(len(X_trainval) * 0.15), 1)
    n_val = min(n_val, max(len(X_trainval) - 1, 0))
    if n_val > 0:
        X_train, X_val = X_trainval[:-n_val], X_trainval[-n_val:]
        y_train, y_val = y_trainval[:-n_val], y_trainval[-n_val:]
    else:
        X_train, X_val = X_trainval, X_trainval
        y_train, y_val = y_trainval, y_trainval

    model = Sequential()
    units = cfg["LSTM_UNITS"]
    for i, u in enumerate(units):
        return_seq = i < len(units) - 1
        if i == 0:
            model.add(LSTM(u, return_sequences=return_seq, input_shape=(lookback, X_seq.shape[2])))
        else:
            model.add(LSTM(u, return_sequences=return_seq))
        model.add(Dropout(cfg["LSTM_DROPOUT"]))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=cfg["LSTM_LEARNING_RATE"]), loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=cfg["LSTM_PATIENCE"], restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(cfg["LSTM_PATIENCE"] // 3, 3), min_lr=1e-5)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["LSTM_EPOCHS"],
        batch_size=cfg["LSTM_BATCH_SIZE"],
        callbacks=[es, rlrop],
        verbose=0,)

    train_pred_scaled = model.predict(X_trainval, verbose=0).flatten()
    test_pred_scaled = model.predict(X_test, verbose=0).flatten()
    train_pred = y_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
    test_pred = y_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    y_train_true = y_scaler.inverse_transform(y_trainval.reshape(-1, 1)).flatten()
    y_test_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    rss_train = np.sum((y_train_true - train_pred) ** 2)
    k = model.count_params()
    aic, bic = pseudo_aic_bic(rss_train, len(y_train_true), k)
    metrics = regression_metrics(y_test_true, test_pred)
    metrics.update({"Model": "LSTM", "AIC": aic, "BIC": bic})

    # overfitting for LSTM
    params_per_sample = k / max(len(X_train), 1)
    print(f"  LSTM trainable parameters : {k}")
    print(f"  LSTM training sequences   : {len(X_train)}")
    print(f"  Parameters per sample     : {params_per_sample:.2f}  "
          f"({'>1 -> capacity exceeds data, overfitting risk' if params_per_sample > 1 else 'OK'})")
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    gap_ratio = final_val_loss / max(final_train_loss, 1e-12)
    print(f"  Final train loss (scaled) : {final_train_loss:.6f}")
    print(f"  Final val loss (scaled)   : {final_val_loss:.6f}")
    print(f"  Val/Train loss ratio      : {gap_ratio:.2f}  "
          f"({'notably >1 -> overfitting' if gap_ratio > 1.2 else 'small gap'})")

    return metrics, test_pred, y_test_true, test_dates, history, model, X_test, y_test, k

# plots

def save_seasonal_decomposition(y, cfg):
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomp = seasonal_decompose(y, model="additive", period=cfg["SEASONAL_DECOMPOSE_PERIOD"])
    fig = decomp.plot()
    fig.set_size_inches(10, 8)
    fig.suptitle("Seasonal Decomposition of NDCI", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], "01_seasonal_decomposition.png"), dpi=cfg["DPI"])
    plt.close()

def save_acf_pacf(y, cfg):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(y, lags=60, ax=axes[0])
    axes[0].set_title("ACF of NDCI")
    plot_pacf(y, lags=60, ax=axes[1])
    axes[1].set_title("PACF of NDCI")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], "02_acf_pacf.png"), dpi=cfg["DPI"])
    plt.close()

def save_forecast_comparison(train_y, test_y, preds_dict, cfg):
    plt.figure(figsize=(13, 5.5))
    plt.plot(train_y.index[-120:], train_y.iloc[-120:], label="Train (actual)", color="gray", linewidth=1)
    plt.plot(test_y.index, test_y.values, label="Test (actual)", color="black", linewidth=2.2)
    colors = {"ARIMAX": "orange", "SARIMAX": "green", "XGBoost": "blue", "LSTM": "crimson"}
    for name, (pred_index, pred_values) in preds_dict.items():
        plt.plot(pred_index, pred_values, label=f"{name} forecast", linestyle="--",
                  linewidth=1.8, color=colors.get(name))

    plt.title("Model Forecasts vs Actual NDCI (Test Period)", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("NDCI")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], "03_forecast_comparison.png"), dpi=cfg["DPI"])
    plt.close()

def save_residual_comparison(test_y, preds_dict, cfg):
    n = len(preds_dict)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=False)
    if n == 1:
        axes = [axes]
    colors = {"ARIMAX": "orange", "SARIMAX": "green", "XGBoost": "royalblue", "LSTM": "crimson"}
    for ax, (name, (pred_index, pred_values)) in zip(axes, preds_dict.items()):
        actual_aligned = test_y.reindex(pred_index) if hasattr(test_y, "reindex") else test_y
        if actual_aligned.isna().any():
            actual_aligned = pd.Series(test_y.values[-len(pred_values):], index=pred_index)
        resid = actual_aligned.values - np.asarray(pred_values)
        ax.plot(pred_index, resid, color=colors.get(name))
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{name} Residuals (test period)", fontsize=11)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], "04_residual_comparison.png"), dpi=cfg["DPI"])
    plt.close()

def save_metrics_bar_charts(results_df, cfg):
    metrics_to_plot = ["RMSE", "MAE", "R2", "MAPE (%)"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics_to_plot):
        bars = ax.bar(results_df["Model"], results_df[metric], color=["orange", "green", "royalblue", "crimson"])
        ax.set_title(metric, fontsize=12)
        ax.grid(alpha=0.3, axis="y")
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)

    plt.suptitle("Model Comparison: Error & Metrics",
                  fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], "05_metrics_comparison.png"), dpi=cfg["DPI"])
    plt.close()


def save_aic_bic_chart(results_df, cfg):
    plt.figure(figsize=(8, 5))
    x = np.arange(len(results_df))
    width = 0.35
    plt.bar(x - width / 2, results_df["AIC"], width, label="AIC")
    plt.bar(x + width / 2, results_df["BIC"], width, label="BIC")
    plt.xticks(x, results_df["Model"])
    plt.ylabel("Score (lower = better)")
    plt.title("Model Fit: AIC / BIC Comparison",
               fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], "06_aic_bic_comparison.png"), dpi=cfg["DPI"])
    plt.close()

def save_statsmodels_feature_importance(fit, feature_cols, name, cfg, chart_num):
    coefs = fit.params.reindex(feature_cols).dropna()
    if coefs.empty:
        return
    importance = coefs.abs().sort_values()
    plt.figure(figsize=(8, max(4, 0.35 * len(importance))))
    importance.plot(kind="barh", color="orange" if name == "ARIMAX" else "green")
    plt.title(f"{name} Feature Importance (|standardized coefficient|)")
    plt.xlabel("Absolute standardized coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], f"{chart_num}_{name.lower()}_feature_importance.png"), dpi=cfg["DPI"])
    plt.close()

def save_lstm_permutation_importance(model, X_test, y_test, feature_cols, cfg, chart_num):
    rng = np.random.default_rng(cfg["RANDOM_SEED"])
    baseline_pred = model.predict(X_test, verbose=0).flatten()
    baseline_mse = np.mean((baseline_pred - y_test) ** 2)
    importances = []
    for i in range(len(feature_cols)):
        X_perm = X_test.copy()
        for t in range(X_perm.shape[1]):
            rng.shuffle(X_perm[:, t, i])
        perm_pred = model.predict(X_perm, verbose=0).flatten()
        perm_mse = np.mean((perm_pred - y_test) ** 2)
        importances.append(perm_mse - baseline_mse)
    imp_series = pd.Series(importances, index=feature_cols).sort_values()
    plt.figure(figsize=(8, max(4, 0.35 * len(imp_series))))
    imp_series.plot(kind="barh", color="crimson")
    plt.title("LSTM Feature Importance (Permutation: increase in test MSE)")
    plt.xlabel("MSE increase when feature is shuffled (scaled target units)")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], f"{chart_num}_lstm_feature_importance.png"), dpi=cfg["DPI"])
    plt.close()

def save_model_complexity_chart(param_counts, n_train, cfg, chart_num):
    models = list(param_counts.keys())
    ratios = [c / max(n_train, 1) for c in param_counts.values()]
    plt.figure(figsize=(8, 5))
    x = np.arange(len(models))
    colors = {"ARIMAX": "orange", "SARIMAX": "green", "XGBoost": "royalblue", "LSTM": "crimson"}
    plt.bar(x, ratios, color=[colors.get(m, "gray") for m in models])
    plt.xticks(x, models)
    plt.yscale("log")
    plt.ylabel("Effective parameters per training sample (log scale)")
    plt.title(f"Model Complexity vs. Training Data Size (n_train = {n_train})\n"
              f"Higher bar = higher overfitting risk")
    plt.grid(alpha=0.3, axis="y")
    for xi, r, c in zip(x, ratios, param_counts.values()):
        plt.annotate(f"k={c}", (xi, r), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], f"{chart_num}_model_complexity_vs_data.png"), dpi=cfg["DPI"])
    plt.close()

def run_learning_curve_experiment(train_y, test_y, train_exog_s, test_exog_s, xgb_train_X, xgb_test_X,
                                   df, feature_cols, target_col, n_test, cfg):
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    n_train_full = len(train_y)
    sizes = [max(int(n_train_full * f), 30) for f in fractions]
    curve = {"ARIMAX": [], "SARIMAX": [], "XGBoost": [], "LSTM": []}
    print("\nLearning curve")
    for n_sub in sizes:
        sub_train_y = train_y.iloc[-n_sub:]
        sub_train_exog_s = train_exog_s.iloc[-n_sub:]
        sub_xgb_train_X = xgb_train_X.iloc[-n_sub:]

        try:
            m, _, _ = run_sarimax_family(sub_train_y, test_y, sub_train_exog_s, test_exog_s,
                                          cfg["ARIMAX_ORDER"], (0, 0, 0, 0), cfg["ARIMAX_TREND"], "ARIMAX", cfg)
            curve["ARIMAX"].append(m["RMSE"])
        except Exception:
            curve["ARIMAX"].append(np.nan)

        try:
            m, _, _ = run_sarimax_family(sub_train_y, test_y, sub_train_exog_s, test_exog_s,
                                          cfg["SARIMAX_ORDER"], cfg["SARIMAX_SEASONAL_ORDER"], cfg["SARIMAX_TREND"], "SARIMAX", cfg)
            curve["SARIMAX"].append(m["RMSE"])
        except Exception:
            curve["SARIMAX"].append(np.nan)

        try:
            m, _, _, _, _ = run_xgboost(sub_train_y, test_y, sub_xgb_train_X, xgb_test_X, cfg)
            curve["XGBoost"].append(m["RMSE"])
        except Exception:
            curve["XGBoost"].append(np.nan)

        try:
            df_subset = df.loc[sub_train_y.index[0]: test_y.index[-1]]
            m, *_ = run_lstm(df_subset, feature_cols, target_col, n_test, cfg)
            curve["LSTM"].append(m["RMSE"])
        except Exception:
            curve["LSTM"].append(np.nan)

        print(f"  n_train={n_sub}: " + ", ".join(f"{k}={v[-1]:.4f}" if not np.isnan(v[-1]) else f"{k}=NaN"
                                                   for k, v in curve.items()))

    plt.figure(figsize=(8.5, 5.5))
    colors = {"ARIMAX": "orange", "SARIMAX": "green", "XGBoost": "royalblue", "LSTM": "crimson"}
    for name, rmse_vals in curve.items():
        plt.plot(sizes, rmse_vals, marker="o", label=name, color=colors.get(name))
    plt.xlabel("Training set size (rows)")
    plt.ylabel("Test RMSE (lower = better)")
    plt.title("Learning Curves: Model Data\nLSTM requires substantially more data but sufficient for ARIMAX, SARIMAX and XGBoost")
    plt.axvspan(sizes[0], sizes[-2], alpha=0.08, color="red")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], "14_learning_curve_comparison.png"), dpi=cfg["DPI"])
    plt.close()
    return curve, sizes

def save_xgb_feature_importance(model, feature_cols, cfg):
    plt.figure(figsize=(8, max(4, 0.35 * len(feature_cols))))
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    importances.plot(kind="barh", color="royalblue")
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], "07_xgboost_feature_importance.png"), dpi=cfg["DPI"])
    plt.close()

def save_lstm_training_curve(history, cfg):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title("LSTM Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (scaled)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], "08_lstm_training_curve.png"), dpi=cfg["DPI"])
    plt.close()

def save_scatter_grid(test_y, preds_dict, cfg):
    n = len(preds_dict)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, (name, (pred_index, pred_values)) in zip(axes, preds_dict.items()):
        actual_aligned = test_y.reindex(pred_index) if hasattr(test_y, "reindex") else test_y
        if actual_aligned.isna().any():
            actual_aligned = pd.Series(test_y.values[-len(pred_values):], index=pred_index)
        a = actual_aligned.values
        p = np.asarray(pred_values)
        ax.scatter(a, p, alpha=0.6, edgecolor="k", s=25)
        lims = [min(a.min(), p.min()), max(a.max(), p.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Actual NDCI")
        ax.set_ylabel("Predicted NDCI")
        ax.grid(alpha=0.3)

    plt.suptitle("Actual vs Predicted NDCI (Test Period)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["OUT_DIR"], "09_actual_vs_predicted_scatter.png"), dpi=cfg["DPI"])
    plt.close()

def main():
    cfg = CONFIG
    df, feature_cols = load_data(cfg)
    y = df[cfg["TARGET_COL"]]
    exog = df[feature_cols]
    n_test = max(int(round(len(df) * cfg["TEST_SIZE_RATIO"])), 1)
    print(f"Loaded data: {len(df)} rows, {len(feature_cols)} feature columns.")
    print(f"Train size: {len(df) - n_test}  |  Test size: {n_test}  (80:20 split)\n")

    # plots
    save_seasonal_decomposition(y, cfg)
    save_acf_pacf(y, cfg)

    # splitting part
    train_y, test_y = y.iloc[:-n_test], y.iloc[-n_test:]
    train_exog, test_exog = exog.iloc[:-n_test], exog.iloc[-n_test:]
    all_metrics = []
    preds_for_plots = {}
    from sklearn.preprocessing import StandardScaler
    exog_scaler = StandardScaler().fit(train_exog)
    train_exog_s = pd.DataFrame(exog_scaler.transform(train_exog), index=train_exog.index, columns=train_exog.columns)
    test_exog_s = pd.DataFrame(exog_scaler.transform(test_exog), index=test_exog.index, columns=test_exog.columns)

    # ARIMAX
    print("ARIMAX")
    arimax_metrics, arimax_pred, arimax_fit = run_sarimax_family(
        train_y, test_y, train_exog_s, test_exog_s,
        cfg["ARIMAX_ORDER"], (0, 0, 0, 0), cfg["ARIMAX_TREND"], "ARIMAX", cfg)
    all_metrics.append(arimax_metrics)
    preds_for_plots["ARIMAX"] = (test_y.index, arimax_pred.values)

    # SARIMAX
    print("SARIMAX")
    sarimax_metrics, sarimax_pred, sarimax_fit = run_sarimax_family(
        train_y, test_y, train_exog_s, test_exog_s,
        cfg["SARIMAX_ORDER"], cfg["SARIMAX_SEASONAL_ORDER"], cfg["SARIMAX_TREND"], "SARIMAX", cfg )
    all_metrics.append(sarimax_metrics)
    preds_for_plots["SARIMAX"] = (test_y.index, sarimax_pred.values)

    # XGBoost
    print("XGBoost")
    xgb_train_X = train_exog.copy()
    xgb_test_X = test_exog.copy()
    xgb_train_X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt") for c in xgb_train_X.columns]
    xgb_test_X.columns = xgb_train_X.columns
    xgb_metrics, xgb_pred, xgb_model, xgb_trend_model, xgb_origin = run_xgboost(train_y, test_y, xgb_train_X, xgb_test_X, cfg)
    all_metrics.append(xgb_metrics)
    preds_for_plots["XGBoost"] = (test_y.index, xgb_pred)

    # LSTM 
    print("LSTM")
    lstm_metrics, lstm_pred, lstm_true, lstm_dates, lstm_history, lstm_model, lstm_X_test, lstm_y_test, lstm_k = run_lstm(
        df, feature_cols, cfg["TARGET_COL"], n_test, cfg )
    all_metrics.append(lstm_metrics)
    preds_for_plots["LSTM"] = (lstm_dates, lstm_pred)

    # results values
    results_df = pd.DataFrame(all_metrics)
    col_order = ["Model", "RMSE", "MAE", "MAPE (%)", "SMAPE (%)", "R2", "Directional Accuracy (%)", "AIC", "BIC"]
    results_df = results_df[col_order]
    print("\nRESULTS")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(cfg["OUT_DIR"], "model_comparison_metrics.csv"), index=False)

    # plots
    save_forecast_comparison(train_y, test_y, preds_for_plots, cfg)
    save_residual_comparison(test_y, preds_for_plots, cfg)
    save_scatter_grid(test_y, preds_for_plots, cfg)
    save_metrics_bar_charts(results_df, cfg)
    save_aic_bic_chart(results_df, cfg)
    save_xgb_feature_importance(xgb_model, list(xgb_train_X.columns), cfg)
    save_lstm_training_curve(lstm_history, cfg)

    # Feature importance for ARIMAX / SARIMAX 
    save_statsmodels_feature_importance(arimax_fit, list(train_exog_s.columns), "ARIMAX", cfg, chart_num=10)
    save_statsmodels_feature_importance(sarimax_fit, list(train_exog_s.columns), "SARIMAX", cfg, chart_num=11)

    # Feature importance for LSTM
    # feature set, i.e. before XGBoost's lag/rolling engineering)
    save_lstm_permutation_importance(lstm_model, lstm_X_test, lstm_y_test, feature_cols, cfg, chart_num=12)

    xgb_k = xgb_effective_k(xgb_model, fallback_k=xgb_train_X.shape[1] + 1)
    param_counts = {
        "ARIMAX": len(arimax_fit.params),
        "SARIMAX": len(sarimax_fit.params),
        "XGBoost": xgb_k,
        "LSTM": lstm_k,
    }
    save_model_complexity_chart(param_counts, n_train=len(train_y), cfg=cfg, chart_num=13)
    if cfg.get("RUN_LEARNING_CURVE", True):
        run_learning_curve_experiment(
            train_y, test_y, train_exog_s, test_exog_s, xgb_train_X, xgb_test_X,
            df, feature_cols, cfg["TARGET_COL"], n_test, cfg)

    output_df = df.copy()
    output_df["Data_Split"] = ["Train"] * (len(df) - n_test) + ["Test"] * n_test  # unseen 20%

    output_df["NDCI_Predicted_ARIMAX"] = np.nan
    output_df.loc[test_y.index, "NDCI_Predicted_ARIMAX"] = arimax_pred.values

    output_df["NDCI_Predicted_SARIMAX"] = np.nan
    output_df.loc[test_y.index, "NDCI_Predicted_SARIMAX"] = sarimax_pred.values

    output_df["NDCI_Predicted_XGBoost"] = np.nan
    output_df.loc[test_y.index, "NDCI_Predicted_XGBoost"] = xgb_pred

    output_df["NDCI_Predicted_LSTM"] = np.nan
    output_df.loc[lstm_dates, "NDCI_Predicted_LSTM"] = lstm_pred

    output_df.to_csv(os.path.join(cfg["OUT_DIR"], "Final_Dataset_with_All_Predictions.csv"))


if __name__ == "__main__":
    main()