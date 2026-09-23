# Rossman/metrics.py — Ortak değerlendirme metrikleri
import numpy as np


def tum_metrikler(y_true, y_pred, isim="Model"):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_pred = np.clip(y_pred, 0, None)   # negatif tahmini sıfırla (satış negatif olamaz)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    m = y_true != 0   # yüzde metrikler için sıfır bölme koruması
    mape = 100 * np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m]))
    rmspe = 100 * np.sqrt(np.mean(((y_true[m] - y_pred[m]) / y_true[m]) ** 2))
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) /
                          (np.abs(y_true) + np.abs(y_pred)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n===== {isim} =====")
    print(f"MAE   : {mae:,.1f} €")
    print(f"RMSE  : {rmse:,.1f} €")
    print(f"MAPE  : {mape:.2f} %")
    print(f"sMAPE : {smape:.2f} %")
    print(f"RMSPE : {rmspe:.2f} %")
    print(f"R²    : {r2:.4f}")

    return {"Model": isim, "MAE": round(mae, 1), "RMSE": round(rmse, 1),
            "MAPE": round(mape, 2), "sMAPE": round(smape, 2),
            "RMSPE": round(rmspe, 2), "R2": round(r2, 4)}