"""Model tabanli onceliklendirme (tarama) faydasi.
Soru: modelin sirasiyla secilen ilk-k aday, rastgele secime gore daha cok
yuksek-verimli hucre icerir mi? -> precision@k ve zenginlestirme (enrichment).
Ek olarak kazanc (gains/lift) egrisi: incelenen oran vs yakalanan yuksek-verim orani."""
import numpy as np

def enrichment_table(y_true, y_pred, thresholds, k_fracs):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); n = len(y_true)
    order = np.argsort(y_pred)[::-1]
    rows = []
    for thr in thresholds:
        base = float((y_true >= thr).mean())
        for kf in k_fracs:
            k = max(1, int(kf * n))
            topk = order[:k]
            prec = float((y_true[topk] >= thr).mean())
            rows.append({"esik": thr, "k_frac": kf, "k": k, "taban_oran": base,
                         "model_isabet": prec, "zenginlestirme": (prec / base) if base > 0 else float("nan")})
    return rows

def gains_curve(y_true, y_pred, threshold, n_points=50):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); n = len(y_true)
    order = np.argsort(y_pred)[::-1]
    pos_total = max(1, int((y_true >= threshold).sum()))
    fracs = np.linspace(1.0/n_points, 1.0, n_points)
    captured = []
    for f in fracs:
        k = max(1, int(f * n))
        captured.append((y_true[order[:k]] >= threshold).sum() / pos_total)
    return fracs, np.array(captured)
