"""Shared evaluation utilities: frame-level metrics, withdrawal-time WMAPE and the paired video bootstrap.

All functions work on frame-level DataFrames with at least the columns
    patient, frame, anatomy (ground truth 0-8) and a prediction column (0-8),
pooled over the folds of a protocol (each video appears once).
"""
import numpy as np
from sklearn.metrics import f1_score, jaccard_score

NAMES = ["outside", "insertion", "cecum", "ileum", "ascending", "transverse", "descending", "sigmoid", "rectum"]
WITHDRAWAL = range(2, 9)          # cecum .. rectum


def confusion_by_video(df, pred_col, n=9):
    """{video: n x n confusion matrix}, so that bootstrap resamples only add matrices."""
    out = {}
    for v, g in df.groupby("patient"):
        cm = np.zeros((n, n), np.int64)
        np.add.at(cm, (g["anatomy"].to_numpy().astype(int), g[pred_col].to_numpy().astype(int)), 1)
        out[v] = cm
    return out


def wf1_from_cm(cm):
    """Support-weighted F1 (sklearn average='weighted') from a confusion matrix, in %."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0)
    return float((f1 * support).sum() / max(support.sum(), 1)) * 100


def wmape(df, pred_col):
    """Withdrawal-time WMAPE (%): mean over videos of |A - P| / A, A/P = actual/predicted withdrawal frames."""
    errs = []
    for _, g in df.groupby("patient"):
        actual = g["anatomy"].isin(WITHDRAWAL).sum()
        pred = g[pred_col].isin(WITHDRAWAL).sum()
        errs.append(abs(actual - pred) / actual * 100)
    return float(np.mean(errs))


def summarize(df, pred_col, labels=range(9)):
    y, p = df["anatomy"].to_numpy(), df[pred_col].to_numpy()
    res = {"wF1": f1_score(y, p, average="weighted") * 100,
           "wJacc": jaccard_score(y, p, average="weighted") * 100,
           "macroF1": f1_score(y, p, labels=list(labels), average="macro", zero_division=0) * 100,
           "F1_per_class": dict(zip([NAMES[c] for c in labels],
                                    (f1_score(y, p, labels=list(labels), average=None, zero_division=0) * 100)
                                    .tolist())),
           "WMAPE_%": wmape(df, pred_col)}
    if "near_lesion" in df:
        res["error_near_lesion_%"] = float((p != y)[df["near_lesion"].to_numpy() == 1].mean() * 100)
    return res


def paired_bootstrap(models_a, model_b, col_a, col_b, n_boot=2000, seed=0):
    """wF1 gain of A over B on the same videos. models_a: list of DataFrames (e.g. one per seed), model_b: one
    DataFrame or a list of the same length (paired by position). The gain is averaged over the A models; the
    95% CI resamples videos (the same resample for every model)."""
    cm_a = [confusion_by_video(d, col_a) for d in models_a]
    if isinstance(model_b, list):
        cm_b = [confusion_by_video(d, col_b) for d in model_b]
    else:
        cm_b = [confusion_by_video(model_b, col_b)] * len(cm_a)
    videos = sorted(cm_a[0])
    rng = np.random.default_rng(seed)

    def gain(sample):
        return float(np.mean([wf1_from_cm(sum(a[v] for v in sample)) - wf1_from_cm(sum(b[v] for v in sample))
                              for a, b in zip(cm_a, cm_b)]))

    boots = [gain(rng.choice(videos, len(videos))) for _ in range(n_boot)]
    return {"gain": gain(videos), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "videos": len(videos), "models_a": len(cm_a)}
