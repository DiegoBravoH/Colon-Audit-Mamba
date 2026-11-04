"""Figure for docs/statistics.md: how the paired video-level bootstrap CI of a wF1 gain is obtained.

Example: Colon-Audit-Mamba (causal, 5 runs) vs. the online Colon-Audit on the REAL-Colon test videos.
  left:  distribution of the 2,000 bootstrap gains, with the gain, the 95% CI (percentiles 2.5 and 97.5) and 0
  right: per-video wF1 gain (runs averaged), coloured by center, to show where the variability comes from
The resampling follows metrics.paired_bootstrap (same seed), so the CI equals the one in the paper.

Run:  python evaluation/bootstrap_figure.py [--protocol 4fold] [--out figures/bootstrap_ci.png]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import COLON_AUDIT_PREDICTIONS, OUTPUT_DIR  # noqa: E402
from metrics import confusion_by_video, wf1_from_cm  # noqa: E402

FOLDS = {"5fold": 5, "4fold": 4}
INK, MUTED, OURS, REF, CI = "#1f2937", "#6b7280", "#1d4ed8", "#9ca3af", "#dbeafe"
CENTERS = {"001": "#1d4ed8", "002": "#059669", "003": "#d97706", "004": "#be185d"}


def load(protocol, root, published):
    n = FOLDS[protocol]
    pub = pd.concat([pd.read_json(published.format(p=protocol, k=k)) for k in range(1, n + 1)], ignore_index=True)
    runs = []
    base = os.path.join(root, protocol)
    for s in sorted(os.listdir(os.path.join(base, "fold1", "mamba12-endomamba"))):
        paths = [os.path.join(base, f"fold{k}", "mamba12-endomamba", s, "frame_predictions_test_val_mcc.json")
                 for k in range(1, n + 1)]
        if all(os.path.exists(p) for p in paths):
            runs.append(pd.concat([pd.read_json(p) for p in paths], ignore_index=True))
    return runs, pub


def bootstrap(cm_a, cm_b, n_boot, seed):
    """Same procedure as metrics.paired_bootstrap, keeping the 2,000 resampled gains."""
    videos = sorted(cm_a[0])
    rng = np.random.default_rng(seed)

    def gain(sample):
        return float(np.mean([wf1_from_cm(sum(a[v] for v in sample)) - wf1_from_cm(sum(b[v] for v in sample))
                              for a, b in zip(cm_a, cm_b)]))

    return gain(videos), np.array([gain(rng.choice(videos, len(videos))) for _ in range(n_boot)]), videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="4fold", choices=list(FOLDS))
    ap.add_argument("--root", default=os.path.join(OUTPUT_DIR, "mamba"))
    ap.add_argument("--published", default=COLON_AUDIT_PREDICTIONS)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--font", default="Poppins")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures",
                                                  "bootstrap_ci.png"))
    a = ap.parse_args()

    runs, pub = load(a.protocol, a.root, a.published)
    cm_a = [confusion_by_video(d, "PredictedClass") for d in runs]
    cm_b = [confusion_by_video(pub, "PredictedClass")] * len(cm_a)
    g, boots, videos = bootstrap(cm_a, cm_b, a.n_boot, a.seed)
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    per_video = pd.Series({v: np.mean([wf1_from_cm(c[v]) for c in cm_a]) - wf1_from_cm(cm_b[0][v]) for v in videos})
    per_video = per_video.sort_values()
    print(f"{a.protocol}: gain {g:+.2f} wF1 points, 95% CI [{lo:+.2f}, {hi:+.2f}], {len(videos)} videos, "
          f"{len(runs)} runs; {(per_video > 0).sum()} of {len(videos)} videos improve")

    plt.rcParams.update({"font.family": a.font, "font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.labelcolor": INK, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1, 1.25]})
    name = "cross-center" if a.protocol == "4fold" else "5-fold"

    # left: bootstrap distribution of the gain
    ax1.axvspan(lo, hi, color=CI, zorder=0)
    ax1.hist(boots, bins=40, color=OURS, alpha=0.85, zorder=1)
    ax1.axvline(g, color=INK, lw=1.2, zorder=2)
    ax1.axvline(0, color="#dc2626", lw=1, ls="--", zorder=2)
    ymax = ax1.get_ylim()[1]
    ax1.text(g, ymax * 1.02, f"gain {g:+.1f}", ha="center", va="bottom", color=INK, fontweight="bold")
    ax1.text(lo, ymax * 0.90, f"{lo:+.1f}\n(2.5%)", ha="right", va="top", color=MUTED, fontsize=7)
    ax1.text(hi, ymax * 0.90, f"{hi:+.1f}\n(97.5%)", ha="left", va="top", color=MUTED, fontsize=7)
    ax1.text(0, ymax * 0.5, "0  ", ha="right", va="center", color="#dc2626", fontsize=7)
    ax1.set_xlim(min(-1.0, boots.min() - 0.5), boots.max() + 1.0)
    ax1.set_ylim(0, ymax * 1.12)
    ax1.set_xlabel("wF1 gain over Colon-Audit (points)")
    ax1.set_ylabel(f"bootstrap resamples (of {a.n_boot:,})")
    ax1.set_title(f"(a) {a.n_boot:,} paired resamples, 95% CI shaded", loc="left", fontsize=8, color=INK)

    # right: per-video gain
    x = np.arange(len(per_video))
    ax2.axhline(0, color="#dc2626", lw=1, ls="--")
    ax2.bar(x, per_video.values, color=[CENTERS.get(v.split("-")[0], OURS) for v in per_video.index], width=0.8)
    ax2.set_xticks([])
    ax2.set_xlabel(f"test videos (n = {len(videos)}), sorted by gain")
    ax2.set_ylabel("per-video wF1 gain (points)")
    ax2.set_title("(b) gain per test video (runs averaged)", loc="left", fontsize=8, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CENTERS.values()]
    ax2.legend(handles, [f"center {c}" for c in CENTERS], frameon=False, fontsize=6.5, loc="upper left",
               handlelength=1.0, handletextpad=0.4)

    fig.suptitle(f"Colon-Audit-Mamba vs. Colon-Audit, REAL-Colon {name} test", x=0.01, ha="left",
                 fontsize=9, color=INK, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print("saved", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
