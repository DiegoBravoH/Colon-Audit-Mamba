"""Figure 2 of the ISBI paper: frame-wise timeline of one complete REAL-Colon test procedure (5-fold protocol).

Rows: ground truth; online = Colon-Audit (588-frame window, published predictions) and Colon-Audit-Mamba causal;
offline = Colon-Audit + HMM (published) and Colon-Audit-Mamba bidirectional. Under each prediction a thin strip
marks the wrong frames; on the right, weighted F1, label changes and predicted withdrawal time.

Case: by default, the test procedure whose causal gain over the online Colon-Audit (mean wF1 over the 5 runs) is
closest to the median gain over the 60 procedures. The paper's Figure 2 instead shows procedure 004-006, chosen
manually as a high-performing example (--video 004-006 --online-only). Within the procedure, the run (seed) with
the median causal wF1 is shown; the bidirectional row uses the same fold and seed.
Class colours follow Colon-Audit (SIPAIM 2026) Fig. 2, with ileum, transverse, descending and rectum re-stepped so
that classes that touch in time stay distinguishable for colour-blind readers (OKLab Delta E checked).

Output: OUTPUT_DIR/figures/fig2.pdf (+ fig2.png preview) and fig2_case.json (selection details)
Run:  python analysis/figure_timeline.py --video 004-006 --online-only
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation"))
from paper_tables import runs_of, PUBLISHED  # noqa: E402
from metrics import WITHDRAWAL  # noqa: E402
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import OUTPUT_DIR  # noqa: E402

MODELS = os.path.join(OUTPUT_DIR, "mamba")
OUT = os.path.join(OUTPUT_DIR, "figures")
FPS = 5
NAMES = ["Outside", "Insertion", "Cecum", "Ileum", "Ascending", "Transverse", "Descending", "Sigmoid", "Rectum"]
COLORS = ["#878680", "#2978d5", "#eb6633", "#ffd43b", "#1aaf79", "#f6b3cf", "#006400", "#493aa6", "#b0203c"]
DARK_TEXT = {3, 5}                       # light fills (ileum, transverse) take dark labels
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#d9d8d3"
X_GROUP, X_MODEL = 0.008, 0.028            # label hierarchy (figure fraction): group headers, indented rows


def wf1(df, col):
    return f1_score(df["anatomy"], df[col], average="weighted") * 100


def select_case(video=None):
    pub = pd.concat([pd.read_json(PUBLISHED.format(p="5fold", k=k)) for k in range(1, 6)], ignore_index=True)
    causal = runs_of(MODELS, "5fold", "mamba12-endomamba", "test")
    bidir = runs_of(MODELS, "5fold", "mamba12-endomamba-bidir", "test")
    rows = []
    for v, g in pub.groupby("patient"):
        per_seed = {s: wf1(d[d.patient == v], "PredictedClass") for s, d in causal.items()}
        rows.append({"video": v, "gain": np.mean(list(per_seed.values())) - wf1(g, "PredictedClass"), "seeds": per_seed})
    gains = np.array([r["gain"] for r in rows])
    case = (next(r for r in rows if r["video"] == video) if video else
            rows[int(np.argmin(np.abs(gains - np.median(gains))))])
    seeds = sorted(case["seeds"], key=case["seeds"].get)
    seed = seeds[len(seeds) // 2]
    v = case["video"]
    pick = lambda d: d[d.patient == v].sort_values("frame").reset_index(drop=True)  # noqa: E731
    p, c, b = pick(pub), pick(causal[seed]), pick(bidir[seed])
    assert (p.frame.values == c.frame.values).all() and (p.frame.values == b.frame.values).all(), "frames differ"
    assert (p.anatomy.values == c.anatomy.values).all()
    info = {"video": v, "seed": seed, "gain_of_case": case["gain"], "median_gain": float(np.median(gains)),
            "causal_wF1_per_seed": case["seeds"]}
    return v, p.anatomy.values, [("Colon-Audit", p.PredictedClass.values), ("Ours, causal", c.PredictedClass.values),
                                 ("Colon-Audit + HMM", p.PredictedClass_HMM.values),
                                 ("Ours, bidirectional", b.PredictedClass.values)], info


def runs(y):
    """(start, end, class) of each constant segment, in frames."""
    cut = np.flatnonzero(np.diff(y)) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(y)]
    return [(s, e, y[s]) for s, e in zip(starts, ends)]


def mmss(frames):
    s = int(round(frames / FPS))
    return f"{s // 60}:{s % 60:02d}"


def draw_bar(ax, y0, h, y, minutes):
    for s, e, k in runs(y):
        ax.add_patch(plt.Rectangle((minutes[s], y0), minutes[e - 1] - minutes[s] + 1 / (FPS * 60), h,
                                   color=COLORS[k], lw=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="show this test procedure instead of the median-gain rule")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--font", default="Poppins")
    ap.add_argument("--online-only", action="store_true", help="ground truth and the two online rows only")
    a = ap.parse_args()
    plt.rcParams.update({"font.family": a.font, "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "font.size": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "axes.labelcolor": MUTED})
    v, gt, preds, info = select_case(a.video)
    if a.online_only:
        preds = preds[:2]
    T = len(gt)
    minutes = np.arange(T) / FPS / 60
    gt_wt = np.isin(gt, WITHDRAWAL).sum()

    bar, err, gap, head = 0.62, 0.13, 0.26, 0.40
    layout, y = [], 0.0                     # rows from top to bottom
    layout.append(("gt", y)); y -= bar + head
    for i, (name, p) in enumerate(preds):
        if i == 2:
            y -= head - gap
        layout.append((i, y)); y -= bar + err + gap
    bottom = y + gap

    H = 2.35 if not a.online_only else 0.66 + 1.69 * (-bottom) / 5.2   # same margins and bar height in inches
    fig, ax = plt.subplots(figsize=(7.0, H))
    fig.subplots_adjust(left=0.172, right=0.80, top=1 - 0.33 / H, bottom=0.33 / H)
    ax.set_xlim(0, minutes[-1]); ax.set_ylim(bottom, 0.02)
    for key, y0 in layout:
        if key == "gt":
            draw_bar(ax, y0 - bar, bar, gt, minutes)
            for s, e, k in runs(gt):
                if (e - s) / T * 0.645 * 7.0 * 72 > 4.2 * len(NAMES[k]) + 6:   # label only where it fits
                    ax.text((minutes[s] + minutes[e - 1]) / 2, y0 - bar / 2, NAMES[k], ha="center", va="center",
                            fontsize=6.5, color=INK if k in DARK_TEXT else "white")
            fig.text(X_GROUP, ax.transData.transform((0, y0 - bar / 2))[1] / fig.dpi / fig.get_figheight(),
                     "Ground truth", ha="left", va="center", color=INK, fontweight="bold")
            ax.text(minutes[-1] * 1.015, y0 - bar / 2, f"withdrawal {mmss(gt_wt)}", ha="left", va="center", color=INK)
            continue
        name, p = preds[key]
        draw_bar(ax, y0 - bar, bar, p, minutes)
        wrong = (p != gt).astype(int)
        for s, e, k in runs(wrong):
            if k:
                ax.add_patch(plt.Rectangle((minutes[s], y0 - bar - err), minutes[e - 1] - minutes[s] + 1 / (FPS * 60),
                                           err * 0.8, color=INK, lw=0))
        df = pd.DataFrame({"anatomy": gt, "p": p})
        stats = (f"wF1 {wf1(df, 'p'):.1f}  ·  {int((np.diff(p) != 0).sum())} changes\n"
                 f"withdrawal {mmss(np.isin(p, WITHDRAWAL).sum())}")
        fig.text(X_MODEL, ax.transData.transform((0, y0 - bar / 2))[1] / fig.dpi / fig.get_figheight(), name,
                 ha="left", va="center", color=INK, fontweight="bold" if name.startswith("Ours") else "normal")
        ax.text(minutes[-1] * 1.015, y0 - bar / 2, stats, ha="left", va="center", color=INK, linespacing=1.15)
    for key, label in ((0, "Online (real time)"), (2, "Offline (after the procedure)")) if len(preds) > 2 else ():
        y0 = dict(layout)[key]
        fig.text(X_GROUP, ax.transData.transform((0, y0 + 0.10))[1] / fig.dpi / fig.get_figheight(), label,
                 ha="left", va="bottom", color=INK, style="italic")

    ax.set_xlim(0, minutes[-1]); ax.set_ylim(bottom, 0.02)
    ax.set_yticks([]); ax.set_xlabel("time (min)", labelpad=1)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", length=2, pad=1.5)
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, NAMES)] +
               [Patch(color=INK, label="error")], loc="upper left", ncol=10, frameon=False,
               bbox_to_anchor=(0.165, 1 + 0.012 / H), handlelength=1.0, handleheight=0.9, columnspacing=0.6, handletextpad=0.3,
               fontsize=5.8)
    fig.text(X_GROUP, 1 - 0.106 / H, f"Procedure {v}", ha="left", va="center", color=INK, fontweight="bold")
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "fig2.pdf"))
    fig.savefig(os.path.join(a.out, "fig2.png"), dpi=220)
    info["rows"] = {n: {"wF1": wf1(pd.DataFrame({"anatomy": gt, "p": p}), "p"),
                        "changes": int((np.diff(p) != 0).sum()),
                        "withdrawal_s": float(np.isin(p, WITHDRAWAL).sum() / FPS)} for n, p in preds}
    info["gt_withdrawal_s"] = float(gt_wt / FPS)
    info["gt_changes"] = int((np.diff(gt) != 0).sum())
    json.dump(info, open(os.path.join(a.out, "fig2_case.json"), "w"), indent=2)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
