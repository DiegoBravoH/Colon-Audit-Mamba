"""Paper tables from the test predictions of every run (test is never used to choose anything).

For each protocol, each configuration and each run (seed) the 60 test videos are pooled over the folds.
  * main table (Table 1 style): per-class F1, wF1, wJacc, macro F1, WMAPE, error near lesions
      - Colon-Audit published, Multi-frame (online) and Multi-frame+HMM (offline), from its released predictions
      - ColonTCN (Biffi et al. 2025), numbers as reported in the Colon-Audit paper
      - Colon-Audit-Mamba (C1), mean +- std over runs (raw causal output, no post-processing)
  * ablation table: every configuration, mean +- std over runs (test and validation wF1), and the paired video
    bootstrap of its wF1 against C1 (runs paired by seed) and against the published Colon-Audit+HMM.
Writes <out>/paper_tables.json, <out>/table_main_<protocol>.csv, <out>/table_ablation_<protocol>.csv

Run:  python evaluation/paper_tables.py [--root OUTPUT_DIR/mamba]
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import COLON_AUDIT_PREDICTIONS as PUBLISHED, LESION_DIR as LESION, OUTPUT_DIR  # noqa: E402
from metrics import summarize, paired_bootstrap, NAMES  # noqa: E402

PROTOCOLS = {"5fold": 5, "4fold": 4}
# ColonTCN, as reported in Colon-Audit (SIPAIM 2026) Table 1 and Section 3.2
COLONTCN = {"4fold": {"F1": [87.9, 84.9, 49.2, 30.1, 41.6, 56.7, 26.2, 53.0, 65.7], "wF1": 62.7, "wJacc": 48.6,
                      "WMAPE_%": 16.8},
            "5fold": {"F1": [95.1, 95.8, 67.3, 56.0, 49.9, 74.8, 47.3, 65.9, 72.5], "wF1": 76.2, "wJacc": 64.7,
                      "WMAPE_%": 3.1}}
CONFIGS = {"C1": "mamba12-endomamba", "C2": "mamba12-endomamba-ctx588", "C3": "mamba12-endomamba-ctx2048",
           "C4": "mamba12-scratch", "C5": "mamba12-endomamba-bidir", "C6": "mamba4-endomamba",
           "C7": "mamba8-endomamba", "C8": "mamba12-endomamba-nospeed"}
DESC = {"C1": "final: 12 causal blocks, EndoMamba init, whole procedure",
        "C2": "context 588 tokens (~2 min)", "C3": "context 2048 tokens (~7 min)",
        "C4": "12 blocks from scratch (no EndoMamba)", "C5": "bidirectional (offline, not real-time)",
        "C6": "4 blocks (EndoMamba)", "C7": "8 blocks (EndoMamba)", "C8": "no speed augmentation"}


def add_lesion(d):
    d["frame"] = d["frame"].astype(int)
    d["near_lesion"] = 0
    for v, g in d.groupby("patient"):
        d.loc[g.index, "near_lesion"] = np.load(os.path.join(LESION, f"{v}.npy"))[g["frame"].to_numpy()]
    return d


def published(protocol):
    n = PROTOCOLS[protocol]
    return add_lesion(pd.concat([pd.read_json(PUBLISHED.format(p=protocol, k=k)) for k in range(1, n + 1)],
                                ignore_index=True))


def runs_of(root, protocol, folder, split):
    """{seed: pooled DataFrame} for the seeds that have the split predicted on every fold."""
    n, out = PROTOCOLS[protocol], {}
    base = os.path.join(root, protocol)
    seeds = set()
    for k in range(1, n + 1):
        d = os.path.join(base, f"fold{k}", folder)
        seeds |= {s for s in (os.listdir(d) if os.path.isdir(d) else []) if s.startswith("seed")}
    for s in sorted(seeds):
        paths = [os.path.join(base, f"fold{k}", folder, s, f"frame_predictions_{split}_val_mcc.json") for k in range(1, n + 1)]
        if all(os.path.exists(p) for p in paths):
            out[s] = pd.concat([pd.read_json(p) for p in paths], ignore_index=True)
    return out


def mean_std(rows, keys):
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join(OUTPUT_DIR, "mamba"))
    p.add_argument("--out", default=None)
    p.add_argument("--n_boot", type=int, default=2000)
    a = p.parse_args()
    out_dir = a.out or a.root
    result = {}
    for protocol in PROTOCOLS:
        pub = published(protocol)
        pub_mf, pub_hmm = summarize(pub, "PredictedClass"), summarize(pub, "PredictedClass_HMM")
        res = {"published_MF": pub_mf, "published_HMM": pub_hmm, "ColonTCN_reported": COLONTCN[protocol], "configs": {}}
        main_rows = [{"model": "Colon-Audit Multi-frame (online)", **pub_mf["F1_per_class"],
                      **{k: pub_mf[k] for k in ("wF1", "wJacc", "macroF1", "WMAPE_%", "error_near_lesion_%")}},
                     {"model": "Colon-Audit Multi-frame+HMM (offline)", **pub_hmm["F1_per_class"],
                      **{k: pub_hmm[k] for k in ("wF1", "wJacc", "macroF1", "WMAPE_%", "error_near_lesion_%")}},
                     {"model": "ColonTCN [reported] (offline)", **dict(zip(NAMES, COLONTCN[protocol]["F1"])),
                      "wF1": COLONTCN[protocol]["wF1"], "wJacc": COLONTCN[protocol]["wJacc"],
                      "WMAPE_%": COLONTCN[protocol]["WMAPE_%"]}]
        abl_rows = []
        c1_test = runs_of(a.root, protocol, CONFIGS["C1"], "test")
        for cid, folder in CONFIGS.items():
            test = runs_of(a.root, protocol, folder, "test")
            if not test:
                continue
            valid = runs_of(a.root, protocol, folder, "valid")
            per_run = [summarize(d, "PredictedClass") for d in test.values()]
            # HMM-decoded predictions of our runs are an extra reference (not in the paper); optional in the inputs
            per_run_hmm = [summarize(d, "PredictedClass_HMM") for d in test.values() if "PredictedClass_HMM" in d]
            ms = mean_std(per_run, ("wF1", "wJacc", "macroF1", "WMAPE_%", "error_near_lesion_%"))
            f1c = {c: float(np.mean([r["F1_per_class"][c] for r in per_run])) for c in NAMES}
            entry = {"description": DESC[cid], "runs": len(test), "test_raw": ms, "test_F1_per_class_mean": f1c,
                     "test_HMM_wF1": mean_std(per_run_hmm, ("wF1",))["wF1"] if per_run_hmm else None,
                     "valid_raw_wF1": mean_std([summarize(d, "PredictedClass") for d in valid.values()], ("wF1",))["wF1"]
                     if valid else None,
                     "vs_published_HMM": paired_bootstrap(list(test.values()), pub, "PredictedClass",
                                                          "PredictedClass_HMM", a.n_boot),
                     "vs_published_MF": paired_bootstrap(list(test.values()), pub, "PredictedClass",
                                                         "PredictedClass", a.n_boot)}
            common = sorted(set(test) & set(c1_test))
            if cid != "C1" and common:
                entry["vs_C1"] = paired_bootstrap([test[s] for s in common], [c1_test[s] for s in common],
                                                  "PredictedClass", "PredictedClass", a.n_boot)
            res["configs"][cid] = entry
            if cid == "C1":
                main_rows.append({"model": f"Colon-Audit-Mamba (online, causal) n={len(test)}", **f1c,
                                  **{k: ms[k][0] for k in ms}, "wF1_std": ms["wF1"][1], "wJacc_std": ms["wJacc"][1]})
            abl_rows.append({"config": cid, "description": DESC[cid], "runs": len(test),
                             "test_wF1": ms["wF1"][0], "test_wF1_std": ms["wF1"][1], "test_wJacc": ms["wJacc"][0],
                             "test_WMAPE": ms["WMAPE_%"][0], "test_err_near_lesion": ms["error_near_lesion_%"][0],
                             "valid_wF1": entry["valid_raw_wF1"][0] if entry["valid_raw_wF1"] else None,
                             "gain_vs_C1": entry.get("vs_C1", {}).get("gain"),
                             "gain_vs_C1_CI": (entry["vs_C1"]["ci_low"], entry["vs_C1"]["ci_high"]) if "vs_C1" in entry else None,
                             "gain_vs_ColonAudit_HMM": entry["vs_published_HMM"]["gain"],
                             "gain_vs_ColonAudit_HMM_CI": (entry["vs_published_HMM"]["ci_low"], entry["vs_published_HMM"]["ci_high"])})
        result[protocol] = res
        pd.DataFrame(main_rows).round(1).to_csv(os.path.join(out_dir, f"table_main_{protocol}.csv"), index=False)  # as in the paper
        abl = pd.DataFrame(abl_rows)
        abl.round(2).to_csv(os.path.join(out_dir, f"table_ablation_{protocol}.csv"), index=False)
        print(f"\n===== {protocol} =====")
        print(pd.DataFrame(main_rows)[["model", "wF1", "wJacc", "macroF1", "WMAPE_%", "error_near_lesion_%"]].round(2).to_string(index=False))
        if len(abl):
            print(abl.round(2).to_string(index=False))
    json.dump(result, open(os.path.join(out_dir, "paper_tables.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
