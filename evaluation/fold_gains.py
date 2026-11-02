"""Per-fold gain of the causal model (C1, runs averaged) over the online Colon-Audit (published predictions), on the
test videos of each fold of both REAL-Colon protocols. Descriptive consistency across folds, no confidence interval;
a cross-center fold holds out one REAL-Colon study (001-004).

Output: OUTPUT_DIR/fold_gains/fold_gains.json and fold_gains.csv
Run:  python evaluation/fold_gains.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_tables import runs_of, published, PROTOCOLS  # noqa: E402
from metrics import wmape  # noqa: E402
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import OUTPUT_DIR, SPLIT_CSV  # noqa: E402

M = OUTPUT_DIR
OUT = os.path.join(M, "fold_gains")
ROOT, FOLDER = os.path.join(M, "mamba"), "mamba12-endomamba"
GRID_CSV = SPLIT_CSV


def wf1(df):
    return f1_score(df["anatomy"], df["PredictedClass"], average="weighted") * 100


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for protocol, n in PROTOCOLS.items():
        folds = [f"fold{k}" for k in range(1, n + 1)]
        split = pd.read_csv(GRID_CSV.format(protocol=protocol), usecols=["patient"] + folds)
        split = split.drop_duplicates("patient").set_index("patient")
        runs, pub = runs_of(ROOT, protocol, FOLDER, "test"), published(protocol)
        for fold in folds:
            videos = split.index[split[fold] == "test"]
            ours = [d[d.patient.isin(videos)] for d in runs.values()]
            ref = pub[pub.patient.isin(videos)]
            rows.append({"protocol": protocol, "fold": fold, "videos": len(videos),
                         "studies": ",".join(sorted({v[:3] for v in videos})),
                         "ours_wF1": float(np.mean([wf1(d) for d in ours])), "colon_audit_wF1": wf1(ref),
                         "ours_WMAPE": float(np.mean([wmape(d, "PredictedClass") for d in ours])),
                         "colon_audit_WMAPE": wmape(ref, "PredictedClass")})
            rows[-1]["gain_wF1"] = rows[-1]["ours_wF1"] - rows[-1]["colon_audit_wF1"]
    table = pd.DataFrame(rows)
    table.round(2).to_csv(os.path.join(OUT, "fold_gains.csv"), index=False)
    json.dump(rows, open(os.path.join(OUT, "fold_gains.json"), "w"), indent=2)
    print(table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
