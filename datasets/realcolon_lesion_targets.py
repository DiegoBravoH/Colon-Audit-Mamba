"""Per-frame lesion targets for the auxiliary lesion head (read-only use of the Labels CSVs).

near_lesion[f] = 1 if a lesion box is visible within +-30 s of native frame f (native fps = 5 * stride),
i.e. the whole inspection / resection manoeuvre, not only the frames where the polyp is on screen.
Output: data/lesion_targets/<video>.npy  uint8 (N_native_frames,), row f = frame f.

Run:  python datasets/realcolon_lesion_targets.py
"""
import sys
import os

import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import SPLIT_CSV, REALCOLON_LABELS, LESION_DIR  # noqa: E402

LABELS_DIR = REALCOLON_LABELS
GRID_5FPS_CSV = SPLIT_CSV.format(protocol="5fold")
OUT = LESION_DIR
NEAR_S = 30

grid = pd.read_csv(GRID_5FPS_CSV, usecols=["patient", "frame_number"])
strides = {v: int(np.bincount(np.diff(np.sort(g.frame_number.to_numpy()))).argmax()) for v, g in grid.groupby("patient")}
total = pos = 0
for f in sorted(os.listdir(LABELS_DIR)):
    if not f.endswith(".csv"):
        continue
    video = f[:-4]
    lesion = pd.read_csv(os.path.join(LABELS_DIR, f))["lesion"].to_numpy().astype(np.int64)
    half = NEAR_S * 5 * strides[video]
    cs = np.concatenate([[0], np.cumsum(lesion)])
    i = np.arange(len(lesion))
    near = ((cs[np.minimum(i + half + 1, len(lesion))] - cs[np.maximum(i - half, 0)]) > 0).astype(np.uint8)
    np.save(os.path.join(OUT, f"{video}.npy"), near)
    total += len(near); pos += int(near.sum())
print(f"60 videos, {total} frames, near_lesion positive {pos / total:.1%}")
