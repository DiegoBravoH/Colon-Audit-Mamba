"""CAS-Colon (Song et al., Sci Data 2025) as an external test set at 5 fps, preprocessed like REAL-Colon.

  extract : every video (60 fps h264) -> frames at t = k / 5 s, sampled by TIMESTAMP (ffmpeg fps=5, so the
            gaps of video 75 cannot shift the labels), full frame (already cropped by the authors) resized to
            224x224 without keeping the aspect ratio (as REAL-Colon), JPEG q=2.
            -> data/CAS-Colon/frames_5fps/<video>/<video>_<k>.jpg   (k starts at 0)
  labels  : -> data/CAS-Colon/labels_5fps.csv  (video, k, t, second, label_raw, cas_label, label, evaluate)
            label = REAL-Colon code: ileum 3, cecum 2, ascending 4, transverse 5 (+ hepatic and splenic flexure),
            descending 6, sigmoid 7, rectum 8 (+ anal canal)  [same merge as the MICCAI 2026 OColT paper].
            evaluate = 0 for: seconds with two labels (overlapping segment boundaries, 4 s), seconds without a
            label, and samples whose nearest real video frame is > 50 ms away (recording gaps, video 75).
            Those frames stay in the sequence (temporal continuity) but are excluded from every metric.

Run:  python datasets/cas_colon.py extract --workers 8      then      python datasets/cas_colon.py labels
"""
import os
import sys
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import CASCOLON_DIR, DATA_DIR  # noqa: E402

SRC = CASCOLON_DIR
OUT = os.path.join(DATA_DIR, "CAS-Colon")
FPS = 5
TO_REAL = {"terminalIleum": 3, "cecum": 2, "ascendingColon": 4, "hepaticFlexure": 5, "transverseColon": 5,
           "splenicFlexure": 5, "descendingColon": 6, "sigmoidColon": 7, "rectum": 8, "analCanal": 8}


def videos():
    return sorted(int(f[:-4]) for f in os.listdir(os.path.join(SRC, "Video")) if f.endswith(".mp4"))


def extract_one(v):
    out = os.path.join(OUT, "frames_5fps", str(v))
    done = os.path.join(out, ".done")
    if os.path.exists(done):
        return v, "skipped"
    os.makedirs(out, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", os.path.join(SRC, "Video", f"{v}.mp4"),
           "-vf", f"fps={FPS},scale=224:224:flags=lanczos", "-q:v", "2", "-start_number", "0",
           os.path.join(out, f"{v}_%d.jpg")]
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        open(done, "w").close()
    return v, "ok" if rc == 0 else f"ffmpeg rc={rc}"


def frame_times(v):
    """Presentation timestamps (s) of every real frame of the source video."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                        "-of", "csv=p=0", os.path.join(SRC, "Video", f"{v}.mp4")], capture_output=True, text=True)
    return np.sort(np.array([float(x) for x in r.stdout.split()]))


def build_labels():
    lab = pd.read_csv(os.path.join(SRC, "Label", "frame", "Frame_Label.csv"))
    lab["second"] = lab["Filename"].str.extract(r"_(\d+)s\.jpg$")[0].astype(int)
    counts = lab.groupby(["VideoID", "second"]).size()
    rows = []
    for v in videos():
        n = len([f for f in os.listdir(os.path.join(OUT, "frames_5fps", str(v))) if f.endswith(".jpg")])
        k = np.arange(n)
        t = k / FPS
        sec = np.floor(t).astype(int)
        real_t = frame_times(v)
        nearest = np.abs(real_t[np.clip(np.searchsorted(real_t, t), 0, len(real_t) - 1)] - t)
        nearest = np.minimum(nearest, np.abs(real_t[np.clip(np.searchsorted(real_t, t) - 1, 0, len(real_t) - 1)] - t))
        g = lab[lab.VideoID == v].drop_duplicates("second", keep=False).set_index("second")["Label_raw"]
        raw = pd.Series(sec).map(g)
        n_lab = pd.Series(sec).map(counts.loc[v] if v in counts.index.get_level_values(0) else {}).fillna(0).to_numpy()
        evaluate = (n_lab == 1) & (nearest <= 0.05)
        rows.append(pd.DataFrame({"video": v, "k": k, "t": t, "second": sec, "label_raw": raw.fillna(""),
                                  "label": raw.map(TO_REAL).fillna(-100).astype(int), "evaluate": evaluate.astype(int),
                                  "reason_excluded": np.where(n_lab == 0, "no label", np.where(n_lab > 1, "two labels",
                                                     np.where(nearest > 0.05, "recording gap", "")))}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(OUT, "labels_5fps.csv"), index=False)
    print(f"{df.video.nunique()} videos, {len(df)} frames at {FPS} fps, evaluated {df.evaluate.sum()}")
    print("excluded:", df[df.evaluate == 0].reason_excluded.value_counts().to_dict())
    print("label distribution (evaluated):", df[df.evaluate == 1].label_raw.value_counts().to_dict())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["extract", "labels"])
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.step == "extract":
        with ThreadPoolExecutor(a.workers) as ex:
            for v, status in ex.map(extract_one, videos()):
                print(v, status, flush=True)
    else:
        build_labels()


if __name__ == "__main__":
    main()
