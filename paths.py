"""Local paths used by every script. Edit the defaults below once, or set environment variables with the same
names. Relative paths are resolved from the repository root."""
import os

REPO = os.path.dirname(os.path.abspath(__file__))


def _path(name, default):
    return os.path.join(REPO, os.environ.get(name, default))


# REAL-Colon (Biffi et al., 2024): per-video label CSVs (Labels/<video>.csv)
REALCOLON_DIR = _path("REALCOLON_DIR", "data/REAL-Colon")
REALCOLON_LABELS = os.path.join(REALCOLON_DIR, "Labels")
# official 5 fps splits: set_type_5fold_5fps.csv and set_type_4fold_5fps.csv
SPLITS_DIR = _path("SPLITS_DIR", "data/REAL-Colon/splits")
SPLIT_CSV = os.path.join(SPLITS_DIR, "set_type_{protocol}_5fps.csv")

# CAS-Colon (Song et al., 2025), as downloaded: Video/<n>.mp4 and Label/frame/Frame_Label.csv
CASCOLON_DIR = _path("CASCOLON_DIR", "data/CAS-Colon-raw")

# derived data written by datasets/: lesion_targets/<video>.npy, CAS-Colon/{frames_5fps/, labels_5fps.csv}
DATA_DIR = _path("DATA_DIR", "data")
LESION_DIR = os.path.join(DATA_DIR, "lesion_targets")
CAS_LABELS = os.path.join(DATA_DIR, "CAS-Colon", "labels_5fps.csv")

# frozen GastroNet-5M ViT-Small embeddings: <variant>/<video>/{features.npy, frame_numbers.npy}
# (variant "original" for REAL-Colon; CAS-Colon/original/CAS-<nn>/ for CAS-Colon)
FEATURES_DIR = _path("FEATURES_DIR", "features/GastroNet-5M/ViT-Small")

# third-party weights
ENDOMAMBA_REPO = _path("ENDOMAMBA_REPO", "third_party/EndoMamba")
ENDOMAMBA_CKPT = _path("ENDOMAMBA_CKPT", "weights/endomamba_small_b48_seqlen16_withteacher_MIX12/checkpoint-499.pth")
GASTRONET_CKPT = _path("GASTRONET_CKPT", "weights/VITS_GastroNet-5M_DINOv1.pth")

# Colon-Audit test predictions (released with Colon-Audit): <protocol>/fold<k>/<file>
COLON_AUDIT_DIR = _path("COLON_AUDIT_DIR", "data/colon_audit_predictions")
COLON_AUDIT_PREDICTIONS = os.path.join(COLON_AUDIT_DIR, "{p}", "fold{k}",
                                       "frame_predictions_soft_vote_val_mcc_hmm.json")

# trained runs, tables and figures: <OUTPUT_DIR>/mamba/<protocol>/fold<k>/<variant>/seed<s>/
OUTPUT_DIR = _path("OUTPUT_DIR", "outputs")
