import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN
import test2_CNN

def set_seeds(seed: int) -> None:
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        if hasattr(tf, "set_random_seed"):
            tf.set_random_seed(seed)
        elif hasattr(tf.random, "set_seed"):
            tf.random.set_seed(seed)
    except Exception:
        pass


def load_sequences(seq_csv: str) -> Dict[str, str]:
    df = pd.read_csv(seq_csv)
    return dict(zip(df["uniprot_id"].astype(str).str.strip(), df["sequence"].astype(str)))


INVALID_TOKENS = {"", "NA", "NAN", "NONE", "NULL"}
COMPLEX_PREFIX = "COMPLEX:"

def _norm(s: str) -> str:
    return str(s).strip()

def _is_invalid_token(x: str) -> bool:
    return _norm(x).upper() in INVALID_TOKENS

def split_members(pid: str) -> List[str]:
    pid = _norm(pid)
    if pid.upper().startswith(COMPLEX_PREFIX):
        pid = pid.split(":", 1)[1]
    
    if "_" in pid:
        return [p.strip() for p in pid.split("_") if p.strip()]
    else:
        return [pid] if pid else []

def normalize_pid(pid: str, seq_map: Dict[str, str]) -> Optional[str]:
    """
    Convert complex with 1 valid member -> that member (regular protein).
    Keep complex with >=2 valid members as 'MEM1_MEM2_...' (no COMPLEX prefix).
    Return None if invalid/unmappable.

    Examples:
      COMPLEX:P43004          -> P43004
      COMPLEX:P43004_Q9UI32   -> P43004_Q9UI32
      O14786_Q9UIW2           -> O14786_Q9UIW2 (if both exist)
      NA_NA                   -> None
    """
    pid = _norm(pid)
    if _is_invalid_token(pid):
        return None

    # Regular protein (exists in seq_map)
    if pid in seq_map:
        return pid

    # Treat underscore or COMPLEX as complex
    if "_" in pid or pid.upper().startswith(COMPLEX_PREFIX):
        members = split_members(pid)

        # If any member token is invalid -> reject
        if any(_is_invalid_token(m) for m in members):
            return None

        # Keep only members that exist in seq_map
        # members = [m for m in members if m in seq_map]

        if len(members) == 0:
            return None
        if len(members) == 1:
            return members[0]  
        return "_".join(members)

    return None


def featurize_sequence_200d(seq: str, dim: int = 200, k: int = 3) -> np.ndarray:
    """
    Deterministic hashed k-mer counts -> 200D, L2 normalized.
    Replace with official CellEnBoost feature extractor if available.
    """
    v = np.zeros(dim, dtype=np.float32)
    if len(seq) < k:
        return v

    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        h = 0
        for ch in kmer:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        v[int(h % dim)] += 1.0

    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    return v


def get_feature(pid_norm: str, seq_map: Dict[str, str], cache: Dict[str, np.ndarray]) -> np.ndarray:
    """
    pid_norm is already normalized:
      - single protein: 'P12345'
      - complex: 'P12345_Q8N123'
    Complex feature = mean(member features.
    """
    pid_norm = _norm(pid_norm)
    if pid_norm in cache:
        return cache[pid_norm]

    if pid_norm in seq_map:
        vec = featurize_sequence_200d(seq_map[pid_norm])
        cache[pid_norm] = vec
        return vec

    # complex (underscore-joined, all members exist in seq_map by construction)
    members = pid_norm.split("_")
    vecs = [get_feature(m, seq_map, cache) for m in members]
    vec = np.mean(np.vstack(vecs), axis=0)
    cache[pid_norm] = vec
    return vec


# -----------------------------
# Build dataset matrices
# -----------------------------
def load_split(split_csv: str, seq_map: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    df = df.copy()

    df["source_norm"] = df["source"].apply(lambda x: normalize_pid(x, seq_map))
    df["target_norm"] = df["target"].apply(lambda x: normalize_pid(x, seq_map))

    mask = df["source_norm"].notna() & df["target_norm"].notna()
    dropped = int((~mask).sum())
    if dropped:
        print(f"[INFO] {os.path.basename(split_csv)}: dropped {dropped} rows (unmappable IDs like NA/invalid).")
        # bad_rows = df.loc[~mask, ["source", "target"]].head(20)
        # print("[INFO] Examples of dropped IDs:\n", bad_rows.to_string(index=False))
    return df.loc[mask].reset_index(drop=True)


def build_xy(
    df: pd.DataFrame,
    seq_map: Dict[str, str],
    cache: Dict[str, np.ndarray],
    pair_mode: str = "avg",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    pair_mode:
      - "avg"   -> X = (L + R)/2         (200D)  [default; safest for CNN]
      - "concat"-> X = [L || R]          (400D)
      - "diff"  -> X = (L - R)           (200D)
      - "hadam" -> X = (L * R)           (200D)
    """
    lig_ids = df["source_norm"].tolist()
    rec_ids = df["target_norm"].tolist()
    y = df["label"].to_numpy()

    L = np.vstack([get_feature(pid, seq_map, cache) for pid in lig_ids])
    R = np.vstack([get_feature(pid, seq_map, cache) for pid in rec_ids])

    if pair_mode == "avg":
        X = (L + R) / 2.0
    elif pair_mode == "diff":
        X = (L - R)
    elif pair_mode == "hadam":
        X = (L * R)
    elif pair_mode == "concat":
        X = np.hstack([L, R])
    else:
        raise ValueError("pair_mode must be one of: avg, diff, hadam, concat")

    return X.astype(np.float32), y.astype(int)


# -----------------------------
# Train / Predict (CellEnBoost ensemble)
# -----------------------------
def train_models(X_train: np.ndarray, y_train: np.ndarray, batch_size: int, cnn_n_features: int):
    # CNN branch
    X_train_r = test2_CNN.reshape_for_CNN(X_train)
    print("CNN reshaped X_train:", X_train_r.shape)
    cnn = Ada_CNN(
        base_estimator=test2_CNN.baseline_model(n_features=cnn_n_features),
        n_estimators=3,
        learning_rate=1,
        epochs=1,
    )
    cnn.fit(X_train_r, y_train, batch_size)

    # LightGBM branch
    print("LightGBM training on X_train:", X_train.shape)
    lgbm = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=200)
    lgbm.fit(X_train, y_train)

    return cnn, lgbm


def predict_scores(cnn, lgbm, X: np.ndarray) -> np.ndarray:
    X_r = test2_CNN.reshape_for_CNN(X)
    c = cnn.predict_proba(X_r)[:, 1]
    l = lgbm.predict_proba(X)[:, 1]
    return 0.4 * c + 0.6 * l


# -----------------------------
# Metrics + outputs
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Tuple[dict, np.ndarray]:
    y_pred = (y_score >= threshold).astype(int)

    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["AUC"] = float(roc_auc_score(y_true, y_score))
        out["AveragePrecision"] = float(average_precision_score(y_true, y_score))
    else:
        out["AUC"] = float("nan")
        out["AveragePrecision"] = float("nan")

    return out, y_pred


def save_predictions(df_test: pd.DataFrame, y_pred: np.ndarray, y_score: np.ndarray, out_dir: str, group: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Required file: ligand,receptor,true_label,pred_label (use normalized IDs as ligand/receptor keys)
    pd.DataFrame(
        {
            "ligand": df_test["source_norm"].astype(str),
            "receptor": df_test["target_norm"].astype(str),
            "true_label": df_test["label"].astype(int),
            "pred_label": y_pred.astype(int),
        }
    ).to_csv(os.path.join(out_dir, f"{group}_test_predictions.csv"), index=False)

    # Optional scores file
    pd.DataFrame(
        {
            "ligand": df_test["source_norm"].astype(str),
            "receptor": df_test["target_norm"].astype(str),
            "true_label": df_test["label"].astype(int),
            "pred_score": y_score.astype(float),
        }
    ).to_csv(os.path.join(out_dir, f"{group}_test_scores.csv"), index=False)


def run_group(group: str, data_dir: str, seq_map: Dict[str, str], out_dir: str, retrain: bool,
              threshold: float, batch_size: int, pair_mode: str) -> dict:
    train_csv = os.path.join(data_dir, f"{group}_train.csv")
    val_csv = os.path.join(data_dir, f"{group}_val.csv")
    test_csv = os.path.join(data_dir, f"{group}_test.csv")

    df_tr = load_split(train_csv, seq_map)
    df_va = load_split(val_csv, seq_map)
    df_te = load_split(test_csv, seq_map)

    cache: Dict[str, np.ndarray] = {}

    X_tr, y_tr = build_xy(df_tr, seq_map, cache, pair_mode=pair_mode)
    X_va, y_va = build_xy(df_va, seq_map, cache, pair_mode=pair_mode)
    X_te, y_te = build_xy(df_te, seq_map, cache, pair_mode=pair_mode)

    if not retrain:
        raise NotImplementedError("Pretrained loading is not implemented. Use --retrain.")

    # Train on train + val (simple and matches your plan)
    X_train = np.vstack([X_tr, X_va])
    y_train = np.hstack([y_tr, y_va])

    # CNN n_features should match X feature width expected by baseline_model.
    # Default pair_mode=avg => X has 200 dims => cnn_n_features=200
    cnn_n_features = X_train.shape[1]

    # Train + predict
    cnn, lgbm = train_models(X_train, y_train, batch_size=batch_size, cnn_n_features=cnn_n_features)
    y_score = predict_scores(cnn, lgbm, X_te)

    metrics, y_pred = compute_metrics(y_te, y_score, threshold)
    save_predictions(df_te, y_pred, y_score, out_dir, group)

    return {
        "Group": group,
        "PairMode": pair_mode,
        "NumTrain": int(len(df_tr) + len(df_va)),
        "NumTest": int(len(df_te)),
        "Threshold": float(threshold),
        **metrics,
        "PredFile": f"{group}_test_predictions.csv",
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to datasets/uniprot")
    ap.add_argument("--out_dir", default="cellenboost_results", help="Output folder")
    ap.add_argument("--retrain", action="store_true", help="Train on train+val then test")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for pred_label")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--batch_size", type=int, default=10, help="Batch size for CNN boosting")
    ap.add_argument(
        "--pair_mode",
        type=str,
        default="avg",
        choices=["avg", "diff", "hadam", "concat"],
        help="How to combine ligand/receptor vectors into pair features. Default=avg (CNN-friendly).",
    )
    args = ap.parse_args()

    set_seeds(args.seed)

    seq_csv = os.path.join(args.data_dir, "protein_sequences_info.csv")
    seq_map = load_sequences(seq_csv)

    # groups = ["SL", "SR", "SRcp", "SLRcp"]
    groups = ["SL"]  # for quick testing; change to all groups for full run
    results = []
    for g in groups:
        print('' + '=' * 40)
        print(f'Running {g}...')
        r = run_group(
            group=g,
            data_dir=args.data_dir,
            seq_map=seq_map,
            out_dir=args.out_dir,
            retrain=args.retrain,
            threshold=args.threshold,
            batch_size=args.batch_size,
            pair_mode=args.pair_mode,
        )
        print(g, r)
        results.append(r)

    print("\nSummary of results:")
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(args.out_dir, "metrics_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()