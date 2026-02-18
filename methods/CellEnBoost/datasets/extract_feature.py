#!/usr/bin/env python3
import argparse
import os
import glob
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# --------------------------
# Utils
# --------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {a: i for i, a in enumerate(AA)}
AA_SET = set(AA)

INVALID_TOKENS = {"", "NA", "NAN", "NONE", "NULL"}
COMPLEX_PREFIX = "COMPLEX:"


def norm(s: str) -> str:
    return str(s).strip()


def is_invalid_token(x: str) -> bool:
    return norm(x).upper() in INVALID_TOKENS


def split_members(pid: str) -> List[str]:
    pid = norm(pid)
    if pid.upper().startswith(COMPLEX_PREFIX):
        pid = pid.split(":", 1)[1]
    if "_" in pid:
        return [p.strip() for p in pid.split("_") if p.strip()]
    return [pid] if pid else []


def normalize_pid(pid: str, seq_map: Dict[str, str]) -> str:
    """
    Normalize:
      - COMPLEX with 1 valid member -> that member
      - COMPLEX with >=2 valid members -> keep joined by underscore
      - If members include NA/invalid -> return None
      - If member doesn't exist in seq_map -> drop it; if nothing remains -> None
    """
    pid = norm(pid)
    if is_invalid_token(pid):
        return None

    # direct hit
    if pid in seq_map:
        return pid

    if "_" in pid or pid.upper().startswith(COMPLEX_PREFIX):
        members = split_members(pid)
        if any(is_invalid_token(m) for m in members):
            return None

        # keep only known proteins
        members = [m for m in members if m in seq_map]

        if len(members) == 0:
            return None
        if len(members) == 1:
            return members[0]
        return "_".join(members)

    return None


def clean_seq(seq: str) -> str:
    seq = (seq or "").strip().upper()
    return "".join([c for c in seq if c in AA_SET])


def load_sequences(protein_csv: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(protein_csv)
    if not {"uniprot_id", "sequence"}.issubset(df.columns):
        raise ValueError("protein_sequences_info.csv must have columns: uniprot_id, sequence")
    df = df.copy()
    df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip()
    df["sequence"] = df["sequence"].astype(str)
    seq_map = dict(zip(df["uniprot_id"], df["sequence"]))
    return df, seq_map


def write_fasta(df_prot: pd.DataFrame, fasta_path: str) -> List[str]:
    """
    Write FASTA in the same order as df_prot rows. Returns list of ids in order.
    """
    ids = df_prot["uniprot_id"].astype(str).str.strip().tolist()
    seqs = df_prot["sequence"].astype(str).tolist()

    with open(fasta_path, "w") as f:
        for pid, seq in zip(ids, seqs):
            f.write(f">{pid}\n{seq.strip()}\n")
    return ids


# --------------------------
# NV (Natural Vector) 60D
# --------------------------
def natural_vector_60(seq: str) -> np.ndarray:
    """
    For each AA:
      count/L, mean_pos/L, second_moment/(L^2)
    20 AAs * 3 = 60 dims
    """
    seq = clean_seq(seq)
    L = len(seq)
    v = np.zeros(60, dtype=np.float32)
    if L == 0:
        return v

    pos = [[] for _ in range(20)]
    for idx, ch in enumerate(seq, start=1):  # 1-based
        pos[AA2I[ch]].append(idx)

    out = []
    for a in range(20):
        p = np.array(pos[a], dtype=np.float32)
        c = float(len(p))
        if c == 0:
            out.extend([0.0, 0.0, 0.0])
            continue
        mu = float(p.mean())
        m2 = float(((p - mu) ** 2).mean())
        out.extend([c / L, mu / L, m2 / (L * L)])

    v[:] = np.array(out, dtype=np.float32)
    return v


def build_nv_table(df_prot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, seq in zip(df_prot["uniprot_id"], df_prot["sequence"]):
        rows.append(natural_vector_60(seq))
    X = np.vstack(rows)
    cols = [f"NV_{i:03d}" for i in range(X.shape[1])]
    out = pd.DataFrame(X, columns=cols)
    out.insert(0, "uniprot_id", df_prot["uniprot_id"].values)
    return out


# --------------------------
# iFeature CTD (273D)
# --------------------------
def run_ifeature_ctd(ifeature_py: str, fasta_path: str, out_dir: str) -> Tuple[str, str, str]:
    """
    Runs CTDC/CTDT/CTDD and returns paths to the outputs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_ctdc = str(out_dir / "CTDC.csv")
    out_ctdt = str(out_dir / "CTDT.csv")
    out_ctdd = str(out_dir / "CTDD.csv")

    cmds = [
        ["python", ifeature_py, "--file", fasta_path, "--type", "CTDC", "--out", out_ctdc],
        ["python", ifeature_py, "--file", fasta_path, "--type", "CTDT", "--out", out_ctdt],
        ["python", ifeature_py, "--file", fasta_path, "--type", "CTDD", "--out", out_ctdd],
    ]
    for c in cmds:
        subprocess.run(c, check=True)
    return out_ctdc, out_ctdt, out_ctdd


def _read_ifeature_table(csv_path: str) -> pd.DataFrame:
    """
    iFeature outputs usually have first column as sequence name/id.
    It may be labeled '#', 'name', or no header. We'll handle robustly.
    """
    df = pd.read_csv(csv_path)
    # If first column isn't 'uniprot_id', rename it
    first = df.columns[0]
    df = df.rename(columns={first: "uniprot_id"})
    df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip()
    return df


def merge_ctd(ctdc: str, ctdt: str, ctdd: str) -> pd.DataFrame:
    a = _read_ifeature_table(ctdc)
    b = _read_ifeature_table(ctdt)
    c = _read_ifeature_table(ctdd)

    df = a.merge(b, on="uniprot_id", how="inner").merge(c, on="uniprot_id", how="inner")

    # sanity: should be 273 feature columns total (excluding id)
    feat_cols = [x for x in df.columns if x != "uniprot_id"]
    if len(feat_cols) != 273:
        print(f"[WARN] CTD feature columns != 273 (got {len(feat_cols)}). "
              f"Still proceeding, but check iFeature outputs.")
    return df


# --------------------------
# PyFeat pseudoKNC + monoMonoKGap
# --------------------------
def write_pyfeat_inputs(df_prot: pd.DataFrame, out_dir: str) -> Tuple[str, str, List[str]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta = out_dir / "FASTA.txt"
    labels = out_dir / "Labels.txt"

    ids = df_prot["uniprot_id"].astype(str).str.strip().tolist()
    seqs = df_prot["sequence"].astype(str).tolist()

    with fasta.open("w") as f:
        for pid, seq in zip(ids, seqs):
            f.write(f">{pid}\n{seq.strip()}\n")

    with labels.open("w") as f:
        for _ in ids:
            f.write("0\n")

    return str(fasta), str(labels), ids


def run_pyfeat(pyfeat_main: str, fasta_txt: str, labels_txt: str, out_dir: str,
              ktuple: int = 3, kgap: int = 1) -> str:
    """
    Runs PyFeat to generate pseudoKNC and monoMonoKGap.
    Returns the produced CSV path (auto-detected as newest CSV in out_dir).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PyFeat writes output into its working directory in many versions.
    # So we run it with cwd=out_dir to keep outputs contained.
    cmd = [
        "python", pyfeat_main,
        "--sequenceType=Protein",
        "--testDataset=1",
        f"--fasta={fasta_txt}",
        f"--label={labels_txt}",
        f"--kTuple={ktuple}",
        f"--kGap={kgap}",
        "--pseudoKNC=1",
        "--monoMono=1",
    ]
    subprocess.run(cmd, check=True, cwd=str(out_dir))

    # auto-detect output CSV (newest .csv)
    csvs = sorted(glob.glob(str(out_dir / "*.csv")), key=os.path.getmtime, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV produced by PyFeat in {out_dir}")
    return csvs[0]


def read_pyfeat_features(pyfeat_csv: str, ids_in_order: List[str]) -> pd.DataFrame:
    """
    PyFeat output often has no header and includes label as last column.
    We'll:
      - read as numeric
      - drop last column (dummy label)
      - attach uniprot_id by the original FASTA order
    """
    df = pd.read_csv(pyfeat_csv, header=None)
    X = df.iloc[:, :-1]  # drop dummy label
    if len(X) != len(ids_in_order):
        raise ValueError(f"PyFeat rows {len(X)} != number of proteins {len(ids_in_order)}. "
                         f"Check FASTA order / output file.")
    out = X.copy()
    out.insert(0, "uniprot_id", ids_in_order)
    return out


# --------------------------
# Merge to 9153D
# --------------------------
def merge_all(nv_df: pd.DataFrame, ctd_df: pd.DataFrame, pyfeat_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure IDs are stripped
    for d in (nv_df, ctd_df, pyfeat_df):
        d["uniprot_id"] = d["uniprot_id"].astype(str).str.strip()

    m = nv_df.merge(ctd_df, on="uniprot_id", how="inner").merge(pyfeat_df, on="uniprot_id", how="inner")

    # expected dims: 60 + 273 + (8420+400)=9153 excluding id
    feat_cols = [c for c in m.columns if c != "uniprot_id"]
    if len(feat_cols) != 9153:
        print(f"[WARN] Total feature dims != 9153 (got {len(feat_cols)}). "
              f"Proceeding, but check PyFeat/iFeature settings or outputs.")
    return m


# --------------------------
# PCA to 200 per group (fit on train+val proteins only)
# --------------------------
def collect_train_val_proteins(data_dir: str, group: str, seq_map: Dict[str, str]) -> List[str]:
    def load_split(path):
        df = pd.read_csv(path)
        df["source_norm"] = df["source"].apply(lambda x: normalize_pid(x, seq_map))
        df["target_norm"] = df["target"].apply(lambda x: normalize_pid(x, seq_map))
        df = df[df["source_norm"].notna() & df["target_norm"].notna()]
        return df

    df_tr = load_split(os.path.join(data_dir, f"{group}_train.csv"))
    df_va = load_split(os.path.join(data_dir, f"{group}_val.csv"))

    ids = set(df_tr["source_norm"]) | set(df_tr["target_norm"]) | set(df_va["source_norm"]) | set(df_va["target_norm"])
    expanded = set()
    for pid in ids:
        for m in pid.split("_"):
            expanded.add(m)
    # keep only ones we have sequences for
    expanded = [i for i in expanded if i in seq_map]
    return sorted(expanded)


def fit_pca_and_save(features_9153: pd.DataFrame, train_ids: List[str],
                     out_path: str, n_components: int = 200, seed: int = 0):
    feats = features_9153.set_index("uniprot_id")
    X_all = feats.values.astype(np.float32)

    # fit on train proteins only (no leakage)
    train_ids = [i for i in train_ids if i in feats.index]
    X_train = feats.loc[train_ids].values.astype(np.float32)

    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_train)
    Z = pca.transform(X_all).astype(np.float32)

    out = pd.DataFrame(Z, index=feats.index, columns=[f"F_{i:03d}" for i in range(n_components)])
    out.insert(0, "uniprot_id", out.index)
    out.reset_index(drop=True, inplace=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein_csv", required=True, help="protein_sequences_info.csv path")
    ap.add_argument("--data_dir", required=True, help="datasets/uniprot path (splits live here)")
    ap.add_argument("--ifeature_py", required=True, help="path to iFeature.py")
    ap.add_argument("--pyfeat_main", required=True, help="path to PyFeat main.py")
    ap.add_argument("--work_dir", default="feature_work", help="temp work directory")
    ap.add_argument("--out_dir", default="features_out", help="output directory for features")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    work = Path(args.work_dir)
    outd = Path(args.out_dir)
    work.mkdir(parents=True, exist_ok=True)
    outd.mkdir(parents=True, exist_ok=True)

    df_prot, seq_map = load_sequences(args.protein_csv)

    # FASTA for iFeature (and also PyFeat, but PyFeat expects FASTA.txt)
    fasta_path = str(work / "proteins.fasta")
    ids_in_order = write_fasta(df_prot, fasta_path)

    # NV
    print("[STEP] Computing Natural Vector (60D)")
    nv_df = build_nv_table(df_prot)

    # iFeature CTD
    print("[STEP] Running iFeature CTD (CTDC/CTDT/CTDD)")
    ctd_dir = work / "ifeature"
    ctdc, ctdt, ctdd = run_ifeature_ctd(args.ifeature_py, fasta_path, str(ctd_dir))
    ctd_df = merge_ctd(ctdc, ctdt, ctdd)

    # PyFeat pseudoKNC + monoMonoKGap
    print("[STEP] Running PyFeat pseudoKNC + monoMonoKGap")
    pyfeat_dir = work / "pyfeat"
    fasta_txt, labels_txt, ids_for_pyfeat = write_pyfeat_inputs(df_prot, str(pyfeat_dir))
    pyfeat_csv = run_pyfeat(args.pyfeat_main, fasta_txt, labels_txt, str(pyfeat_dir), ktuple=3, kgap=1)
    pyfeat_df = read_pyfeat_features(pyfeat_csv, ids_for_pyfeat)

    # Merge all
    print("[STEP] Merging NV + CTD + PyFeat -> 9153D")
    feat9153 = merge_all(nv_df, ctd_df, pyfeat_df)
    feat9153_path = outd / "protein_features_9153.parquet"
    feat9153.to_parquet(feat9153_path, index=False)
    print(f"[OK] Saved 9153D features to {feat9153_path}")

    # PCA per group (fit on train+val proteins)
    groups = ["SL", "SR", "SRcp", "SLRcp"]
    for g in groups:
        print(f"[STEP] PCA->200 for group {g} (fit on {g} train+val proteins)")
        train_ids = collect_train_val_proteins(args.data_dir, g, seq_map)
        out_path = outd / f"protein_features_200_{g}.parquet"
        fit_pca_and_save(feat9153, train_ids, str(out_path), n_components=200, seed=args.seed)
        print(f"[OK] Saved 200D features to {out_path}")


if __name__ == "__main__":
    main()