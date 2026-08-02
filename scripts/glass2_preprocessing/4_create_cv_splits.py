#!/usr/bin/env python3
import argparse
import os

import pandas as pd

N_FOLDS = 10
KEEP_COLS = ["UniProt ID", "smiles", "Label", "Value Type"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="binary_classification_dataset_cleaned_v2.csv")
    p.add_argument("--fold_map", default="../../data/benchmark_receptor_folds.csv",
                   help="UniProt ID -> protein_fold assignment (the released paper split)")
    p.add_argument("--out_dir", default="cv_splits")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    fold_map = pd.read_csv(args.fold_map).set_index("UniProt ID")["protein_fold"].to_dict()
    print(f"input: {len(df)} rows, {df['UniProt ID'].nunique()} receptors, {df['smiles'].nunique()} compounds")

    df = df.copy()
    df["protein_fold"] = df["UniProt ID"].map(fold_map)
    unmapped = df["protein_fold"].isna().sum()
    if unmapped:
        missing = sorted(df.loc[df["protein_fold"].isna(), "UniProt ID"].unique())
        raise SystemExit(f"{unmapped} rows ({len(missing)} receptors) not in fold map, "
                         f"e.g. {missing[:5]} -- dataset and fold map disagree")
    df["protein_fold"] = df["protein_fold"].astype(int)

    keep = [c for c in KEEP_COLS if c in df.columns]
    info = []
    for fold in range(N_FOLDS):
        te = df[df["protein_fold"] == fold]
        tr = df[df["protein_fold"] != fold]
        assert not tr["UniProt ID"].isin(set(te["UniProt ID"])).any(), \
            f"protein fold {fold}: receptor leak train<->test"
        tr[keep].to_csv(f"{args.out_dir}/protein_fold_{fold}_train.csv", index=False)
        te[keep].to_csv(f"{args.out_dir}/protein_fold_{fold}_test.csv", index=False)
        info.append({"fold": fold, "train_size": len(tr), "test_size": len(te),
                     "test_receptors": int(te["UniProt ID"].nunique()),
                     "test_pos": int((te["Label"] == 1).sum())})
        print(f"  protein fold {fold}: train {len(tr):>7d}  test {len(te):>6d}  "
              f"({te['UniProt ID'].nunique()} receptors)")

    pd.DataFrame(info).to_csv(f"{args.out_dir}/protein_split_stats.csv", index=False)
    assert sum(r["test_size"] for r in info) == len(df), "test folds do not partition the dataset"
    print(f"\nprotein split saved to {args.out_dir}/ (exact partition OK)")


if __name__ == "__main__":
    main()
