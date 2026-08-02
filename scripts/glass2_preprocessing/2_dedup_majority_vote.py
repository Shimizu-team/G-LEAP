#!/usr/bin/env python3
import argparse

import pandas as pd


def majority_label(group):
    counts = group["Label"].value_counts()
    label = counts.index[0]
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        label = 0  # tie -> negative
    row = group.iloc[0].copy()
    row["Label"] = label
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="binary_classification_dataset.csv")
    p.add_argument("--output", default="binary_classification_dataset_dedup_majority_vote.csv")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    print(f"input: {len(df)} rows")

    pair_counts = df.groupby(["smiles", "UniProt ID"]).size()
    dup = pair_counts[pair_counts > 1]
    conflicting = df.groupby(["smiles", "UniProt ID"])["Label"].nunique()
    print(f"duplicated pairs: {len(dup)}   label-conflicting pairs: {(conflicting > 1).sum()}")

    dedup = df.groupby(["smiles", "UniProt ID"], sort=False).apply(majority_label).reset_index(drop=True)
    dedup.to_csv(args.output, index=False)
    print(f"\nsaved {args.output}: {len(dedup)} rows "
          f"(removed {len(df) - len(dedup)}), "
          f"{dedup['UniProt ID'].nunique()} receptors, pos_rate={dedup['Label'].mean():.4f}")


if __name__ == "__main__":
    main()
