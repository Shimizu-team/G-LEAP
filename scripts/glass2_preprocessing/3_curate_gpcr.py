#!/usr/bin/env python3
import argparse
import json

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="binary_classification_dataset_dedup_majority_vote.csv")
    p.add_argument("--allowlist", default="curation/uniprots_790.json")
    p.add_argument("--output", default="binary_classification_dataset_cleaned_v2.csv")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    keep = set(json.load(open(args.allowlist)))
    print(f"input: {len(df)} rows, {df['UniProt ID'].nunique()} receptors")
    print(f"allow-list: {len(keep)} receptors")

    out = df[df["UniProt ID"].isin(keep)].copy()
    out.to_csv(args.output, index=False)
    print(f"\nsaved {args.output}: {len(out)} rows, "
          f"{out['UniProt ID'].nunique()} receptors, pos_rate={out['Label'].mean():.4f}")


if __name__ == "__main__":
    main()
