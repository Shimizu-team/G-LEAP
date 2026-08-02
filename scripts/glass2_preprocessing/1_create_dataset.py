#!/usr/bin/env python3
import argparse

import pandas as pd

VALUE_TYPE_MAP = {"IC50": "pIC50", "Ki": "pKi", "EC50": "pEC50", "Kd": "pKd"}
ACTIVE_NM = 10      # <= 10 nM  -> active   (pAff >= 8)
INACTIVE_NM = 1000  # >= 1000 nM -> inactive (pAff <= 6)


def load_inchikey_to_smiles(ligands_tsv):
    lig = pd.read_csv(ligands_tsv, sep="\t")
    lig = lig.dropna(subset=["InChIKey", "SMILES"])
    lig = lig[(lig["InChIKey"] != "") & (lig["SMILES"] != "")]
    mapping = dict(zip(lig["InChIKey"], lig["SMILES"]))
    print(f"InChIKey->SMILES mappings: {len(mapping)}")
    return mapping


def build(args):
    frames = []
    for path in (args.act, args.inact):
        if path:
            df = pd.read_csv(path)
            print(f"loaded {path}: {len(df)} rows")
            frames.append(df)
    if not frames:
        raise SystemExit("no input tables given")
    df_all = pd.concat(frames, ignore_index=True)
    print(f"combined: {len(df_all)} rows")

    df_all["standard_value_numeric"] = pd.to_numeric(df_all["standard_value"], errors="coerce")
    df = df_all.dropna(subset=["standard_value_numeric", "target_uniprot_id", "compound_inchikey"])
    print(f"after dropping NaN value/target/compound: {len(df)} rows")

    pos = df[df["standard_value_numeric"] <= ACTIVE_NM].copy()
    neg = df[df["standard_value_numeric"] >= INACTIVE_NM].copy()
    pos["Label"] = 1
    neg["Label"] = 0
    print(f"active (<= {ACTIVE_NM} nM): {len(pos)}   inactive (>= {INACTIVE_NM} nM): {len(neg)}")

    out = pd.concat([pos, neg], ignore_index=True)
    out["Value Type"] = out["standard_type"].map(VALUE_TYPE_MAP)

    mapping = load_inchikey_to_smiles(args.ligands)
    out["smiles"] = out["compound_inchikey"].map(mapping)
    mapped = out["smiles"].notna().sum()
    print(f"SMILES mapping rate: {mapped}/{len(out)} ({mapped / len(out) * 100:.1f}%)")
    out = out.dropna(subset=["smiles"])

    final = out[["target_uniprot_id", "smiles", "Label", "Value Type"]].rename(
        columns={"target_uniprot_id": "UniProt ID"})
    final.to_csv(args.output, index=False)
    print(f"\nsaved {args.output}: {len(final)} rows, "
          f"{final['UniProt ID'].nunique()} receptors, "
          f"pos_rate={final['Label'].mean():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--act", required=True, help="glass2_reg_act.csv")
    p.add_argument("--inact", default=None,
                   help="glass2_reg_inact.csv (optional; needed for the exact published counts)")
    p.add_argument("--ligands", required=True, help="ligand table (tsv) with InChIKey + SMILES")
    p.add_argument("--output", default="binary_classification_dataset.csv")
    build(p.parse_args())


if __name__ == "__main__":
    main()
