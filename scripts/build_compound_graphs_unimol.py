#!/usr/bin/env python3
"""
Build compound graph dicts using RDKit features + Uni-Mol atomic embeddings.

Output:
- graph dict (.npy): {smiles: (num_nodes, features, edge_index)}
"""

import argparse

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from tqdm import tqdm
from unimol_tools import UniMolRepr


ATOM_LIST = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As",
    "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti",
    "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt",
    "Hg", "Pb", "Unknown",
]
DEGREE_LIST = list(range(0, 11))
H_COUNT_LIST = list(range(0, 11)) + ["other"]
VALENCE_LIST = list(range(0, 11)) + ["other"]
FORMAL_CHARGE_LIST = [-3, -2, -1, 0, +1, +2, +3, "other"]
HYBRID_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    "other",
]
CHIRAL_LIST = ["none", "R", "S"]
RING_SIZE_LIST = list(range(0, 9)) + [">8"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build compound graphs with RDKit + Uni-Mol.")
    parser.add_argument("--csv_file", required=True, help="CSV containing SMILES.")
    parser.add_argument("--smiles_column", default="smiles", help="Column name for SMILES.")
    parser.add_argument("--output_graph", required=True, help="Output .npy graph dict path.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for Uni-Mol.")
    return parser.parse_args()


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"{x} not in allowable set {allowable_set}")
    return [int(x == s) for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def atom_features_with_hydrogen(atom):
    feats = one_of_k_encoding_unk(atom.GetSymbol(), ATOM_LIST)
    feats += one_of_k_encoding(atom.GetDegree(), DEGREE_LIST)
    feats += one_of_k_encoding_unk(atom.GetTotalNumHs(), H_COUNT_LIST)
    feats += one_of_k_encoding_unk(atom.GetImplicitValence(), VALENCE_LIST)
    feats.append(int(atom.GetIsAromatic()))

    g_charge = 0.0
    if atom.HasProp("_GasteigerCharge"):
        charge_val = atom.GetProp("_GasteigerCharge")
        if isinstance(charge_val, (int, float)) and np.isfinite(charge_val):
            g_charge = float(charge_val)
    feats.append(g_charge)

    feats += one_of_k_encoding_unk(atom.GetFormalCharge(), FORMAL_CHARGE_LIST)
    feats += one_of_k_encoding_unk(atom.GetHybridization(), HYBRID_LIST)

    if atom.HasProp("_CIPCode"):
        chir_tag = "R" if atom.GetProp("_CIPCode") == "R" else "S"
    else:
        chir_tag = "none"
    feats += one_of_k_encoding(chir_tag, CHIRAL_LIST)

    ring_flag = atom.IsInRing()
    feats.append(int(ring_flag))
    if ring_flag:
        ring_sizes = [
            len(r) for r in atom.GetOwningMol().GetRingInfo().AtomRings()
            if atom.GetIdx() in r
        ]
        min_ring = min(ring_sizes) if ring_sizes else 0
        ring_size_label = min_ring if min_ring < 9 else ">8"
    else:
        ring_size_label = 0
    feats += one_of_k_encoding(ring_size_label, RING_SIZE_LIST)

    symbol = atom.GetSymbol()
    is_donor = int(symbol in ["N", "O"] and atom.GetTotalNumHs() > 0)
    is_acceptor = int(symbol in ["N", "O", "F", "S"] and atom.GetTotalNumHs() == 0)
    is_halogen = int(symbol in ["F", "Cl", "Br", "I"])
    is_arom_c_hydrophobe = int(symbol == "C" and atom.GetIsAromatic())
    feats += [is_donor, is_acceptor, is_halogen, is_arom_c_hydrophobe]

    return np.asarray(feats, dtype=np.float32)


def create_combined_features(smiles_list, use_gpu):
    print("Initializing Uni-Mol model...")
    unimol_model = UniMolRepr(
        data_type="molecule",
        remove_hs=False,
        model_name="unimolv2",
        model_size="164m",
        use_gpu=use_gpu,
    )
    print("Uni-Mol model initialized (unimolv2, 164m)")

    combined_graph_dict = {}
    problematic_smiles = []

    for smi in tqdm(smiles_list, desc="Processing compounds"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                problematic_smiles.append(smi)
                continue

            try:
                frags = rdmolops.GetMolFrags(mol, asMols=True)
                if len(frags) > 1:
                    mol = max(frags, key=lambda m: m.GetNumAtoms())
                    Chem.SanitizeMol(mol)
            except Exception:
                problematic_smiles.append(smi)
                continue

            mol_no_h = Chem.RemoveHs(mol)
            try:
                AllChem.ComputeGasteigerCharges(mol_no_h)
            except Exception:
                pass

            c_size = mol_no_h.GetNumAtoms()
            if c_size == 0:
                problematic_smiles.append(smi)
                continue

            edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol_no_h.GetBonds()]
            g = nx.Graph(edges).to_directed()
            edge_index = [[e1, e2] for e1, e2 in g.edges]

            rdkit_features = [atom_features_with_hydrogen(atom) for atom in mol_no_h.GetAtoms()]

            unimol_reprs = unimol_model.get_repr([smi], return_atomic_reprs=True)
            unimol_features = unimol_reprs["atomic_reprs"][0]

            if len(unimol_features) < c_size:
                print(f"Warning: Uni-Mol features insufficient for {smi}")
                problematic_smiles.append(smi)
                continue

            features = []
            for j in range(c_size):
                rdkit_feat = np.array(rdkit_features[j], dtype=np.float32)
                unimol_feat = np.array(unimol_features[j], dtype=np.float32)
                features.append(np.concatenate([rdkit_feat, unimol_feat]))

            combined_graph_dict[smi] = (c_size, features, edge_index)

        except Exception as e:
            print(f"Error processing {smi}: {e}")
            problematic_smiles.append(smi)
            continue

    if problematic_smiles:
        print(f"Warning: {len(problematic_smiles)} SMILES could not be processed.")

    print(
        f"Success rate: {len(combined_graph_dict)}/{len(smiles_list)} "
        f"({100 * len(combined_graph_dict) / len(smiles_list):.1f}%)"
    )
    return combined_graph_dict


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_file)
    smiles_list = df[args.smiles_column].dropna().astype(str).unique().tolist()

    graph_dict = create_combined_features(smiles_list, args.use_gpu)
    np.save(args.output_graph, graph_dict)
    print(f"Saved graph dict: {args.output_graph}")
    print(f"Total compounds: {len(graph_dict)}")


if __name__ == "__main__":
    main()
