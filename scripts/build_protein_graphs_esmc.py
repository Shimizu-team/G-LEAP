#!/usr/bin/env python3
"""
Build protein graph dicts from AlphaFold PDBs using ESMC-300M embeddings.

Outputs:
- embeddings (.pt): {protein_id: embeddings}
- graph dict (.npy): {protein_id: (num_nodes, features, edge_index)}
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
from Bio.PDB import PDBParser
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from tqdm import tqdm

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build protein graphs with ESMC embeddings.")
    parser.add_argument("--pdb_dir", required=True, help="Directory containing PDB files.")
    parser.add_argument("--output_graph", required=True, help="Output .npy graph dict path.")
    parser.add_argument("--output_embeddings", required=True, help="Output .pt embeddings path.")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Distance cutoff for edges (A).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def extract_sequence_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_file)
    sequence = ""
    residue_ids = []
    for chain in structure[0]:
        for residue in chain:
            if residue.get_id()[0] != " ":
                continue
            res_name = residue.get_resname()
            if res_name in THREE_TO_ONE:
                sequence += THREE_TO_ONE[res_name]
                residue_ids.append(residue.get_id()[1])
    return sequence, residue_ids


def load_ca_residues(pdb_file):
    residues = []
    with open(pdb_file, "r") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if len(line) <= 54:
                continue
            if line[16] == "B" or " HOH " in line:
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            if atom_name != "CA" or res_name == "UNK":
                continue
            residue_number = int(re.sub(r"[^0-9]", "", line[22:26]))
            res_id = line[17:26]
            coords = [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ]
            residues.append((res_id, residue_number, coords))
    return residues


def build_graph(residues, residue_to_embedding, aa_embeddings, cutoff):
    residue_pairs = []
    uniq = set()
    resnum_by_id = {}
    for res_id, res_num, _ in residues:
        resnum_by_id[res_id] = res_num

    for i in range(len(residues)):
        coords_i = np.array(residues[i][2])
        for j in range(i + 1, len(residues)):
            coords_j = np.array(residues[j][2])
            distance = np.linalg.norm(coords_i - coords_j)
            if distance < cutoff:
                residue_pairs.append((residues[i][0], residues[j][0]))
                uniq.add(residues[i][0])
                uniq.add(residues[j][0])

    if not residue_pairs:
        return 0, [], []

    uniq_list = list(uniq)
    id_to_idx = {res_id: idx for idx, res_id in enumerate(uniq_list)}
    edges = [[id_to_idx[a], id_to_idx[b]] for a, b in residue_pairs]

    zero_feat = torch.zeros_like(aa_embeddings[0]).tolist()
    features = []
    for res_id in uniq_list:
        res_num = resnum_by_id.get(res_id)
        emb_idx = residue_to_embedding.get(res_num)
        if emb_idx is not None and emb_idx < len(aa_embeddings):
            features.append(aa_embeddings[emb_idx].tolist())
        else:
            features.append(zero_feat)

    return len(uniq_list), features, edges


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    pdb_files = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")) + glob.glob(os.path.join(args.pdb_dir, "*.ent")))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {args.pdb_dir}")

    print(f"Found {len(pdb_files)} PDB files.")
    print("Loading ESMC model...")
    client = ESMC.from_pretrained("esmc_300m").to(device)

    embeddings = {}
    graph_dict = {}

    for pdb_file in tqdm(pdb_files, desc="Processing PDBs"):
        protein_id = os.path.splitext(os.path.basename(pdb_file))[0]
        sequence, residue_ids = extract_sequence_from_pdb(pdb_file)
        if not sequence:
            print(f"Skipping {protein_id}: no sequence extracted.")
            continue

        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        logits_output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
        embeddings[protein_id] = logits_output.embeddings

        aa_embeddings = logits_output.embeddings[0][1:-1]
        residue_to_embedding = {res_id: idx for idx, res_id in enumerate(residue_ids)}
        residues = load_ca_residues(pdb_file)
        c_size, features, edge_index = build_graph(
            residues, residue_to_embedding, aa_embeddings, args.cutoff
        )
        if c_size == 0:
            print(f"Skipping {protein_id}: no edges after cutoff.")
            continue
        graph_dict[protein_id] = (c_size, features, edge_index)

    torch.save(embeddings, args.output_embeddings)
    np.save(args.output_graph, graph_dict)
    print(f"Saved embeddings: {args.output_embeddings}")
    print(f"Saved graphs: {args.output_graph}")


if __name__ == "__main__":
    main()
