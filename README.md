# Geometric-Evolutionary Deep Learning Decodes the Human GPCR-Metabolome Interactome and Enables Systematic De-Orphanization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shimizu-team/G-LEAP/blob/main/examples/tutorial_inference.ipynb)

![G-LEAP_GA](https://github.com/user-attachments/assets/94056874-1ecc-433d-9768-79fe3ef52020)


## System Requirements

This software has been developed and tested on:
- **Operating System**: Red Hat Enterprise Linux 9.4
- **Python**: 3.10.19
- **GPU**: NVIDIA H100 (96 GB VRAM)
- **CUDA**: 12.1

An NVIDIA GPU with CUDA support is required for optimal performance. CPU-only inference is possible but significantly slower.

## Repository Structure

```
public_gleap/
├── src/                    # Core inference code
│   ├── inference_cv.py     # 10-fold CV ensemble inference
│   ├── model.py            # GCN + Bilinear attention model
│   └── dataset.py          # PyTorch Geometric dataset
├── models/
│   ├── drug_screening/     # Protein-split benchmark weights (10 folds; Figure 1D/E)
│   └── metabolite_screening/  # Metabolite screening weights (10 folds)
├── scripts/
│   ├── build_protein_graphs_esmc.py   # Protein graph generation
│   ├── build_compound_graphs_unimol.py # Compound graph generation
│   └── glass2_preprocessing/  # GLASS 2.0 benchmark construction (raw tables → CV splits)
├── data/
│   ├── pdb_sample/         # Sample AlphaFold PDB structures
│   ├── protein_graph_sample.npy
│   ├── ligand_graph_sample.npy
│   ├── protein_embeddings_sample.pt
│   └── benchmark_receptor_folds.csv # 790 receptors: protein_fold + pair counts + used_for_model (773)
├── examples/
│   ├── tutorial_inference.ipynb      # Interactive tutorial (GPU; includes Figure 1D/E)
│   ├── tutorial_inference_cpu.ipynb  # CPU variant (sample inference only)
│   ├── input_sample.csv
│   └── ligands_sample.csv
├── environment.yaml        # Conda environment specification
└── requirements.txt        # Pip dependencies
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate gleap
```

Or install with pip:

```bash
pip install -r requirements.txt
```

**Installation time:** Typical installation takes approximately 10-15 minutes on a standard workstation with a stable internet connection.

### Jupyter Kernel (Optional)

```bash
python -m ipykernel install --user --name gleap --display-name "Python (gleap)"
```

## Quick Start

See the interactive tutorial: `examples/tutorial_inference.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shimizu-team/G-LEAP/blob/main/examples/tutorial_inference.ipynb)

**Expected runtime:** The sample-inference tutorial runs in 2-3 minutes on an NVIDIA H100 GPU.
Reproducing the Figure 1D/E benchmark is a separate, GPU-heavy offline step (building graphs for
790 receptors / ~165k compounds, then 10-fold inference), not part of the 2-3 minute run.

### Run Sample Inference

```bash
python src/inference_cv.py \
  --model_dir models/drug_screening \
  --csv_file examples/input_sample.csv \
  --protein_graph data/protein_graph_sample.npy \
  --ligand_graph data/ligand_graph_sample.npy \
  --output_file examples/output.csv
```

All model hyperparameters are fixed to match the pretrained weights.

## Input Format

### CSV File

| Column | Description |
|--------|-------------|
| `UniProt ID` | Protein UniProt accession (e.g., P42866) |
| `smiles` | Compound SMILES string |
| `Label` | (Optional) Ground truth for evaluation |

### Graph Dictionaries

- **Protein graph** (`.npy`): `{UniProt_ID.pdb: (num_nodes, features, edge_index)}`
- **Ligand graph** (`.npy`): `{smiles: (num_nodes, features, edge_index)}`

Feature dimensions:
- Protein: 960 (ESM-C 300M embeddings)
- Ligand: 881 (RDKit descriptors + Uni-Mol embeddings)

## Building Graph Dictionaries

### 1. Protein Graphs (ESM-C + AlphaFold PDB)

Download PDB structures from [AlphaFold DB](https://alphafold.ebi.ac.uk/) (CC-BY 4.0 license). The
AlphaFold DB serves the current model version (v6 at the time of writing); use the API to stay
version-agnostic, or a direct file URL to pin a version:

```bash
# Version-agnostic (recommended): resolve the current model URL via the API
curl -s "https://alphafold.ebi.ac.uk/api/prediction/P42866" \
  | python -c "import sys,json;print(json.load(sys.stdin)[0]['pdbUrl'])" \
  | xargs curl -o P42866.pdb

# ...or pin a specific version directly
curl -o P42866.pdb "https://alphafold.ebi.ac.uk/files/AF-P42866-F1-model_v6.pdb"
```

The *Reproduce Figure 1D, E* section of the tutorial notebook downloads all 790 benchmark
structures this way in a loop.

Generate protein graphs:

```bash
python scripts/build_protein_graphs_esmc.py \
  --pdb_dir data/pdb_sample \
  --output_graph data/protein_graph.npy \
  --output_embeddings data/protein_embeddings.pt
```

### 2. Compound Graphs (RDKit + Uni-Mol)

Prepare a CSV with SMILES:

```csv
smiles
CC(C)NCC(O)c1ccc(O)c(O)c1
```

Generate compound graphs:

```bash
python scripts/build_compound_graphs_unimol.py \
  --csv_file examples/ligands_sample.csv \
  --output_graph data/ligand_graph.npy \
  --use_gpu
```

## Metabolite Screening

For metabolite-protein interaction prediction:

```bash
python src/inference_cv.py \
  --model_dir models/metabolite_screening \
  --csv_file examples/metabolite_input_sample.csv \
  --protein_graph data/metabolite_protein_graph_sample.npy \
  --ligand_graph data/metabolite_ligand_graph_sample.npy \
  --output_file examples/metabolite_output.csv
```

## Output Format

The output CSV contains:

| Column | Description |
|--------|-------------|
| `UniProt_ID` | Protein identifier |
| `SMILES` | Compound SMILES |
| `CV_Ensemble_Score` | Mean prediction across 10 folds |
| `CV_Ensemble_Binary` | Binary prediction (threshold=0.5) |
| `Model_Fold_*_Score` | Individual fold predictions |
| `CV_Score_Std` | Standard deviation across folds |
| `CV_Score_Min/Max` | Min/max scores across folds |

## Dependencies

- Python 3.10
- PyTorch 2.4.1 (CUDA 12.1)
- PyTorch Geometric 2.6.1
- ESM 3.1.3 (for protein embeddings)
- Uni-Mol Tools 0.1.5 (for compound embeddings)
- RDKit (for molecular descriptors)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### Third-party Components

- **AlphaFold structures**: Used in examples are under [CC-BY 4.0](https://alphafold.ebi.ac.uk/faq#faq-7)
- **ESM-C 300M**: [Cambrian Open License](https://github.com/evolutionaryscale/esm/blob/main/LICENSE.md)
- **Uni-Mol Tools**: [MIT License](https://github.com/deepmodeling/Uni-Mol)

## Citation

[Citation information TBD]
