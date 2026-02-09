#!/usr/bin/env python3
"""
Cross-Validation inference script.
Run inference using multiple CV fold models and ensemble the results.

Usage:
python inference_cv.py --model_dir /path/to/cv_models \
    --csv_file /path/to/data.csv \
    --protein_graph /path/to/protein_embeddings.npy \
    --ligand_graph /path/to/ligand_embeddings.npy \
    --output_file results.csv
"""

import torch
import numpy as np
import pandas as pd
import argparse
import os
import glob
from torch_geometric.data import DataLoader
from dataset import MoleculeDataset
from model import DualGNN_Bilinear
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser(description='CV inference script - inference using multiple fold models')

    # Required parameters
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory path where CV models are saved')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='CSV file for inference data')
    parser.add_argument('--protein_graph', type=str, required=True,
                       help='Path to protein graph data (.npy file)')
    parser.add_argument('--ligand_graph', type=str, required=True,
                       help='Path to ligand graph data (.npy file)')

    # Output settings
    parser.add_argument('--output_file', type=str, default='inference_cv_results.csv',
                       help='Output file name (default: inference_cv_results.csv)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')

    # Model configuration parameters
    parser.add_argument('--model_type', type=str, default='gcn_bilinear',
                       choices=['gcn_bilinear'],
                       help='Model type (fixed: gcn_bilinear)')
    parser.add_argument('--split_type', type=str, default='protein',
                       choices=['random', 'protein', 'ligand'],
                       help='CV model split type (default: protein)')
    parser.add_argument('--model_pattern', type=str, default='pretrained_model_fold{fold}.pt',
                       help='Model file pattern (default: pretrained_model_fold{fold}.pt)')

    # CV settings
    parser.add_argument('--n_folds', type=int, default=10,
                       help='Number of CV folds to use (default: 10)')
    parser.add_argument('--ensemble_method', type=str, default='mean',
                       choices=['mean', 'majority'],
                       help='Ensemble method (default: mean)')

    # Model parameters (fixed for pretrained models)
    parser.add_argument('--protein_dim', type=int, default=960,
                       help='Protein feature dimension (fixed: 960 for ESM-C)')
    parser.add_argument('--ligand_dim', type=int, default=881,
                       help='Ligand feature dimension (fixed: 881 for RDKit+UniMol)')
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='Embedding dimension (fixed: 128)')
    parser.add_argument('--output_dim', type=int, default=128,
                       help='Output dimension (fixed: 128)')
    parser.add_argument('--bilinear_hidden', type=int, default=256,
                       help='Bilinear hidden dimension (fixed: 256)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (fixed: 0.2)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GCN layers (fixed: 2)')
    parser.add_argument('--normalization', type=str, default='batch',
                       choices=['none', 'batch', 'layer'],
                       help='Normalization type (fixed: batch)')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'max', 'add'],
                       help='Pooling type (fixed: mean)')

    return parser.parse_args()

def load_cv_models(model_dir, model_type, split_type, n_folds, args):
    """Load all CV models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)

    print(f"Using device: {device}")

    models = []

    for fold in range(n_folds):
        # Model file search pattern
        if args.model_pattern:
            model_pattern = os.path.join(model_dir, args.model_pattern.format(fold=fold))
        else:
            model_pattern = f"{model_dir}/model_{model_type}_{split_type}_fold{fold}.pt"
        model_files = glob.glob(model_pattern)

        if not model_files:
            print(f"Warning: Model file not found for fold {fold}: {model_pattern}")
            continue

        model_path = model_files[0]
        print(f"Loading model for fold {fold}: {model_path}")

        # Model initialization
        model = DualGNN_Bilinear(
            ligand_num_features=args.ligand_dim,
            protein_num_features=args.protein_dim,
            embed_dim=args.embed_dim,
            output_dim=args.output_dim,
            dropout=args.dropout,
            bilinear_hidden=args.bilinear_hidden,
            num_layers=args.num_layers,
            normalization=args.normalization,
            pooling=args.pooling
        )

        # Load model weights
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()
            models.append(model)
            print(f"Loaded fold {fold} model successfully")

        except Exception as e:
            print(f"Error loading model for fold {fold}: {e}")
            continue

    if not models:
        raise ValueError("No models could be loaded successfully!")

    print(f"Successfully loaded {len(models)} models")
    return models, device

def prepare_dataset(csv_file, protein_graph_path, ligand_graph_path):
    """Prepare dataset (following train_cv.py)"""
    print("Loading graph data...")

    # Load graph data
    pdb_graph = np.load(protein_graph_path, allow_pickle=True).item()
    smiles_graph_dict = np.load(ligand_graph_path, allow_pickle=True).item()

    print(f"Protein graph entries: {len(pdb_graph)}")
    print(f"Ligand graph entries: {len(smiles_graph_dict)}")

    # Create mapping dict for UniProt ID to .pdb suffix keys (same as train_cv.py)
    uniprot_to_pdb_key = {}
    for key in pdb_graph.keys():
        if '.pdb' in key:
            uniprot_id = key.split('.')[0]
            uniprot_to_pdb_key[uniprot_id] = key

    print(f"Available protein mappings: {len(uniprot_to_pdb_key)}")
    print(f"Available SMILES: {len(smiles_graph_dict)}")

    # Create protein_graph_dict using the mapping dict
    # Register both with and without .pdb suffix so CSV works either way
    mapped_pdb_graph = {}
    for uniprot_id, pdb_key in uniprot_to_pdb_key.items():
        mapped_pdb_graph[uniprot_id] = pdb_graph[pdb_key]
        mapped_pdb_graph[pdb_key] = pdb_graph[pdb_key]

    # Create dataset
    dataset = MoleculeDataset(
        root='data_inference_cv',
        dataset='inference',
        csv_file=csv_file,
        protein_graph_dict=mapped_pdb_graph,
        smile_graph_dict=smiles_graph_dict
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Warning: Dataset is empty!")
        print("Please check that protein and ligand IDs in CSV match the graph dictionaries")
        return None

    return dataset

def run_cv_inference(models, dataset, device, batch_size=128, ensemble_method='mean'):
    """Run CV inference and return ensemble results"""
    print(f"Running CV inference with {len(models)} models...")

    # Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=['ligand_x', 'protein_x'],
        num_workers=4
    )

    all_predictions = []  # Store predictions from each model
    protein_names = []
    smiles = []

    # Run inference with each model
    for model_idx, model in enumerate(models):
        print(f"Running inference with model {model_idx + 1}/{len(models)}...")
        model.eval()
        model_predictions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = batch.to(device)

                # Run inference
                output = model(batch)
                scores = output.cpu().numpy().flatten()
                model_predictions.extend(scores)

                # Collect metadata only from the first model
                if model_idx == 0:
                    batch_protein_names = []
                    batch_smiles = []

                    current_batch_size = len(scores)
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + current_batch_size, len(dataset))

                    for i in range(start_idx, end_idx):
                        data = dataset[i]
                        batch_protein_names.append(data.protein_name)
                        batch_smiles.append(data.smile)

                    protein_names.extend(batch_protein_names)
                    smiles.extend(batch_smiles)

        all_predictions.append(model_predictions)
        print(f"Model {model_idx + 1} completed. Predictions: {len(model_predictions)}")

    # Ensemble processing
    all_predictions = np.array(all_predictions)  # (n_models, n_samples)

    if ensemble_method == 'mean':
        ensemble_predictions = np.mean(all_predictions, axis=0)
    elif ensemble_method == 'majority':
        # Binary voting (0.5 as threshold)
        binary_predictions = (all_predictions > 0.5).astype(int)
        ensemble_predictions = np.mean(binary_predictions, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")

    print(f"CV inference completed. Total predictions: {len(ensemble_predictions)}")
    print(f"Ensemble method used: {ensemble_method}")

    return ensemble_predictions, protein_names, smiles, all_predictions

def save_cv_results(ensemble_predictions, protein_names, smiles, all_predictions,
                   output_file, input_csv_file, models_info=None):
    """Save CV results to CSV file"""
    print(f"Saving CV results to {output_file}...")

    # Read original CSV file to add information
    original_df = pd.read_csv(input_csv_file)

    # Create basic results dataframe
    results_df = pd.DataFrame({
        'UniProt_ID': protein_names,
        'SMILES': smiles,
        'CV_Ensemble_Score': ensemble_predictions,
        'CV_Ensemble_Binary': [1 if score > 0.5 else 0 for score in ensemble_predictions]
    })

    # Add individual predictions from each model
    for i, model_preds in enumerate(all_predictions):
        results_df[f'Model_Fold_{i}_Score'] = model_preds
        results_df[f'Model_Fold_{i}_Binary'] = [1 if score > 0.5 else 0 for score in model_preds]

    # Add prediction variability statistics
    results_df['CV_Score_Std'] = np.std(all_predictions, axis=0)
    results_df['CV_Score_Min'] = np.min(all_predictions, axis=0)
    results_df['CV_Score_Max'] = np.max(all_predictions, axis=0)

    # Merge with original CSV file to preserve additional information
    if 'UniProt ID' in original_df.columns and 'smiles' in original_df.columns:
        # Unify column names
        original_df = original_df.rename(columns={'UniProt ID': 'UniProt_ID', 'smiles': 'SMILES'})

        # Merge
        merged_df = pd.merge(
            results_df,
            original_df,
            on=['UniProt_ID', 'SMILES'],
            how='left'
        )

        # Adjust column order
        base_columns = ['UniProt_ID', 'SMILES', 'CV_Ensemble_Score', 'CV_Ensemble_Binary']
        cv_columns = [col for col in results_df.columns if col.startswith('Model_Fold_') or col.startswith('CV_Score_')]
        other_columns = [col for col in merged_df.columns if col not in base_columns + cv_columns]
        final_columns = base_columns + cv_columns + other_columns

        merged_df = merged_df[final_columns]
        merged_df.to_csv(output_file, index=False)
    else:
        results_df.to_csv(output_file, index=False)

    print(f"CV results saved successfully!")
    print(f"Total predictions: {len(results_df)}")
    print(f"Positive predictions (ensemble score > 0.5): {sum(results_df['CV_Ensemble_Binary'])}")
    print(f"Negative predictions (ensemble score <= 0.5): {len(results_df) - sum(results_df['CV_Ensemble_Binary'])}")

    # Display statistics
    print(f"\nCV Ensemble Prediction Statistics:")
    print(f"Mean: {np.mean(ensemble_predictions):.4f}")
    print(f"Std:  {np.std(ensemble_predictions):.4f}")
    print(f"Min:  {np.min(ensemble_predictions):.4f}")
    print(f"Max:  {np.max(ensemble_predictions):.4f}")

    print(f"\nCV Prediction Variability:")
    print(f"Average std across samples: {np.mean(results_df['CV_Score_Std']):.4f}")
    print(f"Max std across samples: {np.max(results_df['CV_Score_Std']):.4f}")

def main():
    args = parse_args()

    print("=== GPCR-Ligand Interaction Prediction CV Inference ===")
    print(f"Model directory: {args.model_dir}")
    print(f"Input CSV: {args.csv_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model type: {args.model_type}")
    print(f"Split type: {args.split_type}")
    print(f"N folds: {args.n_folds}")
    print(f"Ensemble method: {args.ensemble_method}")

    # Check file existence
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
    if not os.path.exists(args.protein_graph):
        raise FileNotFoundError(f"Protein graph file not found: {args.protein_graph}")
    if not os.path.exists(args.ligand_graph):
        raise FileNotFoundError(f"Ligand graph file not found: {args.ligand_graph}")

    # Load CV models
    models, device = load_cv_models(
        args.model_dir, args.model_type, args.split_type, args.n_folds, args
    )

    # Prepare dataset
    dataset = prepare_dataset(args.csv_file, args.protein_graph, args.ligand_graph)
    if dataset is None:
        print("Failed to prepare dataset. Exiting.")
        return

    # Run CV inference
    ensemble_predictions, protein_names, smiles, all_predictions = run_cv_inference(
        models, dataset, device, args.batch_size, args.ensemble_method
    )

    # Save results
    save_cv_results(
        ensemble_predictions, protein_names, smiles, all_predictions,
        args.output_file, args.csv_file
    )

    print("\n=== CV Inference completed successfully! ===")

if __name__ == '__main__':
    main()
