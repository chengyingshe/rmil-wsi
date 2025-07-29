#!/usr/bin/env python3
"""
Universal MIL Model Inference Script for Pre-extracted Features
Supports: DSMIL, RMIL, RMIL-Mamba, RMIL-LSTM, RMIL-GRU, RMIL-TTT
Input: .pt files containing [N, feature_dim] tensors
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

import sys
import argparse
import os
import glob
import json
import time
from pathlib import Path
import numpy as np
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings
import logging
import common_utils as mil
import rmil_common

# Suppress warnings
warnings.filterwarnings('ignore')

# Import common utilities
from common_utils import (
    setup_logging, set_random_seed, apply_sparse_init,
    parse_model_params, log_model_info
)

class PTFeatureDataset(Dataset):
    """Dataset class for pre-extracted .pt feature files"""
    def __init__(self, pt_files_list):
        self.pt_files = pt_files_list
        
    def __len__(self):
        return len(self.pt_files)
    
    def __getitem__(self, idx):
        pt_file = self.pt_files[idx]
        
        try:
            # Load pre-extracted features [N, feature_dim]
            features = torch.load(pt_file, map_location='cpu')
            
            # Ensure features are 2D tensor
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            filename = os.path.basename(pt_file)
            
            return {
                'features': features,  # [N, feature_dim]
                'filename': filename,
                'pt_path': pt_file,
                'num_patches': features.shape[0]
            }
            
        except Exception as e:
            logging.error(f"Error loading {pt_file}: {str(e)}")
            # Return empty tensor on error
            return {
                'features': torch.empty(0, 1024),
                'filename': os.path.basename(pt_file),
                'pt_path': pt_file,
                'num_patches': 0
            }

def create_pt_dataset(args, pt_files_list):
    """Create dataset and dataloader for pre-extracted features"""
    if len(pt_files_list) == 0:
        logging.warning("No .pt files found")
        return None, 0
        
    dataset = PTFeatureDataset(pt_files_list)
    
    # For .pt files, we typically process one file at a time
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # Process one .pt file at a time
        shuffle=False, 
        num_workers=0,  # Avoid multiprocessing issues with large tensors
        drop_last=False
    )
    
    return dataloader, len(dataset)

def load_model(args):
    """Load the specified MIL model (without ResNet backbone since features are pre-extracted)"""
    device = torch.device(args.device)
    logging.info(f'Using device: {device}')
    
    # Auto-detect feature dimension from the first .pt file
    if os.path.isfile(args.pt_path) and args.pt_path.endswith('.pt'):
        # Single .pt file
        sample_file = args.pt_path
    elif os.path.isdir(args.pt_path):
        # Directory containing .pt files - use first one
        pt_files = glob.glob(os.path.join(args.pt_path, '*.pt'))
        if len(pt_files) == 0:
            raise ValueError(f"No .pt files found in {args.pt_path}")
        sample_file = pt_files[0]
    else:
        raise ValueError(f"Invalid pt_path: {args.pt_path}")
    
    # Load sample file to determine feature dimension
    try:
        logging.info(f"Auto-detecting feature dimension from: {os.path.basename(sample_file)}")
        sample_features = torch.load(sample_file, map_location='cpu')
        if sample_features.dim() == 1:
            sample_features = sample_features.unsqueeze(0)
        detected_feats_size = sample_features.shape[1]
        logging.info(f"Detected feature dimension: {detected_feats_size}")
    
        # Update feats_size if different from argument
        if detected_feats_size != args.feats_size:
            logging.info(f"Updating feats_size from {args.feats_size} to {detected_feats_size}")
            args.feats_size = detected_feats_size
    except Exception as e:
        logging.error(f"Error loading sample file {sample_file}: {str(e)}")
        logging.info(f"Using default feats_size: {args.feats_size}")
    
    # Parse model parameters
    model_params = parse_model_params(args)
    logging.info(f"Model: {args.model}")
    logging.info(f"Model parameters: {model_params}")
    
    # Initialize i_classifier using rmil_common
    i_classifier = rmil_common.FCLayer(in_size=args.feats_size, out_size=args.num_classes).to(device)
    
    # Initialize BClassifier based on model type
    if args.model.startswith('rmil_ttt'):
        from rmil_ttt import BClassifier
        logging.info('Loading model: RMIL-TTT-Linear')
        ttt_params = parse_model_params(args)
        ttt_params['ttt_layer_type'] = args.model.split('_')[-1]
        b_classifier = BClassifier(input_size=args.feats_size, 
                                      output_class=args.num_classes, 
                                      **ttt_params).to(device)
    elif args.model.startswith('rmil_mamba'):
        from rmil_mamba import BClassifier
        logging.info('Loading model: RMIL-Mamba')
        mamba_params = parse_model_params(args)
        b_classifier = BClassifier(input_size=args.feats_size, 
                                      output_class=args.num_classes, 
                                      **mamba_params).to(device)
    elif args.model.startswith('rmil_gru'):
        from rmil_gru import BClassifier
        logging.info('Loading model: RMIL-GRU')
        gru_params = parse_model_params(args)
        b_classifier = BClassifier(input_size=args.feats_size, 
                                      output_class=args.num_classes, 
                                      **gru_params).to(device)
    elif args.model.startswith('rmil_lstm'):
        from rmil_lstm import BClassifier
        logging.info('Loading model: RMIL-LSTM')
        lstm_params = parse_model_params(args)
        b_classifier = BClassifier(input_size=args.feats_size, 
                                      output_class=args.num_classes, 
                                      **lstm_params).to(device)
    elif args.model == 'snuffy':
        from snuffy import BClassifier
        logging.info('Loading model: Snuffy')
        snuffy_params = parse_model_params(args)
        b_classifier = BClassifier(input_size=args.feats_size, 
                                      output_class=args.num_classes, 
                                      **snuffy_params).to(device)
    elif args.model == 'dsmil':
        from dsmil import BClassifier
        logging.info('Loading model: DSMIL')
        dsmil_params = parse_model_params(args)
        b_classifier = BClassifier(input_size=args.feats_size, 
                                      output_class=args.num_classes, 
                                      **dsmil_params).to(device)
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    # Create MIL network
    milnet = rmil_common.MILNet(i_classifier, b_classifier).to(device)
    
    # Apply sparse initialization
    milnet.apply(lambda m: apply_sparse_init(m))
    
    # Load model weights
    if os.path.exists(args.model_path):
        aggregator_weights = torch.load(args.model_path, map_location=device)
        logging.info(f'Loading milnet weights from: {args.model_path}')
        
        # Try loading with strict=False first, then handle dimension mismatches
        try:
            milnet.load_state_dict(aggregator_weights, strict=False)
            logging.info("Model weights loaded successfully")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                logging.warning("Model dimension mismatch detected, attempting to load compatible weights...")
                
                # Create a new state dict with only compatible weights
                model_state = milnet.state_dict()
                compatible_weights = {}
                
                for key, value in aggregator_weights.items():
                    if key in model_state:
                        if model_state[key].shape == value.shape:
                            compatible_weights[key] = value
                        else:
                            logging.info(f"   Skipping {key}: shape mismatch {model_state[key].shape} vs {value.shape}")
                    else:
                        logging.info(f"   Skipping {key}: not found in current model")
                
                # Load compatible weights
                milnet.load_state_dict(compatible_weights, strict=False)
                logging.info(f"Loaded {len(compatible_weights)}/{len(aggregator_weights)} compatible weights")
                logging.warning("Some layers will use random initialization")
            else:
                raise e
    else:
        logging.warning(f"Model weights not found at {args.model_path}")
        logging.info("Using randomly initialized weights...")
    
    return milnet, None

def process_single_pt_file(args, pt_file, milnet):
    """Process a single .pt feature file and return predictions"""
    logging.info(f"Processing PT file: {os.path.basename(pt_file)}")
    
    try:
        # Load pre-extracted features
        features = torch.load(pt_file, map_location='cpu')
    
        # Ensure features are 2D
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        logging.info(f"Loaded features shape: {features.shape}")
    
        # Check feature dimensions
        if features.shape[1] != args.feats_size:
            logging.warning("Feature dimension mismatch!")
            logging.info(f"   Expected: {args.feats_size}, Got: {features.shape[1]}")
            logging.info(f"   Adjusting feats_size to {features.shape[1]}")
            args.feats_size = features.shape[1]
        
        num_patches = features.shape[0]
        
        if num_patches == 0:
            logging.warning(f"Empty feature tensor in {pt_file}")
            return None
        
        # Move features to device
        bag_feats = features.to(args.device)
        
        # Generate dummy instance predictions (since we only have features)
        # This is needed for some MIL models that expect both features and instance predictions
        with torch.no_grad():
            # Create dummy instance classifications
            ins_classes = torch.zeros(num_patches, args.num_classes).to(args.device)
        
            # Use b_classifier for bag-level prediction
        bag_prediction, attention_weights, _ = milnet.b_classifier(bag_feats, ins_classes)
        bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
        
            # Optional averaging (though less meaningful with pre-extracted features)
        if args.average:
                # Simple averaging of features as fallback
                mean_prediction = torch.sigmoid(torch.mean(bag_feats, dim=0)[:args.num_classes]).cpu().numpy()
                if bag_prediction.ndim == 0:
                    bag_prediction = (bag_prediction + mean_prediction[0] if len(mean_prediction) > 0 else bag_prediction) / 2
                else:
                    bag_prediction = (bag_prediction + mean_prediction[:len(bag_prediction)]) / 2
        
        # Prepare results
        results = {
            'pt_file': os.path.basename(pt_file),
            'pt_path': pt_file,
            'num_patches': int(num_patches),
            'feature_dim': int(features.shape[1]),
            'bag_prediction': bag_prediction.tolist() if bag_prediction.ndim > 0 else [float(bag_prediction)],
            'attention_weights': attention_weights.cpu().numpy().tolist() if attention_weights is not None else [],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Generate attention map if requested (simplified version without spatial positions)
        if args.save_attention_maps and attention_weights is not None:
            save_attention_heatmap(args, results, attention_weights.cpu().numpy())
        
        return results

    except Exception as e:
        logging.error(f"Error processing {pt_file}: {str(e)}")
        return None

def process_pt_directory(args, pt_dir, milnet):
    """Process all .pt files in a directory"""
    logging.info(f"Processing PT directory: {os.path.basename(pt_dir)}")
    
    # Find all .pt files
    pt_files = glob.glob(os.path.join(pt_dir, '*.pt'))
    
    if len(pt_files) == 0:
        logging.warning(f"No .pt files found in {pt_dir}")
        return []
    
    logging.info(f"Found {len(pt_files)} .pt files")
    
    all_results = []
    for i, pt_file in enumerate(pt_files):
        logging.info(f"{'='*30}")
        logging.info(f"Processing file {i+1}/{len(pt_files)}")
        
        results = process_single_pt_file(args, pt_file, milnet)
        if results is not None:
            all_results.append(results)
            
            # Print prediction summary
            pred = results['bag_prediction']
            logging.info(f"Prediction: {pred}")
            if args.num_classes == 2:
                if len(pred) >= 2:
                    logging.info(f"   Class 0: {pred[0]:.4f}, Class 1: {pred[1]:.4f}")
                else:
                    logging.info(f"   Confidence: {pred[0]:.4f}")
    
    return all_results

def save_attention_heatmap(args, results, attention_weights):
    """Save attention weights as a simple heatmap"""
    try:
        import matplotlib.pyplot as plt
        
        # Create a simple 1D heatmap of attention weights
        if len(attention_weights.shape) > 1:
            attention_weights = attention_weights[:, 0]  # Use first class if multi-class
        
        plt.figure(figsize=(12, 3))
        plt.imshow(attention_weights.reshape(1, -1), cmap='Reds', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.title(f'Attention Weights: {results["pt_file"]}')
        plt.xlabel('Patch Index')
        plt.yticks([])
        
        # Save heatmap
        pt_name = results['pt_file'].replace('.pt', '')
        attention_path = os.path.join(args.output_dir, 'attention_maps', f'{pt_name}_attention.png')
        os.makedirs(os.path.dirname(attention_path), exist_ok=True)
        
        plt.savefig(attention_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Attention heatmap saved: {attention_path}")
        
    except ImportError:
        logging.warning("matplotlib not available, skipping attention visualization")
    except Exception as e:
        logging.error(f"Error saving attention heatmap: {str(e)}")

def run_inference(args):
    """Main inference function for .pt files"""
    logging.info("Starting MIL Model Inference for Pre-extracted Features")
    logging.info(f"Model: {args.model}")
    logging.info(f"PT Path: {args.pt_path}")
    logging.info(f"Output: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    milnet, mil_module = load_model(args)
    milnet.eval()
    
    # Determine if processing single file or directory
    all_results = []
    
    if os.path.isfile(args.pt_path) and args.pt_path.endswith('.pt'):
        # Single .pt file
        logging.info("Processing single .pt file")
        results = process_single_pt_file(args, args.pt_path, milnet)
        if results is not None:
            all_results.append(results)
    
    elif os.path.isdir(args.pt_path):
        # Directory containing .pt files
        pt_files = glob.glob(os.path.join(args.pt_path, '*.pt'))
        
        if len(pt_files) == 0:
            logging.error(f"No .pt files found in {args.pt_path}")
            return
        
        logging.info(f"Found {len(pt_files)} .pt files to process")
        
        # Process each .pt file
        for i, pt_file in enumerate(pt_files):
            logging.info(f"{'='*50}")
            logging.info(f"Processing file {i+1}/{len(pt_files)}")
            
            results = process_single_pt_file(args, pt_file, milnet)
            if results is not None:
                all_results.append(results)
            else:
                logging.error(f"Error: PT path {args.pt_path} is not a valid file or directory")
                return
    
    # Save results
    if all_results:
        results_file = os.path.join(args.output_dir, 'inference_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save summary
        total_files = len(all_results)
        summary = {
            'model': args.model,
            'model_path': args.model_path,
            'pt_path': args.pt_path,
            'total_pt_files': total_files,
            'successful_predictions': len(all_results),
            'average_num_patches': np.mean([r['num_patches'] for r in all_results]) if all_results else 0,
            'feature_dimension': all_results[0]['feature_dim'] if all_results else args.feats_size,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'args': vars(args)
        }
        
        summary_file = os.path.join(args.output_dir, 'inference_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info("Inference completed!")
        logging.info(f"Results saved to: {results_file}")
        logging.info(f"Summary saved to: {summary_file}")
        logging.info(f"Successfully processed: {len(all_results)} .pt file(s)")
        
        # Print summary statistics
        if all_results:
            predictions = [r['bag_prediction'] for r in all_results]
            patches_counts = [r['num_patches'] for r in all_results]
            
            logging.info("Summary Statistics:")
            logging.info(f"   Total files processed: {len(all_results)}")
            logging.info(f"   Average patches per file: {np.mean(patches_counts):.1f}")
            logging.info(f"   Feature dimension: {all_results[0]['feature_dim']}")
            
            if args.num_classes == 2:
                pos_predictions = [p[1] if len(p) > 1 else p[0] for p in predictions]
                logging.info(f"   Average positive prediction: {np.mean(pos_predictions):.4f}")
                logging.info(f"   Prediction range: {np.min(pos_predictions):.4f} - {np.max(pos_predictions):.4f}")
    else:
        logging.warning("No successful predictions made")

def main():
    parser = argparse.ArgumentParser(description='Universal MIL Model Inference Script for Pre-extracted Features')
    
    # Model configuration
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]', 
                       choices=['dsmil', 'snuffy', 'abmil', 'rmil_lstm', 'rmil_gru', 'rmil_mamba', 'rmil_ttt_linear', 'rmil_ttt_mlp'])
    parser.add_argument('--model_path', type=str, default='./example_aggregator_weights/tcga_aggregator.pth',
                       help='Path to model weights (.pth file)')
    parser.add_argument('--pt_path', type=str, required=True,
                       help='Path to .pt file or directory containing .pt files with pre-extracted features')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2, 
                       help='Number of output classes')
    parser.add_argument('--feats_size', type=int, default=512,
                       help='Feature size (will be auto-detected from .pt files)')
    
    # Inference parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--average', action='store_true',
                       help='Average bag predictions with mean features')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='inference_output',
                       help='Output directory for results')
    parser.add_argument('--save_attention_maps', action='store_true',
                       help='Save attention weight visualizations')
    
    # Classification thresholds
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    
    # Model parameters
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    
    parser.add_argument('--gru_hidden_size', type=int, default=768, help='GRU hidden size')
    parser.add_argument('--gru_num_layers', type=int, default=2, help='GRU number of layers')
    parser.add_argument('--gru_bidirectional', type=str, default='True', choices=['True', 'False'], help='GRU bidirectional')
    parser.add_argument('--gru_selection_strategy', type=str, default='hybrid', choices=['hybrid', 'original'], help='GRU selection strategy')
    
    parser.add_argument('--lstm_hidden_size', default=768, type=int, help='LSTM hidden size')
    parser.add_argument('--lstm_num_layers', default=2, type=int, help='LSTM number of layers')
    parser.add_argument('--lstm_bidirectional', default='True', type=str, choices=['True', 'False'], help='LSTM bidirectional')
    parser.add_argument('--lstm_selection_strategy', default='hybrid', type=str, choices=['hybrid', 'original'], help='LSTM selection strategy')
    
    parser.add_argument('--mamba_depth', type=int, default=8, help='Mamba depth')
    parser.add_argument('--mamba_d_state', type=int, default=32, help='Mamba d_state')
    parser.add_argument('--mamba_d_conv', type=int, default=4, help='Mamba d_conv')
    parser.add_argument('--mamba_expand', type=int, default=2, help='Mamba expand')
    parser.add_argument('--mamba_selection_strategy', type=str, default='hybrid', choices=['hybrid', 'original'], help='Mamba selection strategy')
    
    parser.add_argument('--ttt_num_heads', type=int, default=16, help='TTT number of heads')
    parser.add_argument('--ttt_num_layers', type=int, default=8, help='TTT number of layers')
    parser.add_argument('--ttt_mini_batch_size', type=int, default=16, help='TTT mini batch size')
    parser.add_argument('--ttt_base_lr', type=float, default=1.0, help='TTT base learning rate')
    parser.add_argument('--ttt_selection_strategy', type=str, default='hybrid', choices=['hybrid', 'original'], help='TTT selection strategy')
    
    parser.add_argument('--big_lambda', type=int, default=64, help='Big lambda (patch selection number)')
    parser.add_argument('--dropout_v', type=float, default=0.2, help='Dropout rate')
    
    # JSON parameter configuration support
    parser.add_argument('--params', type=str, help='JSON format model parameter configuration dictionary')
    
    # Logging parameters
    parser.add_argument('--log_file', default='./inference_log.txt', type=str, help='Log file name')
    parser.add_argument('--random_seed', default=42, type=int, help='Random seed for reproducible results [42]')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file=args.log_file)
    
    # Set random seed
    set_random_seed(args.random_seed)
    
    # Process JSON parameter configuration
    if args.params:
        try:
            import json
            params_dict = json.loads(args.params)
            logging.info(f"Loading JSON parameter configuration: {params_dict}")
            
            # Override args with JSON parameters (JSON config takes priority)
            for key, value in params_dict.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                    logging.info(f"   Override parameter {key}: {value}")
                else:
                    logging.warning(f"   Unknown parameter {key}: {value}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON parameter parsing failed: {e}")
            sys.exit(1)
    
    # Validation
    if not os.path.exists(args.model_path):
        logging.error(f"Model weights not found at {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.pt_path):
        logging.error(f"PT path not found at {args.pt_path}")
        sys.exit(1)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Run inference
    run_inference(args)

if __name__ == '__main__':
    main() 