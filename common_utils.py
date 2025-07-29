import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import time
import json
import os
import copy
import random
import datetime
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.utils import shuffle
import sys

# Logging configuration
def setup_logging(log_file='log.txt'):
    """
    Setup logging configuration
    """
    # Clear previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # Create log format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# Performance monitoring decorator
def log_performance(func_name):
    """Performance monitoring decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logging.info(f"[Performance] {func_name} execution time: {end_time - start_time:.2f}s")
            return result
        return wrapper
    return decorator

# Random seed setting
def set_random_seed(seed=42):
    """
    Set all random seeds for reproducible results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to: {seed}")

# Model initialization helper
def apply_sparse_init(m):
    """Apply sparse initialization to model layers"""
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# ROC and threshold computation
def optimal_thresh(fpr, tpr, thresholds, p=0):
    """Calculate optimal threshold"""
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    """Calculate multi-label ROC curves"""
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)
        
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        
        try:
            c_auc = roc_auc_score(label, prediction)
            logging.info(f"Class {c} ROC AUC score: {c_auc:.4f}")
        except ValueError as e:
            if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                logging.warning(f"Class {c} has only one class present, ROC AUC set to 1.0")
                c_auc = 1
            else:
                raise e

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
        
    return aucs, thresholds, thresholds_optimal

# Model checkpoint management
def save_model_checkpoint(args, fold, save_path, model, epoch, accuracy, auc, thresholds_optimal, model_type='best', additional_metrics=None):
    """Save model checkpoint with training metrics"""
    save_name = os.path.join(save_path, f'fold_{fold}_{model_type}.pth')
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy,
        'auc': auc,
        'fold': fold,
        'model_type': model_type,
        'thresholds_optimal': thresholds_optimal
    }
    
    # Add additional metrics if provided
    if additional_metrics:
        checkpoint.update(additional_metrics)
    
    torch.save(checkpoint, save_name)
    
    # Save thresholds separately
    thresh_file = os.path.join(save_path, f'fold_{fold}_{model_type}.json')
    with open(thresh_file, 'w') as f:
        json.dump([float(x) for x in thresholds_optimal], f)
    
    model_size = os.path.getsize(save_name) / (1024 * 1024)  # MB
    logging.info(f'Model saved: {save_name} (type: {model_type}, size: {model_size:.2f}MB)')
    logging.info(f'Model metrics: Epoch={epoch}, Accuracy={accuracy:.4f}, AUC={auc}')
    logging.info(f'Optimal thresholds: {[float(x) for x in thresholds_optimal]}')
    
    return save_name

def load_checkpoint(resume_path, fold, model_type='last'):
    """Load model checkpoint"""
    checkpoint_path = os.path.join(resume_path, f'fold_{fold}_{model_type}.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logging.info(f"Loaded checkpoint: {checkpoint_path}")
        logging.info(f"   - Epoch: {checkpoint.get('epoch', 'unknown')}")
        logging.info(f"   - Accuracy: {checkpoint.get('accuracy', 'unknown'):.4f}")
        logging.info(f"   - AUC: {checkpoint.get('auc', 'unknown')}")
        return checkpoint
    else:
        logging.warning(f"Checkpoint not found for fold {fold}: {checkpoint_path}")
        return None

# Score calculation
def get_current_score(avg_score, aucs):
    """Calculate current comprehensive score"""
    if isinstance(aucs, list):
        return (sum(aucs) + avg_score) / 2
    else:
        return (aucs + avg_score) / 2

# Print utilities
def print_epoch_info(epoch, total_epochs, train_loss, test_loss, avg_score, aucs, dataset_name=''):
    """Print epoch information"""
    if isinstance(aucs, list) and len(aucs) > 1:
        if dataset_name.startswith('TCGA-lung'):
            info_msg = f'Epoch [{epoch}/{total_epochs}] train_loss: {train_loss:.4f} test_loss: {test_loss:.4f}, avg_score: {avg_score:.4f}, auc_LUAD: {aucs[0]:.4f}, auc_LUSC: {aucs[1]:.4f}'
        else:
            auc_str = '|'.join('class-{}>>{:.4f}'.format(i, auc) for i, auc in enumerate(aucs))
            info_msg = f'Epoch [{epoch}/{total_epochs}] train_loss: {train_loss:.4f} test_loss: {test_loss:.4f}, avg_score: {avg_score:.4f}, AUC: {auc_str}'
    else:
        auc_val = aucs[0] if isinstance(aucs, list) else aucs
        info_msg = f'Epoch [{epoch}/{total_epochs}] train_loss: {train_loss:.4f} test_loss: {test_loss:.4f}, avg_score: {avg_score:.4f}, auc: {auc_val:.4f}'
    
    logging.info(f'[Training] {info_msg}')

def print_final_results(results, training_time, dataset_name=''):
    """Print final training results"""
    if not results:
        logging.warning("No results to display - all folds may have been completed or errors occurred")
        return
        
    accuracies = [r[0] for r in results]
    aucs_list = [r[1] for r in results]
    
    mean_ac = np.mean(accuracies)
    std_ac = np.std(accuracies)
    
    logging.info("=" * 80)
    logging.info("FINAL RESULTS")
    logging.info("=" * 80)
    logging.info(f"Average accuracy: {mean_ac:.4f} ± {std_ac:.4f}")
    
    if isinstance(aucs_list[0], list):
        # Multi-class case
        mean_auc = np.mean(aucs_list, axis=0)
        std_auc = np.std(aucs_list, axis=0)
        for i, (mean_score, std_score) in enumerate(zip(mean_auc, std_auc)):
            logging.info(f"Class {i} average AUC: {mean_score:.4f} ± {std_score:.4f}")
    else:
        # Single class case
        mean_auc = np.mean(aucs_list)
        std_auc = np.std(aucs_list)
        logging.info(f"Average AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    logging.info(f"Total training time: {training_time:.2f}s ({training_time/3600:.2f}h)")
    logging.info(f"Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# WandB integration utilities
def init_wandb(project_name, config, run_name=None):
    """Initialize wandb for experiment tracking"""
    try:
        import wandb
        
        run = wandb.init(
            project=project_name,
            config=config,
            name=run_name,
            reinit=True
        )
        logging.info(f"WandB initialized for project: {project_name}")
        return run
    except ImportError:
        logging.warning("WandB not installed. Install with: pip install wandb")
        return None
    except Exception as e:
        logging.warning(f"Failed to initialize WandB: {e}")
        return None

def log_wandb_metrics(wandb_run, metrics, step=None):
    """Log metrics to wandb"""
    if wandb_run is not None:
        try:
            wandb_run.log(metrics, step=step)
        except Exception as e:
            logging.warning(f"Failed to log to WandB: {e}")

def log_wandb_model(wandb_run, model_path, model_name):
    """Log model artifact to wandb"""
    if wandb_run is not None:
        try:
            import wandb
            artifact = wandb.Artifact(model_name, type='model')
            artifact.add_file(model_path)
            wandb_run.log_artifact(artifact)
            logging.info(f"Model logged to WandB: {model_name}")
        except Exception as e:
            logging.warning(f"Failed to log model to WandB: {e}")

def finish_wandb(wandb_run):
    """Finish wandb run"""
    if wandb_run is not None:
        try:
            wandb_run.finish()
            logging.info("WandB run finished")
        except Exception as e:
            logging.warning(f"Failed to finish WandB run: {e}")

# Model parameter configuration utilities
def parse_model_params(args):
    """Parse model-specific parameters from command line arguments"""
    model_name = args.model
    params = {}
    
    if model_name == 'rmil_gru':
        params.update({
            'gru_hidden_size': getattr(args, 'gru_hidden_size', 768),
            'gru_num_layers': getattr(args, 'gru_num_layers', 2),
            'bidirectional': getattr(args, 'gru_bidirectional', 'True') == 'True',
            'selection_strategy': getattr(args, 'gru_selection_strategy', 'hybrid'),
            'big_lambda': getattr(args, 'big_lambda', 64),
            'dropout_v': getattr(args, 'dropout_v', 0.2),
        })
    elif model_name == 'rmil_lstm':
        params.update({
            'lstm_hidden_size': getattr(args, 'lstm_hidden_size', 256),
            'lstm_num_layers': getattr(args, 'lstm_num_layers', 2),
            'bidirectional': getattr(args, 'lstm_bidirectional', 'True') == 'True',
            'selection_strategy': getattr(args, 'lstm_selection_strategy', 'hybrid'),
            'big_lambda': getattr(args, 'big_lambda', 64),
            'dropout_v': getattr(args, 'dropout_v', 0.2),
        })
    elif model_name == 'rmil_mamba':
        params.update({
            'mamba_depth': getattr(args, 'mamba_depth', 4),
            'd_state': getattr(args, 'mamba_d_state', 32),
            'd_conv': getattr(args, 'mamba_d_conv', 4),
            'expand': getattr(args, 'mamba_expand', 2),
            'selection_strategy': getattr(args, 'mamba_selection_strategy', 'hybrid'),
            'big_lambda': getattr(args, 'big_lambda', 64),
            'dropout_v': getattr(args, 'dropout_v', 0.2),
        })
    elif model_name.startswith('rmil_ttt'):
        params.update({
            'num_heads': getattr(args, 'ttt_num_heads', 16),
            'num_layers': getattr(args, 'ttt_num_layers', 4),
            'mini_batch_size': getattr(args, 'ttt_mini_batch_size', 16),
            'ttt_base_lr': getattr(args, 'ttt_base_lr', 1.0),
            'selection_strategy': getattr(args, 'ttt_selection_strategy', 'hybrid'),
            'big_lambda': getattr(args, 'big_lambda', 64),
            'dropout_v': getattr(args, 'dropout_v', 0.2),
            'ttt_layer_type': model_name.split('_')[-1],
        })
    elif model_name == 'snuffy':
        params.update({
            'num_heads': 2,
            'big_lambda': getattr(args, 'big_lambda', 64),
            'dropout_v': getattr(args, 'dropout_v', 0.2),
        })
    
    return params

def log_model_info(model, model_name, params=None):
    """Log detailed model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model architecture: {model_name}")
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Estimated model size: {total_params * 4 / (1024*1024):.2f} MB")
    
    if params:
        logging.info("Model specific parameters:")
        for key, value in params.items():
            logging.info(f"   {key}: {value}")

# PT file generation utility
@log_performance("PT file generation")
def generate_pt_files_generic(args, df, get_bag_feats_func, temp_dir="temp_train"):
    """Generic PT file generation function"""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        logging.info(f'Created intermediate training directory: {temp_dir}')
        logging.info(f'Number of samples to process: {len(df)}')
        
        for i in range(len(df)):
            try:
                result = get_bag_feats_func(df.iloc[i], args)
                
                if len(result) == 5:  # Camelyon16 format with patch labels and positions
                    label, feats, feats_csv_path, feats_labels, positions = result
                elif len(result) == 3:  # Standard format
                    label, feats, feats_csv_path = result
                    feats_labels, positions = None, None
                else:
                    raise ValueError(f"Unexpected return format from get_bag_feats_func: {len(result)} elements")
                
                pt_file_path = os.path.join(temp_dir, os.path.splitext(os.path.basename(feats_csv_path))[0] + ".pt")
                
                bag_label = torch.tensor(np.array([label]), dtype=torch.float32)
                bag_feats = torch.tensor(np.array(feats), dtype=torch.float32)
                repeated_label = bag_label.repeat(bag_feats.size(0), 1)
                
                if feats_labels is not None and positions is not None:
                    # Camelyon16 format: [features, bag_label, patch_label, position_x, position_y]
                    positions_tensor = torch.tensor([[eval(pos)[0], eval(pos)[1]] if isinstance(pos, str) else pos for pos in positions], dtype=torch.float32)
                    feats_labels_tensor = torch.tensor(feats_labels, dtype=torch.float32).unsqueeze(1)
                    stacked_data = torch.cat((bag_feats, repeated_label, feats_labels_tensor, positions_tensor), dim=1)
                else:
                    # Standard format: [features, bag_label]
                    stacked_data = torch.cat((bag_feats, repeated_label), dim=1)
                
                torch.save(stacked_data, pt_file_path)
                
                if (i + 1) % 100 == 0:
                    logging.info(f'Processed {i + 1}/{len(df)} samples')
                    
            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                continue
        
        logging.info(f'PT file generation completed, generated {len(df)} files')
    else:
        logging.info('Intermediate training files already exist, skipping generation') 