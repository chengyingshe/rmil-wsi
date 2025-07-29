import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score, hamming_loss
from sklearn.model_selection import KFold
from collections import OrderedDict
from tqdm import tqdm
import logging
import time
import random
import common_utils as mil
import rmil_common

# Import common utilities
from common_utils import (
    setup_logging, log_performance, set_random_seed, apply_sparse_init,
    multi_label_roc, optimal_thresh, save_model_checkpoint, load_checkpoint,
    get_current_score, print_epoch_info, print_final_results,
    init_wandb, log_wandb_metrics, log_wandb_model, finish_wandb,
    parse_model_params, log_model_info
)
from data import PatchFeaturesDataset


def train(args, train_loader, milnet, criterion, optimizer, wandb_run=None, epoch=None):
    milnet.train()
    total_loss = 0
    device = torch.device(args.device)
    
    epoch_start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # Extract features and labels from batch
        patch_features = batch['patch_features'].to(device)
        labels = batch['label'].to(device)
        
        # Process each bag in the batch (typically batch size is 1 for MIL)
        for j in range(patch_features.size(0)):
            bag_feats = patch_features[j]
            
            # Convert single label to proper format for BCEWithLogitsLoss
            if args.num_classes == 1:
                bag_label = labels[j].float().unsqueeze(0).unsqueeze(0)
            else:
                bag_label = torch.zeros(1, args.num_classes, device=device)
                if labels[j] < args.num_classes:
                    bag_label[0, labels[j]] = 1
            
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)        
            
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label)
            max_loss = criterion(max_prediction.view(1, -1), bag_label)
            loss = 0.5*bag_loss + 0.5*max_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_loader), loss.item()))
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / len(train_loader)
    
    logging.info(f'Training completed - Average loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s, Training samples: {len(train_loader.dataset)}')
    
    # Log to WandB if available
    if wandb_run and epoch is not None:
        log_wandb_metrics(wandb_run, {
            'train_loss': avg_loss,
            'train_time': epoch_time,
            'epoch': epoch
        })
    
    return avg_loss

def test(args, test_loader, milnet, criterion, thresholds=None, return_predictions=False, wandb_run=None, epoch=None):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    device = torch.device(args.device)
    
    test_start_time = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Extract features and labels from batch
            patch_features = batch['patch_features'].to(device)
            labels = batch['label'].to(device)
            
            # Process each bag in the batch (typically batch size is 1 for MIL)
            for j in range(patch_features.size(0)):
                bag_feats = patch_features[j]
                
                # Convert single label to proper format for BCEWithLogitsLoss
                if args.num_classes == 1:
                    bag_label = labels[j].float().unsqueeze(0).unsqueeze(0)
                else:
                    bag_label = torch.zeros(1, args.num_classes, device=device)
                    if labels[j] < args.num_classes:
                        bag_label[0, labels[j]] = 1
                
                bag_feats = bag_feats.view(-1, args.feats_size)
                
                ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
                max_prediction, _ = torch.max(ins_prediction, 0)  
                
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label)
                max_loss = criterion(max_prediction.view(1, -1), bag_label)
                loss = 0.5*bag_loss + 0.5*max_loss
                total_loss += loss.item()
                
                # Store predictions and labels
                test_labels.append(bag_label.squeeze().cpu().numpy().astype(int))
                if args.average:
                    test_predictions.append((torch.sigmoid(max_prediction)+torch.sigmoid(bag_prediction)).squeeze().cpu().numpy())
                else: 
                    test_predictions.append(torch.sigmoid(bag_prediction).squeeze().cpu().numpy())
            
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_loader), loss.item()))
    
    test_time = time.time() - test_start_time
    avg_loss = total_loss / len(test_loader)
    
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if thresholds: thresholds_optimal = thresholds
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_labels)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_labels)
    
    logging.info(f'Testing completed - Average loss: {avg_loss:.4f}, Accuracy: {avg_score:.4f}, AUC: {auc_value}, Time: {test_time:.2f}s, Test samples: {len(test_loader.dataset)}')
    
    # Log to WandB if available
    if wandb_run and epoch is not None:
        log_wandb_metrics(wandb_run, {
            'test_loss': avg_loss,
            'test_acc': avg_score,
            'test_auc': auc_value[0] if isinstance(auc_value, list) else auc_value,
            'test_time': test_time,
            'epoch': epoch
        })
    
    if return_predictions:
        return avg_loss, avg_score, auc_value, thresholds_optimal, test_predictions, test_labels
    return avg_loss, avg_score, auc_value, thresholds_optimal

def main():
    parser = argparse.ArgumentParser(description='Train model on dataset')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size [1]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers [4]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [100]')
    parser.add_argument('--stop_epochs', default=5, type=int, help='Skip remaining epochs if training has not improved after N epochs [5]')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device [cuda:0]')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [1e-3]')
    parser.add_argument('--dataset_dir', default='datasets/mydatasets/TCGA-lung', type=str, help='Dataset folder name')
    parser.add_argument('--label_file', default='TCGA.csv', type=str, help='CSV label file name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    
    parser.add_argument('--model', default='rmil_gru', type=str, help='MIL model [dsmil]', choices=['dsmil', 'snuffy', 'abmil', 'rmil_lstm', 'rmil_gru', 'rmil_mamba', 'rmil_ttt_linear', 'rmil_ttt_mlp'])
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=False, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--eval_scheme', default='5-fold-cv', type=str, help='Evaluation scheme [5-fold-cv | 5-fold-cv-standalone-test | 5-time-train+valid+test ]')
    parser.add_argument('--log_file', default='./log.txt', type=str, help='Log file name')
    parser.add_argument('--weights_save_root', default='./weights', type=str, help='Weights save root')
    
    parser.add_argument('--random_seed', default=42, type=int, help='Random seed for reproducible data splits [42]')
    
    # Model parameters
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
    
    # WandB parameters
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB for experiment tracking')
    parser.add_argument('--wandb_project', default='rmil-tcga', type=str, help='WandB project name')
    parser.add_argument('--wandb_run_name', default=None, type=str, help='WandB run name')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file=args.log_file)
    
    # Set random seed
    set_random_seed(args.random_seed)
    
    # Process JSON parameter configuration
    if args.params:
        try:
            import json
            if os.path.isfile(args.params):
                with open(args.params, 'r') as f:
                    params_dict = json.load(f)
                    params_dict['model'] = args.model
            else:
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
    
    # Initialize WandB if requested - will be initialized per fold
    wandb_run = None
    if args.use_wandb and not args.wandb_run_name:
        args.wandb_run_name = args.model
    
    
    # Parse model parameters
    model_params = parse_model_params(args)
    logging.info(f"Model: {args.model}")
    logging.info(f"Model parameters: {model_params}")
    
    def init_model(args):
        model_start_time = time.time()
        device = torch.device(args.device)  
        
        i_classifier = rmil_common.FCLayer(in_size=args.feats_size, out_size=args.num_classes).to(device)
        
        # Initialize BClassifier based on model type
        if args.model.startswith('rmil_ttt'):
            from rmil_ttt import BClassifier
            ttt_params = parse_model_params(args)
            ttt_params['ttt_layer_type'] = args.model.split('_')[-1]
            b_classifier = BClassifier(input_size=args.feats_size, 
                                          output_class=args.num_classes, 
                                          **ttt_params).to(device)
        elif args.model == 'rmil_mamba':
            from rmil_mamba import BClassifier
            mamba_params = parse_model_params(args)
            b_classifier = BClassifier(input_size=args.feats_size, 
                                          output_class=args.num_classes, 
                                          **mamba_params).to(device)
        elif args.model == 'rmil_gru':
            from rmil_gru import BClassifier
            gru_params = parse_model_params(args)
            b_classifier = BClassifier(input_size=args.feats_size, 
                                          output_class=args.num_classes, 
                                          **gru_params).to(device)
        elif args.model == 'rmil_lstm':
            from rmil_lstm import BClassifier
            lstm_params = parse_model_params(args)
            b_classifier = BClassifier(input_size=args.feats_size, 
                                          output_class=args.num_classes, 
                                          **lstm_params).to(device)
        elif args.model == 'dsmil':
            from dsmil import BClassifier
            dsmil_params = parse_model_params(args)
            b_classifier = BClassifier(input_size=args.feats_size, 
                                          output_class=args.num_classes, 
                                          **dsmil_params).to(device)    
        else:
            raise ValueError(f"Model {args.model} not supported")
        
        milnet = rmil_common.MILNet(i_classifier, b_classifier).to(device)
        
        # Apply sparse initialization
        milnet.apply(lambda m: apply_sparse_init(m))
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
        
        model_time = time.time() - model_start_time
        log_model_info(milnet, model_time)
        
        return milnet, criterion, optimizer, scheduler
    
    # Dataset setup
    patch_features_dir = os.path.join(args.dataset_dir, 'pt_files')
    label_file = os.path.join(args.dataset_dir, args.label_file)
    dataset = PatchFeaturesDataset(patch_features_dir, label_file)
    
    logging.info(f"Dataset loaded with {len(dataset)} samples")

    save_path = os.path.join(args.weights_save_root, f'{args.model}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(save_path, exist_ok=True)
    logging.info(f"New training, model save path: {save_path}")

    if args.eval_scheme == '5-fold-cv':
        logging.info(f"5-fold cross-validation - Total samples: {len(dataset)}")
        
        # Use fixed random seed for consistent data splits
        kf = KFold(n_splits=5, shuffle=True, random_state=args.random_seed)
        fold_results = []

        training_start_time = time.time()
        
        # Create indices for cross-validation
        dataset_indices = list(range(len(dataset)))
        
        for fold, (train_index, test_index) in enumerate(kf.split(dataset_indices)):
            fold_start_time = time.time()
            
            # Initialize WandB for this fold
            fold_wandb_run = None
            if args.use_wandb:
                fold_run_name = f"{args.wandb_run_name}_fold_{fold + 1}"
                fold_wandb_run = init_wandb(args.wandb_project, args.__dict__, fold_run_name)
                logging.info(f"Initialized WandB run for fold {fold + 1}: {fold_run_name}")
            
            logging.info(f"Starting fold {fold + 1}/5 cross-validation")
            
            milnet, criterion, optimizer, scheduler = init_model(args)
            
            # Create subset datasets for this fold
            train_dataset = torch.utils.data.Subset(dataset, train_index)
            test_dataset = torch.utils.data.Subset(dataset, test_index)
            
            # Create data loaders - batch size 1 for MIL (each sample is a bag)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            
            logging.info(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")
            
            # Initialize training state
            fold_best_score = 0
            best_ac = 0
            best_auc = 0
            counter = 0
            start_epoch = 1

            for epoch in range(start_epoch, args.num_epochs+1):
                epoch_start_time = time.time()
                counter += 1
                
                train_loss_bag = train(args, train_loader, milnet, criterion, optimizer, fold_wandb_run, epoch)
                test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, test_loader, milnet, criterion, wandb_run=fold_wandb_run, epoch=epoch)
                
                print_epoch_info(epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs)
                scheduler.step()

                current_score = get_current_score(avg_score, aucs)
                epoch_time = time.time() - epoch_start_time
                
                logging.info(f"Epoch {epoch} total time: {epoch_time:.2f}s, Current score: {current_score:.4f}")
                
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = avg_score
                    best_auc = aucs
                    save_model_checkpoint(args, fold, save_path, milnet, epoch, avg_score, aucs, thresholds_optimal, 'best')
                    logging.info(f"Fold {fold + 1} best model updated! Score: {current_score:.4f}")
                    
                # Save last model every epoch (for resuming training)
                save_model_checkpoint(args, fold, save_path, milnet, epoch, avg_score, aucs, thresholds_optimal, 'last')
                    
                if counter > args.stop_epochs: 
                    logging.info(f"Early stopping triggered - No improvement for {args.stop_epochs} epochs")
                    break
                    
            fold_time = time.time() - fold_start_time
            fold_results.append((best_ac, best_auc))
            logging.info(f"Fold {fold + 1} completed - Best accuracy: {best_ac:.4f}, Best AUC: {best_auc}, Time: {fold_time:.2f}s")
            if fold_wandb_run:
                finish_wandb(fold_wandb_run)

        # Final results statistics
        if fold_results:
            print_final_results(fold_results, time.time() - training_start_time)
        else:
            logging.warning("No fold produced results, possibly all folds completed or errors occurred")

    elif args.eval_scheme == '5-time-train+valid+test':
        logging.info(f"5-time train+valid+test - Total samples: {len(dataset)}")
        
        fold_results = []
        training_start_time = time.time()

        for iteration in range(5):
            iteration_start_time = time.time()
            
            logging.info(f"Starting iteration {iteration + 1}/5")
            
            milnet, criterion, optimizer, scheduler = init_model(args)

            # Create shuffled indices for this iteration
            dataset_indices = list(range(len(dataset)))
            random.seed(args.random_seed + iteration)
            random.shuffle(dataset_indices)
            
            total_samples = len(dataset_indices)
            train_end = int(total_samples * (1-args.split-0.1))
            val_end = train_end + int(total_samples * 0.1)

            train_indices = dataset_indices[:train_end]
            val_indices = dataset_indices[train_end:val_end]
            test_indices = dataset_indices[val_end:]
            
            # Create subset datasets
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
            logging.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

            # Initialize training state
            fold_best_score = 0
            best_ac = 0
            best_auc = 0
            counter = 0
            start_epoch = 1
            best_model = None

            for epoch in range(start_epoch, args.num_epochs + 1):
                epoch_start_time = time.time()
                counter += 1
                
                train_loss_bag = train(args, train_loader, milnet, criterion, optimizer, wandb_run, epoch)
                test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, val_loader, milnet, criterion, wandb_run=wandb_run, epoch=epoch)
                
                print_epoch_info(epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs)
                scheduler.step()

                current_score = get_current_score(avg_score, aucs)
                epoch_time = time.time() - epoch_start_time
                
                logging.info(f"Epoch {epoch} total time: {epoch_time:.2f}s, Current score: {current_score:.4f}")
                
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = avg_score
                    best_auc = aucs
                    save_model_checkpoint(args, iteration, save_path, milnet, epoch, avg_score, aucs, thresholds_optimal, 'best')
                    best_model = copy.deepcopy(milnet.cpu())
                    milnet.cuda()
                    logging.info(f"Iteration {iteration + 1} best model updated! Score: {current_score:.4f}")
                    
                # Save last model every epoch (for resuming training)
                save_model_checkpoint(args, iteration, save_path, milnet, epoch, avg_score, aucs, thresholds_optimal, 'last')
                    
                if counter > args.stop_epochs: 
                    logging.info(f"Early stopping triggered - No improvement for {args.stop_epochs} epochs")
                    break
                    
            # Evaluate on test set
            if best_model is not None:
                logging.info(f"Evaluating iteration {iteration + 1} best model on test set")
                test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, test_loader, best_model.cuda(), criterion)
            
            iteration_time = time.time() - iteration_start_time
            fold_results.append((best_ac, best_auc))
            logging.info(f"Iteration {iteration + 1} completed - Validation accuracy: {best_ac:.4f}, Test accuracy: {avg_score:.4f}, Time: {iteration_time:.2f}s")

        # Final results statistics
        if fold_results:
            print_final_results(fold_results, time.time() - training_start_time)
        else:
            logging.warning("No iteration produced results, possibly all iterations completed or errors occurred")
        
    elif args.eval_scheme == '5-fold-cv-standalone-test':
        # Create shuffled indices for consistent splits
        dataset_indices = list(range(len(dataset)))
        random.seed(args.random_seed)
        random.shuffle(dataset_indices)
        
        # Reserve test set
        test_split_size = int(args.split * len(dataset_indices))
        reserved_testing_indices = dataset_indices[:test_split_size]
        training_indices = dataset_indices[test_split_size:]
        
        logging.info(f"5-fold cross-validation + standalone test set")
        logging.info(f"Training+validation samples: {len(training_indices)}, Independent test samples: {len(reserved_testing_indices)}")
        
        # Use fixed random seed for consistent data splits
        kf = KFold(n_splits=5, shuffle=True, random_state=args.random_seed)
        fold_results = []
        fold_models = []
        
        training_start_time = time.time()

        for fold, (train_index, test_index) in enumerate(kf.split(training_indices)):
            fold_start_time = time.time()
            
            logging.info(f"Starting fold {fold + 1}/5 cross-validation")
            
            milnet, criterion, optimizer, scheduler = init_model(args)
            
            # Create actual indices for this fold
            train_indices = [training_indices[i] for i in train_index]
            val_indices = [training_indices[i] for i in test_index]
            
            # Create subset datasets
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
            logging.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
            
            # Initialize training state
            fold_best_score = 0
            best_ac = 0
            best_auc = 0
            counter = 0
            start_epoch = 1
            best_model = []

            for epoch in range(start_epoch, args.num_epochs+1):
                epoch_start_time = time.time()
                counter += 1
                
                train_loss_bag = train(args, train_loader, milnet, criterion, optimizer, wandb_run, epoch)
                test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, val_loader, milnet, criterion, wandb_run=wandb_run, epoch=epoch)
                
                print_epoch_info(epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs)
                scheduler.step()

                current_score = get_current_score(avg_score, aucs)
                epoch_time = time.time() - epoch_start_time
                
                logging.info(f"Epoch {epoch} total time: {epoch_time:.2f}s, Current score: {current_score:.4f}")
                
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = avg_score
                    best_auc = aucs
                    save_model_checkpoint(args, fold, save_path, milnet, epoch, avg_score, aucs, thresholds_optimal, 'best')
                    best_model = [copy.deepcopy(milnet.cpu()), thresholds_optimal]
                    milnet.cuda()
                    logging.info(f"Fold {fold + 1} best model updated! Score: {current_score:.4f}")
                    
                # Save last model every epoch (for resuming training)
                save_model_checkpoint(args, fold, save_path, milnet, epoch, avg_score, aucs, thresholds_optimal, 'last')
                    
                if counter > args.stop_epochs: 
                    logging.info(f"Early stopping triggered - No improvement for {args.stop_epochs} epochs")
                    break
                    
            fold_time = time.time() - fold_start_time
            fold_results.append((best_ac, best_auc))
            if best_model:
                fold_models.append(best_model)
            logging.info(f"Fold {fold + 1} completed - Best accuracy: {best_ac:.4f}, Best AUC: {best_auc}, Time: {fold_time:.2f}s")

        # Ensemble prediction on independent test set
        if fold_models:
            logging.info("Starting ensemble prediction on independent test set")
            
            # Create test dataset and loader
            test_dataset = torch.utils.data.Subset(dataset, reserved_testing_indices)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
            fold_predictions = []
            for i, item in enumerate(fold_models):
                best_model = item[0]
                optimal_thresh = item[1]
                logging.info(f"Using model {i + 1} for prediction")
                test_loss_bag, avg_score, aucs, thresholds_optimal, test_predictions, test_labels = test(args, test_loader, best_model.cuda(), criterion, thresholds=optimal_thresh, return_predictions=True)
                fold_predictions.append(test_predictions)
                
            predictions_stack = np.stack(fold_predictions, axis=0)
            mode_result = mode(predictions_stack, axis=0)
            combined_predictions = mode_result.mode[0]
            combined_predictions = combined_predictions.squeeze()

            # Ensemble prediction results
            logging.info("Independent test set ensemble prediction results:")
            if args.num_classes > 1:
                hammingloss = hamming_loss(test_labels, combined_predictions)
                subset_accuracy = accuracy_score(test_labels, combined_predictions)
                logging.info(f"Hamming Loss: {hammingloss:.4f}")
                logging.info(f"Subset Accuracy (exact match rate): {subset_accuracy:.4f}")
            else:
                accuracy = accuracy_score(test_labels, combined_predictions)
                balanced_accuracy = balanced_accuracy_score(test_labels, combined_predictions)
                logging.info(f"Accuracy: {accuracy:.4f}")
                logging.info(f"Balanced Accuracy: {balanced_accuracy:.4f}")

        # Final results statistics
        if fold_results:
            print_final_results(fold_results, time.time() - training_start_time)

            # Save test files and models
            os.makedirs('test', exist_ok=True)
            test_files = [dataset.patch_features_files[i] for i in reserved_testing_indices]
            with open("test/test_list.json", "w") as file:
                json.dump(test_files, file)
            logging.info(f"Test sample list saved: test/test_list.json ({len(test_files)} samples)")

            for i, item in enumerate(fold_models):
                best_model = item[0]
                optimal_thresh = item[1]
                model_path = f"test/mil_weights_fold_{i}.pth"
                thresh_path = f"test/mil_threshold_fold_{i}.json"
                
                torch.save(best_model.state_dict(), model_path)
                with open(thresh_path, "w") as file:
                    optimal_thresh = [float(i) for i in optimal_thresh]
                    json.dump(optimal_thresh, file)
                
                logging.info(f"Model {i + 1} saved: {model_path}")
                logging.info(f"Threshold {i + 1} saved: {thresh_path}")
        else:
            logging.warning("No fold produced results, possibly all folds completed or errors occurred")

    logging.info("=" * 80)
    logging.info("Training completed!")
    logging.info("=" * 80)
    
    # Finish WandB run
    if wandb_run:
        finish_wandb(wandb_run)

if __name__ == '__main__':
    main()