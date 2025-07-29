import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, datetime, time
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
import logging
import json

# Import common utilities
from common_utils import (
    setup_logging, log_performance, set_random_seed, apply_sparse_init,
    multi_label_roc, optimal_thresh, save_model_checkpoint, load_checkpoint,
    dropout_patches, get_current_score, print_epoch_info, print_final_results,
    init_wandb, log_wandb_metrics, log_wandb_model, finish_wandb,
    parse_model_params, log_model_info
)


def get_data(file_path):
    df = pd.read_csv(file_path)
    df = pd.DataFrame(df)
    df = df[df.columns[0]]
    data_list = []    
    for i in range(0, df.shape[0]):  
        data = str(df.iloc[i]).split(' ')
        ids = data[0].split(':')
        idi = int(ids[0])
        idb = int(ids[1])
        idc = int(ids[2])
        data = data[1:]
        feature_vector = np.zeros(len(data))  
        for i, feature in enumerate(data):
            feature_data = feature.split(':')
            if len(feature_data) == 2:
                feature_vector[i] = feature_data[1]
        data_list.append([idi, idb, idc, feature_vector])
    return data_list

def get_bag(data, idb):
    data_array = np.array(data, dtype=object)
    bag_id = data_array[:, 1]
    return data_array[np.where(bag_id == idb)]

@log_performance("Training one epoch")
def epoch_train(bag_ins_list, optimizer, criterion, milnet, args, wandb_run=None, epoch=None):
    milnet.train()
    epoch_loss = 0
    train_start_time = time.time()
    device = torch.device(args.device)
    
    for i, data in enumerate(bag_ins_list):
        optimizer.zero_grad()
        data_bag_list = shuffle(data[1])
        data_tensor = torch.from_numpy(np.stack(data_bag_list)).float().to(device)
        data_tensor = data_tensor[:, 0:args.num_feats]
        label = torch.from_numpy(np.array(int(np.clip(data[0], 0, 1)))).float().to(device)
        classes, bag_prediction, _, _ = milnet(data_tensor) # n X L
        max_prediction, index = torch.max(classes, 0)
        loss_bag = criterion(bag_prediction.view(1, -1), label.view(1, -1))
        loss_max = criterion(max_prediction.view(1, -1), label.view(1, -1))
        loss_total = 0.5*loss_bag + 0.5*loss_max
        loss_total = loss_total.mean()
        loss_total.backward()
        optimizer.step()  
        epoch_loss = epoch_loss + loss_total.item()
        
        # Log progress every 10 bags
        if (i + 1) % 10 == 0:
            sys.stdout.write('\r Training bag [%d/%d] avg loss: %.4f' % (i+1, len(bag_ins_list), epoch_loss/(i+1)))
    
    train_time = time.time() - train_start_time
    avg_loss = epoch_loss / len(bag_ins_list)
    logging.info(f'Training completed - Average loss: {avg_loss:.4f}, Time: {train_time:.2f}s, Training bags: {len(bag_ins_list)}')
    
    # Log to wandb
    if wandb_run and epoch is not None:
        log_wandb_metrics(wandb_run, {
            'train_loss': avg_loss,
            'train_time': train_time,
            'epoch': epoch
        })
    
    return avg_loss

@log_performance("Testing one epoch")
def epoch_test(bag_ins_list, criterion, milnet, args, wandb_run=None, epoch=None):
    milnet.eval()
    bag_labels = []
    bag_predictions = []
    epoch_loss = 0
    test_start_time = time.time()
    device = torch.device(args.device)
    
    with torch.no_grad():
        for i, data in enumerate(bag_ins_list):
            bag_labels.append(np.clip(data[0], 0, 1))
            data_tensor = torch.from_numpy(np.stack(data[1])).float().to(device)
            data_tensor = data_tensor[:, 0:args.num_feats]
            label = torch.from_numpy(np.array(int(np.clip(data[0], 0, 1)))).float().to(device)
            classes, bag_prediction, _, _ = milnet(data_tensor) # n X L
            max_prediction, index = torch.max(classes, 0)
            loss_bag = criterion(bag_prediction.view(1, -1), label.view(1, -1))
            loss_max = criterion(max_prediction.view(1, -1), label.view(1, -1))
            loss_total = 0.5*loss_bag + 0.5*loss_max
            loss_total = loss_total.mean()
            bag_predictions.append(torch.sigmoid(bag_prediction).cpu().squeeze().numpy())
            epoch_loss = epoch_loss + loss_total.item()
            
            # Log progress every 10 bags
            if (i + 1) % 10 == 0:
                sys.stdout.write('\r Testing bag [%d/%d] avg loss: %.4f' % (i+1, len(bag_ins_list), epoch_loss/(i+1)))
    
    test_time = time.time() - test_start_time
    epoch_loss = epoch_loss / len(bag_ins_list)
    
    # Calculate test metrics
    accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_predictions)
    
    logging.info(f'Testing completed - Average loss: {epoch_loss:.4f}, Time: {test_time:.2f}s, Testing bags: {len(bag_ins_list)}')
    logging.info(f'Test metrics - Accuracy: {accuracy:.4f}, AUC: {auc_value:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fscore:.4f}')
    
    # Log to wandb
    if wandb_run and epoch is not None:
        log_wandb_metrics(wandb_run, {
            'test_loss': epoch_loss,
            'test_time': test_time,
            'test_acc': accuracy,
            'test_auc': auc_value,
            'epoch': epoch
        })
    
    return epoch_loss, bag_labels, bag_predictions

def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore

def cross_validation_set(in_list, fold, index):
    csv_list = copy.deepcopy(in_list)
    n = int(len(csv_list)/fold)
    chunked = [csv_list[i:i+n] for i in range(0, len(csv_list), n)]
    test_list = chunked.pop(index)
    return list(itertools.chain.from_iterable(chunked)), test_list

def compute_pos_weight(bags_list):
    pos_count = 0
    for item in bags_list:
        pos_count = pos_count + np.clip(item[0], 0, 1)
    return (len(bags_list)-pos_count)/pos_count

def init_model(args, mil):
    """Initialize model"""
    model_start_time = time.time()
    
    i_classifier = mil.FCLayer(args.num_feats, 1)
    
    # Parse model-specific parameters
    model_params = parse_model_params(args)
    
    # Initialize BClassifier based on model type
    if args.model == 'rmil_gru':
        b_classifier = mil.BClassifier(input_size=args.num_feats, 
                                        output_class=1,
                                        **model_params)
    elif args.model == 'rmil_lstm':
        b_classifier = mil.BClassifier(input_size=args.num_feats, 
                                        output_class=1,
                                        **model_params)
    elif args.model == 'rmil_mamba':
        b_classifier = mil.BClassifier(input_size=args.num_feats, 
                                        output_class=1,
                                        **model_params)
    elif args.model.startswith('rmil_ttt'):
        b_classifier = mil.BClassifier(input_size=args.num_feats, 
                                        output_class=1,
                                        **model_params)
    elif args.model == 'snuffy':
        b_classifier = mil.BClassifier(input_size=args.num_feats, 
                                        output_class=1,
                                        **model_params)
    else:
        # Other models use default parameters
        b_classifier = mil.BClassifier(input_size=args.num_feats, output_class=1)
    device = torch.device(args.device)
    milnet = mil.MILNet(i_classifier, b_classifier).to(device)
    milnet.apply(lambda m: apply_sparse_init(m))
    
    pos_weight = torch.tensor(1.0)  # Default weight, will be recalculated during training
    criterion = nn.BCEWithLogitsLoss(pos_weight)
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    
    model_time = time.time() - model_start_time
    
    logging.info(f"Model initialization completed (Time: {model_time:.2f}s)")
    log_model_info(milnet, args.model, model_params)
    
    # Output detailed model architecture parameters
    logging.info(f"Model architecture parameters:")
    logging.info(f"   Model type: {args.model}")
    logging.info(f"   Dataset: {args.datasets}")
    logging.info(f"   Feature dimension: {args.num_feats}")
    
    # Get BClassifier parameters
    if hasattr(b_classifier, '__dict__'):
        for attr_name in dir(b_classifier):
            if not attr_name.startswith('_') and not callable(getattr(b_classifier, attr_name)):
                try:
                    attr_value = getattr(b_classifier, attr_name)
                    if isinstance(attr_value, (int, float, str, bool)):
                        logging.info(f"   {attr_name}: {attr_value}")
                except:
                    pass
    
    # Output specific parameters based on model type
    try:
        if args.model == 'rmil_gru':
            if hasattr(b_classifier, 'gru_hidden_size'):
                logging.info(f"   gru_hidden_size: {getattr(b_classifier, 'gru_hidden_size', 'Not found')}")
            if hasattr(b_classifier, 'gru_num_layers'):
                logging.info(f"   gru_num_layers: {getattr(b_classifier, 'gru_num_layers', 'Not found')}")
            if hasattr(b_classifier, 'bidirectional'):
                logging.info(f"   bidirectional: {getattr(b_classifier, 'bidirectional', 'Not found')}")
            if hasattr(b_classifier, 'big_lambda'):
                logging.info(f"   big_lambda: {getattr(b_classifier, 'big_lambda', 'Not found')}")
            if hasattr(b_classifier, 'dropout_v'):
                logging.info(f"   dropout_v: {getattr(b_classifier, 'dropout_v', 'Not found')}")
                
        elif args.model == 'rmil_lstm':
            if hasattr(b_classifier, 'lstm_hidden_size'):
                logging.info(f"   lstm_hidden_size: {getattr(b_classifier, 'lstm_hidden_size', 'Not found')}")
            if hasattr(b_classifier, 'lstm_num_layers'):
                logging.info(f"   lstm_num_layers: {getattr(b_classifier, 'lstm_num_layers', 'Not found')}")
            if hasattr(b_classifier, 'bidirectional'):
                logging.info(f"   bidirectional: {getattr(b_classifier, 'bidirectional', 'Not found')}")
            if hasattr(b_classifier, 'big_lambda'):
                logging.info(f"   big_lambda: {getattr(b_classifier, 'big_lambda', 'Not found')}")
            if hasattr(b_classifier, 'dropout_v'):
                logging.info(f"   dropout_v: {getattr(b_classifier, 'dropout_v', 'Not found')}")
                
        elif args.model == 'rmil_mamba':
            if hasattr(b_classifier, 'mamba_depth'):
                logging.info(f"   mamba_depth: {getattr(b_classifier, 'mamba_depth', 'Not found')}")
            if hasattr(b_classifier, 'd_state'):
                logging.info(f"   d_state: {getattr(b_classifier, 'd_state', 'Not found')}")
            if hasattr(b_classifier, 'd_conv'):
                logging.info(f"   d_conv: {getattr(b_classifier, 'd_conv', 'Not found')}")
            if hasattr(b_classifier, 'expand'):
                logging.info(f"   expand: {getattr(b_classifier, 'expand', 'Not found')}")
            if hasattr(b_classifier, 'big_lambda'):
                logging.info(f"   big_lambda: {getattr(b_classifier, 'big_lambda', 'Not found')}")
            if hasattr(b_classifier, 'dropout_v'):
                logging.info(f"   dropout_v: {getattr(b_classifier, 'dropout_v', 'Not found')}")
                
        elif args.model.startswith('rmil_ttt'):
            if hasattr(b_classifier, 'num_heads'):
                logging.info(f"   num_heads: {getattr(b_classifier, 'num_heads', 'Not found')}")
            if hasattr(b_classifier, 'num_layers'):
                logging.info(f"   num_layers: {getattr(b_classifier, 'num_layers', 'Not found')}")
            if hasattr(b_classifier, 'mini_batch_size'):
                logging.info(f"   mini_batch_size: {getattr(b_classifier, 'mini_batch_size', 'Not found')}")
            if hasattr(b_classifier, 'ttt_base_lr'):
                logging.info(f"   ttt_base_lr: {getattr(b_classifier, 'ttt_base_lr', 'Not found')}")
            if hasattr(b_classifier, 'ttt_layer_type'):
                logging.info(f"   ttt_layer_type: {getattr(b_classifier, 'ttt_layer_type', args.model.split('_')[-1])}")
            if hasattr(b_classifier, 'big_lambda'):
                logging.info(f"   big_lambda: {getattr(b_classifier, 'big_lambda', 'Not found')}")
            if hasattr(b_classifier, 'dropout_v'):
                logging.info(f"   dropout_v: {getattr(b_classifier, 'dropout_v', 'Not found')}")
                
        elif args.model == 'snuffy':
            if hasattr(b_classifier, 'num_heads'):
                logging.info(f"   num_heads: {getattr(b_classifier, 'num_heads', 2)}")
            if hasattr(b_classifier, 'big_lambda'):
                logging.info(f"   big_lambda: {getattr(b_classifier, 'big_lambda', 'Not found')}")
            if hasattr(b_classifier, 'dropout_v'):
                logging.info(f"   dropout_v: {getattr(b_classifier, 'dropout_v', 'Not found')}")
    except Exception as e:
        logging.warning(f"Error getting model parameters: {e}")
    
    return milnet, criterion, optimizer, scheduler

def save_model(args, fold, model, epoch, accuracy, auc, precision, recall, fscore, save_path, model_type='best'):
    """Save model and training metrics"""
    # Construct save filename: fold_X_type.pth
    save_name = os.path.join(save_path, f'fold_{fold}_{model_type}.pth')
    
    # Build complete model checkpoint with model weights and training metrics
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
        'fold': fold,
        'model_type': model_type,
        'dataset': args.datasets,
        'model_name': args.model,
        'num_feats': args.num_feats
    }
    
    torch.save(checkpoint, save_name)
    
    model_size = os.path.getsize(save_name) / (1024 * 1024)  # MB
    logging.info(f'Model saved: {save_name} (Type: {model_type}, Size: {model_size:.2f}MB)')
    logging.info(f'Model metrics: Epoch={epoch}, Accuracy={accuracy:.4f}, AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={fscore:.4f}')
    
    return save_name

def main():
    parser = argparse.ArgumentParser(description='Train model on classical MIL datasets')
    parser.add_argument('--datasets', default='musk1', type=str, help='Choose MIL datasets from: musk1, musk2, elephant, fox, tiger [musk1]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [100]')
    parser.add_argument('--cv_fold', default=5, type=int, help='Number of cross validation fold [5]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]', choices=['dsmil', 'snuffy', 'abmil', 'rmil_lstm', 'rmil_gru', 'rmil_mamba', 'rmil_ttt_linear', 'rmil_ttt_mlp'])
    parser.add_argument('--early_stopping', default=5, type=int, help='Early stopping patience [5]')
    parser.add_argument('--log_file', default='./log_mil.txt', type=str, help='Log file name')
    parser.add_argument('--weights_save_root', default='./weights/mil', type=str, help='Weights save root')
    
    # Add model parameter support
    # GRU parameters
    parser.add_argument('--gru_hidden_size', type=int, default=768, help='GRU hidden size')
    parser.add_argument('--gru_num_layers', type=int, default=2, help='GRU number of layers')
    parser.add_argument('--gru_bidirectional', type=str, choices=['True', 'False'], default='True', help='GRU bidirectional')
    parser.add_argument('--gru_selection_strategy', type=str, choices=['hybrid', 'original'], default='hybrid', help='GRU selection strategy')
    
    # LSTM parameters
    parser.add_argument('--lstm_hidden_size', type=int, default=768, help='LSTM hidden size')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='LSTM number of layers')
    parser.add_argument('--lstm_bidirectional', type=str, choices=['True', 'False'], default='True', help='LSTM bidirectional')
    parser.add_argument('--lstm_selection_strategy', type=str, choices=['hybrid', 'original'], default='hybrid', help='LSTM selection strategy')
    
    # Mamba parameters
    parser.add_argument('--mamba_depth', type=int, default=8, help='Mamba depth')
    parser.add_argument('--mamba_d_state', type=int, default=32, help='Mamba d_state')
    parser.add_argument('--mamba_d_conv', type=int, default=4, help='Mamba d_conv')
    parser.add_argument('--mamba_expand', type=int, default=2, help='Mamba expand')
    parser.add_argument('--mamba_selection_strategy', type=str, choices=['hybrid', 'original'], default='hybrid', help='Mamba selection strategy')
    
    # TTT parameters
    parser.add_argument('--ttt_num_heads', type=int, default=16, help='TTT number of heads')
    parser.add_argument('--ttt_num_layers', type=int, default=8, help='TTT number of layers')
    parser.add_argument('--ttt_mini_batch_size', type=int, default=16, help='TTT mini batch size')
    parser.add_argument('--ttt_base_lr', type=float, default=1.0, help='TTT base learning rate')
    parser.add_argument('--ttt_selection_strategy', type=str, choices=['hybrid', 'original'], default='hybrid', help='TTT selection strategy')
    
    # Common parameters
    parser.add_argument('--big_lambda', type=int, default=64, help='Big lambda (patch selection number)')
    parser.add_argument('--dropout_v', type=float, default=0.2, help='Dropout rate')
    
    # JSON parameter configuration support
    parser.add_argument('--params', type=str, help='JSON format model parameter configuration dictionary')
    
    # WandB parameters
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB for experiment tracking')
    parser.add_argument('--wandb_project', default='rmil-mil', type=str, help='WandB project name')
    parser.add_argument('--wandb_run_name', default=None, type=str, help='WandB run name')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device')
    parser.add_argument('--random_seed', default=42, type=int, help='Random seed')
        
    args = parser.parse_args()
    
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
    
    # Setup logging system
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logger = setup_logging(args.log_file)
    
    # Log training start information
    logging.info("=" * 80)
    logging.info("Starting training on MIL classic datasets")
    logging.info("=" * 80)
    logging.info(f"Training start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Training parameters:")
    for arg, value in vars(args).items():
        logging.info(f"   {arg}: {value}")
    
    # Initialize WandB if requested - will be initialized per fold
    if args.use_wandb and not args.wandb_run_name:
        args.wandb_run_name = args.model

    # Model loading
    if args.model == 'dsmil':
        import dsmil as mil
        logging.info("Loading model: DSMIL")
    elif args.model == 'abmil':
        import abmil as mil
        logging.info("Loading model: ABMIL")
    elif args.model == 'snuffy':
        import snuffy as mil
        logging.info("Loading model: Snuffy")
    elif args.model == 'rmil_lstm':
        import rmil_lstm as mil
        logging.info("Loading model: RMIL-LSTM")
    elif args.model == 'rmil_gru':
        import rmil_gru as mil
        logging.info("Loading model: RMIL-GRU")
    elif args.model == 'rmil_mamba':
        import rmil_mamba as mil
        logging.info("Loading model: RMIL-Mamba")
    elif args.model == 'rmil_ttt_linear':
        import rmil_ttt as mil
        logging.info("Loading model: RMIL-TTT-Linear")
    elif args.model == 'rmil_ttt_mlp':
        import rmil_ttt as mil
        logging.info("Loading model: RMIL-TTT-MLP")
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
        
    # Dataset loading
    logging.info(f"Loading dataset: {args.datasets}")
    data_start_time = time.time()
    
    if args.datasets == 'musk1':
        data_all = get_data('datasets/mil_dataset/Musk/musk1norm.svm')
        args.num_feats = 166
    elif args.datasets == 'musk2':
        data_all = get_data('datasets/mil_dataset/Musk/musk2norm.svm')
        args.num_feats = 166
    elif args.datasets == 'elephant':
        data_all = get_data('datasets/mil_dataset/Elephant/data_100x100.svm')
        args.num_feats = 230
    elif args.datasets == 'fox':
        data_all = get_data('datasets/mil_dataset/Fox/data_100x100.svm')
        args.num_feats = 230
    elif args.datasets == 'tiger':
        data_all = get_data('datasets/mil_dataset/Tiger/data_100x100.svm')
        args.num_feats = 230
    else:
        raise ValueError(f"Unsupported dataset: {args.datasets}")
    
    data_time = time.time() - data_start_time
    logging.info(f"Dataset loading completed (Time: {data_time:.2f} seconds)")
    logging.info(f"Feature dimension: {args.num_feats}")
    
    # Create model save directory (reference train_tcga.py)
    save_path = os.path.join(args.weights_save_root, f'{args.model}_{args.datasets}_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}')
    os.makedirs(save_path, exist_ok=True)
    logging.info(f"Model save path: {save_path}")
    
    bag_ins_list = []
    num_bag = data_all[-1][1]+1
    for i in range(num_bag):
        bag_data = get_bag(data_all, i)
        bag_label = bag_data[0, 2]
        bag_vector = bag_data[:, 3]
        bag_ins_list.append([bag_label, bag_vector])
    bag_ins_list = shuffle(bag_ins_list)
    
    logging.info(f"Total bags: {len(bag_ins_list)}")
    
    ### Check if there are two classes in the test bags
    valid_bags = 1
    shuffle_count = 0
    while(valid_bags):
        shuffle_count += 1
        bag_ins_list = shuffle(bag_ins_list)
        for k in range (0, args.cv_fold):
            bags_list, test_list = cross_validation_set(bag_ins_list, fold=args.cv_fold, index=k)
            bag_labels = 0
            for i, data in enumerate(test_list):
                bag_labels = np.clip(data[0], 0, 1) + bag_labels
            if bag_labels > 0:
                valid_bags = 0
        if shuffle_count > 100:  # Prevent infinite loop
            logging.warning("Dataset might be unbalanced, exiting check loop")
            break
    
    logging.info(f"Number of data shuffles: {shuffle_count}")
    
    acs = []
    aucs = []
    precisions = []
    recalls = []
    fscores = []
    
    training_start_time = time.time()
    
    logging.info(f'Dataset: {args.datasets}')
    logging.info(f'Starting {args.cv_fold} fold cross-validation')
    
    for k in range(0, args.cv_fold):
        fold_start_time = time.time()
        
        # Initialize WandB for this fold
        fold_wandb_run = None
        if args.use_wandb:
            fold_run_name = f"{args.wandb_run_name}_fold_{k + 1}"
            project_name = f'{args.wandb_project}_{args.datasets}'
            fold_wandb_run = init_wandb(project_name, args.__dict__, fold_run_name)
            logging.info(f"Initialized WandB run for fold {k + 1}: {fold_run_name}")
        
        logging.info(f'Starting fold {k+1}/{args.cv_fold} cross-validation')
        
        bags_list, test_list = cross_validation_set(bag_ins_list, fold=args.cv_fold, index=k)
        logging.info(f"Training bags: {len(bags_list)}, Test bags: {len(test_list)}")
        
        milnet, criterion, optimizer, scheduler = init_model(args, mil)
        
        # Recalculate positive sample weight
        pos_weight = torch.tensor(compute_pos_weight(bags_list))
        criterion = nn.BCEWithLogitsLoss(pos_weight)
        logging.info(f"Positive sample weight: {pos_weight.item():.4f}")
        
        optimal_ac = 0
        best_auc = 0
        best_precision = 0
        best_recall = 0
        best_fscore = 0
        patience_counter = 0
        
        for epoch in range(1, args.num_epochs + 1):
            epoch_start_time = time.time()
            
            train_loss = epoch_train(bags_list, optimizer, criterion, milnet, args, wandb_run=fold_wandb_run, epoch=epoch)
            test_loss, bag_labels, bag_predictions = epoch_test(test_list, criterion, milnet, args, wandb_run=fold_wandb_run, epoch=epoch)
            accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_predictions)
            
            epoch_time = time.time() - epoch_start_time
            
            logging.info(f'Epoch [{epoch}/{args.num_epochs}] '
                        f'train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}, '
                        f'accuracy: {accuracy:.4f}, auc: {auc_value:.4f}, '
                        f'precision: {precision:.4f}, recall: {recall:.4f}, fscore: {fscore:.4f}, '
                        f'Time: {epoch_time:.2f} seconds')
            
            if accuracy > optimal_ac:
                optimal_ac = accuracy
                best_auc = auc_value
                best_precision = precision
                best_recall = recall
                best_fscore = fscore
                patience_counter = 0
                
                # Save best model
                save_model(args, k, milnet, epoch, accuracy, auc_value, precision, recall, fscore, save_path, 'best')
                logging.info(f"Best model updated for fold {k+1}! Accuracy: {accuracy:.4f}")
            else:
                patience_counter += 1
                
            scheduler.step()
            
            # Early stopping
            if patience_counter >= args.early_stopping:
                logging.info(f"Early stopping triggered - No improvement for {args.early_stopping} consecutive epochs")
                break
                
        # Save last generation model after training
        save_model(args, k, milnet, epoch, accuracy, auc_value, precision, recall, fscore, save_path, 'last')
        logging.info(f"Last generation model for fold {k+1} saved")
        
        fold_time = time.time() - fold_start_time
        logging.info(f'Best accuracy for fold {k+1}: {optimal_ac:.4f}, Time: {fold_time:.2f} seconds')
        
        if fold_wandb_run:
            finish_wandb(fold_wandb_run)
        
        acs.append(optimal_ac)
        aucs.append(best_auc)
        precisions.append(best_precision)
        recalls.append(best_recall)
        fscores.append(best_fscore)
    
    total_training_time = time.time() - training_start_time
    
    # Final results statistics
    logging.info("=" * 80)
    logging.info("Final results statistics")
    logging.info("=" * 80)
    logging.info(f'Cross-validation accuracy: {np.mean(np.array(acs)):.4f} ± {np.std(np.array(acs)):.4f}')
    logging.info(f'Cross-validation AUC: {np.mean(np.array(aucs)):.4f} ± {np.std(np.array(aucs)):.4f}')
    logging.info(f'Cross-validation Precision: {np.mean(np.array(precisions)):.4f} ± {np.std(np.array(precisions)):.4f}')
    logging.info(f'Cross-validation Recall: {np.mean(np.array(recalls)):.4f} ± {np.std(np.array(recalls)):.4f}')
    logging.info(f'Cross-validation F1 score: {np.mean(np.array(fscores)):.4f} ± {np.std(np.array(fscores)):.4f}')
    logging.info(f'Total training time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)')
    logging.info(f'Training end time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    print(f'Cross validation accuracy mean: {np.mean(np.array(acs)):.4f}, std {np.std(np.array(acs)):.4f}')
    print(f'Cross validation AUC mean: {np.mean(np.array(aucs)):.4f}, std {np.std(np.array(aucs)):.4f}')
    
    logging.info("=" * 80)
    logging.info("Training completed!")
    logging.info("=" * 80)
    
    
if __name__ == '__main__':
    main()