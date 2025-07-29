"""
Common components for Multiple Instance Learning (MIL) models.

This module contains shared components used across different MIL architectures
including DSMIL, RMIL-GRU, RMIL-LSTM, RMIL-Mamba, RMIL-TTT, and Snuffy.
"""

import copy
import math
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FCLayer(nn.Module):
    """Fully connected layer for instance-level classification."""
    
    def __init__(self, in_size: int, out_size: int = 1):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            feats: Input features [N, D]
            
        Returns:
            Tuple of (features, predictions)
        """
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    """Instance-level classifier base class."""
    
    def __init__(self, feature_extractor: nn.Module, feature_size: int, output_class: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [N, D]
            
        Returns:
            Tuple of (features, classification scores)
        """
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class IntelligentPatchSelector(nn.Module):
    """
    Intelligent patch selection module for selecting important patches
    from a bag of instances using different strategies.
    """
    
    def __init__(self, selection_strategy: str = 'hybrid', 
                 diversity_weight: float = 0.3, 
                 uncertainty_weight: float = 0.3):
        super().__init__()
        self.selection_strategy = selection_strategy
        self.diversity_weight = diversity_weight
        self.uncertainty_weight = uncertainty_weight
    
    def forward(self, x: Tensor, c: Tensor, big_lambda: int, 
                top_big_lambda_share: float = 0.7, 
                random_patch_share: float = 0.3) -> Tuple[Tensor, Tensor]:
        """
        Select patches based on different strategies.
        
        Args:
            x: Input features [B, N, D]
            c: Instance classification scores [B, N, C]
            big_lambda: Total number of patches to select
            top_big_lambda_share: Ratio of top patches to select
            random_patch_share: Ratio of random patches to select
        
        Returns:
            Tuple of (selected_indices, attention_weights)
        """
        B, N, D = x.shape
        B, N, C = c.shape
        
        if self.selection_strategy == 'original':
            return self._original_selection(x, c, big_lambda, top_big_lambda_share, random_patch_share)
        elif self.selection_strategy == 'hybrid':
            return self._hybrid_selection(x, c, big_lambda)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
    
    def _original_selection(self, x: Tensor, c: Tensor, big_lambda: int, 
                          top_big_lambda_share: float, random_patch_share: float) -> Tuple[Tensor, Tensor]:
        """Original Snuffy selection strategy based on the reference implementation."""
        B, N, _ = x.shape
        
        # Ensure big_lambda does not exceed available patch count
        big_lambda = min(big_lambda, N)
        
        # Sort patches by instance-level scores
        _, m_indices = torch.sort(c, 1, descending=True)
        
        # Select top-k patches following Snuffy's implementation
        top_k = max(1, min(int(big_lambda * top_big_lambda_share), N))
        top_big_lambda_share_indices = m_indices[:, :top_k, :].squeeze()
        
        # Handle scalar tensor case
        if top_big_lambda_share_indices.dim() == 0:
            top_big_lambda_share_indices = top_big_lambda_share_indices.unsqueeze(0)
        
        # For batch processing, handle each sample
        selected_indices = []
        for b in range(B):
            if B == 1:
                current_top_indices = top_big_lambda_share_indices
            else:
                current_top_indices = top_big_lambda_share_indices[b]
            
            # Convert to list for set operations (following Snuffy's approach)
            current_top_list = current_top_indices.tolist()
            # Ensure it's a flat list of integers
            if not isinstance(current_top_list, list):
                current_top_list = [current_top_list]
            else:
                # Flatten any nested lists recursively
                def flatten_list(lst):
                    result = []
                    for item in lst:
                        if isinstance(item, list):
                            result.extend(flatten_list(item))
                        else:
                            result.append(int(item))  # Ensure it's an integer
                    return result
                current_top_list = flatten_list(current_top_list)
            
            # Calculate remaining indices
            remaining_indices = list(set(range(N)) - set(current_top_list))
            
            # Calculate random share
            randoms_share = min(
                int(big_lambda * random_patch_share),
                max(0, N - top_k)
            )
            
            # Select random indices from remaining
            if randoms_share > 0 and len(remaining_indices) > 0:
                random_count = min(randoms_share, len(remaining_indices))
                random_selected = random.sample(remaining_indices, random_count)
                combined_indices = current_top_list + random_selected
            else:
                combined_indices = current_top_list
            
            # Convert to tensor
            selected_indices.append(torch.tensor(combined_indices, device=x.device, dtype=torch.long))
        
        # Pad to same length for batch processing
        max_len = max(len(idx) for idx in selected_indices)
        max_len = min(max_len, big_lambda)
        padded_indices = []
        for idx in selected_indices:
            if len(idx) < max_len:
                padding = torch.full((max_len - len(idx),), -1, device=x.device, dtype=torch.long)
                idx = torch.cat([idx, padding])
            elif len(idx) > max_len:
                idx = idx[:max_len]
            padded_indices.append(idx)
        
        selected_indices = torch.stack(padded_indices)
        
        # Compute attention weights (simple softmax)
        attention_weights = F.softmax(c.squeeze(-1), dim=1)
        
        return selected_indices, attention_weights
    
    def _hybrid_selection(self, x: Tensor, c: Tensor, big_lambda: int) -> Tuple[Tensor, Tensor]:
        """
        Hybrid selection strategy combining relevance, diversity, and uncertainty.
        """
        B, N, _ = x.shape
        
        # Ensure big_lambda does not exceed available patch count
        big_lambda = min(big_lambda, N)
        
        # Relevance scores (based on instance classification scores)
        if c.shape[-1] == 1:
            relevance_scores = c.squeeze(-1)  # [B, N]
        else:
            relevance_scores = c.max(dim=-1)[0]  # [B, N] - use max class score
        
        # Diversity scores (based on feature distances)
        diversity_scores = self._compute_diversity_scores(x)  # [B, N]
        
        # Uncertainty scores (based on prediction entropy)
        uncertainty_scores = self._compute_uncertainty_scores(c)  # [B, N]
        
        # Combine scores
        combined_scores = (
            relevance_scores + 
            self.diversity_weight * diversity_scores +
            self.uncertainty_weight * uncertainty_scores
        )
        
        # Select top-k (ensure not exceeding available count)
        _, selected_indices = torch.topk(combined_scores, big_lambda, dim=1)
        
        # Compute attention weights
        attention_weights = F.softmax(combined_scores, dim=1)
        
        return selected_indices, attention_weights
    
    def _compute_diversity_scores(self, x: Tensor) -> Tensor:
        """Compute diversity scores based on feature distances."""
        B, N, D = x.shape
        
        # Compute average distance of each patch to all other patches
        x_norm = F.normalize(x, p=2, dim=-1)  # L2 normalization
        similarity_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B, N, N]
        
        # Diversity = 1 - average similarity
        avg_similarity = (similarity_matrix.sum(dim=-1) - 1) / (N - 1)  # exclude self
        diversity_scores = 1 - avg_similarity
        
        return diversity_scores
    
    def _compute_uncertainty_scores(self, c: Tensor) -> Tensor:
        """Compute uncertainty scores based on prediction entropy."""
        # Entropy-based uncertainty from prediction probabilities
        if c.shape[-1] == 1:
            # For binary classification, create [prob, 1-prob] form
            probs = torch.sigmoid(c.squeeze(-1))  # [B, N]
            probs_full = torch.stack([1-probs, probs], dim=-1)  # [B, N, 2]
        else:
            probs_full = F.softmax(c, dim=-1)  # [B, N, C]
        
        uncertainty_scores = -torch.sum(probs_full * torch.log(probs_full + 1e-8), dim=-1)  # [B, N]
        return uncertainty_scores


class BaseBClassifier(nn.Module):
    """
    Base bag-level classifier that provides common functionality
    for different MIL architectures.
    """
    
    def __init__(self, input_size: int, output_class: int, 
                 dropout_v: float = 0.0, nonlinear: bool = True, 
                 passing_v: bool = False, big_lambda: int = 64,
                 selection_strategy: str = 'hybrid', 
                 use_intelligent_selector: bool = True):
        super().__init__()
        
        # Save parameters
        self.input_size = input_size
        self.output_class = output_class
        self.big_lambda = big_lambda
        self.passing_v = passing_v
        
        # Value transformation layer (consistent with DSMIL)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU() if nonlinear else nn.Identity()
            )
        else:
            self.v = nn.Identity()

        # Intelligent patch selector
        self.use_intelligent_selector = use_intelligent_selector
        if use_intelligent_selector:
            self.patch_selector = IntelligentPatchSelector(selection_strategy=selection_strategy)

        # Final classification head
        self.classifier = nn.Linear(input_size, output_class)

    def _extract_selected_features(self, x: Tensor, selected_indices: Tensor, 
                                 big_lambda: int) -> Tensor:
        """
        Extract features based on selected indices.
        
        Args:
            x: Input features [1, N, D]
            selected_indices: Selected indices [big_lambda] or [1, big_lambda]
            big_lambda: Number of selected patches
        
        Returns:
            Extracted features [1, big_lambda, D]
        """
        B, N, D = x.shape
        device = x.device
        
        # Handle selected_indices dimensions
        if selected_indices.dim() == 2:
            selected_indices = selected_indices.squeeze(0)  # [big_lambda]
        
        # Ensure big_lambda does not exceed actual selectable count
        actual_big_lambda = min(big_lambda, len(selected_indices), N)
        
        # Extract selected features
        valid_indices = selected_indices[selected_indices >= 0]
        valid_indices = valid_indices[valid_indices < N]  # ensure indices are in valid range
        
        # Further limit valid index count
        if len(valid_indices) > actual_big_lambda:
            valid_indices = valid_indices[:actual_big_lambda]
        
        # Create output tensor
        m_feats = torch.zeros(1, actual_big_lambda, D, device=device)
        
        if len(valid_indices) > 0:
            selected_feats = x[0, valid_indices]  # [len(valid_indices), D]
            m_feats[0, :len(valid_indices)] = selected_feats
        
        return m_feats

    def _apply_patch_selection(self, feats: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply patch selection strategy.
        
        Args:
            feats: Input features [N, D]
            c: Instance classification scores [N, C]
            
        Returns:
            Tuple of (selected_features, attention_weights)
        """
        device = feats.device
        N, D = feats.shape
        
        # Dynamic adjustment of big_lambda
        effective_big_lambda = min(self.big_lambda, N)
        
        # Apply value transformation
        V = self.v(feats)  # [N, D]
        
        # Expand dimensions for batch processing (add batch dimension)
        V = V.unsqueeze(0)  # [1, N, D]
        c_expanded = c.unsqueeze(0)  # [1, N, C]

        # Intelligent patch selection
        if self.use_intelligent_selector:
            selected_indices, attention_weights = self.patch_selector(V, c_expanded, effective_big_lambda)
            selected_indices = selected_indices.squeeze(0)  # [effective_big_lambda]
            attention_weights = attention_weights.squeeze(0)  # [N]
        else:
            # Use simple top-k selection strategy
            if c.shape[-1] == 1:
                c_for_sort = c.squeeze(-1)  # [N]
            else:
                c_for_sort = c.max(dim=-1)[0]  # [N] - use max class score
            
            _, m_indices = torch.sort(c_for_sort, 0, descending=True)
            selected_indices = m_indices[:effective_big_lambda]
            attention_weights = F.softmax(c_for_sort, dim=0)

        # Extract selected features
        m_feats = self._extract_selected_features(V, selected_indices, effective_big_lambda)
        
        return m_feats, attention_weights

    def _format_output(self, bag_prediction: Tensor, attention_weights: Tensor, 
                      bag_representation: Tensor, c: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Format output to match DSMIL interface.
        
        Args:
            bag_prediction: Bag-level prediction [1, C]
            attention_weights: Attention weights [N] or [N, C]
            bag_representation: Bag representation [1, D]
            c: Original instance scores [N, C]
            
        Returns:
            Formatted tuple (bag_prediction, attention_weights, bag_representation)
        """
        # Adjust output format to match DSMIL
        # DSMIL expects: (bag_prediction [1, C], attention_weights [N, C], bag_representation [1, D])
        if attention_weights.dim() == 1:
            # Expand attention_weights to match expected format [N, C]
            attention_weights = attention_weights.unsqueeze(-1).expand(-1, c.shape[1])
        
        return bag_prediction, attention_weights, bag_representation


class MILNet(nn.Module):    
    def __init__(self, i_classifier: nn.Module, b_classifier: nn.Module):
        super().__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """        
        Args:
            x: Input features [N, D]
        
        Returns:
            Tuple: (instance_predictions, bag_prediction, attention_weights, bag_representation)
        """
        # Instance-level classification
        feats, classes = self.i_classifier(x)
        # Bag-level classification
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag, A, B


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def initialize_weights(module: nn.Module) -> None:
    """Initialize model weights using standard methods."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1 for LSTM
                    if isinstance(m, nn.LSTM) and 'bias' in name:
                        n = param.size(0)
                        param.data[(n//4):(n//2)].fill_(1)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def format_number(num: Union[int, float]) -> str:
    """Format number for display."""
    if num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.0f}"


def create_model_args(model_name: str, params: dict):
    """Create mock args object for model initialization."""
    class Args:
        def __init__(self):
            for key, value in params.items():
                setattr(self, key, value)
            self.model = model_name
            # Handle bidirectional flags for RNN models
            if 'gru_bidirectional' in params:
                self.gru_bidirectional = 'True' if params['gru_bidirectional'] else 'False'
            if 'lstm_bidirectional' in params:
                self.lstm_bidirectional = 'True' if params['lstm_bidirectional'] else 'False'
    return Args() 