# rmil_gru.py - RMIL with GRU Encoder
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rmil_common import (
    FCLayer, IClassifier, IntelligentPatchSelector, BaseBClassifier, 
    MILNet, initialize_weights, clones
)


class GRUBlock(nn.Module):
    """
    GRU-based encoding block for replacing Transformer self-attention mechanism
    """
    def __init__(
        self, 
        input_size, 
        hidden_size=None,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        batch_first=True,
        **kwargs
    ):
        """
        GRU encoding block
        
        Args:
            input_size: Input feature dimension
            hidden_size: GRU hidden layer dimension, default to input_size
            num_layers: Number of GRU layers
            dropout: Dropout ratio
            bidirectional: Whether to use bidirectional GRU
            batch_first: Whether batch dimension comes first
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else input_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        
        # Output dimension calculation
        gru_output_size = self.hidden_size * (2 if bidirectional else 1)
        
        # Projection layer to map GRU output back to original dimension
        if gru_output_size != input_size:
            self.projection = nn.Linear(gru_output_size, input_size)
        else:
            self.projection = nn.Identity()
        
        # Normalization and Dropout
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden_state=None):
        """
        Forward pass
        
        Args:
            x: Input features [B, N, D]
            hidden_state: Optional initial hidden state
        
        Returns:
            Tuple[Tensor, Tensor]: (output features, h_n)
        """
        # Save input for residual connection
        residual = x
        
        # Through GRU
        gru_output, hidden_state = self.gru(x, hidden_state)
        
        # Project to original dimension
        output = self.projection(gru_output)
        
        # Residual connection and normalization
        output = self.layer_norm(output + residual)
        output = self.dropout(output)
        
        return output, hidden_state


class BClassifier(BaseBClassifier):
    """
    GRU-based bag-level classifier replacing original Transformer
    """
    def __init__(self, input_size, output_class, dropout_v=0.2, nonlinear=True, passing_v=False, 
                 # GRU specific parameters
                 gru_hidden_size=1024, gru_num_layers=3, big_lambda=64,
                 selection_strategy='hybrid', use_intelligent_selector=True,
                 bidirectional=True):
        super().__init__(
            input_size=input_size,
            output_class=output_class,
            dropout_v=dropout_v,
            nonlinear=nonlinear,
            passing_v=passing_v,
            big_lambda=big_lambda,
            selection_strategy=selection_strategy,
            use_intelligent_selector=use_intelligent_selector
        )
        
        # GRU encoder layers
        self.gru_hidden_size = gru_hidden_size if gru_hidden_size is not None else input_size
        self.gru_num_layers = gru_num_layers
        
        # Build GRU layer
        self.gru_encoder = GRUBlock(
            input_size=input_size,
            hidden_size=self.gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=dropout_v,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, feats, c):
        """
        Forward pass maintaining DSMIL-consistent interface
        
        Args:
            feats: Input features [N, D] 
            c: Instance classification scores [N, C]
        
        Returns:
            Tuple: (bag prediction [1, C], attention weights [N, C], bag representation [1, D])
        """
        # Apply patch selection
        m_feats, attention_weights = self._apply_patch_selection(feats, c)
        
        # Through GRU encoder
        encoded_feats, _ = self.gru_encoder(m_feats)
        
        # Global pooling: average pooling
        bag_representation = encoded_feats.mean(dim=1)  # [1, D]
        
        # Final classification
        bag_prediction = self.classifier(bag_representation)  # [1, C]
        
        # Format output to match DSMIL
        return self._format_output(bag_prediction, attention_weights, bag_representation, c)


def create_rmil_gru_model(input_size=1024, output_class=1, dropout_v=0.2, 
                           gru_hidden_size=1024, gru_num_layers=3, 
                           selection_strategy='hybrid', bidirectional=True):
    """
    Factory function: Create RMIL-GRU model
    
    Args:
        input_size: Input feature dimension
        output_class: Number of output classes
        dropout_v: Dropout ratio
        gru_hidden_size: GRU hidden layer dimension
        gru_num_layers: Number of GRU layers
        selection_strategy: Patch selection strategy ('original' or 'hybrid')
        bidirectional: Whether to use bidirectional GRU
    
    Returns:
        MILNet: Complete model
    """
    # Instance-level classifier
    i_classifier = FCLayer(input_size, output_class)
    
    # Bag-level classifier (GRU-based)
    b_classifier = BClassifier(
        input_size=input_size,
        output_class=output_class,
        dropout_v=dropout_v,
        nonlinear=True,
        passing_v=False,
        gru_hidden_size=gru_hidden_size,
        gru_num_layers=gru_num_layers,
        selection_strategy=selection_strategy,
        use_intelligent_selector=True,
        bidirectional=bidirectional
    )
    
    # Complete model
    model = MILNet(i_classifier, b_classifier)
    
    # Weight initialization
    initialize_weights(model)
    
    return model


# Usage example
if __name__ == "__main__":
    # Model configuration
    input_size = 1024
    output_class = 2
    big_lambda = 64
    batch_size = 2
    num_patches = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = create_rmil_gru_model(
        input_size=input_size,
        output_class=output_class,
        dropout_v=0.1,
        gru_hidden_size=512,
        gru_num_layers=3,
        selection_strategy='hybrid',
        bidirectional=True
    ).to(device)

    print(f"RMIL-GRU Model Created!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test data
    x = torch.randn(num_patches, input_size).to(device)  # Note: [N, D] format
    
    print(f"\nTesting with input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        classes, prediction_bag, attention_weights, bag_representation = model(x)
    
    print(f"Instance predictions shape: {classes.shape}")
    print(f"Bag prediction shape: {prediction_bag.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Bag representation shape: {bag_representation.shape}")
    
    print(f"\nModel output shapes match DSMIL interface!")
    print(f"Attention weight range: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
    
    print("\nRMIL-GRU hybrid model test completed successfully!") 