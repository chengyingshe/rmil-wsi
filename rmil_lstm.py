# rmil_lstm.py - RMIL with LSTM Encoder
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


class LSTMBlock(nn.Module):
    """
    LSTM-based encoding block for replacing Transformer self-attention mechanism
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
        LSTM encoding block
        
        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden layer dimension, default to input_size
            num_layers: Number of LSTM layers
            dropout: Dropout ratio
            bidirectional: Whether to use bidirectional LSTM
            batch_first: Whether batch dimension comes first
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else input_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        
        # Output dimension calculation
        lstm_output_size = self.hidden_size * (2 if bidirectional else 1)
        
        # Projection layer to map LSTM output back to original dimension
        if lstm_output_size != input_size:
            self.projection = nn.Linear(lstm_output_size, input_size)
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
            Tuple[Tensor, Tuple]: (output features, (h_n, c_n))
        """
        # Save input for residual connection
        residual = x
        
        # Through LSTM
        lstm_output, hidden_state = self.lstm(x, hidden_state)
        
        # Project to original dimension
        output = self.projection(lstm_output)
        
        # Residual connection and normalization
        output = self.layer_norm(output + residual)
        output = self.dropout(output)
        
        return output, hidden_state


class BClassifier(BaseBClassifier):
    """
    LSTM-based bag-level classifier replacing original Transformer
    """
    def __init__(self, input_size, output_class, dropout_v=0.1, nonlinear=True, passing_v=False, 
                 # LSTM specific parameters
                 lstm_hidden_size=512, lstm_num_layers=4, big_lambda=64,
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
        
        # LSTM encoder layers
        self.lstm_hidden_size = lstm_hidden_size if lstm_hidden_size is not None else input_size
        self.lstm_num_layers = lstm_num_layers
        
        # Build LSTM layer
        self.lstm_encoder = LSTMBlock(
            input_size=input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=lstm_num_layers,
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
        
        # Through LSTM encoder
        encoded_feats, _ = self.lstm_encoder(m_feats)
        
        # Global pooling: average pooling
        bag_representation = encoded_feats.mean(dim=1)  # [1, D]
        
        # Final classification
        bag_prediction = self.classifier(bag_representation)  # [1, C]
        
        # Format output to match DSMIL
        return self._format_output(bag_prediction, attention_weights, bag_representation, c)


def create_rmil_lstm_model(input_size=1024, output_class=1, dropout_v=0.1, 
                            lstm_hidden_size=512, lstm_num_layers=4, 
                            selection_strategy='hybrid', bidirectional=True):
    """
    Factory function: Create RMIL-LSTM model
    
    Args:
        input_size: Input feature dimension
        output_class: Number of output classes
        dropout_v: Dropout ratio
        lstm_hidden_size: LSTM hidden layer dimension
        lstm_num_layers: Number of LSTM layers
        selection_strategy: Patch selection strategy ('original' or 'hybrid')
        bidirectional: Whether to use bidirectional LSTM
    
    Returns:
        MILNet: Complete model
    """
    # Instance-level classifier
    i_classifier = FCLayer(input_size, output_class)
    
    # Bag-level classifier (LSTM-based)
    b_classifier = BClassifier(
        input_size=input_size,
        output_class=output_class,
        dropout_v=dropout_v,
        nonlinear=True,
        passing_v=False,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
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
    model = create_rmil_lstm_model(
        input_size=input_size,
        output_class=output_class,
        dropout_v=0.1,
        lstm_hidden_size=512,
        lstm_num_layers=4,
        selection_strategy='hybrid',
        bidirectional=True
    ).to(device)

    print(f"RMIL-LSTM Model Created!")
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
    
    print("\nRMIL-LSTM hybrid model test completed successfully!") 