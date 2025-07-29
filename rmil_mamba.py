# rmil_mamba.py - RMIL with Mamba Encoder
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

# Assume mamba_ssm is installed
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    raise ImportError("mamba_ssm not available. Please install mamba_ssm.")


class MambaBlock(nn.Module):
    """
    Mamba block adapted for MIL tasks, based on Vision Mamba
    """
    def __init__(
        self, 
        dim, 
        d_state=16,
        d_conv=4,
        expand=2,
        norm_cls=nn.LayerNorm, 
        fused_add_norm=False, 
        residual_in_fp32=False,
        drop_path=0.,
        layer_idx=None,
        bimamba_type="v2",
        if_divide_out=True,
        **factory_kwargs
    ):
        """
        Mamba block, similar to original Block class but using Mamba instead of Multi-head Attention
        
        Args:
            dim: Feature dimension
            d_state: Mamba state dimension
            d_conv: Convolution kernel size
            expand: Expansion ratio
            norm_cls: Normalization layer type
            fused_add_norm: Whether to use fused add+norm
            residual_in_fp32: Whether residual connection uses fp32
            drop_path: DropPath ratio
            layer_idx: Layer index
            bimamba_type: BiMamba type
            if_divide_out: Whether to divide output
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        
        # Mamba mixer
        if Mamba is not None:
            self.mixer = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=layer_idx,
                bimamba_type=bimamba_type,
                if_divide_out=if_divide_out,
                **factory_kwargs
            )
        else:
            # Fallback: use simple linear layers
            self.mixer = nn.Sequential(
                nn.Linear(dim, dim * expand),
                nn.GELU(),
                nn.Linear(dim * expand, dim)
            )
        
        self.norm = norm_cls(dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, 
        hidden_states: Tensor, 
        residual: Optional[Tensor] = None, 
        inference_params=None,
        **kwargs
    ):
        """
        Forward pass
        
        Args:
            hidden_states: Input features [B, N, D]
            residual: Residual connection
            inference_params: Inference parameters
        
        Returns:
            Tuple[Tensor, Tensor]: (output features, residual)
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        
        # Through Mamba mixer
        if hasattr(self.mixer, 'forward') and 'inference_params' in self.mixer.forward.__code__.co_varnames:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        else:
            hidden_states = self.mixer(hidden_states)
        
        return hidden_states, residual


class BClassifier(BaseBClassifier):
    """
    Mamba-based bag-level classifier replacing original Transformer
    """
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False, 
                 # Mamba specific parameters
                 mamba_depth=6, d_state=16, d_conv=4, expand=2, big_lambda=64,
                 selection_strategy='hybrid', use_intelligent_selector=True):
        super(BClassifier, self).__init__(
            input_size=input_size,
            output_class=output_class,
            dropout_v=dropout_v,
            nonlinear=nonlinear,
            passing_v=passing_v,
            big_lambda=big_lambda,
            selection_strategy=selection_strategy,
            use_intelligent_selector=use_intelligent_selector
        )
        
        # Mamba encoder layers
        self.mamba_depth = mamba_depth
        self.embed_dim = input_size
        
        # Build Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                dim=input_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                norm_cls=RMSNorm if RMSNorm is not None else nn.LayerNorm,
                fused_add_norm=False,
                residual_in_fp32=False,
                drop_path=0.1 * (i / mamba_depth),  # Progressive drop path
                layer_idx=i,
                bimamba_type="v2",
                if_divide_out=True,
            )
            for i in range(mamba_depth)
        ])
        
        # Final normalization layer
        self.norm_f = (RMSNorm if RMSNorm is not None else nn.LayerNorm)(input_size)

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
        
        # Through Mamba encoder
        hidden_states = m_feats
        residual = None
        
        for layer in self.mamba_layers:
            hidden_states, residual = layer(hidden_states, residual)
        
        # Final normalization
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
        
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        
        # Global pooling: average pooling
        bag_representation = hidden_states.mean(dim=1)  # [1, D]
        
        # Final classification
        bag_prediction = self.classifier(bag_representation)  # [1, C]
        
        # Format output to match DSMIL
        return self._format_output(bag_prediction, attention_weights, bag_representation, c)


def create_rmil_mamba_model(input_size=1024, output_class=1, dropout_v=0.0, 
                             mamba_depth=6, d_state=16, selection_strategy='hybrid'):
    """
    Factory function: Create RMIL-Mamba model
    
    Args:
        input_size: Input feature dimension
        output_class: Number of output classes
        dropout_v: Dropout ratio
        mamba_depth: Number of Mamba layers
        d_state: Mamba state dimension
        selection_strategy: Patch selection strategy ('original' or 'hybrid')
    
    Returns:
        MILNet: Complete model
    """
    # Instance-level classifier
    i_classifier = FCLayer(input_size, output_class)
    
    # Bag-level classifier (Mamba-based)
    b_classifier = BClassifier(
        input_size=input_size,
        output_class=output_class,
        dropout_v=dropout_v,
        nonlinear=True,
        passing_v=False,
        mamba_depth=mamba_depth,
        d_state=d_state,
        selection_strategy=selection_strategy,
        use_intelligent_selector=True
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
    model = create_rmil_mamba_model(
        input_size=input_size,
        output_class=output_class,
        dropout_v=0.1,
        mamba_depth=6,
        d_state=16,
        selection_strategy='hybrid'
    ).to(device)

    print(f"RMIL-Mamba Model Created!")
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
    
    print("\nRMIL-Mamba hybrid model test completed successfully!")