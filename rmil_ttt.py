# rmil_ttt.py - RMIL with TTT Encoder
import random
from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from rmil_common import (
    FCLayer, IClassifier, IntelligentPatchSelector, BaseBClassifier, 
    MILNet, initialize_weights
)



def ln_fwd(x, gamma, beta, eps=1e-6):
    """Batch forward for LayerNorm."""
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    return y


def gelu_bwd(x):
    """GELU backward function."""
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff





class TTTBlock(nn.Module):
    """TTT Block that can switch between TTTLinear and TTTMLP."""
    
    def __init__(self, config_dict):
        super(TTTBlock, self).__init__()
        
        # Configuration parameters
        self.width = config_dict['hidden_size']
        self.hidden_size = config_dict['hidden_size']
        self.num_heads = config_dict['num_heads']
        
        # Handle dimension mismatch: ensure head_dim * num_heads equals hidden_size
        if self.width % self.num_heads != 0:
            # Adjust num_heads to be a divisor of width, or use projection
            possible_heads = [h for h in [1, 2, 4, 8, 16] if self.width % h == 0]
            if possible_heads:
                original_num_heads = self.num_heads
                self.num_heads = max([h for h in possible_heads if h <= self.num_heads])
                logging.warning(f"调整注意力头数: {original_num_heads} -> {self.num_heads} (适应特征维度 {self.width})")
            else:
                # Use projection to make dimensions compatible
                self.use_projection = True
                # Find the closest multiple of num_heads that's <= width
                self.projected_dim = (self.width // self.num_heads) * self.num_heads
                self.input_projection = nn.Linear(self.width, self.projected_dim, bias=False)
                self.output_projection = nn.Linear(self.projected_dim, self.width, bias=False)
                self.head_dim = self.projected_dim // self.num_heads
                logging.warning(f"使用投影层处理维度不匹配: {self.width} -> {self.projected_dim} (头数: {self.num_heads})")
        else:
            self.use_projection = False
            self.head_dim = self.width // self.num_heads
        
        if not hasattr(self, 'head_dim'):
            self.head_dim = self.width // self.num_heads
            
        self.mini_batch_size = config_dict.get('mini_batch_size', 16)
        self.ttt_layer_type = config_dict.get('ttt_layer_type', 'linear')
        self.ttt_base_lr = config_dict.get('ttt_base_lr', 1.0)
        
        # Token scaling factors
        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size,)))
        
        # QKV projection matrices for TTT - use projected dimensions
        effective_dim = self.projected_dim if hasattr(self, 'use_projection') and self.use_projection else self.width
        self.WQ = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, effective_dim, self.head_dim)))
        self.WK = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, effective_dim, self.head_dim)))
        self.WV = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, effective_dim, self.head_dim)))
        
        # Projections (keeping for compatibility)
        self.q_proj = nn.Linear(effective_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(effective_dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(effective_dim, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(effective_dim, self.num_heads * self.head_dim, bias=False)
        
        # TTT learning rate gate
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=(self.width, 1)) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros(1) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        
        # TTT layer normalization
        self.ttt_norm_weight = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        
        # TTT model parameters
        if self.ttt_layer_type == "linear":
            self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
            self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        elif self.ttt_layer_type == "mlp":
            self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
            self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
            self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
            self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        
        # Post normalization
        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    def get_eta(self, X):
        """
        Compute learning rate scaling factors.
        Args:
            X: Input tensor [B, H, N, D]
        Returns:
            eta: Learning rate scaling factors [B, H, N, 1]
        """
        B, H, N, D = X.shape
        
        # Simple learning rate computation
        # Use a small fixed learning rate scaled by input norm
        input_norm = torch.norm(X, dim=-1, keepdim=True)  # [B, H, N, 1]
        
        # Normalize to prevent extreme values
        input_norm = torch.clamp(input_norm, min=1e-6)
        
        # Base learning rate
        base_lr = 0.01
        
        # Scale learning rate inversely with input norm for stability
        eta = base_lr / (1.0 + input_norm)
        
        return eta

    def ttt_linear(self, inputs, mini_batch_size):
        """TTT Linear computation - simplified version."""
        # Extract inputs
        XQ = inputs["XQ"]  # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        XK = inputs["XK"]
        XV = inputs["XV"]
        eta = inputs["eta"]  # [B, num_heads, num_mini_batch, mini_batch_size, 1]
        
        B, num_heads, num_mini_batch, actual_mini_batch_size, head_dim = XQ.shape
        
        # Initialize parameters
        W1_init = self.W1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, num_heads, head_dim, head_dim]
        b1_init = self.b1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, num_heads, 1, head_dim]
        
        outputs = []
        
        for t in range(num_mini_batch):
            # Current mini-batch
            XQ_t = XQ[:, :, t]  # [B, num_heads, mini_batch_size, head_dim]
            XK_t = XK[:, :, t]
            XV_t = XV[:, :, t]
            eta_t = eta[:, :, t]  # [B, num_heads, mini_batch_size, 1]
            
            # Forward pass
            X1 = XK_t
            Z1 = torch.einsum('bhkd,bhdf->bhkf', X1, W1_init) + b1_init
            reconstruction_target = XV_t - XK_t
            
            # Layer normalization
            ln_weight = self.ttt_norm_weight.reshape(1, num_heads, 1, head_dim)
            ln_bias = self.ttt_norm_bias.reshape(1, num_heads, 1, head_dim)
            Z1_normalized = ln_fwd(Z1, ln_weight, ln_bias)
            
            # Loss computation (L2 reconstruction loss)
            loss = torch.sum((Z1_normalized - reconstruction_target) ** 2, dim=-1, keepdim=True)  # [B, num_heads, K, 1]
            grad_Z1 = 2 * (Z1_normalized - reconstruction_target)  # [B, num_heads, K, head_dim]
            
            # Compute gradients w.r.t. parameters
            grad_W1 = torch.einsum('bhkd,bhkf->bhdf', X1, grad_Z1)  # [B, num_heads, head_dim, head_dim]
            grad_b1 = torch.sum(grad_Z1, dim=2, keepdim=True)  # [B, num_heads, 1, head_dim]
            
            # Apply learning rate scaling
            eta_scaled = eta_t.squeeze(-1)  # [B, num_heads, mini_batch_size]
            
            # Update parameters for dual form computation
            W1_updated = W1_init - torch.mean(eta_scaled.unsqueeze(-1).unsqueeze(-1) * grad_W1.unsqueeze(2), dim=2)
            b1_updated = b1_init - torch.mean(eta_scaled.unsqueeze(-1).unsqueeze(-1) * grad_b1.unsqueeze(2), dim=2)
            
            # Dual form forward pass
            Z1_dual = torch.einsum('bhkd,bhdf->bhkf', XQ_t, W1_updated) + b1_updated
            Z1_dual_normalized = ln_fwd(Z1_dual, ln_weight, ln_bias)
            
            # Output
            XQW_t = XQ_t + Z1_dual_normalized
            outputs.append(XQW_t)
            
            # Update parameters for next iteration (use last token's gradient)
            if actual_mini_batch_size > 0:
                last_eta = eta_t[:, :, -1:, :]  # [B, num_heads, 1, 1]
                last_grad_W1 = torch.einsum('bhkd,bhkf->bhdf', X1[:, :, -1:, :], grad_Z1[:, :, -1:, :])
                last_grad_b1 = grad_Z1[:, :, -1:, :]
                
                W1_init = W1_init - last_eta.squeeze(-1).unsqueeze(-1) * last_grad_W1
                b1_init = b1_init - torch.sum(last_eta * last_grad_b1, dim=2, keepdim=True)
        
        return torch.stack(outputs, dim=2)  # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]

    def ttt_mlp(self, inputs, mini_batch_size):
        """TTT MLP computation - simplified version."""
        # Extract inputs
        XQ = inputs["XQ"]
        XK = inputs["XK"] 
        XV = inputs["XV"]
        eta = inputs["eta"]  # [B, num_heads, num_mini_batch, mini_batch_size, 1]
        
        B, num_heads, num_mini_batch, actual_mini_batch_size, head_dim = XQ.shape
        
        # Initialize parameters
        W1_init = self.W1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, num_heads, head_dim, 4*head_dim]
        b1_init = self.b1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, num_heads, 1, 4*head_dim]
        W2_init = self.W2.unsqueeze(0).expand(B, -1, -1, -1)  # [B, num_heads, 4*head_dim, head_dim]
        b2_init = self.b2.unsqueeze(0).expand(B, -1, -1, -1)  # [B, num_heads, 1, head_dim]
        
        outputs = []
        
        for t in range(num_mini_batch):
            # Current mini-batch
            XQ_t = XQ[:, :, t]
            XK_t = XK[:, :, t]
            XV_t = XV[:, :, t]
            eta_t = eta[:, :, t]  # [B, num_heads, mini_batch_size, 1]
            
            # Forward pass
            X1 = XK_t
            Z1 = torch.einsum('bhkd,bhdf->bhkf', X1, W1_init) + b1_init
            X2 = F.gelu(Z1, approximate="tanh")
            Z2 = torch.einsum('bhkf,bhfg->bhkg', X2, W2_init) + b2_init
            reconstruction_target = XV_t - XK_t
            
            # Layer normalization
            ln_weight = self.ttt_norm_weight.reshape(1, num_heads, 1, head_dim)
            ln_bias = self.ttt_norm_bias.reshape(1, num_heads, 1, head_dim)
            Z2_normalized = ln_fwd(Z2, ln_weight, ln_bias)
            
            # Loss and gradients
            loss = torch.sum((Z2_normalized - reconstruction_target) ** 2, dim=-1, keepdim=True)
            grad_Z2 = 2 * (Z2_normalized - reconstruction_target)  # [B, num_heads, K, head_dim]
            grad_Z1 = torch.einsum('bhkg,bhfg->bhkf', grad_Z2, W2_init) * gelu_bwd(Z1)  # [B, num_heads, K, 4*head_dim]
            
            # Compute parameter gradients
            grad_W1 = torch.einsum('bhkd,bhkf->bhdf', X1, grad_Z1)
            grad_b1 = torch.sum(grad_Z1, dim=2, keepdim=True)
            grad_W2 = torch.einsum('bhkf,bhkg->bhfg', X2, grad_Z2)
            grad_b2 = torch.sum(grad_Z2, dim=2, keepdim=True)
            
            # Apply learning rate scaling
            eta_scaled = eta_t.squeeze(-1)  # [B, num_heads, mini_batch_size]
            
            # Update parameters for dual form
            W1_updated = W1_init - torch.mean(eta_scaled.unsqueeze(-1).unsqueeze(-1) * grad_W1.unsqueeze(2), dim=2)
            b1_updated = b1_init - torch.mean(eta_scaled.unsqueeze(-1).unsqueeze(-1) * grad_b1.unsqueeze(2), dim=2)
            W2_updated = W2_init - torch.mean(eta_scaled.unsqueeze(-1).unsqueeze(-1) * grad_W2.unsqueeze(2), dim=2)
            b2_updated = b2_init - torch.mean(eta_scaled.unsqueeze(-1).unsqueeze(-1) * grad_b2.unsqueeze(2), dim=2)
            
            # Dual form forward pass
            Z1_dual = torch.einsum('bhkd,bhdf->bhkf', XQ_t, W1_updated) + b1_updated
            X2_dual = F.gelu(Z1_dual, approximate="tanh")
            Z2_dual = torch.einsum('bhkf,bhfg->bhkg', X2_dual, W2_updated) + b2_updated
            Z2_dual_normalized = ln_fwd(Z2_dual, ln_weight, ln_bias)
            
            # Output
            XQW_t = XQ_t + Z2_dual_normalized
            outputs.append(XQW_t)
            
            # Update parameters for next iteration
            if actual_mini_batch_size > 0:
                last_eta = eta_t[:, :, -1:, :]  # [B, num_heads, 1, 1]
                last_grad_W1 = torch.einsum('bhkd,bhkf->bhdf', X1[:, :, -1:, :], grad_Z1[:, :, -1:, :])
                last_grad_b1 = grad_Z1[:, :, -1:, :]
                last_grad_W2 = torch.einsum('bhkf,bhkg->bhfg', X2[:, :, -1:, :], grad_Z2[:, :, -1:, :])
                last_grad_b2 = grad_Z2[:, :, -1:, :]
                
                W1_init = W1_init - last_eta.squeeze(-1).unsqueeze(-1) * last_grad_W1
                b1_init = b1_init - torch.sum(last_eta * last_grad_b1, dim=2, keepdim=True)
                W2_init = W2_init - last_eta.squeeze(-1).unsqueeze(-1) * last_grad_W2
                b2_init = b2_init - torch.sum(last_eta * last_grad_b2, dim=2, keepdim=True)
        
        return torch.stack(outputs, dim=2)

    def forward(self, X):
        """
        Forward pass of TTT block.
        Args:
            X: Input tensor [B, N, D]
        Returns:
            Output tensor [B, N, D]
        """
        B, N, D = X.shape
        
        # Apply input projection if needed
        if hasattr(self, 'use_projection') and self.use_projection:
            X_projected = self.input_projection(X)  # [B, N, projected_dim]
            working_X = X_projected
            working_D = self.projected_dim
        else:
            working_X = X
            working_D = D
        
        # Ensure N is divisible by mini_batch_size for mini-batch processing
        remainder = N % self.mini_batch_size
        if remainder != 0:
            # Pad sequence to make it divisible by mini_batch_size
            padding_size = self.mini_batch_size - remainder
            padding = torch.zeros(B, padding_size, working_D, device=X.device, dtype=X.dtype)
            X_padded = torch.cat([working_X, padding], dim=1)
            N_padded = N + padding_size
        else:
            X_padded = working_X
            N_padded = N
        
        # Reshape into mini-batches: [B, N_padded, D] -> [B, num_mini_batch, mini_batch_size, D]
        num_mini_batch = N_padded // self.mini_batch_size
        actual_mini_batch_size = self.mini_batch_size
        X_mb = X_padded.view(B, num_mini_batch, actual_mini_batch_size, working_D)
        
        # Compute QKV using einsum: [B, num_mini_batch, actual_mini_batch_size, D] x [num_heads, D, head_dim] -> [B, num_heads, num_mini_batch, actual_mini_batch_size, head_dim]
        XQ = torch.einsum("bnkd,hdf->bhnkf", X_mb, self.WQ)
        XK = torch.einsum("bnkd,hdf->bhnkf", X_mb, self.WK)
        XV = torch.einsum("bnkd,hdf->bhnkf", X_mb, self.WV)
        
        # Get learning rate factors
        # Reshape X_padded for eta computation: [B, N_padded, D] -> [B, H, N_padded, D_head]
        X_for_eta = X_padded.view(B, N_padded, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        eta = self.get_eta(X_for_eta)  # [B, H, N_padded, 1]
        
        # Reshape eta to mini-batch structure: [B, H, N_padded, 1] -> [B, H, num_mini_batch, actual_mini_batch_size, 1]
        eta = eta.view(B, self.num_heads, num_mini_batch, actual_mini_batch_size, 1)
        
        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
        }
        
        if self.ttt_layer_type == "linear":
            output = self.ttt_linear(inputs, actual_mini_batch_size)
        elif self.ttt_layer_type == "mlp":
            output = self.ttt_mlp(inputs, actual_mini_batch_size)
        else:
            raise ValueError(f"Unknown TTT layer type: {self.ttt_layer_type}")
        
        # Reshape output back to original sequence length
        # output: [B, num_heads, num_mini_batch, actual_mini_batch_size, head_dim] -> [B, N_padded, D]
        output = output.permute(0, 2, 3, 1, 4).contiguous()  # [B, num_mini_batch, actual_mini_batch_size, num_heads, head_dim]
        output = output.view(B, N_padded, working_D)  # [B, N_padded, working_D]
        
        # Remove padding if it was added
        if remainder > 0:
            output = output[:, :N, :]  # Remove padding
        
        # Apply output projection if needed
        if hasattr(self, 'use_projection') and self.use_projection:
            output = self.output_projection(output)  # [B, N, D]
        
        # Apply post-normalization
        output = self.post_norm(output)
        
        return output


class BClassifier(BaseBClassifier):
    """
    Modified BClassifier using TTT blocks instead of Transformer.
    """
    
    def __init__(self, input_size, output_class, 
                 dropout_v=0.2, nonlinear=True, passing_v=False,
                 # TTT specific parameters
                 num_heads=4, num_layers=6, big_lambda=128, mini_batch_size=8,
                 ttt_layer_type='linear', ttt_base_lr=2.0, 
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
        
        # TTT configuration
        ttt_config = {
            'hidden_size': input_size,
            'num_heads': num_heads,
            'mini_batch_size': mini_batch_size,
            'ttt_layer_type': ttt_layer_type,
            'ttt_base_lr': ttt_base_lr,
        }
        
        # Build TTT encoder
        self.ttt_layers = nn.ModuleList([
            TTTBlock(ttt_config) for _ in range(num_layers)
        ])

    def forward(self, feats, c):
        """
        Forward pass maintaining DSMIL interface.
        
        Args:
            feats: Input features [N, D]
            c: Instance classification scores [N, C]
            
        Returns:
            Tuple: (bag_prediction [1, C], attention_weights [N, C], bag_representation [1, D])
        """
        # Apply patch selection using BaseBClassifier method
        m_feats, attention_weights = self._apply_patch_selection(feats, c)
        
        # Apply TTT layers
        encoded_features = m_feats
        for ttt_layer in self.ttt_layers:
            encoded_features = ttt_layer(encoded_features)
        
        # Global average pooling
        bag_representation = encoded_features.mean(dim=1)  # [1, D]
        
        # Final classification
        bag_prediction = self.classifier(bag_representation)  # [1, C]
        
        # Format output to match DSMIL
        return self._format_output(bag_prediction, attention_weights, bag_representation, c)





def create_rmil_ttt_model(input_size=1024, output_class=1, dropout_v=0.2,
                           num_heads=4, num_layers=6, mini_batch_size=8,
                           ttt_layer_type='linear', ttt_base_lr=2.0,
                           selection_strategy='hybrid'):
    """
    Factory function to create RMIL-TTT model.
    
    Args:
        input_size: Input feature dimension
        output_class: Number of output classes
        dropout_v: Dropout rate
        num_heads: Number of attention heads
        num_layers: Number of TTT layers
        mini_batch_size: Mini-batch size for TTT
        ttt_layer_type: 'linear' or 'mlp'
        ttt_base_lr: Base learning rate for TTT
        selection_strategy: Patch selection strategy
        
    Returns:
        Configured MILNet model
    """
    # Instance classifier (identity feature extractor)
    i_classifier = IClassifier(
        feature_extractor=nn.Identity(),
        feature_size=input_size,
        output_class=output_class
    )
    
    # Bag classifier with TTT
    b_classifier = BClassifier(
        input_size=input_size,
        output_class=output_class,
        dropout_v=dropout_v,
        num_heads=num_heads,
        num_layers=num_layers,
        mini_batch_size=mini_batch_size,
        ttt_layer_type=ttt_layer_type,
        ttt_base_lr=ttt_base_lr,
        selection_strategy=selection_strategy
    )
    
    # Create MIL network
    model = MILNet(i_classifier, b_classifier)
    
    # Initialize weights
    initialize_weights(model)
    
    return model


if __name__ == "__main__":
    # Test the model
    print("🧪 Testing RMIL-TTT model...")
    
    # Test parameters
    batch_size = 1
    num_patches = 100
    feature_dim = 1024
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test data
    test_input = torch.randn(num_patches, feature_dim).to(device)
    
    # Test TTTLinear model
    print("\n🔬 Testing TTTLinear model...")
    model_linear = create_rmil_ttt_model(
        input_size=feature_dim,
        output_class=num_classes,
        num_heads=4,
        num_layers=2,
        mini_batch_size=8,
        ttt_layer_type='linear',
        selection_strategy='hybrid'
    ).to(device)
    
    with torch.no_grad():
        classes_linear, bag_pred_linear, attention_linear, bag_repr_linear = model_linear(test_input)
    
    print(f"TTTLinear - Classes shape: {classes_linear.shape}")
    print(f"TTTLinear - Bag prediction shape: {bag_pred_linear.shape}")
    print(f"TTTLinear - Attention shape: {attention_linear.shape}")
    print(f"TTTLinear - Bag representation shape: {bag_repr_linear.shape}")
    print(f"TTTLinear - Total parameters: {sum(p.numel() for p in model_linear.parameters()):,}")
    
    # Test TTTMLP model
    print("\n🔬 Testing TTTMLP model...")
    model_mlp = create_rmil_ttt_model(
        input_size=feature_dim,
        output_class=num_classes,
        num_heads=4,
        num_layers=2,
        mini_batch_size=8,
        ttt_layer_type='mlp',
        selection_strategy='hybrid'
    )
    
    with torch.no_grad():
        classes_mlp, bag_pred_mlp, attention_mlp, bag_repr_mlp = model_mlp(test_input)
    
    print(f"TTTMLP - Classes shape: {classes_mlp.shape}")
    print(f"TTTMLP - Bag prediction shape: {bag_pred_mlp.shape}")
    print(f"TTTMLP - Attention shape: {attention_mlp.shape}")
    print(f"TTTMLP - Bag representation shape: {bag_repr_mlp.shape}")
    print(f"TTTMLP - Total parameters: {sum(p.numel() for p in model_mlp.parameters()):,}")
    
    print("\n✅ RMIL-TTT model test completed successfully!")
    print("\n📊 Model Summary:")
    print("- Successfully replaced Transformer with TTT blocks")
    print("- Supports both TTTLinear and TTTMLP variants")
    print("- Maintains DSMIL-compatible interface")
    print("- Includes intelligent patch selection")
    print("- Ready for training and evaluation") 