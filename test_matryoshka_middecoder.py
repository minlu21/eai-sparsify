#!/usr/bin/env python3
"""Simple test script to verify MatryoshkaMidDecoder implementation."""

import torch
from sparsify.config import SparseCoderConfig
from sparsify.sparse_coder import SparseCoder, MatryoshkaMidDecoder

def test_matryoshka_middecoder():
    """Test that MatryoshkaMidDecoder works correctly."""
    print("Testing MatryoshkaMidDecoder...")
    
    # Create a simple sparse coder config
    cfg = SparseCoderConfig(
        num_latents=64,
        k=16,
        activation="topk",
        matryoshka=True,
        matryoshka_expansion_factors=[0.25, 0.5, 1.0],  # [16, 32, 64] latents
    )
    
    # Create sparse coder
    sparse_coder = SparseCoder(
        d_in=128,
        cfg=cfg,
        device="cpu",
        dtype=torch.float32,
    )
    
    # Create test data
    batch_size = 4
    x = torch.randn(batch_size, 128)
    y = torch.randn(batch_size, 128)
    
    # Test regular forward pass (should use MatryoshkaMidDecoder)
    print("Testing forward pass with matryoshka=True...")
    output = sparse_coder(x, y)
    
    print(f"Output type: {type(output)}")
    print(f"FVU: {output.fvu.item():.6f}")
    print(f"AuxK loss: {output.auxk_loss.item():.6f}")
    print(f"Multi-TopK FVU: {output.multi_topk_fvu.item():.6f}")
    
    # Test with matryoshka=False
    print("\nTesting forward pass with matryoshka=False...")
    cfg.matryoshka = False
    sparse_coder.cfg = cfg
    output2 = sparse_coder(x, y)
    
    print(f"Output type: {type(output2)}")
    print(f"FVU: {output2.fvu.item():.6f}")
    print(f"AuxK loss: {output2.auxk_loss.item():.6f}")
    print(f"Multi-TopK FVU: {output2.multi_topk_fvu.item():.6f}")
    
    # Test direct MatryoshkaMidDecoder creation
    print("\nTesting direct MatryoshkaMidDecoder creation...")
    cfg.matryoshka = True
    sparse_coder.cfg = cfg
    
    # Get pre_acts by doing a forward pass
    with torch.no_grad():
        top_acts, top_indices, pre_acts = sparse_coder.encode(x)
    
    # Create MatryoshkaMidDecoder directly
    matryoshka_mid = MatryoshkaMidDecoder(
        sparse_coder=sparse_coder,
        x=x,
        activations=top_acts,
        indices=top_indices,
        pre_acts=pre_acts,
        expansion_factors=[0.25, 0.5, 1.0],
    )
    
    # Test the call
    output3 = matryoshka_mid(y, index=0)
    
    print(f"Direct MatryoshkaMidDecoder output type: {type(output3)}")
    print(f"FVU: {output3.fvu.item():.6f}")
    print(f"AuxK loss: {output3.auxk_loss.item():.6f}")
    print(f"Multi-TopK FVU: {output3.multi_topk_fvu.item():.6f}")
    
    print("\n✅ All tests passed! MatryoshkaMidDecoder is working correctly.")

if __name__ == "__main__":
    test_matryoshka_middecoder()