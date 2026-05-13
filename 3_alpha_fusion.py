import platform
import os
from pathlib import Path
import numpy as np
import shutil
from safetensors import safe_open
from safetensors.torch import save_file

# ==========================================
# PHASE 3: ALPHA FUSION
# ==========================================
# This script intelligently detects your hardware and automatically uses 
# the correct backend: MLX for Apple Silicon, or PyTorch for Windows/Linux.
# It mathematically bakes the T matrix into the model's head for zero-overhead inference.

MODEL_PATH = "google/gemma-2-27b-it"
CACHE_DIR = Path("./vacuum_cache")
ALPHA = 0.02
OUTPUT_DIR = Path("./fused_gemma_model")

def find_safetensor_file_with_key(model_path, key_options):
    safetensor_files = list(model_path.glob("*.safetensors"))
    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in key_options:
                if key in f.keys():
                    return st_file, key
    return None, None

def clone_model_structure():
    print("\n[1/3] Cloning base model structure...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(MODEL_PATH))
        
    for item in model_path.iterdir():
        if item.is_file() and item.suffix != ".safetensors":
            shutil.copy2(item, OUTPUT_DIR / item.name)
            
    print(f"  Cloned config to {OUTPUT_DIR}")
    return model_path

def run_fusion(model_path, backend):
    print(f"\n[2/3] Performing Alpha-Coupled Superposition Blending ({backend})...")
    
    t_path = CACHE_DIR / "T_matrix.npy"
    if not t_path.exists():
        raise FileNotFoundError("T_matrix.npy not found! Run Phase 2 first.")
    
    T_np = np.load(t_path)
    
    keys_to_try = ["lm_head.weight", "language_model.lm_head.weight", "model.embed_tokens.weight"]
    target_shard, head_key = find_safetensor_file_with_key(model_path, keys_to_try)
    
    if target_shard is None:
        raise ValueError("Could not find lm_head matrix in safetensors.")
        
    print(f"  Located target head '{head_key}' in shard: {target_shard.name}")
    
    safetensor_files = list(model_path.glob("*.safetensors"))
    
    if backend == "MLX":
        import mlx.core as mx
        T = mx.array(T_np, dtype=mx.float32)
    else:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        T = torch.tensor(T_np, dtype=torch.float32, device=device)
    
    for st_file in safetensor_files:
        print(f"  Processing {st_file.name}...")
        tensors = {}
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
                
        if head_key in tensors:
            print(f"    >>> FUSING {head_key} <<<")
            original_dtype = tensors[head_key].dtype
            
            if backend == "MLX":
                W = mx.array(tensors[head_key].numpy(), dtype=mx.float32)
                W_translated = W @ T.T
                W_final = (ALPHA * W) + ((1.0 - ALPHA) * W_translated)
                # Convert back to torch tensor for saving via safetensors
                tensors[head_key] = torch.from_numpy(np.array(W_final)).to(original_dtype)
            else:
                import torch
                W = tensors[head_key].to(torch.float32).to(device)
                W_translated = torch.matmul(W, T.t())
                W_final = (ALPHA * W) + ((1.0 - ALPHA) * W_translated)
                tensors[head_key] = W_final.to(original_dtype).cpu()
                
            print(f"    >>> Fusion Complete. Shape: {tensors[head_key].shape} <<<")
            
        save_file(tensors, OUTPUT_DIR / st_file.name)
        
    print(f"\n[3/3] Success! Fully fused model saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    print("=== ARC-SPACE VACUUM TRAINING ===")
    base_path = clone_model_structure()
    
    if platform.system() == "Darwin":
        print("Hardware Detected: Apple Silicon (Mac)")
        run_fusion(base_path, backend="MLX")
    else:
        print("Hardware Detected: Windows/Linux")
        run_fusion(base_path, backend="PyTorch")
