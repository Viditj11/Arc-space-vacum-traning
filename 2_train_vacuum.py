import platform
import os
from pathlib import Path
import numpy as np

# ==========================================
# PHASE 2: TRAIN VACUUM
# ==========================================
# This script intelligently detects your hardware and automatically uses 
# the correct backend: MLX for Apple Silicon, or PyTorch for Windows/Linux.

MODEL_PATH = "google/gemma-2-27b-it" 
CACHE_DIR = Path("./vacuum_cache")
LEARNING_RATE = 1e-4
EPOCHS = 100
BATCH_SIZE = 1024

def run_mlx():
    import mlx.core as mx
    import mlx.optimizers as optim
    from safetensors import safe_open
    
    print("\n[1/3] Locating isolated lm_head via MLX...")
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(MODEL_PATH, allow_patterns=["*.safetensors"]))
        
    safetensor_files = list(model_path.glob("*.safetensors"))
    keys_to_try = ["lm_head.weight", "language_model.lm_head.weight", "model.embed_tokens.weight"]
    
    W_tensor = None
    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in keys_to_try:
                if key in f.keys():
                    W_tensor = f.get_tensor(key).numpy()
                    break
        if W_tensor is not None:
            break
            
    if W_tensor is None:
        raise ValueError("Could not find lm_head matrix.")
        
    W = mx.array(W_tensor, dtype=mx.float32)
    HIDDEN_DIM = W.shape[1]
    
    print("\n[2/3] Mapping cached hidden states...")
    h_mmap = np.lib.format.open_memmap(str(CACHE_DIR / "hidden_states.npy"), mode='r')
    t_mmap = np.lib.format.open_memmap(str(CACHE_DIR / "target_ids.npy"), mode='r')
    total_samples = h_mmap.shape[0]
    
    T = mx.eye(HIDDEN_DIM, dtype=mx.float32)
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    
    def compute_arcface_loss(logits, targets, margin=0.5, scale=30.0):
        cos_theta = mx.linalg.norm(logits, axis=-1, keepdims=True)
        cos_theta = logits / mx.maximum(cos_theta, 1e-8)
        
        target_mask = mx.zeros_like(logits)
        target_mask[mx.arange(targets.shape[0]), targets] = 1.0
        
        target_logits = mx.sum(cos_theta * target_mask, axis=-1, keepdims=True)
        sin_theta = mx.sqrt(1.0 - mx.clip(target_logits**2, 0, 1))
        cos_theta_m = target_logits * np.cos(margin) - sin_theta * np.sin(margin)
        
        final_logits = mx.where(target_mask == 1.0, cos_theta_m, cos_theta)
        final_logits = final_logits * scale
        
        ce_loss = mx.mean(mx.losses.cross_entropy(final_logits, targets))
        return ce_loss

    def loss_fn(T_mat, X, Y):
        X_trans = X @ T_mat
        logits = X_trans @ W.T
        loss_ce = compute_arcface_loss(logits, Y)
        procrustes = mx.mean((T_mat.T @ T_mat - mx.eye(HIDDEN_DIM))**2)
        return loss_ce + procrustes
        
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    
    print("\n[3/3] Commencing Vacuum Training (MLX)...")
    for epoch in range(EPOCHS):
        indices = np.random.choice(total_samples, size=BATCH_SIZE, replace=False)
        X_batch = mx.array(h_mmap[indices])
        Y_batch = mx.array(t_mmap[indices])
        
        loss, grads = loss_and_grad_fn(T, X_batch, Y_batch)
        T = optimizer.apply_gradients({'T': grads}, {'T': T})['T']
        mx.eval(T)
        
        if epoch % 10 == 0:
            mem_info = mx.metal.get_active_memory() / (1024**3)
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | Peak VRAM: {mem_info:.2f} GB")
            
    print("\nTraining Complete!")
    np.save(CACHE_DIR / "T_matrix.npy", np.array(T))
    print("Saved T_matrix.npy to disk.")

def run_pytorch():
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/3] Locating isolated lm_head via PyTorch ({DEVICE})...")
    
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(MODEL_PATH, allow_patterns=["*.safetensors"]))
        
    safetensor_files = list(model_path.glob("*.safetensors"))
    keys_to_try = ["lm_head.weight", "language_model.lm_head.weight", "model.embed_tokens.weight"]
    
    W_tensor = None
    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in keys_to_try:
                if key in f.keys():
                    W_tensor = f.get_tensor(key)
                    break
        if W_tensor is not None:
            break
            
    if W_tensor is None:
        raise ValueError("Could not find lm_head matrix.")
        
    W = W_tensor.to(torch.float32).to(DEVICE)
    HIDDEN_DIM = W.shape[1]
    
    print("\n[2/3] Mapping cached hidden states...")
    h_mmap = np.lib.format.open_memmap(str(CACHE_DIR / "hidden_states.npy"), mode='r')
    t_mmap = np.lib.format.open_memmap(str(CACHE_DIR / "target_ids.npy"), mode='r')
    total_samples = h_mmap.shape[0]
    
    T = torch.eye(HIDDEN_DIM, dtype=torch.float32, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.AdamW([T], lr=LEARNING_RATE)
    
    def compute_arcface_loss(logits, targets, margin=0.5, scale=30.0):
        cos_theta = F.normalize(logits, p=2, dim=-1)
        target_logits = cos_theta.gather(1, targets.view(-1, 1))
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logits, 2).clamp(0, 1))
        cos_theta_m = target_logits * np.cos(margin) - sin_theta * np.sin(margin)
        final_logits = cos_theta.scatter(1, targets.view(-1, 1), cos_theta_m)
        return F.cross_entropy(final_logits * scale, targets)

    print("\n[3/3] Commencing Vacuum Training (PyTorch)...")
    for epoch in range(EPOCHS):
        indices = np.random.choice(total_samples, size=BATCH_SIZE, replace=False)
        X_batch = torch.tensor(h_mmap[indices], dtype=torch.float32, device=DEVICE)
        Y_batch = torch.tensor(t_mmap[indices], dtype=torch.long, device=DEVICE)
        
        X_translated = torch.matmul(X_batch, T)
        logits = torch.matmul(X_translated, W.t())
        
        loss_ce = compute_arcface_loss(logits, Y_batch)
        procrustes_loss = torch.mean((torch.matmul(T.t(), T) - torch.eye(HIDDEN_DIM, device=DEVICE))**2)
        total_loss = loss_ce + procrustes_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            print(f"Epoch {epoch:04d} | Loss: {total_loss.item():.4f} | Peak VRAM: {peak_vram:.2f} GB")
            
    print("\nTraining Complete!")
    np.save(CACHE_DIR / "T_matrix.npy", T.detach().cpu().numpy())
    print("Saved T_matrix.npy to disk.")

if __name__ == "__main__":
    print("=== ARC-SPACE VACUUM TRAINING ===")
    if platform.system() == "Darwin":
        print("Hardware Detected: Apple Silicon (Mac)")
        run_mlx()
    else:
        print("Hardware Detected: Windows/Linux")
        run_pytorch()
