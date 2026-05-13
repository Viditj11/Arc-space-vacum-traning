import platform
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import gc

# ==========================================
# PHASE 1: EXTRACT CACHE
# ==========================================
# This script intelligently detects your hardware and automatically uses 
# the correct backend: MLX for Apple Silicon, or PyTorch for Windows/Linux.

MODEL_PATH = "google/gemma-2-27b-it"  # Replace with actual model ID/path
CACHE_DIR = Path("./vacuum_cache")
CACHE_DIR.mkdir(exist_ok=True)

def generate_dummy_data_mlx(tokenizer, num_samples=50):
    dataset = []
    for i in range(num_samples):
        prompt = f"<bos><start_of_turn>user\nExplain concept {i}.<end_of_turn>\n<start_of_turn>model\nConcept {i} is an important mathematical principle.<end_of_turn><eos>"
        encoded = tokenizer.encode(prompt)
        dataset.append(encoded)
    return dataset

def generate_dummy_data_pt(tokenizer, num_samples=50):
    dataset = []
    for i in range(num_samples):
        prompt = f"<bos><start_of_turn>user\nExplain concept {i}.<end_of_turn>\n<start_of_turn>model\nConcept {i} is an important mathematical principle.<end_of_turn><eos>"
        encoded = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        dataset.append(encoded)
    return dataset

def run_mlx():
    import mlx.core as mx
    from mlx_lm import load
    
    print(f"\n[1/3] Loading base model ({MODEL_PATH}) via MLX...")
    model, tokenizer = load(str(MODEL_PATH))
    mx.eval(model.parameters())
    
    dataset = generate_dummy_data_mlx(tokenizer)
    
    print("\n[2/3] Extracting hidden states...")
    lm = model.language_model if hasattr(model, 'language_model') else (model.model if hasattr(model, 'model') else model)
    transformer = lm.model if hasattr(lm, 'model') else lm
    
    # Infer hidden dim
    dummy_h = transformer.embed_tokens(mx.array([[tokenizer.eos_token_id]]))
    hidden_dim = dummy_h.shape[-1]
    total_tokens = sum(len(seq) - 1 for seq in dataset)
    
    print(f"Total tokens: {total_tokens} | Hidden dim: {hidden_dim}")
    
    h_path = CACHE_DIR / "hidden_states.npy"
    t_path = CACHE_DIR / "target_ids.npy"
    h_mmap = np.lib.format.open_memmap(str(h_path), mode='w+', dtype=np.float32, shape=(total_tokens, hidden_dim))
    t_mmap = np.lib.format.open_memmap(str(t_path), mode='w+', dtype=np.int32, shape=(total_tokens,))
    
    ptr = 0
    for seq in tqdm(dataset, desc="Caching"):
        input_ids = mx.array([seq])
        hidden = transformer.embed_tokens(input_ids)
        for layer in transformer.layers:
            hidden = layer(hidden)
        if hasattr(transformer, 'norm'):
            hidden = transformer.norm(hidden)
            
        h_f32 = hidden[0, :-1, :].astype(mx.float32)
        t_ids = input_ids[0, 1:].astype(mx.int32)
        mx.eval(h_f32, t_ids)
        
        seq_len = h_f32.shape[0]
        h_mmap[ptr : ptr+seq_len] = np.array(h_f32, copy=False)
        t_mmap[ptr : ptr+seq_len] = np.array(t_ids, copy=False)
        ptr += seq_len
        
    h_mmap.flush()
    t_mmap.flush()
    print(f"Successfully cached {total_tokens} tokens to {CACHE_DIR}/")
    
    print("\n[3/3] Purging base model from VRAM...")
    del model
    del transformer
    del lm
    mx.eval(mx.zeros(1))
    print("VRAM Purge Complete! Ready for Phase 2.")

def run_pytorch():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n[1/3] Loading base model ({MODEL_PATH}) via PyTorch/CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    
    dataset = generate_dummy_data_pt(tokenizer)
    
    print("\n[2/3] Extracting hidden states...")
    with torch.no_grad():
        dummy_input = torch.tensor([[tokenizer.bos_token_id]]).to(model.device)
        dummy_out = model(dummy_input, output_hidden_states=True)
        hidden_dim = dummy_out.hidden_states[-1].shape[-1]
        
    total_tokens = sum(len(seq) - 1 for seq in dataset)
    print(f"Total tokens: {total_tokens} | Hidden dim: {hidden_dim}")
    
    h_path = CACHE_DIR / "hidden_states.npy"
    t_path = CACHE_DIR / "target_ids.npy"
    h_mmap = np.lib.format.open_memmap(str(h_path), mode='w+', dtype=np.float32, shape=(total_tokens, hidden_dim))
    t_mmap = np.lib.format.open_memmap(str(t_path), mode='w+', dtype=np.int32, shape=(total_tokens,))
    
    ptr = 0
    for input_ids in tqdm(dataset, desc="Caching"):
        input_ids_gpu = input_ids.unsqueeze(0).to(model.device)
        with torch.no_grad():
            outputs = model(input_ids_gpu, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            
        h_f32 = hidden[0, :-1, :].to(torch.float32).cpu().numpy()
        t_ids = input_ids[1:].cpu().numpy().astype(np.int32)
        
        seq_len = h_f32.shape[0]
        h_mmap[ptr : ptr+seq_len] = h_f32
        t_mmap[ptr : ptr+seq_len] = t_ids
        ptr += seq_len
        
    h_mmap.flush()
    t_mmap.flush()
    print(f"Successfully cached {total_tokens} tokens to {CACHE_DIR}/")
    
    print("\n[3/3] Purging base model from VRAM...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VRAM Purge Complete! Ready for Phase 2.")

if __name__ == "__main__":
    print("=== ARC-SPACE VACUUM TRAINING ===")
    if platform.system() == "Darwin":
        print("Hardware Detected: Apple Silicon (Mac)")
        run_mlx()
    else:
        print("Hardware Detected: Windows/Linux")
        run_pytorch()
