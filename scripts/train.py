import torch
import random
import os
from src.model import BigramLanguageModel
from src.tokenizer import CharacterTokenizer
from src.data_loader import get_internet_data_stream, get_streaming_batch

# --- 1. HARDWARE DETECTION ---
device = None
is_tpu = False

try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    is_tpu = True
    print(f"🚀 Hardware Found: TPU ({device})")
except ImportError:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎮 Hardware Found: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("💻 Hardware Found: CPU")

# --- 2. HYPERPARAMETERS ---
# Balanced for 8GB RAM and GTX 860M
batch_size = 16 
block_size = 128 
max_iters = 5000
learning_rate = 1e-3
eval_interval = 100
save_interval = 500

# --- 3. DATA & TOKENIZER SETUP ---
print("📡 Initializing internet data stream...")
stream = get_internet_data_stream()

# Grab a small sample to build the initial vocabulary
# In a real scenario, you'd use a pre-built vocab file for consistency
sample_data = next(stream)
tokenizer = CharacterTokenizer(sample_data['text'])
vocab_size = tokenizer.vocab_size

# --- 4. MODEL INITIALIZATION ---
model = BigramLanguageModel(
    vocab_size=vocab_size, 
    n_embd=128, 
    n_head=4, 
    n_layer=4, 
    block_size=block_size
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Ensure checkpoint directories exist
os.makedirs('checkpoints/best_model', exist_ok=True)

# --- 5. TRAINING LOOP ---
print(f"🔥 Starting training on {vocab_size} unique characters...")

for iter in range(max_iters):
    
    # Get a batch of data from the internet
    xb, yb = get_streaming_batch(stream, tokenizer, block_size, batch_size, device)
    
    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Hardware-specific step execution
    if is_tpu:
        xm.optimizer_step(optimizer)
        xm.mark_step()
    else:
        optimizer.step()

    # Progress logging
    if iter % eval_interval == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")

    # Periodic Saving
    if iter % save_interval == 0 and iter > 0:
        torch.save(model.state_dict(), 'checkpoints/best_model/model.pt')
        print(f"💾 Checkpoint saved at step {iter}")

# --- 6. FINAL SAVE ---
torch.save(model.state_dict(), 'checkpoints/best_model/model.pt')
print("✅ Training complete. Model saved to checkpoints/best_model/model.pt")