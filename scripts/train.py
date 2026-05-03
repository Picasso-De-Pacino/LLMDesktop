import sys
import os
import torch
import random

# This adds the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can safely import from src
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

# --- 2. HYPERPARAMETERS (SCALED FOR 2B) ---
# For a 2B model on a TPU v2-8, we must keep batch_size small per core
batch_size = 4 if is_tpu else 1 
block_size = 1024       # Increased context for a 2B model
max_iters = 10000       # 2B models need many more steps to converge
learning_rate = 3e-4    # Lowered learning rate for stability at large scale
eval_interval = 50
save_interval = 250

# --- 3. DATA & TOKENIZER SETUP ---
print("📡 Initializing internet data stream...")
stream = get_internet_data_stream()

all_chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
tokenizer = CharacterTokenizer(all_chars) 
vocab_size = tokenizer.vocab_size

# --- 4. MODEL INITIALIZATION (2 BILLION PARAMETERS) ---
# n_embd=2048, n_head=16, n_layer=24 creates ~2.1B parameters
model = BigramLanguageModel(
    vocab_size=vocab_size, 
    n_embd=2048, 
    n_head=16, 
    n_layer=24, 
    block_size=block_size
).to(device)

# --- 🧠 RESUME LOGIC ---
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'model.pt')

if os.path.exists(checkpoint_path):
    print(f"📂 Found existing brain at {checkpoint_path}. Loading weights...")
    try:
        # map_location='cpu' is critical when switching hardware or scaling
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("✅ Resume successful. Continuing training...")
    except Exception as e:
        print(f"⚠️ Could not load checkpoint: {e}. Starting fresh (likely due to size change).")
else:
    print("🌱 No existing brain found. Starting from scratch.")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- 5. TRAINING LOOP ---
print(f"🔥 Starting training 2B parameter model on {vocab_size} unique characters...")

model.train() 
for iter in range(max_iters):
    
    # Get a batch of data
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
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Checkpoint saved at step {iter}")

# --- 6. FINAL SAVE ---
torch.save(model.state_dict(), checkpoint_path)
print(f"✅ Training complete. 2B Model saved to {checkpoint_path}")