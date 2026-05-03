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

# --- 2. HYPERPARAMETERS (SCALED FOR TPU STABILITY) ---
# We use a very small batch size to prioritize model depth over breadth
batch_size = 2 if is_tpu else 1 
block_size = 512        # Reduced context to save activation memory
max_iters = 10000       
learning_rate = 5e-4    # Slightly higher for a smaller model
eval_interval = 100
save_interval = 500

# --- 3. DATA & TOKENIZER SETUP ---
print("📡 Initializing internet data stream...")
stream = get_internet_data_stream()

all_chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
tokenizer = CharacterTokenizer(all_chars) 
vocab_size = tokenizer.vocab_size

# --- 4. MODEL INITIALIZATION (BALANCED FOR 8GB TPU) ---
# n_embd=1024, n_head=16, n_layer=12 fits comfortably in 8GB HBM
model = BigramLanguageModel(
    vocab_size=vocab_size, 
    n_embd=1024, 
    n_head=16, 
    n_layer=12, 
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
print(f"🔥 Starting training 1B parameter model on {vocab_size} unique characters...")

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