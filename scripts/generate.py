import torch
from src.model import BigramLanguageModel
from src.tokenizer import CharacterTokenizer
from src.data_loader import get_internet_data_stream

# 1. Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 128

# 2. Get the Tokenizer 
stream = get_internet_data_stream()
# Pull a few examples to make sure we have a real vocabulary
vocab_text = ""
for _ in range(10):
    try:
        example = next(stream)
        vocab_text += example['text']
    except StopIteration:
        break

tokenizer = CharacterTokenizer(vocab_text)
print(f"Vocab size initialized: {tokenizer.vocab_size}")

# 3. Initialize the "Untrained" Model
model = BigramLanguageModel(
    vocab_size=tokenizer.vocab_size, 
    n_embd=128, 
    n_head=4, 
    n_layer=4, 
    block_size=block_size
).to(device)

# 4. Generation Function
def generate_text(model, max_new_tokens=500):
    # Start with a "newline" character as the prompt
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    model.eval() # Put model in evaluation mode
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop the context if it exceeds block_size
            idx_cond = idx[:, -block_size:]
            # Get predictions
            logits, _ = model(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
    
    return tokenizer.decode(idx[0].tolist())

print("--- GENERATING FROM UNTRAINED BRAIN ---")
print(generate_text(model))