import torch
import random
from datasets import load_dataset

def get_internet_data_stream():
    """
    Streams high-quality educational data (Math, Science, Tech) 
    using the FineWeb-Edu dataset.
    """
    # FineWeb-Edu is the gold standard for avoiding 'junk' and 'abstract' text.
    # We use 'sample-10BT' which is a massive, high-quality technical subset.
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    
    # These keywords act as a 'hard filter' to keep the model strictly technical
    target_keywords = [
        # --- CONSTRUCTION & CIVIL ENGINEERING ---
        "structural engineering", "architecture", "concrete", "steel", "masonry",
        "foundation", "load-bearing", "geotechnical", "blueprint", "scaffolding",
        "hvac", "plumbing", "electrical wiring", "urban planning", "infrastructure",
        "bim", "cad", "materials science", "statics", "dynamics", "surveying",
        
        # --- HARD SCIENCES & MATH ---
        "physics", "quantum", "mathematics", "calculus", "algebra", "chemistry",
        "thermodynamics", "electromagnetism", "stochastic",
        
        # --- HUMAN MEDICINE ---
        "pathology", "physiology", "pharmacology", "neuroscience", "anatomy", 
        "immunology", "cardiology", "surgery", "diagnostics",
        
        # --- BOTANY & AGRICULTURE ---
        "agronomy", "soil science", "irrigation", "botany", "silviculture", 
        "husbandry", "veterinary", "nitrogen cycle", "hydroponics",
        
        # --- TECHNICAL & BUSINESS ---
        "accounting", "pib", "law", "programming", "algorithm", "mechanics",
        "cryptography", "semiconductor", "logistics", "supply chain"
    ]

    def tech_filter(iterable):
        for example in iterable:
            text_lower = example['text'].lower()
            # If the article doesn't talk about your science/tech interests, skip it
            if any(word in text_lower for word in target_keywords):
                yield example

    return iter(tech_filter(dataset))

def get_streaming_batch(stream_iterator, tokenizer, block_size, batch_size, device):
    """
    Fetches the next chunk of technical data.
    """
    xs, ys = [], []
    count = 0
    
    while count < batch_size:
        try:
            example = next(stream_iterator)
            text = example['text']
            
            # Convert text to tokens
            tokens = tokenizer.encode(text)
            
            # Only use if text is long enough for our block_size
            if len(tokens) > block_size:
                # Pick a random starting point in this technical article
                start_idx = random.randint(0, len(tokens) - block_size - 1)
                
                x = torch.tensor(tokens[start_idx : start_idx + block_size], dtype=torch.long)
                y = torch.tensor(tokens[start_idx + 1 : start_idx + block_size + 1], dtype=torch.long)
                
                xs.append(x)
                ys.append(y)
                count += 1
        except StopIteration:
            # Note: FineWeb-Edu is so massive, you will likely never hit the end.
            break
            
    # Safety check: if we couldn't fill a batch, we return what we have or handle error
    if not xs:
        return None, None
        
    return torch.stack(xs).to(device), torch.stack(ys).to(device)