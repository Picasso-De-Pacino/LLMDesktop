import torch
from datasets import load_dataset

def get_internet_data_stream():
    """
    Streams wikitext data directly from Hugging Face.
    This version uses the modern Parquet-based format.
    """
    # Using 'wikitext-103-v1' as a reliable alternative to the old wiki script
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    return iter(dataset)

def get_streaming_batch(stream_iterator, tokenizer, block_size, batch_size, device):
    """
    Fetches the next chunk of data from the internet stream.
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
                # Pick a random starting point in this article
                import random
                start_idx = random.randint(0, len(tokens) - block_size - 1)
                
                x = torch.tensor(tokens[start_idx : start_idx + block_size], dtype=torch.long)
                y = torch.tensor(tokens[start_idx + 1 : start_idx + block_size + 1], dtype=torch.long)
                
                xs.append(x)
                ys.append(y)
                count += 1
        except StopIteration:
            # If the stream ends, we'd need to re-initialize or break
            break
            
    return torch.stack(xs).to(device), torch.stack(ys).to(device)