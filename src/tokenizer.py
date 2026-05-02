class CharacterTokenizer:
    def __init__(self, text=None):
        # Default characters to ensure vocab_size is never 0
        default_chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\n")
        if text:
            all_chars = sorted(list(set(list(text) + default_chars)))
        else:
            all_chars = sorted(default_chars)
            
        self.chars = all_chars
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s):
        # Use .get(c, 0) to handle unknown characters gracefully
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, l):
        return ''.join([self.itos.get(i, '<UNK>') for i in l])