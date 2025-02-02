import torch
import torch.nn.functional as F
from typing import List

class MLPLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, block_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Initialize layers
        self.C = torch.nn.Parameter(torch.randn((vocab_size, embedding_dim)))
        self.W1 = torch.nn.Parameter(torch.randn((block_size * embedding_dim, hidden_dim)))
        self.b1 = torch.nn.Parameter(torch.randn(hidden_dim))
        self.W2 = torch.nn.Parameter(torch.randn((hidden_dim, vocab_size)))
        self.b2 = torch.nn.Parameter(torch.randn(vocab_size))
        
    def forward(self, idx):
        emb = self.C[idx]  # (batch, block_size, embedding_dim)
        x = emb.view(-1, self.block_size * self.C.shape[1])  # (batch, block_size * embedding_dim)
        h = torch.tanh(x @ self.W1 + self.b1)  # (batch, hidden_dim)
        logits = h @ self.W2 + self.b2  # (batch, vocab_size)
        return logits
    
    def generate(self, context: List[int], max_tokens: int, temperature: float = 1.0) -> List[int]:
        for _ in range(max_tokens):
            x = torch.tensor([context[-self.block_size:]])
            logits = self.forward(x)
            probs = F.softmax(logits * (1.0/temperature), dim=-1)
            next_ix = torch.multinomial(probs, num_samples=1).item()
            context.append(next_ix)
            if next_ix == 0:  # end token
                break
        return context
