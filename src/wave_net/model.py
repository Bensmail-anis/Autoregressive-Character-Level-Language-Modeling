import torch
import torch.nn.functional as F
from layers import (
    Sequential, Embedding, FlattenConsecutive, 
    Linear, BatchNorm1d, Tanh
)

class NameGeneratorModel:
    def __init__(self, vocab_size, n_embd=24, n_hidden=128):
        self.model = Sequential([
            Embedding(vocab_size, n_embd),
            FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), 
            BatchNorm1d(n_hidden), Tanh(),
            FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), 
            BatchNorm1d(n_hidden), Tanh(),
            FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), 
            BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, vocab_size),
        ])
        
        # Initialize parameters
        with torch.no_grad():
            self.model.layers[-1].weight *= 0.1
        
        self.parameters = self.model.parameters()
        for p in self.parameters:
            p.requires_grad = True
    
    def __call__(self, x):
        return self.model(x)
    
    def generate_names(self, itos, block_size=8, num_names=20, temperature=1.0):
        generated_names = []
        
        for _ in range(num_names):
            out = []
            context = [0] * block_size
            
            while True:
                # Forward pass
                logits = self(torch.tensor([context]))
                
                # Apply temperature scaling for better probability distribution
                scaled_logits = logits / temperature
                
                # Numerical stability: clip extreme values
                scaled_logits = torch.clamp(scaled_logits, min=-20, max=20)
                
                # Softmax with numerical stability
                probs = F.softmax(scaled_logits, dim=1)
                
                # Safe sampling
                try:
                    ix = torch.multinomial(probs, num_samples=1).item()
                except RuntimeError:
                    print("Sampling failed, using argmax")
                    ix = torch.argmax(probs, dim=1).item()
                
                context = context[1:] + [ix]
                out.append(ix)
                
                if ix == 0 or len(out) > 20:  # Prevent infinite loops
                    break
            
            generated_names.append(''.join(itos[i] for i in out))
        
        return generated_names