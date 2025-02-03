import torch
import torch.nn.functional as F

class NameGenerator:
    def __init__(self, vocab_size, block_size, n_embd=10, n_hidden=100):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.g = torch.Generator().manual_seed(2147483647)
        
        # Embedding matrix
        self.C = torch.randn((vocab_size, n_embd), generator=self.g)
        
        # Network layers
        from layers import Linear, BatchNorm1d, Tanh
        self.layers = [
            Linear(n_embd * block_size, n_hidden, bias=False, generator=self.g), 
            BatchNorm1d(n_hidden), 
            Tanh(),
            Linear(n_hidden, n_hidden, bias=False, generator=self.g), 
            BatchNorm1d(n_hidden), 
            Tanh(),
            Linear(n_hidden, n_hidden, bias=False, generator=self.g), 
            BatchNorm1d(n_hidden), 
            Tanh(),
            Linear(n_hidden, n_hidden, bias=False, generator=self.g), 
            BatchNorm1d(n_hidden), 
            Tanh(),
            Linear(n_hidden, n_hidden, bias=False, generator=self.g), 
            BatchNorm1d(n_hidden), 
            Tanh(),
            Linear(n_hidden, vocab_size, bias=False, generator=self.g), 
            BatchNorm1d(vocab_size)
        ]

        # Initialize layer parameters
        with torch.no_grad():
            self.layers[-1].gamma *= 0.1
            for layer in self.layers[:-1]:
                if isinstance(layer, Linear):
                    layer.weight *= 1.0

        # Collect all parameters
        self.parameters = [self.C] + [p for layer in self.layers for p in layer.parameters()]
        for p in self.parameters:
            p.requires_grad = True

    def forward(self, X):
        emb = self.C[X]
        x = emb.view(emb.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x

    def train_step(self, Xb, Yb, lr=0.1):
        # Forward pass
        x = self.forward(Xb)
        loss = F.cross_entropy(x, Yb)

        # Backward pass
        for p in self.parameters:
            p.grad = None
        loss.backward()

        # Update
        for p in self.parameters:
            p.data += -lr * p.grad

        return loss.item()

    def generate_names(self, num_names=20, max_length=10, itos=None):
        generated_names = []
        for _ in range(num_names):
            context = [0] * self.block_size
            out = []
            while True:
                emb = self.C[torch.tensor([context])]
                x = emb.view(emb.shape[0], -1)
                for layer in self.layers:
                    x = layer(x)
                
                # Add temperature and clipping to prevent inf/nan
                logits = x / 1.0  # optional temperature
                probs = F.softmax(torch.clip(logits, min=-10, max=10), dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
                
                context = context[1:] + [ix]
                out.append(ix)
                
                if ix == 0 or len(out) >= max_length:
                    break
            
            if itos:
                generated_names.append(''.join(itos[i] for i in out))
        
        return generated_names