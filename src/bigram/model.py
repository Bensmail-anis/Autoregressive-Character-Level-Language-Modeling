import torch
import torch.nn.functional as F

class BigramLanguageModel:
    def __init__(self, num_chars=27):
        self.num_chars = num_chars
        self.W = None
        self.stoi = None
        self.itos = None
    
    def create_mapping(self, words):
        # Create character to index mapping
        chars = sorted(list(set(''.join(words))))
        self.stoi = {s:i+1 for i,s in enumerate(chars)}
        self.stoi['.'] = 0  # Special token for start/end
        self.itos = {i:s for s,i in self.stoi.items()}
        
    def prepare_training_data(self, words):
        # Create training pairs
        xs, ys = [], []
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                xs.append(ix1)
                ys.append(ix2)
        return torch.tensor(xs), torch.tensor(ys)
    
    def train(self, words, learning_rate=50, num_epochs=100, seed=2147483647):
        # Initialize model
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn((self.num_chars, self.num_chars), 
                           generator=g, requires_grad=True)
        
        # Prepare data
        xs, ys = self.prepare_training_data(words)
        num = xs.nelement()
        
        # Training loop
        for epoch in range(num_epochs):
            # Forward pass
            xenc = F.one_hot(xs, num_classes=self.num_chars).float()
            logits = xenc @ self.W
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)
            loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(self.W**2).mean()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
            
            # Backward pass
            self.W.grad = None
            loss.backward()
            
            # Update
            self.W.data += -learning_rate * self.W.grad
    
    def generate_names(self, num_names=5, seed=2147483647):
        g = torch.Generator().manual_seed(seed)
        names = []
        
        for _ in range(num_names):
            out = []
            ix = 0
            while True:
                # Get next character probabilities
                xenc = F.one_hot(torch.tensor([ix]), num_classes=self.num_chars).float()
                logits = xenc @ self.W
                counts = logits.exp()
                p = counts / counts.sum(1, keepdims=True)
                
                # Sample next character
                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                if ix == 0:  # End token
                    break
                out.append(self.itos[ix])
            names.append(''.join(out))
        return names