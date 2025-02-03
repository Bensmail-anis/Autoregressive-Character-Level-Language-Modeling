import torch
import torch.nn.functional as F

class NameGeneratorMLP:
    def __init__(self, vocab_size, block_size, n_embd=10, n_hidden=200):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_hidden = n_hidden

        g = torch.Generator().manual_seed(2147483647)
        
        # Model parameters
        self.C = torch.randn((vocab_size, n_embd), generator=g)
        self.W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
        self.W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
        self.b2 = torch.randn(vocab_size, generator=g) * 0

        # BatchNorm parameters
        self.bngain = torch.ones((1, n_hidden))
        self.bnbias = torch.zeros((1, n_hidden))
        self.bnmean_running = torch.zeros((1, n_hidden))
        self.bnstd_running = torch.ones((1, n_hidden))

        self.parameters = [self.C, self.W1, self.W2, self.b2, self.bngain, self.bnbias]
        for p in self.parameters:
            p.requires_grad = True

    def forward(self, X):
        # Embedding
        emb = self.C[X]
        embcat = emb.view(emb.shape[0], -1)
        
        # Linear layer
        hpreact = embcat @ self.W1
        
        # BatchNorm layer
        bnmeani = hpreact.mean(0, keepdim=True)
        bnstdi = hpreact.std(0, keepdim=True)
        hpreact = self.bngain * (hpreact - bnmeani) / bnstdi + self.bnbias
        
        # Update running stats
        with torch.no_grad():
            self.bnmean_running = 0.999 * self.bnmean_running + 0.001 * bnmeani
            self.bnstd_running = 0.999 * self.bnstd_running + 0.001 * bnstdi
        
        # Non-linearity and output
        h = torch.tanh(hpreact)
        logits = h @ self.W2 + self.b2
        
        return logits

    def calculate_loss(self, logits, targets):
        return F.cross_entropy(logits, targets)