import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, dataset, max_steps=200000, batch_size=32):
        self.model = model
        self.dataset = dataset
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.lossi = []
    
    def train(self):
        for i in range(self.max_steps):
            # Minibatch
            ix = torch.randint(0, self.dataset.Xtr.shape[0], (self.batch_size,))
            Xb, Yb = self.dataset.Xtr[ix], self.dataset.Ytr[ix]
            
            # Forward pass
            logits = self.model(Xb)
            loss = F.cross_entropy(logits, Yb)
            
            # Backward pass
            for p in self.model.parameters:
                p.grad = None
            loss.backward()
            
            # Update
            lr = 0.1 if i < 150000 else 0.01
            for p in self.model.parameters:
                p.data += -lr * p.grad
            
            # Track stats
            if i % 10000 == 0:
                print(f'{i:7d}/{self.max_steps:7d}: {loss.item():.4f}')
            
            self.lossi.append(loss.log10().item())
        
        return self.lossi
    
    def plot_loss(self):
        plt.plot(torch.tensor(self.lossi).view(-1, 1000).mean(1))
        plt.title('Training Loss')
        plt.xlabel('Iteration (per 1000 steps)')
        plt.ylabel('Log Loss')
        plt.show()
    
    def evaluate_loss(self, split='val'):
        x, y = self.dataset.get_splits()[split]
        with torch.no_grad():
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
        print(f'{split} loss: {loss.item()}')