import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.losses = []
        self.eval_metrics = {'train_loss': [], 'val_loss': [], 'test_loss': None}
        
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Evaluate the model on given data."""
        with torch.no_grad():
            logits = self.model(X)
            loss = F.cross_entropy(logits, Y)
            return loss.item()
    
    def train(self, Xtr: torch.Tensor, Ytr: torch.Tensor, Xdev: torch.Tensor, Ydev: torch.Tensor) -> Dict[str, List[float]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
        for i in range(self.config.NUM_EPOCHS):
            # Get random minibatch
            ix = torch.randint(0, Xtr.shape[0], (self.config.BATCH_SIZE,))
            Xbatch, Ybatch = Xtr[ix], Ytr[ix]
            
            # Forward pass
            logits = self.model(Xbatch)
            loss = F.cross_entropy(logits, Ybatch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track stats
            self.losses.append(loss.item())
            
            # Evaluate on train and validation sets periodically
            if i % 10000 == 0:
                train_loss = self.evaluate(Xtr, Ytr)
                val_loss = self.evaluate(Xdev, Ydev)
                self.eval_metrics['train_loss'].append(train_loss)
                self.eval_metrics['val_loss'].append(val_loss)
                print(f"Step {i}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Learning rate decay
            if i == self.config.LR_DECAY_AFTER:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.config.LR_DECAY_VALUE
                    
        return self.eval_metrics
    
    def test(self, Xte: torch.Tensor, Yte: torch.Tensor) -> float:
        """Evaluate the model on test set."""
        test_loss = self.evaluate(Xte, Yte)
        self.eval_metrics['test_loss'] = test_loss
        return test_loss
    
    def plot_loss(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(12, 5))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.losses)
        plt.title('Training Loss Over Time (Every Step)')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        
        # Plot evaluation metrics
        plt.subplot(1, 2, 2)
        steps = list(range(0, self.config.NUM_EPOCHS, 10000))
        plt.plot(steps, self.eval_metrics['train_loss'], label='Train Loss')
        plt.plot(steps, self.eval_metrics['val_loss'], label='Validation Loss')
        plt.title('Train vs Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()