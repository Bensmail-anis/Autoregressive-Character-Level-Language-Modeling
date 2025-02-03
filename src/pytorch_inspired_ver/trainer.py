import torch
import matplotlib.pyplot as plt
from model import NameGenerator
import torch.nn.functional as F

def train_model(Xtr, Ytr, Xdev, Ydev, Xte, Yte, vocab_size, block_size):
    model = NameGenerator(vocab_size, block_size)
    lossi = []
    
    max_steps = 200000
    batch_size = 32
    g = torch.Generator().manual_seed(2147483647)

    for i in range(max_steps):
        # Minibatch
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        # Learning rate decay
        lr = 0.1 if i < 150000 else 0.01
        
        # Train step
        loss = model.train_step(Xb, Yb, lr)
        lossi.append(loss)

        # Periodic logging
        if i % 10000 == 0:
            print(f'{i:7d}/{max_steps:7d}: {loss:.4f}')

        # Early stopping for demonstration
        if i >= 1000:
            break

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(lossi)
    plt.title('Training Loss')
    plt.show()

    return model

def evaluate_splits(model, Xtr, Ytr, Xdev, Ydev, Xte, Yte):
    model.layers[-1].training = False
    
    for split, (X, Y) in [('train', (Xtr, Ytr)), 
                          ('val', (Xdev, Ydev)), 
                          ('test', (Xte, Yte))]:
        emb = model.C[X]
        x = emb.view(emb.shape[0], -1)
        for layer in model.layers:
            x = layer(x)
        loss = F.cross_entropy(x, Y)
        print(f'{split} loss: {loss.item()}')