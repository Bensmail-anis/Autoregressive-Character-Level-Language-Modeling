import torch
import matplotlib.pyplot as plt
from nn import NameGeneratorMLP
from data_loader import DataLoader

def train(data_path, max_steps=200000, batch_size=32, lr_decay_point=100000):
    # Load and split data
    loader = DataLoader(data_path)
    (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = loader.split_data()

    # Initialize model
    model = NameGeneratorMLP(
        vocab_size=loader.vocab_size, 
        block_size=loader.block_size
    )

    # Training loop
    lossi = []
    g = torch.Generator().manual_seed(2147483647)

    for i in range(max_steps):
        # Minibatch
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        # Forward pass
        logits = model.forward(Xb)
        loss = model.calculate_loss(logits, Yb)

        # Backward pass
        for p in model.parameters:
            p.grad = None
        loss.backward()

        # Update
        lr = 0.1 if i < lr_decay_point else 0.01
        for p in model.parameters:
            p.data += -lr * p.grad

        # Track stats
        if i % 10000 == 0:
            print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())

    # Plotting
    plt.plot(lossi)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Log Loss')
    plt.show()

    return model, loader

def evaluate_split_loss(model, loader, split):
    x, y = {
        'train': loader.split_data()[0],
        'val': loader.split_data()[1],
        'test': loader.split_data()[2],
    }[split]

    with torch.no_grad():
        emb = model.C[x]
        embcat = emb.view(emb.shape[0], -1)
        hpreact = embcat @ model.W1
        hpreact = model.bngain * (hpreact - model.bnmean_running) / model.bnstd_running + model.bnbias
        h = torch.tanh(hpreact)
        logits = h @ model.W2 + model.b2
        loss = model.calculate_loss(logits, y)
        print(f'{split} loss: {loss.item()}')