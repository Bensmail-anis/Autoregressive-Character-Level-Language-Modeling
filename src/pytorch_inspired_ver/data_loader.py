import torch
import random

class DataLoader:
    def __init__(self, filename, block_size=3):
        self.block_size = block_size
        self.words = open(filename, 'r').read().splitlines()
        
        # Build vocabulary
        self.chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s:i+1 for i,s in enumerate(self.chars)}
        self.stoi['.'] = 0
        self.itos = {i:s for s,i in self.stoi.items()}
        self.vocab_size = len(self.itos)

    def build_dataset(self, words):
        X, Y = [], []

        for w in words:
            context = [0] * self.block_size
            for ch in w + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    def split_data(self, test_ratio=0.2, val_ratio=0.1):
        random.seed(42)
        random.shuffle(self.words)
        
        n1 = int((1-test_ratio-val_ratio)*len(self.words))
        n2 = int((1-test_ratio)*len(self.words))

        Xtr, Ytr = self.build_dataset(self.words[:n1])
        Xdev, Ydev = self.build_dataset(self.words[n1:n2])
        Xte, Yte = self.build_dataset(self.words[n2:])

        return (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte)