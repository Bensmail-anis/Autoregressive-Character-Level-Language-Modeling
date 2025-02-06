import torch
import random

class NameDataset:
    def __init__(self, filename='../../data/names.txt', block_size=8):
        # Read words
        self.words = open(filename, 'r').read().splitlines()
        self.block_size = block_size
        
        # Build vocabulary
        self.chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s:i+1 for i,s in enumerate(self.chars)}
        self.stoi['.'] = 0
        self.itos = {i:s for s,i in self.stoi.items()}
        self.vocab_size = len(self.itos)
        
        # Shuffle and split data
        random.seed(42)
        random.shuffle(self.words)
        
        # Split ratios
        n1 = int(0.8*len(self.words))
        n2 = int(0.9*len(self.words))
        
        self.Xtr, self.Ytr = self.build_dataset(self.words[:n1])
        self.Xdev, self.Ydev = self.build_dataset(self.words[n1:n2])
        self.Xte, self.Yte = self.build_dataset(self.words[n2:])
    
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
    
    def get_splits(self):
        return {
            'train': (self.Xtr, self.Ytr),
            'val': (self.Xdev, self.Ydev),
            'test': (self.Xte, self.Yte)
        }