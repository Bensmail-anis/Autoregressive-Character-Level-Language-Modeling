import torch
import random
from typing import Tuple, List, Dict

class DataProcessor:
    def __init__(self, filename: str):
        self.words = open(filename, 'r').read().splitlines()
        self.chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s:i+1 for i,s in enumerate(self.chars)}
        self.stoi['.'] = 0
        self.itos = {i:s for s,i in self.stoi.items()}
        
    def build_dataset(self, words: List[str], block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X, Y = [], []
        for w in words:
            context = [0] * block_size
            for ch in w + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        return torch.tensor(X), torch.tensor(Y)
    
    def get_train_val_test_split(self, block_size: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        random.seed(seed)
        words = self.words.copy()
        random.shuffle(words)
        
        n1 = int(0.8 * len(words))
        n2 = int(0.9 * len(words))
        
        Xtr, Ytr = self.build_dataset(words[:n1], block_size)
        Xdev, Ydev = self.build_dataset(words[n1:n2], block_size)
        Xte, Yte = self.build_dataset(words[n2:], block_size)
        
        return Xtr, Ytr, Xdev, Ydev, Xte, Yte
