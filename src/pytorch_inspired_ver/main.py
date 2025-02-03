import torch
import torch.nn.functional as F
from data_loader import DataLoader
from trainer import train_model, evaluate_splits

def main():
    # Load data
    data_loader = DataLoader('../../data/names.txt')
    (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = data_loader.split_data()

    # Train model
    model = train_model(
        Xtr, Ytr, Xdev, Ydev, Xte, Yte, 
        vocab_size=data_loader.vocab_size, 
        block_size=data_loader.block_size
    )

    # Evaluate splits
    evaluate_splits(model, Xtr, Ytr, Xdev, Ydev, Xte, Yte)

    # Generate names
    generated_names = model.generate_names(
        num_names=20, 
        itos=data_loader.itos
    )
    print("Generated Names:")
    print(generated_names)

if __name__ == "__main__":
    main()