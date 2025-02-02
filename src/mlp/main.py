from config import Config
from data import DataProcessor
from model import MLPLanguageModel
from trainer import Trainer

def main():
    # Initialize data processor and load data
    data_processor = DataProcessor('../../data/names.txt')
    Xtr, Ytr, Xdev, Ydev, Xte, Yte = data_processor.get_train_val_test_split(
        block_size=Config.BLOCK_SIZE
    )
    
    # Initialize model
    model = MLPLanguageModel(
        vocab_size=len(data_processor.itos),
        block_size=Config.BLOCK_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM
    )
    
    # Train model
    trainer = Trainer(model, Config)
    metrics = trainer.train(Xtr, Ytr, Xdev, Ydev)
    
    # Evaluate on test set
    test_loss = trainer.test(Xte, Yte)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    
    # Plot losses
    trainer.plot_loss()
    
    # Generate some names
    print("\nGenerated names:")
    for _ in range(10):
        context = [0] * Config.BLOCK_SIZE
        generated = model.generate(context, max_tokens=20, temperature=0.8)
        name = ''.join(data_processor.itos[ix] for ix in generated)
        print(name)

if __name__ == '__main__':
    main()