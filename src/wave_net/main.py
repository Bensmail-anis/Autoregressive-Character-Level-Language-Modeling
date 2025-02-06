import sys
import torch
sys.path.append('.')

from data import NameDataset
from model import NameGeneratorModel
from train import Trainer


def main():
    # Load dataset
    dataset = NameDataset()
    
    # Create model
    model = NameGeneratorModel(
        vocab_size=dataset.vocab_size, 
        n_embd=24, 
        n_hidden=128
    )
    
    # Train with potential early stopping
    trainer = Trainer(model, dataset, max_steps=200000)
    
    # Validate loss before generation
    trainer.train()
    trainer.evaluate_loss('train')
    trainer.evaluate_loss('val')
    
    # Ensure model is in evaluation mode
    for layer in model.model.layers:
        layer.training = False
    
    # Generate names with different temperatures
    temperatures = [0.7, 1.0, 1.5]
    for temp in temperatures:
        print(f"\nGeneration with temperature {temp}:")
        generated_names = model.generate_names(
            dataset.itos, 
            block_size=8, 
            num_names=20,
            temperature=temp
        )
        for name in generated_names:
            print(name)

if __name__ == '__main__':
    main()