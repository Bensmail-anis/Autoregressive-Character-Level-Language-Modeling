from model import BigramLanguageModel
from data_loader import load_names

def main():
    # Configuration
    DATA_FILE = "../../data/names.txt"
    NUM_EPOCHS = 100
    NUM_NAMES = 10
    
    # Load data
    words = load_names(DATA_FILE)
    if not words:
        return
    
    print(f"Loaded {len(words)} names from dataset")
    
    # Create and train model
    model = BigramLanguageModel()
    model.create_mapping(words)
    print("Training model...")
    model.train(words, num_epochs=NUM_EPOCHS)
    
    # Generate names
    print("\nGenerated names:")
    names = model.generate_names(num_names=NUM_NAMES)
    for i, name in enumerate(names, 1):
        print(f"{i}. {name}")

if __name__ == "__main__":
    main()