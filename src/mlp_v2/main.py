from trainer import train, evaluate_split_loss

def main():
    # Train the model
    model, loader = train("../../data/names.txt")

    # Evaluate on different splits
    evaluate_split_loss(model, loader, 'train')
    evaluate_split_loss(model, loader, 'val')

if __name__ == "__main__":
    main()