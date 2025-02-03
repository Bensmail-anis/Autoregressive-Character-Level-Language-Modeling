# Model Hyperparameters
MODEL_CONFIG = {
    'vocab_size': None,  # Will be set dynamically
    'block_size': 3,
    'n_embd': 10,
    'n_hidden': 200
}

# Training Hyperparameters
TRAINING_CONFIG = {
    'max_steps': 200000,
    'batch_size': 32,
    'lr_decay_point': 100000,
    'initial_lr': 0.1,
    'final_lr': 0.01
}

# Data Configuration
DATA_CONFIG = {
    'data_path': 'data/names.txt',
    'test_ratio': 0.2,
    'val_ratio': 0.1
}