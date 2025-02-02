import torch

class Config:
    BLOCK_SIZE = 3  # context length
    HIDDEN_DIM = 200
    EMBEDDING_DIM = 10
    LEARNING_RATE = 0.1
    LR_DECAY_AFTER = 100000
    LR_DECAY_VALUE = 0.01
    BATCH_SIZE = 32
    NUM_EPOCHS = 200000
    SEED = 2147483647