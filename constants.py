import torch

# Preprocessing
TRAIN_PATH = './data/train'

PROCESSED_TRAIN_PATH = './data/train/processed_train'
PROCESSED_VAL_PATH = './data/train/processed_val'

TEST_PATH = './data/test'
PROCESSED_TEST_PATH = './data/processed_test'

CLEANED_PATH = './data/train/cleaned'
CLEANED_PROCESSED_PATH = r'data/train/processed_train/cleaned'

DIRTY_PATH = './data/train/dirty'
DIRTY_PROCESSED_PATH = r'data/train/processed_train/dirty'

# Training
BATCH_SIZE = 64
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
