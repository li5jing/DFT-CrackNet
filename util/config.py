import torch
import os 

### train on total dataset
NUM_EPOCHS = 100
DATASET_SIZE = {'train' : 9600, 'val' : 1200, 'test' : 1200}
dataset = '/amax/sxd/jing/DFT-CrackNet/Crack600/split_dataset_final/' # Path to get the dataset, relative path to the dataset's root directory

### train on sample dataset
# NUM_EPOCHS = 1
# DATASET_SIZE = {'train' : 360, 'val' : 120, 'test' : 120}
# dataset = os.path.join('../', 'sample_dataset/') # or split_dataset_final

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True # Enable loading data into memory to speed up data loading when using GPU.
LOAD_MODEL = False  # Indicates whether to load a previously saved model or start training from scratch.
# Dataset directories
TRAIN_IMG_DIR = dataset+"train/IMG"
TRAIN_MASK_DIR = dataset+"train/GT"
VAL_IMG_DIR = dataset+"val/IMG"
VAL_MASK_DIR = dataset+"val/GT"
TEST_IMG_DIR = dataset+"test/IMG"
TEST_MASK_DIR = dataset+"test/GT"
