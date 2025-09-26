import torch
import os 

### train on total dataset
NUM_EPOCHS = 100
DATASET_SIZE = {'train' : 9600, 'val' : 1200, 'test' : 1200}
dataset = '/amax/sxd/jing/DFT-CrackNet/Crack600/split_dataset_final/' #表示获取数据集的路径，通过相对路径获得数据集的根目录

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
PIN_MEMORY = True #数据加载在内存中，提高数据加载的速度，在使用GOU时，将其设置为true
LOAD_MODEL = False  #表示不加载已保存的模型，从头开始训练；若设置为true，模型将从之前的训练状态恢复，以便继续进行训练。
# Dataset dir
TRAIN_IMG_DIR = dataset+"train/IMG"
TRAIN_MASK_DIR = dataset+"train/GT"
VAL_IMG_DIR = dataset+"val/IMG"
VAL_MASK_DIR = dataset+"val/GT"
TEST_IMG_DIR = dataset+"test/IMG"
TEST_MASK_DIR = dataset+"test/GT"