import os

from torch.nn import PReLU

from common import *

model = 'vit_tiny_patch16_224'
dataset = 'imagenet1k' # cifar10, cifar100, imagenet1k

image_size = 224
batch_size = 256
lr = 0.0005 * (batch_size / 512)
output_dir = f"{output_root}/{dataset}/{model}/{os.path.basename(__file__).split('.')[0]}"
model_kwargs = dict(act_layer=PReLU)
