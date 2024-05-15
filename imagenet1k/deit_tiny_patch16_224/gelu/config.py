GELU = <class 'torch.nn.modules.activation.GELU'>
amp = True
batch_size = 256
clip_max_norm = 1.0
criterion = 'ce'
data_root = './data'
dataset = 'imagenet1k'
device = 'cuda'
dist_backend = 'nccl'
dist_url = 'env://'
distributed = True
drop_last = False
drop_lr_now = False
epochs = 100
eval_aug_kwargs = {}
eval_interval = 1
evaluator = 'default'
find_unused_params = False
gamma = 0.1
image_size = 224
load_pos = None
lr = 0.00025
lr_drop = -1
milestones = None
min_lr = 1e-05
model = 'deit_tiny_patch16_224'
model_kwargs = {'act_layer': <class 'torch.nn.modules.activation.GELU'>}
model_lib = 'torchvision-ex'
momentum = 0.9
need_targets = False
no_dist = False
no_pretrain = True
note = 'dataset: imagenet1k | model: deit_tiny_patch16_224 | output_dir: ./runs/__exp__/imagenet1k/deit_tiny_patch16_224/gelu'
num_workers = 8
optimizer = 'adamw'
output_dir = './runs/__exp__/imagenet1k/deit_tiny_patch16_224/gelu'
output_root = './runs/__exp__'
pin_memory = True
print_freq = 50
resume = None
save_interval = 50
save_pos = None
scheduler = 'cosine'
seed = 42
step_size = None
sync_bn = True
train_aug_kwargs = {}
warmup_epochs = 20
warmup_lr = 1e-06
warmup_steps = 0
weight_decay = 0.05
world_size = 4
