A Method on Searching Better Activation Functions
=================================================
Image Classification Task
=================================================
The code is partly based on https://github.com/horrible-dong/DNRT. We thank their works.
## Installation

The environment is based on `python 3.7 & pytorch 1.11.0+cu113`.

Create and activate the environment :

```bash
cd ./EAFO-Image_classification
```

```bash
conda env create -f environment.yaml
```

```bash
conda activate EAFO
```

## Training

The corresponding config file should be imported correctly when reproducing the experiments.

**single-gpu**

```bash
python main.py --config /path/to/config.py
```

**multi-gpu**

```bash
torchrun --nproc_per_node=$num_gpus main.py --config /path/to/config.py
```

The `cifar10` and `cifar100` datasets will be automatically downloaded to
the `--data_root` directory. For `imagenet1k`, please refer to the guidelines outlined in["Way to put datasets"](data/README_data.md) for manual dataset preparation.

During the training, the config file, checkpoints (.pth), logs and all other outputs will be stored in `--output_dir`.

## Evaluation

If you want to evaluate the output checkpoints (.pth), you can execute the following command.

**single-gpu**

```bash
python main.py --config /path/to/config.py --resume /path/to/checkpoint.pth --eval
```

**multi-gpu**

```bash
torchrun --nproc_per_node=$num_gpus main.py --config /path/to/config.py --resume /path/to/checkpoint.pth --eval
```

