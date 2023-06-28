# Climber Motion Classification

This project aims to classify climber motions based on sensor data. The script provides functionality to train a model, pretrain a model, or finetune a model using pretraining.

## Getting Started

### Prerequisites

- Python 3.7 or later
- PyTorch
- pandas
- numpy

### Installation

1. Clone the repository

git clone https://github.com/LucVermeer/grud


## Running the script

This script accepts a variety of command line arguments:

- `--mode`: Specifies whether to `train`, `pretrain`, `coarse_pretrain` or `finetune` the model (default is `train`)
- `--seed`: The seed for reproducibility (default is `42`)
- `--batch_size`: Batch size for training (default is `32`)
- `--num_workers`: Number of workers for the DataLoader (default is `0`)
- `--data_path`: Path to the preprocessed data (default is `grud/data.csv`)
- `--seq_len`: Sequence length for training (default is `10`)
- `--step_size`: Step size for training (default is `1`)
- `--model_name`: Name of the pretrained model (default is `coarse_model_42`)
- `--freeze_gru`: Whether to freeze the weights of the GRU layers (default is `False`)

To run the script with the default parameters, use:

```bash
python -m grud.train
```
Example:

```bash
python -m grud.train --mode=finetune --seed=0 --batch_size=64
```

## Experiments

Evaluation in experiments.ipynb
experiments.ipynb primarily focuses on the evaluation of our model's performance.

Authors
Luc Vermeer
