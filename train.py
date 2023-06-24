import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import random_split
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd

from .model import GRUD, APCModel, GRUD_ML
from .data import TimeSeriesDataset

from datetime import datetime
import argparse


early_stop_callback = EarlyStopping(
    monitor="val_acc",
    patience=3,
    verbose=True,
    mode="max",
)


def split_data(
    df,
    start_test="2023-06-14 07:36:33.297403",
    end_test="2023-06-14 08:03:30.700492",
):
    """The timestamps are in column 'Time (s)"""
    # Definne the test set as the rows in between the start_test and end_test timestamps
    test_df = df[(df["Time (s)"] >= start_test) & (df["Time (s)"] <= end_test)]
    # Define the train set as the rows before the start_test timestamp and after the end_test timestamp
    train_df = df[(df["Time (s)"] < start_test) | (df["Time (s)"] > end_test)]
    return train_df, test_df


def train_multilayer(args):
    # Instantiate the dataset
    df = pd.read_csv(args.data_path)

    # Split the dataframe into train, validation, and test
    train_df, test_df = split_data(df)
    # create the training dataset
    train_dataset = TimeSeriesDataset(
        train_df, seq_len=args.seq_len, step=args.step_size
    )

    # create the test dataset using the scaler from the training dataset
    test_dataset = TimeSeriesDataset(
        test_df,
        seq_len=args.seq_len,
        step=args.step_size,
        scaler=train_dataset.get_scaler(),
        label_encoder=train_dataset.get_label_encoder(),
    )

    # Create the DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Instantiate the model
    print("Instantiating the model...")
    model = GRUD_ML(
        input_size=train_dataset.x.shape[2],
        hidden_size=34,
        output_size=6,
    )
    print("Model instantiated.")

    logger = TensorBoardLogger("lightning_logs", name="my_model")

    # Instantiate the trainer
    print("Instantiating the trainer...")
    trainer = Trainer(
        max_epochs=50, callbacks=[early_stop_callback], logger=logger
    )
    print("Trainer instantiated.")

    # Train the model
    print("Training the model...")
    trainer.fit(model, train_dataloader, val_dataloader)
    print("Training finished.")

    # Save the mode
    print("Saving the model...")
    torch.save(
        model.state_dict(),
        "grud/models/model_{}_{}.pt".format(
            args.seed, datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
    )
    print("Model saved.")

    # Evaluate the model on the test set
    print("Evaluating the model...")
    trainer.test(model, test_dataloader)
    print("Evaluation finished.")


def train(args):
    # Instantiate the dataset
    df = pd.read_csv(args.data_path)

    # Split the dataframe into train, validation, and test
    train_df, test_df = split_data(df)
    # create the training dataset
    train_dataset = TimeSeriesDataset(
        train_df, seq_len=args.seq_len, step=args.step_size
    )

    # create the test dataset using the scaler from the training dataset
    test_dataset = TimeSeriesDataset(
        test_df,
        seq_len=args.seq_len,
        step=args.step_size,
        scaler=train_dataset.get_scaler(),
        label_encoder=train_dataset.get_label_encoder(),
    )

    # Create the DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Instantiate the model
    print("Instantiating the model...")
    model = GRUD(
        input_size=train_dataset.x.shape[2],
        hidden_size=train_dataset.x.shape[2],
        output_size=6,
    )
    print("Model instantiated.")

    logger = TensorBoardLogger("lightning_logs", name="my_model")

    # Instantiate the trainer
    print("Instantiating the trainer...")
    trainer = Trainer(
        max_epochs=50, callbacks=[early_stop_callback], logger=logger
    )
    print("Trainer instantiated.")

    # Train the model
    print("Training the model...")
    trainer.fit(model, train_dataloader, val_dataloader)
    print("Training finished.")

    # Save the mode
    print("Saving the model...")
    torch.save(
        model.state_dict(),
        "grud/models/model_{}_{}.pt".format(
            args.seed, datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
    )
    print("Model saved.")

    # Evaluate the model on the test set
    print("Evaluating the model...")
    trainer.test(model, test_dataloader)
    print("Evaluation finished.")


def pretrain(args):
    # Load the unlabeled data
    unlabeled_df = pd.read_csv("unlabeled_data.csv")

    # Create the unlabeled dataset for pretraining
    unlabeled_dataset = TimeSeriesDataset(
        unlabeled_df, seq_len=args.seq_len, step=args.step_size
    )

    # Create the DataLoader
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Instantiate the APC model
    print("Instantiating the APC model...")
    model = APCModel(
        input_size=unlabeled_dataset.x.shape[2],
        hidden_size=unlabeled_dataset.x.shape[2],
        output_size=unlabeled_dataset.x.shape[2],
    )
    print("APC Model instantiated.")

    logger = TensorBoardLogger("lightning_logs", name="my_model")

    # Instantiate the trainer
    print("Instantiating the trainer...")
    trainer = Trainer(
        max_epochs=50, callbacks=[early_stop_callback], logger=logger
    )
    print("Trainer instantiated.")

    # Pretrain the model
    print("Pretraining the model...")
    trainer.fit(model, unlabeled_dataloader)
    print("Pretraining finished.")

    # Save the pre-trained model
    print("Saving the pre-trained model...")
    torch.save(
        model.state_dict(),
        "grud/models/apc_model_{}_{}.pt".format(
            args.seed, datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
    )
    print("Pre-trained model saved.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="train", help="Train, pretrain or test"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the DataLoader",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="grud/data.csv",
        help="Path to the preprocessed data",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=50,
        help="Sequence length for training",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Step size for training",
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    if args.mode == "train":
        print("Training model...")
        train(args)
    elif args.mode == "train_ml":
        print("Training multilayer model...")
        train_multilayer(args)
    elif args.mode == "pretrain":
        print("Pretraining model...")
        pretrain(args)
