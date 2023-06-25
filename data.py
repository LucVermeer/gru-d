from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df,
        label_col="Label",
        time_col="Time (s).1",
        exclude_cols=[
            "Time (s)",
            "Time (s).1",
            "Velocity (m/s)",
            "Direction (°)",
            "Distance (cm)",
            "Horizontal Accuracy (m)",
            "Time_Until_Next_Label",
            "Time_Since_Previous_Label",
            "Latitude (°)",
            "Longitude (°)",
        ],
        seq_len=100,
        step=10,
        label_encoder=None,
        scaler=None,
        coarse_labels=False,
    ):
        self.seq_len = seq_len
        self.step = step

        if coarse_labels:
            df[label_col] = df[label_col].apply(self.convert_labels)

        # Separate the features and labels
        x = df.drop(columns=[label_col] + exclude_cols).values
        y = df[label_col].values

        # Encode the labels
        if not label_encoder:
            label_encoder = LabelEncoder()

        y = label_encoder.fit_transform(y)

        self.label_encoder = label_encoder

        # Generate 'x_mean'
        x_mean = np.nanmean(x, axis=0)

        # Generate 'mask' (1 for non-NaN, 0 for NaN)
        mask = ~np.isnan(x)

        # Copy 'x' and replace NaNs with 0
        x[np.isnan(x)] = 0

        # Generate 'delta' (time elapsed since the last measurement)
        time = df[time_col].values
        delta = np.zeros_like(x)
        delta[1:] = np.subtract(time[1:], time[:-1])[:, None]

        # Normalize 'x' and 'x_mean' for each feature using the mean and std of the training data
        if scaler is None:
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
        else:
            x = scaler.transform(x)

        x_mean = scaler.transform([x_mean])[0]
        self.scaler = scaler

        # Reshape data into overlapping sequences
        num_sequences = (len(df) - seq_len) // step
        self.x = np.zeros((num_sequences, seq_len, x.shape[1]))
        self.x_mean = np.zeros((num_sequences, seq_len, x_mean.shape[0]))
        self.mask = np.zeros((num_sequences, seq_len, mask.shape[1]))
        self.delta = np.zeros((num_sequences, seq_len, delta.shape[1]))
        self.y = np.zeros(
            num_sequences
        )  # initialize with zeros of size num_sequences, not y.shape[0]

        for i in range(num_sequences):
            start_idx = i * step
            self.x[i] = x[start_idx : start_idx + seq_len]
            self.x_mean[i] = np.repeat(x_mean[np.newaxis, :], seq_len, axis=0)
            self.mask[i] = mask[start_idx : start_idx + seq_len]
            self.delta[i] = delta[start_idx : start_idx + seq_len]
            # make the label the most common label in the sequence
            sequence_labels = y[start_idx : start_idx + seq_len]
            self.y[i] = np.argmax(
                np.bincount(sequence_labels)
            )  # argmax of bincount gives the most common label

        # Convert to tensors
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.x_mean = torch.tensor(self.x_mean, dtype=torch.float32)
        self.mask = torch.tensor(self.mask, dtype=torch.float32)
        self.delta = torch.tensor(self.delta, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def convert_labels(self, label):
        if label in ["break-btn", "lowering-btn", "falling-btn"]:
            return "not-climbing"
        elif label in ["straight-btn", "slab-btn", "overhanging-btn"]:
            return "climbing"
        else:
            return label  # just in case there are other labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.x_mean[idx],
            self.mask[idx],
            self.delta[idx],
            self.y[idx],
        )

    def get_label_encoder(self):
        return self.label_encoder

    def get_scaler(self):
        return self.scaler

    def get_labels(self):
        return self.y


class TimeSeriesPretrainDataset(Dataset):
    def __init__(
        self,
        df,
        time_col="Time (s).1",
        exclude_cols=[
            "Time (s)",
            "Time (s).1",
            "Velocity (m/s)",
            "Direction (°)",
            "Distance (cm)",
            "Horizontal Accuracy (m)",
            "Latitude (°)",
            "Longitude (°)",
        ],
        seq_len=100,
        step=10,
        scaler=None,
    ):
        self.seq_len = seq_len
        self.step = step

        # Get features
        x = df.drop(columns=exclude_cols).values

        # Generate 'x_mean'
        x_mean = np.nanmean(x, axis=0)

        # Generate 'mask' (1 for non-NaN, 0 for NaN)
        mask = ~np.isnan(x)

        # Copy 'x' and replace NaNs with 0
        x[np.isnan(x)] = 0

        # Generate 'delta' (time elapsed since the last measurement)
        time = df[time_col].values
        delta = np.zeros_like(x)
        delta[1:] = np.subtract(time[1:], time[:-1])[:, None]

        # Normalize 'x' and 'x_mean' for each feature using the mean and std of the training data
        if scaler is None:
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
        else:
            x = scaler.transform(x)

        x_mean = scaler.transform([x_mean])[0]
        self.scaler = scaler

        # Reshape data into overlapping sequences
        num_sequences = (
            len(df) - seq_len - 1
        ) // step  # -1 because the last element in sequence becomes target
        self.x = np.zeros((num_sequences, seq_len, x.shape[1]))
        self.x_mean = np.zeros((num_sequences, seq_len, x_mean.shape[0]))
        self.mask = np.zeros((num_sequences, seq_len, mask.shape[1]))
        self.delta = np.zeros((num_sequences, seq_len, delta.shape[1]))
        self.y = np.zeros(
            (num_sequences, x.shape[1])
        )  # initialize with zeros, y will be of size (num_sequences, features)

        for i in range(num_sequences):
            start_idx = i * step
            self.x[i] = x[start_idx : start_idx + seq_len]
            self.x_mean[i] = np.repeat(x_mean[np.newaxis, :], seq_len, axis=0)
            self.mask[i] = mask[start_idx : start_idx + seq_len]
            self.delta[i] = delta[start_idx : start_idx + seq_len]
            # target is the next item in sequence
            self.y[i] = x[start_idx + seq_len]

        # Convert to tensors
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.x_mean = torch.tensor(self.x_mean, dtype=torch.float32)
        self.mask = torch.tensor(self.mask, dtype=torch.float32)
        self.delta = torch.tensor(self.delta, dtype=torch.float32)
        self.y = torch.tensor(
            self.y, dtype=torch.float32
        )  # y is now a tensor of floats since it's the next item

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.x_mean[idx],
            self.mask[idx],
            self.delta[idx],
            self.y[idx],
        )

    def get_scaler(self):
        return self.scaler
