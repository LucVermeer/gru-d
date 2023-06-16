from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        csv_file,
        label_col="Label",
        time_col="Time (s).1",
        exclude_cols=["Time (s)", "Time (s).1"],
        seq_len=100,
        step=10,
    ):
        self.seq_len = seq_len
        self.step = step

        # Load the csv file
        df = pd.read_csv(csv_file)

        # Separate the features and labels
        x = df.drop(columns=[label_col] + exclude_cols).values
        y = df[label_col].values

        # Encode the labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Generate 'x_mean'
        x_mean = np.nanmean(x, axis=0)

        # Generate 'mask' (1 for non-NaN, 0 for NaN)
        mask = ~np.isnan(x)

        # Copy 'x' and replace NaNs with 0
        x[np.isnan(x)] = 0

        # Generate 'delta' (time elapsed since the last measurement)
        time = df[time_col].values
        print(time.shape)
        delta = np.zeros_like(x)
        delta[1:] = np.subtract(time[1:], time[:-1])[:, None]

        # Reshape data into overlapping sequences
        num_sequences = (len(df) - seq_len) // step
        self.x = np.zeros((num_sequences, seq_len, x.shape[1]))
        self.x_mean = np.zeros((num_sequences, seq_len, x_mean.shape[0]))
        self.mask = np.zeros((num_sequences, seq_len, mask.shape[1]))
        self.delta = np.zeros((num_sequences, seq_len, delta.shape[1]))
        self.y = np.zeros((num_sequences, y.shape[0]))

        for i in range(num_sequences):
            start_idx = i * step
            self.x[i] = x[start_idx : start_idx + seq_len]
            self.x_mean[i] = np.repeat(x_mean[np.newaxis, :], seq_len, axis=0)
            self.mask[i] = mask[start_idx : start_idx + seq_len]
            self.delta[i] = delta[start_idx : start_idx + seq_len]
            self.y[i] = y[
                start_idx + seq_len - 1
            ]  # label is from the end of the sequence

        # Convert to tensors
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.x_mean = torch.tensor(self.x_mean, dtype=torch.float32)
        self.mask = torch.tensor(self.mask, dtype=torch.float32)
        self.delta = torch.tensor(self.delta, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        # Move to device
        self.x = self.x.to(device)
        self.x_mean = self.x_mean.to(device)
        self.mask = self.mask.to(device)
        self.delta = self.delta.to(device)
        self.y = self.y.to(device)

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
