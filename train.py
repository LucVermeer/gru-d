import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule

import sys
import os

from .model import GRUDCell
from .data import TimeSeriesDataset


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask):
        loss = self.mse(pred, target)
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()


class GRUD(LightningModule):
    def __init__(self, input_size, hidden_size, output_size=6):
        super(GRUD, self).__init__()
        self.hidden_size = hidden_size
        self.grud_cell = GRUDCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.loss_fn = MaskedMSELoss()

    def forward(self, x, x_mean, mask, delta):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        for t in range(seq_len):
            h = self.grud_cell(
                x[:, t, :], h, x_mean[:, t, :], mask[:, t, :], delta[:, t, :]
            )
        out = self.fc(h)
        return F.softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, x_mean, mask, delta, y = batch
        pred = self.forward(x, x_mean, mask, delta)
        loss = self.loss_fn(pred, y, mask)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


if __name__ == "__main__":
    # Instantiate the dataset
    print("Instantiating the dataset...")
    dataset = TimeSeriesDataset("grud/data.csv")
    print("Dataset instantiated.")

    # Create a DataLoader
    batch_size = 32  # choose the batch size that suits your needs
    # Shuffle??
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    print("Instantiating the model...")
    model = GRUD(input_size=dataset.x.shape[1], hidden_size=50, output_size=6)
    print("Model instantiated.")

    # Instantiate the trainer
    print("Instantiating the trainer...")
    trainer = Trainer(max_epochs=1)
    print("Trainer instantiated.")

    # Train the model
    print("Training the model...")
    trainer.fit(model, dataloader)
    print("Training finished.")
