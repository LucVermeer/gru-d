import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule

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
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, x_mean, mask, delta):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        for t in range(seq_len):
            h = self.grud_cell(
                x[:, t, :], h, x_mean[:, t, :], mask[:, t, :], delta[:, t, :]
            )
        out = self.fc(h)
        # print(out.shape)
        return out

    def training_step(self, batch, batch_idx):
        x, x_mean, mask, delta, y = batch
        pred = self.forward(x, x_mean, mask, delta)
        y = torch.argmax(y, dim=1)
        # pred = torch.argmax(pred, dim=1)
        loss = self.loss_fn(pred, y)
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
    batch_size = 32
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Instantiate the model
    print("Instantiating the model...")
    model = GRUD(input_size=dataset.x.shape[2], hidden_size=25, output_size=6)
    print("Model instantiated.")

    # Instantiate the trainer
    print("Instantiating the trainer...")
    trainer = Trainer(max_epochs=1)
    print("Trainer instantiated.")

    # Train the model
    print("Training the model...")
    trainer.fit(model, dataloader)
    print("Training finished.")

    # Save the mode
    print("Saving the model...")
    torch.save(model.state_dict(), "grud/model.pt")
    print("Model saved.")
