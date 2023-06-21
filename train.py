import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import random_split
from sklearn.metrics import f1_score
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping


from .model import GRUDCell
from .data import TimeSeriesDataset

from datetime import datetime
import argparse


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

    def validation_step(self, batch, batch_idx):
        x, x_mean, mask, delta, y = batch
        pred = self.forward(x, x_mean, mask, delta)
        y = torch.argmax(y, dim=1)
        loss = self.loss_fn(pred, y)

        # Calculate accuracy
        pred_classes = torch.argmax(pred, dim=1)
        acc = (pred_classes == y).float().mean()

        # Log loss and accuracy
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, x_mean, mask, delta, y = batch
        pred = self(x, x_mean, mask, delta)
        y = torch.argmax(y, dim=1)  # convert one-hot to class indices

        # Compute loss
        loss = self.loss_fn(pred, y)

        pred = torch.argmax(
            pred, dim=1
        )  # get the class with highest predicted probability

        # Compute accuracy
        correct_predictions = (pred == y).float()
        acc = correct_predictions.sum() / len(correct_predictions)

        # Compute F1-score
        f1 = f1_score(y.detach().cpu(), pred.detach().cpu(), average="macro")

        # Log the loss, accuracy, and F1-score
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log(
            "test_f1", torch.tensor(f1), prog_bar=True
        )  # f1_score returns a numpy float, need to convert to tensor

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


early_stop_callback = EarlyStopping(
    monitor="val_acc",  # Metric to monitor
    patience=3,  # Number of epochs with no improvement after which training will be stopped
    verbose=True,  # Report when the training has been stopped
    mode="min",  # 'min' indicates that training will be stopped when the quantity monitored has stopped decreasing
)


def train():
    # Instantiate the dataset
    print("Instantiating the dataset...")
    dataset = TimeSeriesDataset(
        "grud/data.csv",
        seq_len=100,
        step=10,
    )
    print("Dataset instantiated.")

    # Split the dataset into training and test sets
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders for training and test sets
    batch_size = 32
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    # Instantiate the model
    print("Instantiating the model...")
    model = GRUD(
        input_size=dataset.x.shape[2],
        hidden_size=dataset.x.shape[2],
        output_size=6,
    )
    print("Model instantiated.")

    # Instantiate the trainer
    print("Instantiating the trainer...")
    trainer = Trainer(max_epochs=50, callbacks=[early_stop_callback])
    print("Trainer instantiated.")

    # Train the model
    print("Training the model...")
    trainer.fit(model, train_dataloader, val_dataloader)
    print("Training finished.")

    # Save the mode
    print("Saving the model...")
    torch.save(
        model.state_dict(),
        "grud/models/model_{}.pt".format(
            datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
    )
    print("Model saved.")

    # Evaluate the model on the test set
    print("Evaluating the model...")
    trainer.test(model, test_dataloader)
    print("Evaluation finished.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducibility"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model", default=True
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    if args.train:
        train()
