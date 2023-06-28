import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import OneCycleLR


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask):
        loss = self.mse(pred, target)
        masked_loss = loss * mask
        if mask.sum() == 0:
            return masked_loss.sum()
        return masked_loss.sum() / mask.sum()


class GRUD(LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=6,
        learning_rate=0.001,
        finetune=False,
    ):
        super(GRUD, self).__init__()
        self.hidden_size = hidden_size
        self.grud_cell = GRUDCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.finetune = finetune

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
        # y = torch.argmax(y, dim=1)
        # pred = torch.argmax(pred, dim=1)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_mean, mask, delta, y = batch
        pred = self.forward(x, x_mean, mask, delta)
        # y = torch.argmax(y, dim=1)
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
        # y = torch.argmax(y, dim=1)  # convert one-hot to class indices

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
        if self.finetune:
            # Initialize a warm up phase for the first 20% of training steps
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
            steps_per_epoch = len(self.train_dataloader())
            num_training_steps = steps_per_epoch * self.trainer.max_epochs
            lr_scheduler = {
                "scheduler": OneCycleLR(
                    optimizer,
                    0.001,
                    total_steps=num_training_steps,
                    anneal_strategy="linear",
                    pct_start=0.2,
                ),
                "name": "learning_rate",
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [lr_scheduler]
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class APCModel(LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(APCModel, self).__init__()
        self.hidden_size = hidden_size
        self.grud_cell = GRUDCell(input_size, hidden_size)
        self.fc = nn.Linear(
            hidden_size, output_size
        )  # output_size is the same as input_size for APC
        self.loss_fn = MaskedMSELoss()

    def forward(self, x, x_mean, mask, delta):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        outputs = []
        for t in range(seq_len):
            h = self.grud_cell(
                x[:, t, :], h, x_mean[:, t, :], mask[:, t, :], delta[:, t, :]
            )
            output = self.fc(h)
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def training_step(self, batch, batch_idx):
        x, x_mean, mask, delta, y = batch
        pred = self.forward(x, x_mean, mask, delta)
        pred = pred[:, -1, :]
        mask = mask[:, -1, :]
        loss = self.loss_fn(pred, y, mask)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class GRUDCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUDCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define the parameters for the GRU-D
        self.weight_z = nn.Parameter(
            torch.Tensor(hidden_size, input_size + hidden_size)
        )
        self.weight_r = nn.Parameter(
            torch.Tensor(hidden_size, input_size + hidden_size)
        )
        self.weight_h_tilde = nn.Parameter(
            torch.Tensor(hidden_size, input_size + hidden_size)
        )

        # Define the parameters for the decay rates
        self.weight_gamma_x = nn.Parameter(torch.Tensor(input_size))
        self.weight_gamma_h = nn.Parameter(torch.Tensor(hidden_size))

        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_z)
        nn.init.xavier_uniform_(self.weight_r)
        nn.init.xavier_uniform_(self.weight_h_tilde)

        # Initialize the decay rates with uniform because they have less than 2 dimensions
        nn.init.uniform_(self.weight_gamma_x, a=0.0, b=1.0)
        nn.init.uniform_(self.weight_gamma_h, a=0.0, b=1.0)
        nn.init.zeros_(self.bias)

    def forward(self, x, h_prev, x_mean, mask, delta):
        """
        x: current input
        h_prev: previous hidden state
        x_mean: the mean value of the input x
        mask: binary mask that indicates the presence of missing values
        delta: time intervals between two consecutive visits
        """

        gamma_x = torch.exp(
            -torch.max(
                torch.zeros_like(delta),
                self.weight_gamma_x * delta
                # + self.bias_gamma_x,
            )
        )
        gamma_h = torch.exp(
            -torch.max(
                torch.zeros_like(delta),
                self.weight_gamma_h * delta,
            )
        )

        x_hat = gamma_x * x + (1 - gamma_x) * mask * x_mean
        h_prev = gamma_h * h_prev

        z = torch.sigmoid(
            F.linear(
                torch.cat((x_hat, h_prev), dim=1), self.weight_z, self.bias
            )
        )
        r = torch.sigmoid(
            F.linear(
                torch.cat((x_hat, h_prev), dim=1), self.weight_r, self.bias
            )
        )
        h_tilde = torch.tanh(
            F.linear(
                torch.cat((x_hat, r * h_prev), dim=1),
                self.weight_h_tilde,
                self.bias,
            )
        )
        h_next = (1 - z) * h_prev + z * h_tilde

        return h_next
