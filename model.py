import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, LightningModule


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
        self.weight_gamma_x = nn.Parameter(
            torch.Tensor(input_size, input_size)
        )
        self.weight_gamma_h = nn.Parameter(
            torch.Tensor(hidden_size, input_size + hidden_size)
        )

        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_z)
        nn.init.xavier_uniform_(self.weight_r)
        nn.init.xavier_uniform_(self.weight_h_tilde)
        nn.init.xavier_uniform_(self.weight_gamma_x)
        nn.init.xavier_uniform_(self.weight_gamma_h)
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
            -torch.max(torch.zeros_like(delta), self.weight_gamma_x * delta)
        )
        gamma_h = torch.exp(
            -torch.max(torch.zeros_like(delta), self.weight_gamma_h * delta)
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
