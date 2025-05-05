"""
Model definitions for convolutional LSTM flood predictor.
"""
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Single convolutional LSTM cell.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=padding)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h_prev], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            self.conv(combined), self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur


class ConvLSTM(nn.Module):
    """
    Multi-step convolutional LSTM.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, height, width = input_seq.size()
        h = torch.zeros(batch_size, self.cell.hidden_dim,
                        height, width, device=input_seq.device)
        c = h.clone()

        for t in range(seq_len):
            h, c = self.cell(input_seq[:, t], h, c)
        return h


class FloodPredictor(nn.Module):
    """
    Flood prediction model combining ConvLSTM and fully-connected layers.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 16,
                 kernel_size: int = 3) -> None:
        super().__init__()
        self.convlstm = ConvLSTM(input_dim, hidden_dim, kernel_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        spatial: torch.Tensor,
        gage_scalar: torch.Tensor,
        prev_runoff: torch.Tensor,
    ) -> torch.Tensor:
        h = self.convlstm(spatial)
        h_pool = self.pool(h).view(h.size(0), -1)
        x = torch.cat([
            h_pool,
            gage_scalar.unsqueeze(1),
            prev_runoff.unsqueeze(1),
        ], dim=1)
        return self.fc(x).squeeze(1)