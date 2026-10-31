"""PyTorch autoencoders scored by reconstruction error."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import ModelConfig, TrainingConfig
from .base import Detector

log = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder with a low-dimensional bottleneck."""

    def __init__(self, n_features: int, hidden: int, latent: int, layers: int, dropout: float):
        super().__init__()
        rnn_dropout = dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=rnn_dropout)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=rnn_dropout)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        seed = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(seed)
        return self.head(out)


class ConvAutoencoder(nn.Module):
    """1-D convolutional autoencoder over time (strided encoder, transposed decoder)."""

    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        mid = max(hidden // 2, 4)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, hidden, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, mid, 7, stride=2, padding=3),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(mid, mid, 7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(mid, hidden, 7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, n_features, 7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        h = self.encoder(x.transpose(1, 2))
        out = self.decoder(h)
        if out.size(-1) < length:  # odd window lengths: pad back to the input length
            out = nn.functional.pad(out, (0, length - out.size(-1)), mode="replicate")
        return out[..., :length].transpose(1, 2)


class TorchAutoencoderDetector(Detector):
    """Trains an autoencoder on normal windows and scores by reconstruction error.

    Training minimises MSE over the whole window. The anomaly score is the mean
    absolute error of the newest reading (``score_on="last"``), which localises
    alarms in time and is less sensitive to a single drifting sensor than MSE, or
    the mean absolute error over the full window (``score_on="window"``).
    """

    def __init__(
        self,
        name: str,
        n_features: int,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
    ):
        self.name = name
        self.n_features = n_features
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = resolve_device(train_cfg.device)
        torch.manual_seed(train_cfg.seed)
        self.net = self._build().to(self.device)
        self.history: list[float] = []

    def _build(self) -> nn.Module:
        m = self.model_cfg
        if self.name == "lstm_ae":
            return LSTMAutoencoder(
                self.n_features, m.hidden_dim, m.latent_dim, m.num_layers, m.dropout
            )
        if self.name == "conv_ae":
            return ConvAutoencoder(self.n_features, m.hidden_dim, m.dropout)
        raise ValueError(f"unknown autoencoder '{self.name}'")

    def fit(self, windows: np.ndarray) -> TorchAutoencoderDetector:
        cfg = self.train_cfg
        torch.manual_seed(cfg.seed)
        generator = torch.Generator().manual_seed(cfg.seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(windows.astype(np.float32))),
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimiser = torch.optim.Adam(
            self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        loss_fn = nn.MSELoss()
        self.net.train()
        self.history = []
        for epoch in range(cfg.epochs):
            total = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimiser.zero_grad()
                loss = loss_fn(self.net(batch), batch)
                loss.backward()
                optimiser.step()
                total += loss.item() * len(batch)
            self.history.append(total / len(windows))
            log.debug("%s epoch %d loss %.5f", self.name, epoch + 1, self.history[-1])
        return self

    @torch.no_grad()
    def score(self, windows: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.net.eval()
        out = []
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i : i + batch_size].astype(np.float32)).to(self.device)
            err = (self.net(batch) - batch).abs()
            if self.model_cfg.score_on == "last":
                err = err[:, -1, :].mean(dim=1)
            else:
                err = err.mean(dim=(1, 2))
            out.append(err.cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    def save(self, directory: Path) -> None:
        torch.save(self.net.state_dict(), Path(directory) / "model.pt")

    def load(self, directory: Path) -> TorchAutoencoderDetector:
        state = torch.load(
            Path(directory) / "model.pt", map_location=self.device, weights_only=True
        )
        self.net.load_state_dict(state)
        self.net.eval()
        return self
