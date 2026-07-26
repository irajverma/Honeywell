"""
sequence_model.py — PyTorch LSTM Autoencoder sequence model (Phase 3).

Learns normal sequential patterns from causal sliding windows (L=10) and flags
anomalies based on sequence reconstruction error.
"""

import time
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class LSTMAutoencoder(nn.Module):
    """
    Sequence Autoencoder using LSTM encoder-decoder architecture.
    Compresses input sequence X (B, L, D) into hidden representation and reconstructs X.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        _, (hidden, cell) = self.encoder(x)

        # Use final layer's hidden state as compressed representation
        # hidden[-1] shape: (batch_size, hidden_dim)
        seq_len = x.size(1)
        dec_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

        # Decoder initialized with encoder's final hidden/cell states
        dec_out, _ = self.decoder(dec_input, (hidden, cell))

        # Project back to feature dimension
        reconstruction = self.projection(dec_out)
        return reconstruction


class SequenceAnomalyDetector:
    """
    Wrapper for PyTorch LSTM Autoencoder training and anomaly scoring.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        window_size: int = 10,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 5,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.window_size = window_size
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)

        self.model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)

    def fit(self, train_windows: np.ndarray, val_split: float = 0.1):
        """Train LSTM Autoencoder on normal sequence windows using MSE loss."""
        print(f"  [LSTM] Preparing training dataset ({len(train_windows):,} windows of shape {train_windows.shape[1:]})...")
        dataset = TensorDataset(torch.from_numpy(train_windows).float())

        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        print(f"  [LSTM] Starting training for {self.epochs} epochs on device: {self.device}...")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)
            train_loss /= len(train_set)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(self.device)
                    recon = self.model(batch_x)
                    loss = criterion(recon, batch_x)
                    val_loss += loss.item() * len(batch_x)
            val_loss /= len(val_set)

            print(f"    Epoch {epoch:02d}/{self.epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        print(f"  [LSTM] Training complete in {time.time() - start_time:.2f}s")

    def score(self, windows: np.ndarray, batch_size: int = 512, score_mode: str = "final_step") -> np.ndarray:
        """
        Compute sequence anomaly scores.
        
        score_mode:
          - "final_step": Reconstruction error at final event index (-1) ONLY. Prevents preceding normal steps from diluting the score.
          - "blend": 50/50 blend of final step error and overall window error.
          - "window": Mean reconstruction error across the entire L-step window.
        """
        self.model.eval()
        dataset = TensorDataset(torch.from_numpy(windows).float())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        scores_list = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                recon = self.model(batch_x)

                step_error = ((batch_x[:, -1, :] - recon[:, -1, :]) ** 2).mean(dim=-1)
                window_error = ((batch_x - recon) ** 2).mean(dim=[1, 2])

                if score_mode == "final_step":
                    batch_scores = step_error
                elif score_mode == "window":
                    batch_scores = window_error
                else: # blend
                    batch_scores = 0.5 * step_error + 0.5 * window_error

                scores_list.append(batch_scores.cpu().numpy())

        return np.concatenate(scores_list, axis=0)
