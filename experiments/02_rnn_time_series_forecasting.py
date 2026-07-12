from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. Generate synthetic time series
# ============================================================

def make_synthetic_load_series(n_hours=24 * 60):
    """
    Create a synthetic hourly electricity-load-like time series.

    Components:
        - baseline
        - daily seasonality
        - weekly seasonality
        - small upward trend
        - random noise

    Output:
        t shape: (n_hours,)
        x shape: (n_hours,)
    """
    t = torch.arange(n_hours, dtype=torch.float32)

    baseline = 10.0
    daily = 2.0 * torch.sin(2 * torch.pi * t / 24)
    weekly = 0.8 * torch.sin(2 * torch.pi * t / (24 * 7))
    trend = 0.002 * t
    noise = 0.3 * torch.randn(n_hours)

    x = baseline + daily + weekly + trend + noise

    return t, x


# ============================================================
# 2. Create sliding-window dataset
# ============================================================

def make_sliding_windows(x, window_size=24):
    """
    Convert a 1D time series into supervised learning examples.

    Each example:
        X[i] = [x_i, x_{i+1}, ..., x_{i+window_size-1}]
        y[i] = x_{i+window_size}

    X shape: (n_samples, window_size, 1)
    y shape: (n_samples, 1)
    """
    X = []
    y = []

    n_time = len(x)

    for start in range(n_time - window_size):
        input_window = x[start:start + window_size]
        target_value = x[start + window_size]

        X.append(input_window.unsqueeze(-1))
        y.append(target_value.unsqueeze(0))

    X = torch.stack(X)
    y = torch.stack(y)

    return X, y


# ============================================================
# 3. Define RNN model
# ============================================================

class RNNForecaster(nn.Module):
    """
    Sequence-to-one RNN forecaster.

    Input:
        x shape: (batch_size, window_size, input_size)

    Output:
        y_pred shape: (batch_size, 1)
    """

    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        output_size=1,
        nonlinearity="tanh",
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity=nonlinearity,
        )

        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h_last = self.rnn(x)

        # h_last shape: (num_layers, batch_size, hidden_size)
        # For one layer, h_last[-1] has shape (batch_size, hidden_size).
        last_hidden = h_last[-1]

        y_pred = self.readout(last_hidden)

        return y_pred


# ============================================================
# 4. Main experiment
# ============================================================

def main():
    torch.manual_seed(0)

    # --------------------------------------------------------
    # Create data
    # --------------------------------------------------------

    n_hours = 24 * 60
    window_size = 24

    t, x = make_synthetic_load_series(n_hours=n_hours)

    X, y = make_sliding_windows(x, window_size=window_size)

    print("Original time series:")
    print("t shape:", t.shape)
    print("x shape:", x.shape)

    print("\nSliding-window dataset before normalization:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # --------------------------------------------------------
    # Time-based train/test split
    # --------------------------------------------------------

    n_samples = X.shape[0]
    train_size = int(0.8 * n_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[train_size:]
    y_test = y[train_size:]

    print("\nTime-based train/test split:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # --------------------------------------------------------
    # Normalization using train statistics only
    # --------------------------------------------------------

    # Important:
    # Only use the training period to compute mean and std.
    # This avoids leaking information from the future test period.
    train_mean = X_train.mean()
    train_std = X_train.std()

    X_train_norm = (X_train - train_mean) / train_std
    y_train_norm = (y_train - train_mean) / train_std

    X_test_norm = (X_test - train_mean) / train_std
    y_test_norm = (y_test - train_mean) / train_std

    print("\nNormalization statistics from training data:")
    print("train_mean:", train_mean.item())
    print("train_std:", train_std.item())

    print("\nAfter normalization:")
    print("X_train_norm mean:", X_train_norm.mean().item())
    print("X_train_norm std:", X_train_norm.std().item())

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_dataset = TensorDataset(X_train_norm, y_train_norm)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    # --------------------------------------------------------
    # Model, optimizer, loss
    # --------------------------------------------------------

    model = RNNForecaster(
        input_size=1,
        hidden_size=32,
        output_size=1,
        nonlinearity="tanh",
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    loss_fn = nn.MSELoss()

    # --------------------------------------------------------
    # Training loop on normalized data
    # --------------------------------------------------------

    num_epochs = 50

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            y_batch_pred = model(X_batch)
            batch_loss = loss_fn(y_batch_pred, y_batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_train_loss += batch_loss.item()

        average_train_loss = total_train_loss / len(train_loader)

        model.eval()

        with torch.no_grad():
            y_test_pred_norm = model(X_test_norm)
            test_loss_norm = loss_fn(y_test_pred_norm, y_test_norm)

        train_losses.append(average_train_loss)
        test_losses.append(test_loss_norm.item())

        if epoch % 5 == 0:
            print(
                f"epoch={epoch:3d} | "
                f"train_loss_norm={average_train_loss:.6f} | "
                f"test_loss_norm={test_loss_norm.item():.6f}"
            )

    # --------------------------------------------------------
    # Final predictions in normalized and original scale
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():
        y_test_pred_norm = model(X_test_norm)

    # Convert RNN predictions back to original scale.
    y_test_pred = y_test_pred_norm * train_std + train_mean

    # RNN losses
    final_rnn_test_loss = loss_fn(y_test_pred, y_test)
    final_rnn_test_loss_norm = loss_fn(y_test_pred_norm, y_test_norm)

    # --------------------------------------------------------
    # Baseline 1: persistence baseline
    #
    # Predict next value by copying the last observed value
    # in the input window.
    # --------------------------------------------------------

    persistence_pred = X_test[:, -1, :]
    persistence_test_loss = loss_fn(persistence_pred, y_test)

    persistence_pred_norm = X_test_norm[:, -1, :]
    persistence_test_loss_norm = loss_fn(persistence_pred_norm, y_test_norm)

    # --------------------------------------------------------
    # Baseline 2: daily seasonal baseline
    #
    # Window:
    #   X[i] = [x_i, ..., x_{i+23}]
    #
    # Target:
    #   y[i] = x_{i+24}
    #
    # Same hour yesterday relative to target x_{i+24}
    # is x_i, i.e. the first value in the window.
    # --------------------------------------------------------

    daily_pred = X_test[:, 0, :]
    daily_test_loss = loss_fn(daily_pred, y_test)

    daily_pred_norm = X_test_norm[:, 0, :]
    daily_test_loss_norm = loss_fn(daily_pred_norm, y_test_norm)

    # --------------------------------------------------------
    # Print final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("Final comparison in ORIGINAL scale")
    print("=" * 80)

    print("\nRNN test loss:")
    print(final_rnn_test_loss)

    print("\nPersistence baseline test loss:")
    print(persistence_test_loss)

    print("\nDaily baseline test loss:")
    print(daily_test_loss)

    print("\nRNN improvement over persistence:")
    print(persistence_test_loss - final_rnn_test_loss)

    print("\nRNN improvement over daily baseline:")
    print(daily_test_loss - final_rnn_test_loss)

    print("\n" + "=" * 80)
    print("Final comparison in NORMALIZED scale")
    print("=" * 80)

    print("\nRNN test loss normalized:")
    print(final_rnn_test_loss_norm)

    print("\nPersistence baseline test loss normalized:")
    print(persistence_test_loss_norm)

    print("\nDaily baseline test loss normalized:")
    print(daily_test_loss_norm)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Original time series
    plt.figure(figsize=(10, 4))
    plt.plot(t.numpy(), x.numpy())
    plt.xlabel("hour")
    plt.ylabel("synthetic load")
    plt.title("Synthetic hourly load time series")
    plt.tight_layout()

    figure_path = FIGURE_DIR / "synthetic_load_series.png"
    plt.savefig(figure_path, dpi=200)
    print(f"\nSaved plot to {figure_path}")

    # Train/test loss on normalized scale
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss on normalized data")
    plt.title("RNN forecasting loss with normalization")
    plt.legend()
    plt.tight_layout()

    figure_path = FIGURE_DIR / "rnn_forecasting_loss_normalized.png"
    plt.savefig(figure_path, dpi=200)
    print(f"Saved plot to {figure_path}")

    # True vs predicted test values in original scale
    plt.figure(figsize=(10, 4))

    n_plot = 200

    plt.plot(
        y_test[:n_plot].squeeze().numpy(),
        label="true next-hour load",
    )

    plt.plot(
        y_test_pred[:n_plot].squeeze().numpy(),
        label="RNN prediction",
    )

    plt.plot(
        persistence_pred[:n_plot].squeeze().numpy(),
        label="persistence baseline",
        linestyle="--",
    )

    plt.plot(
        daily_pred[:n_plot].squeeze().numpy(),
        label="daily baseline",
        linestyle=":",
    )

    plt.xlabel("test time index")
    plt.ylabel("synthetic load")
    plt.title("Next-hour forecasting: RNN vs baselines")
    plt.legend()
    plt.tight_layout()

    figure_path = FIGURE_DIR / "rnn_forecasting_predictions_with_baselines.png"
    plt.savefig(figure_path, dpi=200)
    print(f"Saved plot to {figure_path}")


if __name__ == "__main__":
    main()