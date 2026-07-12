from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. Dataset: copy-first memory task
# ============================================================

def make_copy_first_dataset(n_samples=1000, seq_len=10):
    """
    Create examples of the form:

        input  = [x_1, x_2, ..., x_T]
        target = x_1

    where x_t are random numbers.

    This is a simple memory task:
    the model must remember the first input over seq_len steps.

    X shape: (n_samples, seq_len, 1)
    y shape: (n_samples, 1)
    """
    X = torch.randn(n_samples, seq_len, 1)
    y = X[:, 0, :]

    return X, y


# ============================================================
# 2. Model
# ============================================================

class RNNRegressor(nn.Module):
    """
    Sequence-to-one vanilla RNN model.

    Input:
        x shape: (batch_size, seq_len, input_size)

    Output:
        y_pred shape: (batch_size, output_size)
    """

    def __init__(
        self,
        input_size=1,
        hidden_size=16,
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
# 3. Experiment function
# ============================================================

def run_experiment(
    seq_len=10,
    hidden_size=16,
    lr=0.01,
    nonlinearity="tanh",
    batch_size=32,
    num_epochs=20,
    n_train=1000,
    n_test=200,
):
    """
    Train a vanilla RNN on the copy-first task for a fixed sequence length.
    """
    print("\n" + "=" * 80)
    print(
        f"Running experiment: "
        f"seq_len={seq_len}, "
        f"hidden_size={hidden_size}, "
        f"lr={lr}, "
        f"nonlinearity={nonlinearity}, "
        f"batch_size={batch_size}"
    )
    print("=" * 80)

    X_train, y_train = make_copy_first_dataset(
        n_samples=n_train,
        seq_len=seq_len,
    )
    X_test, y_test = make_copy_first_dataset(
        n_samples=n_test,
        seq_len=seq_len,
    )

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    train_dataset = TensorDataset(X_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = RNNRegressor(
        input_size=1,
        hidden_size=hidden_size,
        output_size=1,
        nonlinearity=nonlinearity,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    loss_fn = nn.MSELoss()

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
            y_test_pred = model(X_test)
            test_loss = loss_fn(y_test_pred, y_test)

        train_losses.append(average_train_loss)
        test_losses.append(test_loss.item())

        print(
            f"epoch={epoch:3d} | "
            f"train_loss={average_train_loss:.6f} | "
            f"test_loss={test_loss.item():.6f}"
        )

    return {
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "lr": lr,
        "nonlinearity": nonlinearity,
        "batch_size": batch_size,
        "final_test_loss": test_losses[-1],
        "train_losses": train_losses,
        "test_losses": test_losses,
    }


# ============================================================
# 4. Main experiment
# ============================================================

def main():
    torch.manual_seed(0)

    experiments = [
        {"seq_len": 4, "hidden_size": 16, "lr": 0.01, "nonlinearity": "tanh"},
        {"seq_len": 10, "hidden_size": 16, "lr": 0.01, "nonlinearity": "tanh"},
        {"seq_len": 20, "hidden_size": 16, "lr": 0.01, "nonlinearity": "tanh"},
        {"seq_len": 50, "hidden_size": 16, "lr": 0.01, "nonlinearity": "tanh"},
    ]

    results = []

    for config in experiments:
        result = run_experiment(
            seq_len=config["seq_len"],
            hidden_size=config["hidden_size"],
            lr=config["lr"],
            nonlinearity=config["nonlinearity"],
            batch_size=32,
            num_epochs=20,
        )

        results.append(result)

    # --------------------------------------------------------
    # Print ranked summary table
    # --------------------------------------------------------

    results_sorted = sorted(
        results,
        key=lambda r: r["final_test_loss"],
    )

    print("\n" + "=" * 88)
    print("Summary ranked by final test loss")
    print("=" * 88)

    print(
        f"{'rank':>4} | "
        f"{'seq_len':>8} | "
        f"{'hidden_size':>12} | "
        f"{'lr':>8} | "
        f"{'nonlinearity':>12} | "
        f"{'final_test_loss':>16}"
    )
    print("-" * 88)

    for rank, r in enumerate(results_sorted, start=1):
        print(
            f"{rank:>4} | "
            f"{r['seq_len']:>8} | "
            f"{r['hidden_size']:>12} | "
            f"{r['lr']:>8} | "
            f"{r['nonlinearity']:>12} | "
            f"{r['final_test_loss']:>16.8f}"
        )

    best = results_sorted[0]

    print("\nBest configuration:")
    print(
        f"seq_len={best['seq_len']}, "
        f"hidden_size={best['hidden_size']}, "
        f"lr={best['lr']}, "
        f"nonlinearity={best['nonlinearity']}, "
        f"final_test_loss={best['final_test_loss']:.8f}"
    )

    # --------------------------------------------------------
    # Plot test loss curves
    # --------------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for r in results:
        label = (
            f"T={r['seq_len']}, "
            f"H={r['hidden_size']}, "
            f"lr={r['lr']}"
        )

        plt.plot(r["test_losses"], label=label)

    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("test loss")
    plt.title("Copy-first memory task: effect of sequence length")
    plt.legend()
    plt.tight_layout()

    figure_path = FIGURE_DIR / "copy_first_memory_test_losses.png"
    plt.savefig(figure_path, dpi=200)

    print(f"\nSaved plot to {figure_path}")


if __name__ == "__main__":
    main()