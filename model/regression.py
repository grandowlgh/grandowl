import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class LinearRegression(nn.Module):
    """PyTorch-based Simple Linear Regression Model optimized via SGD."""

    def __init__(self, learning_rate: float = 0.001, num_epochs: int = 1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Initialize parameters w0 (intercept) and w1 (slope)
        self.w1 = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))
        self.w0 = nn.Parameter(torch.zeros(1, dtype=torch.float32, requires_grad=True))

        # Loss function & Optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD([self.w0, self.w1], lr=self.learning_rate)

        # Optimization history
        self.w0_history = []
        self.w1_history = []
        self.loss_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes response y = w0 + w1 * x."""
        return self.w0 + self.w1 * x

    def _to_tensor(self, data) -> torch.Tensor:
        """Helper function to convert arrays/lists to float32 PyTorch Tensors."""
        if isinstance(data, torch.Tensor):
            return data.float().view(-1, 1)
        return torch.tensor(np.array(data), dtype=torch.float32).view(-1, 1)

    def fit(self, X_train, y_train, X_test, y_test):
        """Trains the model using SGD and prints test set R^2 score."""
        X_tr = self._to_tensor(X_train)
        y_tr = self._to_tensor(y_train)
        X_te = self._to_tensor(X_test)
        y_te = self._to_tensor(y_test)

        self.w0_history.clear()
        self.w1_history.clear()
        self.loss_history.clear()

        # Training loop
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            
            y_pred = self.forward(X_tr)
            loss = self.loss_fn(y_pred, y_tr)
            
            loss.backward()
            self.optimizer.step()

            # Record metrics
            self.w0_history.append(self.w0.item())
            self.w1_history.append(self.w1.item())
            self.loss_history.append(loss.item())

        # Compute R^2 score on test set
        self.eval()
        with torch.no_grad():
            y_test_pred = self.forward(X_te)
            ss_res = torch.sum((y_te - y_test_pred) ** 2)
            ss_tot = torch.sum((y_te - torch.mean(y_te)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot)

        print(f"Training Complete ({self.num_epochs} Epochs)")
        print(f"Final Estimated Parameters -> w0 (Intercept): {self.w0.item():.6f}, w1 (Slope): {self.w1.item():.6f}")
        print(f"Test Set R^2 Score: {r2.item():.6f}")
        return r2.item()

    def predict(self, x) -> np.ndarray:
        """Predicts response for new independent variable inputs."""
        self.eval()
        X_tensor = self._to_tensor(x)
        with torch.no_grad():
            predictions = self.forward(X_tensor)
        return predictions.numpy().flatten()

    def analysis_plot(self, X_train, y_train):
        """Plots training data, fitted line, parameter trajectories, and loss curve."""
        X_tr = self._to_tensor(X_train).numpy().flatten()
        y_tr = self._to_tensor(y_train).numpy().flatten()

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Subplot 1: Fitted Line
        x_line = np.linspace(X_tr.min(), X_tr.max(), 100)
        y_line = self.predict(x_line)
        axs[0].scatter(X_tr, y_tr, color="blue", alpha=0.6, label="Original Data")
        axs[0].plot(x_line, y_line, color="red", linewidth=2, label="Fitted Regression Line")
        axs[0].set_xlabel("Benefit-Cost-Ratio (BCR)")
        axs[0].set_ylabel("Annual Production")
        axs[0].set_title("Fitted Model vs Data")
        axs[0].legend()
        axs[0].grid(True)

        # Subplot 2: Parameter Trajectories
        epochs = range(1, len(self.loss_history) + 1)
        axs[1].plot(epochs, self.w0_history, label="$w_0$ (Intercept)", color="green")
        axs[1].plot(epochs, self.w1_history, label="$w_1$ (Slope)", color="purple")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Parameter Value")
        axs[1].set_title("Parameter Trajectories ($w_0, w_1$)")
        axs[1].legend()
        axs[1].grid(True)

        # Subplot 3: Loss Trajectory
        axs[2].plot(epochs, self.loss_history, color="orange", label="MSE Loss")
        axs[2].set_xlabel("Epochs")
        axs[2].set_ylabel("Mean Squared Error")
        axs[2].set_title("Loss Convergence Curve")
        axs[2].set_yscale("log")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()