import time
import torch
import accelerate
import numpy as np
from enum import Enum
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import scipy.linalg as la
import scipy.sparse as sp
import torch.optim as optim
import matplotlib.pyplot as plt
from neuralop.models import FNO
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.sparse.linalg as spla
from typing import Tuple, List, Optional, Callable, TypeVar, Generic


T = TypeVar("T", bound="Result")


class Result(Generic[T]):
    def save_to_file(self, filepath: str | Path) -> None:
        assert isinstance(filepath, str) or isinstance(
            filepath, Path
        ), "filepath must be a string or Path object"

        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.suffix == ".pth":
            filepath = filepath.with_suffix(".pth")

        torch.save(self.__dict__, filepath)

    @classmethod
    def load_from_file(cls: type[T], filepath: str | Path) -> T:
        assert isinstance(filepath, str) or isinstance(
            filepath, Path
        ), "filepath must be a string or Path object"

        if isinstance(filepath, str):
            filepath = Path(filepath)

        checkpoint = torch.load(filepath, weights_only=False)
        return cls(**checkpoint)



@dataclass
class Configuration(Result["Configuration"]):
    num_samples: int
    grid_size: int
    num_eigenvalues: int
    learning_rate: float
    num_epochs: int
    n_modes: Tuple[int, int]
    hidden_channels: int
    in_channels: int
    out_channels: int
    lifting_channels: int
    projection_channels: int
    n_layers: int


@dataclass
class DataConfiguration(Result["DataConfiguration"]):
    dx: float
    dy: float
    max_x_extent: float
    max_y_extent: float
    potential_type: str


@dataclass
class TrainInput:
    train_potentials: torch.Tensor
    train_eigen_vals: torch.Tensor
    num_epochs: int
    model: FNO
    optimizer: optim.Optimizer
    loss_fn: Callable[..., float] = nn.MSELoss()
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None


@dataclass
class TrainResult(Result["TrainResult"]):
    losses: List[float]
    model: FNO
    train_duration: float

    def save_model(self, filepath: str | Path) -> None:
        # Ensure the filepath is a Path object
        if isinstance(filepath, str):
            filepath = Path(filepath)

        torch.save(self.model.state_dict(), filepath)

    def plot_losses(
        self,
        *,
        save: bool = True,
        save_file: str = "training_loss_harmonic_to_anharmonic.png",
        title: str = "Training Loss across Training Epochs",
    ) -> None:
        plt.figure(figsize=(10, 5))
        plt_losses = [i.item() for i in self.losses]
        plt.plot(plt_losses)
        plt.hlines(
            plt_losses[-1],
            0,
            len(plt_losses),
            linestyles="dashed",
            color="r",
            label=f"final loss ({round(plt_losses[-1], 5)})",
        )
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Mean Square Error")
        plt.title(title)
        plt.legend()

        if save:
            plt.savefig(save_file)

        plt.show()


class ErrorType(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class PlotType(Enum):
    SCATTER = "scatter"
    BOX = "box"
    VIOLIN = "violin"
    BOX_ON_VIOLIN = "box_on_violin"
    HISTOGRAM = "histogram"


@dataclass
class ValidationErrors:
    MRE: float
    MSE: float
    MAE: float
    R2: torch.Tensor


@dataclass
class ValidationInput:
    val_potentials: torch.Tensor
    val_eigen_vals: torch.Tensor
    model: FNO
    loss_fn: Callable[..., float] = nn.MSELoss()


@dataclass
class ValidationResult(Result["ValidationResult"]):
    val_eigen_vals: torch.Tensor
    val_loss: float
    predicted_val_eigen_vals: torch.Tensor

    def errors(self, *, epsilon=1e-8) -> ValidationErrors:
        predicted_gaps = (
            self.predicted_val_eigen_vals[:, 1] - self.predicted_val_eigen_vals[:, 0]
        ).cpu()
        # predicted_gaps = predicted_gaps
        actual_gaps = (self.val_eigen_vals[:, 1] - self.val_eigen_vals[:, 0]).cpu()
        # actual_gaps = actual_gaps

        # Compute the mean absolute error
        absolute_errors = torch.abs(predicted_gaps - actual_gaps)
        mae = torch.mean(absolute_errors)

        # Compute the mean relative error
        relative_errors = absolute_errors / (torch.abs(actual_gaps) + epsilon)
        mre = torch.mean(relative_errors)

        # Pearson correlation coefficient
        r2 = torch.corrcoef(torch.stack([actual_gaps, predicted_gaps]))[0, 1]

        # Compute the mean square error
        squared_errors = (predicted_gaps - actual_gaps) ** 2
        mse = torch.mean(squared_errors)

        return ValidationErrors(
            MRE=mre.item(),
            MSE=mse.item(),
            MAE=mae.item(),
            R2=r2,
        )

    def plot_loss(
        self,
        *,
        save_fig: bool = True,
        save_file: str = "validation_loss.png",
        title: str = "Validation Loss",
    ) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.val_loss)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()

        if save_fig:
            plt.savefig(save_file)

        plt.show()

    def plot(
        self,
        error_type: ErrorType = ErrorType.ABSOLUTE,
        *,
        save_fig: bool = True,
        save_file: str = "validation_error.png",
        title: str = "Validation Error (Log Scale)",
    ) -> None:
        """
        Plot the error (either absolute or relative) on a log scale.
        """
        predicted_gaps = (
            self.predicted_val_eigen_vals[:, 1] - self.predicted_val_eigen_vals[:, 0]
        ).cpu()
        # predicted_gaps = predicted_gaps
        actual_gaps = (self.val_eigen_vals[:, 1] - self.val_eigen_vals[:, 0]).cpu()
        # Compute absolute or relative error
        if error_type == ErrorType.ABSOLUTE:
            errors = torch.abs(predicted_gaps - actual_gaps)
            ylabel = "Absolute Error (Log Scale)"
        elif error_type == ErrorType.RELATIVE:
            # To avoid division by zero, we add a small constant where true values are zero
            epsilon = 1e-8
            errors = torch.abs(predicted_gaps - actual_gaps) / (actual_gaps + epsilon)
            ylabel = "Relative Error (Log Scale)"

        # Plotting the error on a log scale
        plt.figure(figsize=(10, 5))
        plt.plot(errors.cpu().numpy(), label="Error")
        plt.yscale("log")  # Log scale for the error
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()

        if save_fig:
            plt.savefig(save_file)

        plt.show()

    def plot_eigen_values(
        self,
        *,
        save_fig: bool = True,
        save_file: str = "validation_plot.png",
        title: str = "Predicted vs Actual Eigenvalues",
    ) -> None:
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.predicted_val_eigen_vals[:, 0].cpu(), label='Predicted Eigenvalues - Level 0', marker='o')
        plt.plot(self.predicted_val_eigen_vals[:, 1].cpu(), label='Predicted Eigenvalues - Level 1', marker='o')
        plt.plot(self.val_eigen_vals[:, 0].cpu(), label='True Eigenvalues - Level 0', marker='x')
        plt.plot(self.val_eigen_vals[:, 1].cpu(), label='True Eigenvalues - Level 1', marker='x')
        plt.title(title)
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue')
        plt.legend()
        

        if save_fig:
            plt.savefig(save_file)
        plt.show()

    def visualize_energy_gap(
        self,
        plot_type: PlotType,
        *,
        save_fig: bool = True,
        save_file: str = "energy_gap.png",
        title: str = "Predicted vs. Actual Energy Gaps",
    ) -> None:
        # Calculate the energy gap
        predicted_energy_gaps = (
            self.predicted_val_eigen_vals[:, 1]
            - self.predicted_val_eigen_vals[:, 0]
        ).cpu()
        
        true_energy_gaps = (
            self.val_eigen_vals[:, 1] - self.val_eigen_vals[:, 0]
        ).cpu()

        match plot_type:
            case PlotType.SCATTER:
                plt.figure(figsize=(8, 6))
                plt.scatter(
                    true_energy_gaps.cpu(),
                    predicted_energy_gaps.cpu(),
                    alpha=0.7,
                    label="Predicted vs Actual Gaps",
                )
                plt.plot(
                    [min(true_energy_gaps), max(true_energy_gaps)],
                    [min(true_energy_gaps), max(true_energy_gaps)],
                    "r--",
                    label="Perfect Prediction (y = x)",
                )
                plt.xlabel("Actual Energy Gap")
                plt.ylabel("Predicted Energy Gaps")
                plt.title("Predicted vs. Actual Energy Gaps")
            case PlotType.BOX:
                absolute_errors = abs(predicted_energy_gaps - true_energy_gaps)

                plt.figure(figsize=(6, 5))
                sns.boxplot(data=absolute_errors, color="skyblue")
                plt.title("Box Plot of Absolute Errors")
                plt.ylabel("Absolute Error")
            case PlotType.HISTOGRAM:
                absolute_errors = abs(predicted_energy_gaps - true_energy_gaps)

                plt.figure(figsize=(8, 6))
                sns.histplot(absolute_errors, kde=True, color="blue", alpha=0.6)
                plt.title("Distribution of Absolute Errors")
                plt.xlabel("Absolute Error")
                plt.ylabel("Frequency")
            case PlotType.VIOLIN:
                absolute_errors = abs(predicted_energy_gaps - true_energy_gaps)

                plt.figure(figsize=(6, 5))
                sns.violinplot(data=absolute_errors, color="lightgreen")
                plt.title("Violin Plot of Absolute Errors")
                plt.ylabel("Absolute Error")
            case PlotType.BOX_ON_VIOLIN:
                absolute_errors = abs(predicted_energy_gaps - true_energy_gaps)

                plt.figure(figsize=(6, 5))
                sns.violinplot(data=absolute_errors, color="lightgreen", inner=None)
                sns.boxplot(data=absolute_errors, color="skyblue", width=0.3)
                plt.title("Violin + Box Plot of Absolute Errors")
                plt.ylabel("Absolute Error")

            case _:
                raise ValueError(f"Unknown plot type: {plot_type}")

        plt.legend()
        plt.grid(True)

        if save_fig:
            plt.savefig(save_file)

        plt.show()
