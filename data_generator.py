import numpy as np
from typing import Tuple, Union


def generate_data(
    n_samples: int,
    *,
    nx: Union[int, float] = 5,
    hbar: int = 1,
    m: Union[int, float] = 1,
    omega: Union[int, float] = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates data for a quantum mechanical system with a 2D harmonic oscillator potential.

    Args:
        n_samples (int): The number of samples to generate.
        nx (Union[int, float], optional): The maximum x-coordinate of the grid. Defaults to 5.
        hbar (int, optional): The reduced Planck constant. Defaults to 1.
        m (Union[int, float], optional): The mass of the particle. Defaults to 1.
        omega (Union[int, float], optional): The angular frequency of the harmonic oscillator. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]: A tuple containing:
            - energy_indices (np.ndarray): The indices of the first 5 energy levels.
            - true_energy_levels (np.ndarray): The true energy levels of the first 5 energy levels.
            - levels (List[Tuple[int, int]]): The first 5 energy levels in the (n_x, n_y) format.
    """

    def energy_levels(n1: int, n2: int) -> np.ndarray:
        return (n1 + n2 + 1) * hbar * omega

    # Define first 5 levels (n_x, n_y)
    levels = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)]

    # Generate true wave functions and energy levels
    true_energy_levels = np.zeros(len(levels))

    # Generate meshgrid
    x = np.linspace(-nx, nx, n_samples)
    y = np.linspace(-nx, nx, n_samples)

    # X, Y = np.meshgrid(x, y)
    X, Y = np.ogrid[-nx : nx : len(x) * 1j, -nx : nx : len(y) * 1j]

    for i, (nx, ny) in enumerate(levels):
        true_energy_levels[i] = energy_levels(nx, ny)

    energy_indices = np.arange(len(levels)).reshape(-1, 1)

    return energy_indices, true_energy_levels, levels
