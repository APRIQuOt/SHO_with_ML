import torch
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla


def generate_potential_data(
    num_samples,
    dx,
    dy,
    max_x_extent,
    max_y_extent,
    potential_type,
    min_x_extent=2,
    min_y_extent=2,
    lambda_value=0.1,
    num_eigenvalues=2,
    *,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    potentials = []
    eigenvalues = []

    # Compute the maximum grid size based on max_x and max_y
    max_x = int(max_x_extent / dx) + 1
    max_y = int(max_y_extent / dy) + 1

    for _ in range(num_samples):
        # Random physical extent for X and Y, ensuring it's within min and max bounds
        x_extent = np.random.uniform(min_x_extent, max_x_extent)
        y_extent = np.random.uniform(min_y_extent, max_y_extent)

        # Compute the number of grid points based on the extent and step size
        grid_size_x = int(x_extent / dx) + 1  # Number of grid points along X
        grid_size_y = int(y_extent / dy) + 1  # Number of grid points along Y

        # Create the grid
        x = np.linspace(-x_extent / 2, x_extent / 2, grid_size_x)
        y = np.linspace(-y_extent / 2, y_extent / 2, grid_size_y)
        X, Y = np.meshgrid(x, y)

        if potential_type == "harmonic":
            V = 0.5 * (X**2 + Y**2)
        elif potential_type == "anharmonic":
            V = 0.5 * (X**2 + Y**2) + lambda_value * (X**4 + Y**4)
        elif potential_type == "well":
            V = np.zeros_like(X)
            well_depth = np.random.uniform(-3, -1)
            well_width_x = np.random.uniform(
                min_x_extent, max_x_extent
            )  # Random width in X direction for the rectangular well
            well_width_y = np.random.uniform(
                min_y_extent, max_y_extent
            )  # Random width in Y direction for the rectangular well
            V[(np.abs(X) <= well_width_x / 2) & (np.abs(Y) <= well_width_y / 2)] = (
                well_depth
            )
        else:
            raise ValueError(
                "Unknown potential type: choose 'harmonic', 'anharmonic', or 'well'"
            )

        # Generate finite difference matrices for the potential grid
        D2_x = (
            sp.diags([1, -2, 1], [-1, 0, 1], shape=(grid_size_x, grid_size_x)) / dx**2
        )
        D2_y = (
            sp.diags([1, -2, 1], [-1, 0, 1], shape=(grid_size_y, grid_size_y)) / dy**2
        )
        I_x = sp.identity(grid_size_x)
        I_y = sp.identity(grid_size_y)

        D2_2D = sp.kron(D2_x, I_y) + sp.kron(I_x, D2_y)

        # Hamiltonian with potential
        V_flat = V.flatten()
        H = -0.5 * D2_2D + sp.diags(V_flat, 0)

        # Determine the matrix size N
        N = H.shape[0]

        try:
            # If k >= N, use dense matrix solver scipy.linalg.eigh, otherwise use sparse solver eigsh
            if num_eigenvalues >= N:
                E, _ = la.eigh(H.toarray())
                E = np.sort(E)[:num_eigenvalues]
            else:
                E, _ = spla.eigsh(
                    H, k=num_eigenvalues, which="SM", maxiter=50000, tol=1e-4
                )
                E = np.sort(E)
        except spla.ArpackNoConvergence:
            # Fall back to dense solver in case of convergence failure
            print("Warning: ARPACK failed to converge, using dense solver instead.")
            E, _ = la.eigh(H.toarray())
            E = np.sort(E)[:num_eigenvalues]

        # Pad the potential to max_x and max_y dimensions
        padded_V = pad_potential(V, max_x, max_y)

        potentials.append(padded_V)
        eigenvalues.append(E)

    # Convert potentials and eigenvalues to NumPy arrays first, then to torch tensors
    potentials = np.array(potentials)
    potentials = torch.tensor(potentials).float().to(device=device)

    eigenvalues = np.array(eigenvalues)
    eigenvalues = torch.tensor(eigenvalues).float().to(device=device)

    return potentials, eigenvalues


def pad_potential(potential, max_x, max_y):
    """Pad or truncate the potential grid to the maximum size."""
    padded = np.zeros((max_x, max_y))  # Initialize with zeros
    # Get the size of the current potential
    x_size, y_size = potential.shape

    # Ensure the potential is correctly padded or truncated
    x_end = min(x_size, max_x)  # Use the smaller size to avoid truncating incorrectly
    y_end = min(y_size, max_y)

    padded[:x_end, :y_end] = potential[:x_end, :y_end]  # Copy the original potential
    return padded


def generate_potential_data_feeto(
    num_samples, grid_size, num_eigenvalues, device, dtype=torch.float32
):
    # Create the grid and potential
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    V = 0.5 * (X**2 + Y**2)  # Harmonic potential
    V_flat = V.flatten()  # Flatten once

    # Generate finite difference matrices for the potential grid
    dx = x[1] - x[0]
    D2_1D = sp.diags([1, -2, 1], [-1, 0, 1], shape=(grid_size, grid_size)) / dx**2
    I = sp.identity(grid_size)
    D2_2D = sp.kron(D2_1D, I) + sp.kron(I, D2_1D)

    # Precompute Hamiltonian
    H = -0.5 * D2_2D

    # Compute eigenvalues for all samples
    eigenvalues = []
    potentials = [
        V
    ] * num_samples  # Create a list of the same potential for all samples

    for _ in range(num_samples):
        # Add the potential to the Hamiltonian
        H_with_potential = H + sp.diags(V_flat, 0)

        # Compute eigenvalues
        E, _ = spla.eigsh(H_with_potential, k=num_eigenvalues, which="SM")
        eigenvalues.append(np.sort(E))

    return torch.tensor(np.array(potentials), dtype=dtype).to(device), torch.tensor(
        np.array(eigenvalues), dtype=dtype
    ).to(device)
