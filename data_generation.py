import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def generate_potential_data(num_samples, grid_size, num_eigenvalues):
    potentials = []
    eigenvalues = []

    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    V = 0.5 * (X**2 + Y**2)  # Harmonic potential

    for _ in range(num_samples):
        # Generate finite difference matrices for the potential grid
        dx = x[1] - x[0]
        D2_1D = sp.diags([1, -2, 1], [-1, 0, 1], shape=(grid_size, grid_size)) / dx**2
        I = sp.identity(grid_size)
        D2_2D = sp.kron(D2_1D, I) + sp.kron(I, D2_1D)

        # Hamiltonian with potential
        V_flat = V.flatten()
        H = -0.5 * D2_2D + sp.diags(V_flat, 0)

        # Compute eigenvalues
        E, _ = spla.eigsh(H, k=num_eigenvalues, which='SM')
        E = np.sort(E)

        potentials.append(V)
        eigenvalues.append(E)

    potentials = np.array(potentials)
    eigenvalues = np.array(eigenvalues)
    return potentials, eigenvalues

if __name__ == "__main__":
    num_samples = 500
    grid_size = 30
    num_eigenvalues = 25

    X_train, y_train = generate_potential_data(num_samples, grid_size, num_eigenvalues)
    X_val, y_val = generate_potential_data(10, grid_size, num_eigenvalues)
    X_test, y_test = generate_potential_data(10, grid_size, num_eigenvalues)

    # For testing, print the shapes of the generated data
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
