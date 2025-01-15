import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, Generator, Optional
from enum import Enum
from itertools import chain
import os
from pathlib import Path

dtype = torch.float32


# Define Potential Types
class PotentialType(Enum):
    HARMONIC = "Harmonic"
    ANHARMONIC = "Anharmonic"
    WELL = "Well"


# Fixed Parameters
GRID_SIZE = 64  # Fixed N x N grid size for FNO
MAX_EIGENVALUES = 2
POTENTIAL_KEY = "potentials"
EIGENVALUE_KEY = "eigenvalues"
LABEL_KEY = "labels"



# Dataset Class for Precomputed Potentials
class PotentialDataset(Dataset):
    def __init__(self, potentials: np.ndarray, eigenvalues: np.ndarray, labels: np.ndarray):
        self.potentials = torch.tensor(potentials, dtype=dtype)
        self.eigenvalues = torch.tensor(eigenvalues, dtype=dtype)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.potentials)

    def __getitem__(self, idx):
        return self.potentials[idx], self.eigenvalues[idx], self.labels[idx]


# Generate Potentials and Eigenvalues
def generate_potential_data(
    num_samples: int,
    dx: float,
    dy: float,
    potential_type: PotentialType,
    lambda_value: float = 0.1,
    well_depth_range: Tuple[float, float] = (-3, -1),
    num_eigenvalues: int = MAX_EIGENVALUES,
) -> Tuple[np.ndarray, np.ndarray]:
    
    assert isinstance(num_samples, int), "num_samples has to be an integer"
    assert num_samples >= 1, "num_samples cannot be less than 1"
    
    potentials = []
    eigenvalues = []

    x = np.linspace(-1, 1, GRID_SIZE)
    y = np.linspace(-1, 1, GRID_SIZE)
    X, Y = np.meshgrid(x, y)

    for _ in range(num_samples):
        if potential_type == PotentialType.HARMONIC:
            V = 0.5 * (X**2 + Y**2)
        elif potential_type == PotentialType.ANHARMONIC:
            V = 0.5 * (X**2 + Y**2) + lambda_value * (X**4 + Y**4)
        elif potential_type == PotentialType.WELL:
            V = np.zeros_like(X)
            well_depth = np.random.uniform(*well_depth_range)
            well_width = np.random.uniform(0.2, 0.8)
            V[(np.abs(X) <= well_width) & (np.abs(Y) <= well_width)] = well_depth
        else:
            raise ValueError("Unknown potential type")

        # Finite Difference Matrix
        D2_x = sp.diags([1, -2, 1], [-1, 0, 1], shape=(GRID_SIZE, GRID_SIZE)) / dx**2
        D2_y = sp.diags([1, -2, 1], [-1, 0, 1], shape=(GRID_SIZE, GRID_SIZE)) / dy**2
        I = sp.identity(GRID_SIZE)
        D2_2D = sp.kron(D2_x, I) + sp.kron(I, D2_y)

        # Hamiltonian
        H = -0.5 * D2_2D + sp.diags(V.flatten(), 0)

        # Eigenvalues
        try:
            E, _ = spla.eigsh(H, k=num_eigenvalues, which="SM")
        except spla.ArpackNoConvergence:
            E, _ = la.eigh(H.toarray())
        E = np.sort(E)[:num_eigenvalues]

        potentials.append(V)
        eigenvalues.append(E)

    return np.array(potentials), np.array(eigenvalues)


# Precompute Dataset and Save
def precompute_data(
    output_path: str,
    num_samples: int,
    dx: float,
    dy: float,
    potential_type: PotentialType,
    *,
    mix_ratio: float = 1.0
):
    """
    NOTE: Zero label for Harmonic and 1 label for Well Potential
    """
    assert mix_ratio >= 0.1 and mix_ratio <= 1.0, "mix ratio should be from 0.1 to 1.0"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    match mix_ratio:
        case 1.0:
            potentials, eigenvalues = generate_potential_data(
                num_samples, dx, dy, potential_type
            )
            match potential_type:
                case PotentialType.HARMONIC:
                    labels = np.zeros(num_samples)
                case PotentialType.WELL:
                    labels = np.ones(num_samples)
                case _:
                    labels = np.zeros(num_samples) + 2
            # labels = np.zeros(num_samples) if potential_type == PotentialType.HARMONIC else np.ones(num_samples)
            np.savez(output_path, potentials=potentials, eigenvalues=eigenvalues, labels=labels)

        case _:
            num_samples_1 = int(mix_ratio * num_samples)
            potentials_1, eigenvalues_1 = generate_potential_data(
                num_samples_1, dx, dy, potential_type
            )
            labels_1 = np.zeros(num_samples_1) if potential_type == PotentialType.HARMONIC else np.ones(num_samples_1)

            other_type = PotentialType.HARMONIC if potential_type == PotentialType.WELL else PotentialType.WELL
            num_samples_2 = num_samples - num_samples_1
            potentials_2, eigenvalues_2 = generate_potential_data(
                num_samples_2, dx, dy, other_type
            )
            labels_2 = np.zeros(num_samples_2) if other_type == PotentialType.HARMONIC else np.ones(num_samples_2)

            
            # Concatenate the potentials and eigenvalues from both types
            potentials = np.concatenate([potentials_1, potentials_2], axis=0)
            eigenvalues = np.concatenate([eigenvalues_1, eigenvalues_2], axis=0)
            labels = np.concatenate([labels_1, labels_2], axis=0)

            # Save the mixed dataset
            np.savez(output_path, potentials=potentials, eigenvalues=eigenvalues, labels=labels)



class BalancedSampler(Sampler):
    def __init__(self, dataset: Dataset, harmonic_ratio: float, batch_size: int):
        self.dataset = dataset
        self.harmonic_ratio = harmonic_ratio
        self.batch_size = batch_size

        # Extract labels and separate indices
        self.labels = dataset.labels.numpy()
        self.harmonic_indices = np.where(self.labels == 0)[0]
        self.well_indices = np.where(self.labels == 1)[0]

        # Shuffle indices initially
        np.random.shuffle(self.harmonic_indices)
        np.random.shuffle(self.well_indices)

    def __iter__(self):
        harmonic_per_batch = int(self.harmonic_ratio * self.batch_size)
        well_per_batch = self.batch_size - harmonic_per_batch

        # Ensure we have enough samples
        harmonic_batches = self.harmonic_indices[:len(self.harmonic_indices) - (len(self.harmonic_indices) % harmonic_per_batch)]
        well_batches = self.well_indices[:len(self.well_indices) - (len(self.well_indices) % well_per_batch)]

        # Create balanced mini-batches
        batches = []
        for i in range(0, len(harmonic_batches), harmonic_per_batch):
            batch_harmonic = harmonic_batches[i : i + harmonic_per_batch]
            batch_well = well_batches[i : i + well_per_batch]

            if len(batch_harmonic) == harmonic_per_batch and len(batch_well) == well_per_batch:
                batch = np.concatenate([batch_harmonic, batch_well])
                np.random.shuffle(batch)  # Shuffle within the batch
                batches.append(batch)

        # Flatten the list of batches into a single iterable
        return iter(np.concatenate(batches))

    def __len__(self):
        # The number of mini-batches in the dataset
        total_samples = len(self.harmonic_indices) + len(self.well_indices)
        return total_samples // self.batch_size
    

def get_balanced_dataloader(file_path: Path, batch_size: int, harmonic_ratio: float = 0.5):
    # Load the data from the .npz file
    data = np.load(file_path)
    potentials = data[POTENTIAL_KEY]
    eigenvalues = data[EIGENVALUE_KEY]
    labels = data[LABEL_KEY]

    # Create the dataset
    dataset = PotentialDataset(potentials, eigenvalues, labels)

    # Create a BalancedSampler to enforce the mix ratio in mini-batches
    # sampler = BalancedSampler(dataset, harmonic_ratio)
    sampler = BalancedSampler(dataset, harmonic_ratio, batch_size)

    # Create and return the DataLoader with the custom sampler
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    return dataloader


# Create DataLoader for Training
def get_dataloader(file_path: str, batch_size: int, shuffle: bool = True, balance_batches: bool = False, harmonic_ratio: float = 0.5):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if balance_batches is False:
        data = np.load(file_path)
        dataset = PotentialDataset(data[POTENTIAL_KEY], data[EIGENVALUE_KEY], data[LABEL_KEY])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    assert harmonic_ratio >= 0.1 and harmonic_ratio <= 0.9, "harmonic_ratio should be between 0.1 and 0.9"
    return get_balanced_dataloader(
        file_path=file_path,
        batch_size=batch_size,
        harmonic_ratio=harmonic_ratio
    )


def generate_data(type_1: PotentialType, num_samples_1: int, type_2: PotentialType, num_samples_2: int, **kw_args):
    gen_1 = generate_potential_data(num_samples=num_samples_1, **kw_args, potential_type=type_1)
    gen_2 = generate_potential_data(num_samples=num_samples_2, **kw_args, potential_type=type_2)
    return chain(gen_1, gen_2)
