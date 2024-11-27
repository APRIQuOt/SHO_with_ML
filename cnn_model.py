import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_generation import generate_potential_data, pad_potential

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#CNN model
class CNNModel(nn.Module):
    def __init__(self, input_shape):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16, momentum=0.9230123048499552)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32, momentum=0.9230123048499552)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9230123048499552)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128, momentum=0.9230123048499552)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256, momentum=0.9230123048499552)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256, momentum=0.9230123048499552)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256, momentum=0.9230123048499552)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * input_shape[0] * input_shape[1], 512)
        self.fc2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=0.05)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = torch.relu(self.bn7(self.conv7(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Function to apply random rotation with padding
def rotate_with_padding(image, angle):
    """ Rotate the input image by the given angle and apply necessary padding. """
    h, w = image.shape[-2], image.shape[-1]
    diagonal = int(np.ceil(np.sqrt(h**2 + w**2)))

    # Pad the image to diagonal size
    padding_top = (diagonal - h) // 2
    padding_left = (diagonal - w) // 2
    padding = (padding_left, padding_top, diagonal - w - padding_left, diagonal - h - padding_top)
    padded_image = nn.functional.pad(image, padding, mode='constant', value=0)

    # Rotate the image by the given angle
    rotated_image = TF.rotate(padded_image, angle)

    # Crop back to original size (center crop)
    start_x = (rotated_image.shape[-2] - h) // 2
    start_y = (rotated_image.shape[-1] - w) // 2
    cropped_image = rotated_image[..., start_x:start_x + h, start_y:start_y + w]
    
    return cropped_image

# Parameters
num_samples = 16000
dx = 0.1  # Fixed grid step size
dy = 0.1  # Fixed grid step size
x_max = 5  # Maximum physical extent for X
y_max = 5  # Maximum physical extent for Y
lambda_value = 0.1 # Anharmonic strength

# Generate harmonic data
train_potentials, train_eigenvalues = generate_potential_data(num_samples, dx, dy, x_max, y_max, potential_type="harmonic")

# Generate harmonic and potential well training data
num_harmonic_samples = 2*(num_samples // 3) 
num_well_samples = num_samples - num_harmonic_samples

harmonic_potentials, harmonic_eigenvalues = generate_potential_data(num_harmonic_samples, dx, dy, x_max, y_max, potential_type="harmonic")
well_potentials, well_eigenvalues = generate_potential_data(num_well_samples, dx, dy, x_max, y_max, potential_type="well")

# Combine harmonic and well potentials for training
train_potentials = torch.cat([harmonic_potentials, well_potentials], dim=0)
train_eigenvalues = torch.cat([harmonic_eigenvalues, well_eigenvalues], dim=0)

# Generate validation and test data (anharmonic)
num_val_test_samples = int(num_samples * 0.1)
val_potentials, val_eigenvalues = generate_potential_data(num_val_test_samples, dx, dy, x_max, y_max, potential_type="anharmonic", lambda_value=lambda_value)
test_potentials, test_eigenvalues = generate_potential_data(num_val_test_samples, dx, dy, x_max, y_max, potential_type="anharmonic", lambda_value=lambda_value)

# Prepare data
train_potentials = [pad_potential(p.numpy(), int(x_max / dx) + 1, int(y_max / dy) + 1) for p in train_potentials]
val_potentials = [pad_potential(p.numpy(), int(x_max / dx) + 1, int(y_max / dy) + 1) for p in val_potentials]
test_potentials = [pad_potential(p.numpy(), int(x_max / dx) + 1, int(y_max / dy) + 1) for p in test_potentials]

# Convert potentials to NumPy arrays first, then to a PyTorch tensor
train_potentials = torch.tensor(np.array(train_potentials)).float().unsqueeze(1)
val_potentials = torch.tensor(np.array(val_potentials)).float().unsqueeze(1)
test_potentials = torch.tensor(np.array(test_potentials)).float().unsqueeze(1)

# Convert eigenvalues to NumPy arrays
y_train = torch.tensor(np.array([e[:2] for e in train_eigenvalues])).float()
y_val = torch.tensor(np.array([e[:2] for e in val_eigenvalues])).float()
y_test = torch.tensor(np.array([e[:2] for e in test_eigenvalues])).float()

# Compute the maximum grid size from the maximum extents
max_x = int(x_max / dx) + 1
max_y = int(y_max / dy) + 1

# Move data to GPU if available
train_potentials = train_potentials.to(device)
val_potentials = val_potentials.to(device)
test_potentials = test_potentials.to(device)

y_train = y_train.to(device)
y_val = y_val.to(device)
y_test = y_test.to(device)

# Create PyTorch Datasets and DataLoaders
train_dataset = TensorDataset(train_potentials, y_train)
val_dataset = TensorDataset(val_potentials, y_val)
test_dataset = TensorDataset(test_potentials, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = CNNModel(input_shape=(max_x, max_y)).to(device)

# Define loss function and optimizer with L2 regularization
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.000937238199960265, weight_decay=0.009)  # L2 regularization with weight decay

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training the model
accumulation_steps = 2  # Accumulate gradients over 4 batches before updating weights
epochs = 2000
train_losses, val_losses = [], []
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for i, (X_batch, y_batch) in enumerate(train_loader):
        angle = np.random.uniform(-180, 180)
        X_batch_rotated = rotate_with_padding(X_batch, angle)
        y_pred = model(X_batch_rotated)

        loss = criterion(y_pred, y_batch) / accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulation_steps

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(val_losses, label='Validation Loss')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.title('Loss vs. Epoch')
plt.show()

# Test the model
model.eval()
test_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.6f}")

# Relative error calculation
relative_errors = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        error = torch.abs(y_pred - y_batch) / torch.abs(y_batch)
        relative_errors.append(error)

relative_errors = torch.cat(relative_errors)
mean_relative_error = torch.mean(relative_errors).item()
mean_relative_error_percentage = mean_relative_error * 100
print(f"Mean Relative Error: {mean_relative_error_percentage:.6f}%")

# Evaluate the model on anharmonic test data and compute energy difference errors
energy_diff_errors_anharmonic = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)

        # Calculate true and predicted energy differences for anharmonic potential
        true_energy_diff = y_batch[:, 1] - y_batch[:, 0]
        pred_energy_diff = y_pred[:, 1] - y_pred[:, 0]

        # Calculate the absolute error in the energy differences
        energy_diff_error = torch.abs(true_energy_diff - pred_energy_diff)
        energy_diff_errors_anharmonic.append(energy_diff_error)

energy_diff_errors_anharmonic = torch.cat(energy_diff_errors_anharmonic)
mean_energy_diff_error_anharmonic = torch.mean(energy_diff_errors_anharmonic).item()
print(f"Mean Absolute Error in Energy Difference for Anharmonic Potentials (E1 - E0): {mean_energy_diff_error_anharmonic:.6f}")