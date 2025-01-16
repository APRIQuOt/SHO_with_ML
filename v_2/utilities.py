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

from structs import TrainInput, TrainResult, ValidationInput, ValidationResult




def train(train_input: TrainInput, *, accelerator: Optional[accelerate.Accelerator] = None) -> TrainResult:
  train_potentials = train_input.train_potentials
  train_eigen_vals = train_input.train_eigen_vals
  num_epochs = train_input.num_epochs
  optimizer = train_input.optimizer
  model = train_input.model
  device = train_input.device
  loss_fn = train_input.loss_fn

  model.train()

  num_epochs = 500

  start = time.time()
  losses = []
  running_loss = 0.0

  with tqdm(total=num_epochs, desc="Training", unit="epoch", unit_scale=True) as pbar:
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        predicted_eigen_vals = model(train_potentials.unsqueeze(1))

        # Extract the eigenvalues by averaging over the spatial dimensions
        predicted_eigen_vals = predicted_eigen_vals.mean(dim=[2, 3])  # Shape: [500, 25]

        # Compute the loss
        # train_eigen_vals = torch.tensor(train_eigen_vals, dtype=torch.float32).to(device)
        loss = loss_fn(predicted_eigen_vals, train_eigen_vals)
        # compute relative error
        # loss = loss -  torch.mean(train_eigen_vals)
        losses.append(loss)
        running_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients

        if accelerator is not None:
            accelerator.backward(loss)        # Compute gradients
        else:
           loss.backward()
        
        optimizer.step()       # Update model parameters

        # Update tqdm display
        avg_loss = running_loss / (epoch + 1)  # Average loss up to current epoch
        pbar.set_postfix({
            "loss": loss.item(),
            "avg_loss": avg_loss,
            "time_elapsed": f"{time.time() - start:.2f}s"
        }, refresh=True)
        pbar.update(1)

  duration = time.time() - start
  return TrainResult(losses=losses, model=model, train_duration=duration)



def validate_model(val_input: ValidationInput) -> ValidationResult:
  val_potentials = val_input.val_potentials
  val_eigen_vals = val_input.val_eigen_vals
  model = val_input.model
  loss_fn = val_input.loss_fn

  # Validation Phase
  model.eval()  # Set the model to evaluation mode

  # Forward pass: compute predicted eigenvalues for validation data
  with torch.no_grad():  # Disable gradient calculation
      predicted_val_eigen_vals = model(val_potentials.unsqueeze(1))
      predicted_val_eigen_vals = predicted_val_eigen_vals.mean(dim=[2, 3])  # Shape: [10, 25]

  # Compute validation loss
  val_loss = loss_fn(predicted_val_eigen_vals, val_eigen_vals)

  # Print validation loss
  print(f'Validation Loss: {val_loss.item():.6f}')

  return ValidationResult(
      val_eigen_vals=val_eigen_vals,
      val_loss=val_loss.item(),
      predicted_val_eigen_vals=predicted_val_eigen_vals,
  )