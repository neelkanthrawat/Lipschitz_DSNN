### contains code to generate data points from different 1D distributions# Function to 

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# list of functions
# 1. create_dataloaders
# 2. geenrate gaussian data
# 3. generate laplace data



# Function to convert datasets into DataLoaders
def create_dataloaders(train_data, val_data, test_data, batch_size=32):
    """
    Convert numpy arrays into PyTorch DataLoaders.

    Parameters:
    - train_data: Training dataset (numpy array)
    - val_data: Validation dataset (numpy array)
    - test_data: Test dataset (numpy array)
    - batch_size: Batch size for the DataLoaders

    Returns:
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - test_loader: DataLoader for test data
    """
    # Convert numpy arrays to PyTorch tensors
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    val_tensor = torch.tensor(val_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    test_dataset = TensorDataset(test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# 1. generate data from a 1D Gaussian distribution
def generate_gaussian_data(mean, 
            std_dev, total_samples, train_ratio=0.6, val_ratio=0.2):
    """
    Generate training, validation, and test datasets from a 1D Gaussian distribution.

    Parameters:
    - mean: Mean of the Gaussian distribution
    - std_dev: Standard deviation of the Gaussian distribution
    - total_samples: Total number of samples to generate
    - train_ratio: Proportion of data to use for training (default is 70%)
    - val_ratio: Proportion of data to use for validation (default is 15%)

    Returns:
    - train_data: Training set
    - val_data: Validation set
    - test_data: Test set
    """
    # Generate data
    data = np.random.normal(loc=mean, scale=std_dev, size=total_samples)

    # Shuffle data
    np.random.shuffle(data)

    # Split data based on ratios
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data

# 2.Function to generate data from a 1D Laplace distribution
def generate_laplace_data(mean, scale, total_samples, num_test_data=15000, train_ratio=0.75, val_ratio=0.25):
    """
    Generate training, validation, and test datasets from a 1D Laplace distribution.

    Parameters:
    - mean: Mean (location parameter) of the Laplace distribution
    - scale: Scale (diversity) parameter of the Laplace distribution
    - total_samples: Total number of samples (excluding test data)
    - num_test_data: Number of samples to reserve for the test dataset
    - train_ratio: Proportion of remaining data to use for training (default is 70%)
    - val_ratio: Proportion of remaining data to use for validation (default is 30%)

    Returns:
    - train_data: Training set
    - val_data: Validation set
    - test_data: Test set
    """
    # Generate full dataset including test data
    total_data = np.random.laplace(loc=mean, scale=scale, size=total_samples + num_test_data)

    # Shuffle data
    np.random.shuffle(total_data)

    # Separate test data
    test_data = total_data[:num_test_data]
    remaining_data = total_data[num_test_data:]

    # Split remaining data into train and validation sets
    train_end = int(len(remaining_data) * train_ratio)
    
    train_data = remaining_data[:train_end]
    val_data = remaining_data[train_end:]

    return train_data, val_data, test_data

