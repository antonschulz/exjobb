import torch

# Import required modules and functions
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .util import read_dataset, compute_and_scale_deltas
import random

# Custom Dataset with optional transform.
class InsectDataset(Dataset):
    def __init__(self, tracks_dict, transform=None):
        self.tracks_dict = tracks_dict
        self.label_encoder = LabelEncoder()
        self.labels = []
        self.data = []
        self.transform = transform
        
        # Encode labels and prepare data
        for track_name, track_info in tracks_dict.items():
            df = track_info['dataframe']
            label = track_info['label']
            self.data.append(df.values)  # NumPy array of shape (seq_len, 2)
            self.labels.append(label)
        
        # Encode labels to integers
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert the stored array into a tensor.
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        # Apply augmentation if a transform is provided.
        if self.transform:
            x = self.transform(x)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

import math

def augment_sequence(seq, noise_std=0.01, prob_rotate=0.5, max_rotate_angle=math.pi/2, translation_range=0.1):
    """
    Augments a sequence of (x, y) coordinates (each in [0,1]) by:
      - With probability `prob_rotate`, rotating the entire sequence by a random angle.
      - Applying a random translation in both x and y.
      - Adding Gaussian noise.

    Args:
        seq (torch.Tensor): Tensor of shape (seq_len, 2) containing (x, y) coordinates.
        noise_std (float): Standard deviation of Gaussian noise to add.
        prob_rotate (float): Probability of applying a random rotation.
        max_rotate_angle (float): Maximum rotation angle in radians (random angle ∈ [-max, max]).
        translation_range (float): Maximum absolute translation offset for both x and y.

    Returns:
        torch.Tensor: The augmented sequence with values clamped to [0, 1].
    """
    seq_aug = seq.clone()
    seq_len = seq_aug.size(0)

    # Random rotation
    if random.random() < prob_rotate:
        angle = random.uniform(-max_rotate_angle, max_rotate_angle)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Rotate around the sequence center (0.5, 0.5)
        center = torch.tensor([0.5, 0.5], device=seq_aug.device)
        seq_centered = seq_aug - center
        rot_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=seq_aug.device)
        seq_rot = torch.matmul(seq_centered, rot_matrix.T) + center
        seq_aug = seq_rot

    # Apply a random translation in both x and y directions.
    shift_x = random.uniform(-translation_range, translation_range)
    shift_y = random.uniform(-translation_range, translation_range)
    seq_aug[:, 0] += shift_x
    seq_aug[:, 1] += shift_y

    # Add small Gaussian noise to both dimensions.
    noise = torch.randn_like(seq_aug) * noise_std
    seq_aug += noise

    # Clamp values to ensure they remain within [0, 1]
    seq_aug = seq_aug.clamp(0, 1)

    return seq_aug

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0) 
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, labels, lengths

def load_dataset(data_dir: str, diff: bool=False, scaler: str=None, augment: bool=True) -> tuple[InsectDataset, InsectDataset, InsectDataset, InsectDataset]:
    """
    Loads dataset in path data_dir. Assumes train/test folder exists in said directory.
    Defaults to 0.8/0.2 train/val split and 0.8/0.2 train,val/test split.
    """
    train_path, test_path, val_path = f'{data_dir}/train', f'{data_dir}/test', f'{data_dir}/val'
    
    train_dict = read_dataset(train_path, interpolate=True, normalize=True)
    val_dict = read_dataset(val_path, interpolate=True, normalize=True)
    test_dict = read_dataset(test_path, interpolate=True, normalize=True)


    if diff:
        scaler = compute_and_scale_deltas(train_dict, scaler_str=scaler)
        _ = compute_and_scale_deltas(val_dict, scaler=scaler)
        _ = compute_and_scale_deltas(test_dict, scaler=scaler)

    # Create training dataset with augmentation.
    if augment:
        train_dataset = InsectDataset(train_dict, transform=augment_sequence)
    else:
        train_dataset = InsectDataset(train_dict, transform=None)

    # Create test dataset without augmentation.
    val_dataset = InsectDataset(val_dict, transform=None)
    test_dataset = InsectDataset(test_dict, transform=None)
    # also create train set of the entire set
    entire_training_dict = train_dict | val_dict
    full_training_dataset = InsectDataset(entire_training_dict, transform=augment_sequence)
    
    return train_dataset, val_dataset, test_dataset, full_training_dataset




