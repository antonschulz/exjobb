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
    def __init__(self, tracks_dict, transform=None, p_splice=0.1, min_len=5):
        self.tracks_dict = tracks_dict
        self.label_encoder = LabelEncoder()
        self.labels = []
        self.data = []
        self.p_splice   = p_splice
        self.min_len    = min_len
        self.transform = transform
        
        # Encode labels and prepare data
        for track_name, track_info in tracks_dict.items():
            df = track_info['dataframe']
            label = track_info['label']
            self.data.append(df.values)  # NumPy array of shape (seq_len, 2)
            self.labels.append(label)
        
        # Encode labels to integers
        self.labels = self.label_encoder.fit_transform(self.labels)

        # Build a map from label → list of indices for fast same‐label sampling
        if self.transform:
            self.indices_by_label = self._build_index_map()

    def _build_index_map(self):
        m = {}
        for idx, lbl in enumerate(self.labels):
            m.setdefault(lbl, []).append(idx)
        return m

    def _load_delta(self, idx: int) -> torch.Tensor:
        """
        Load the Δ‐sequence at index idx, return a torch.Tensor of shape (T,2).
        Converts from NumPy if needed.
        """
        seq = self.data[idx]
        if not isinstance(seq, torch.Tensor):
            seq = torch.as_tensor(seq, dtype=torch.float32)
        return seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert the stored array into a tensor.
        x = self._load_delta(idx)
        # Apply augmentation if a transform is provided.
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            # with prob p_splice, pick another same-label seq and splice
            if random.random() < self.p_splice:
                same_lbl_idxs = self.indices_by_label[self.labels[idx]]
                # avoid picking itself if only one sample per label
                if len(same_lbl_idxs) > 1:
                    j = idx
                    while j == idx:
                        j = random.choice(same_lbl_idxs)
                    delta2 = self._load_delta(j)
                    x = splice_deltas(x, delta2, min_len=self.min_len)
            x = self.transform(x)
        return x, y

import math

def splice_deltas(delta1: torch.Tensor,
                  delta2: torch.Tensor,
                  min_len: int = 5) -> torch.Tensor:
    """
    Splice two delta‐sequences by cutting each in [min_len, T-min_len]
    and concatenating the first part of the first to the second part of the second.
    If either sequence is shorter than 2*min_len, just concatenate them whole.
    """
    T1, T2 = delta1.size(0), delta2.size(0)

    # Only cut if each sequence is at least 2*min_len long
    if T1 < 2 * min_len or T2 < 2 * min_len:
        return torch.cat([delta1, delta2], dim=0)

    # safe to pick cut points in [min_len, T - min_len]
    cut1 = random.randint(min_len, T1 - min_len)
    cut2 = random.randint(min_len, T2 - min_len)
    return torch.cat([delta1[:cut1], delta2[cut2:]], dim=0)

def augment_sequence(seq, noise_std=0.01, prob_rotate=0.5, max_rotate_angle=math.pi):
    """
    Treat `seq` as a sequence of standardized (Δx, Δy) pairs.
    Randomly rotates around (0,0) and adds Gaussian noise.
    """
    # ensure torch.Tensor
    if not isinstance(seq, torch.Tensor):
        seq = torch.as_tensor(seq, dtype=torch.float32)

    seq_aug = seq.clone()

    # random rotation
    if random.random() < prob_rotate:
        angle = random.uniform(-max_rotate_angle, max_rotate_angle)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = torch.tensor([[cos_a, -sin_a],
                          [ sin_a,  cos_a]],
                         dtype=seq_aug.dtype,
                         device=seq_aug.device)
        seq_aug = seq_aug @ R.T

    # add noise
    if noise_std > 0:
        seq_aug = seq_aug + torch.randn_like(seq_aug) * noise_std

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
    if diff == True:
        normalize = False
    else:
        normalize=True
    train_dict = read_dataset(train_path, interpolate=True, normalize=normalize)
    val_dict = read_dataset(val_path, interpolate=True, normalize=normalize)
    test_dict = read_dataset(test_path, interpolate=True, normalize=normalize)


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




