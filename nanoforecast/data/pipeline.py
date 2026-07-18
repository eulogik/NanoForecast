import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Tuple, Optional

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset wrapping time series records.
    Provides option for real-time data augmentations.
    """
    def __init__(
        self, 
        records: List[Dict], 
        augment: bool = False,
        augment_prob: float = 0.5
    ):
        self.records = records
        self.augment = augment
        self.augment_prob = augment_prob

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        
        # Extract features
        # Add channel dimension: [C, L] -> here C=1 (univariate targets)
        x = torch.tensor(rec["context"], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(rec["prediction"], dtype=torch.float32).unsqueeze(0)
        
        freq_id = torch.tensor(rec["freq_id"], dtype=torch.long)
        
        # Handle covariates
        covariates = torch.tensor(rec["context_covariates"], dtype=torch.float32)
        
        # Apply data augmentation if requested
        if self.augment and np.random.rand() < self.augment_prob:
            x, covariates = self._apply_augmentations(x, covariates)
            
        return {
            "x": x,
            "y": y,
            "freq_id": freq_id,
            "covariates": covariates
        }

    def _apply_augmentations(
        self, 
        x: torch.Tensor, 
        covariates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies random scale adjustments, shifting, and jitter noise.
        """
        # Random scale multiplier: multiply target by [0.5, 2.0]
        scale = np.random.uniform(0.5, 2.0)
        x = x * scale
        
        # Random shifting offset: add constant offset to target
        shift = np.random.uniform(-2.0, 2.0)
        x = x + shift
        
        # Add random noise jitter
        if np.random.rand() > 0.5:
            noise = torch.randn_like(x) * 0.05
            x = x + noise
            
        # Randomly mask 5-15% of values (simulate missing values)
        if np.random.rand() > 0.7:
            seq_len = x.shape[-1]
            mask_len = int(np.random.uniform(0.05, 0.15) * seq_len)
            mask_start = np.random.randint(0, seq_len - mask_len)
            x[..., mask_start:mask_start+mask_len] = 0.0
            
        return x, covariates


class ResolutionBatchSampler(Sampler):
    """
    Resolution-Aware Batch Sampler.
    Groups indices by their frequency ID and yields batches containing
    series of ONLY one frequency. This allows the model to learn frequency-specific
    priors cleanly within a batch.

    By default `drop_last=False` and `min_batch_size=1`, so partial remainder
    batches are emitted rather than dropped silently. Set `drop_last=True` to
    recover the original strict behaviour.
    """
    def __init__(
        self,
        freq_ids: List[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        min_batch_size: int = 1,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.min_batch_size = max(1, min_batch_size)

        self.freq_to_indices = {}
        for idx, freq_id in enumerate(freq_ids):
            self.freq_to_indices.setdefault(freq_id, []).append(idx)

    def __iter__(self):
        batches = []
        for indices in self.freq_to_indices.values():
            indices_copy = list(indices)
            if self.shuffle:
                np.random.shuffle(indices_copy)

            for i in range(0, len(indices_copy), self.batch_size):
                batch = indices_copy[i:i + self.batch_size]
                if len(batch) < self.min_batch_size:
                    continue
                if self.drop_last and len(batch) != self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)

        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for indices in self.freq_to_indices.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                # ceil(n / batch_size) with min_batch_size filtering
                full, rem = divmod(n, self.batch_size)
                total += full + (1 if rem >= self.min_batch_size else 0)
        return total


def create_dataloader(
    records: List[Dict],
    batch_size: int,
    augment: bool = False,
    shuffle: bool = True,
    drop_last: bool = False,
    min_batch_size: int = 1,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Helper function to wrap dataset in a DataLoader using ResolutionBatchSampler.
    """
    dataset = TimeSeriesDataset(records, augment=augment)
    freq_ids = [rec["freq_id"] for rec in records]

    sampler = ResolutionBatchSampler(
        freq_ids,
        batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        min_batch_size=min_batch_size,
    )

    # pin_memory only helps when a GPU is present; it's a no-op (and prints a
    # warning) on CPU. Parallel num_workers overlap data prep with compute so
    # the training threads are not starved waiting on the loader.
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    return loader
