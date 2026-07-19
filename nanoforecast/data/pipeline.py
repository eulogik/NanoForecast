import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Tuple, Optional


class TimeSeriesDataset(Dataset):
    """Dataset wrapping time series records with Reverso-style augmentation
    and multi-horizon slicing support.
    """
    def __init__(
        self,
        records: List[Dict],
        augment: bool = False,
        augment_prob: float = 0.5,
        multi_horizon: bool = False,
        min_horizon: int = 12,
    ):
        self.records = records
        self.augment = augment
        self.augment_prob = augment_prob
        self.multi_horizon = multi_horizon
        self.min_horizon = min_horizon

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        ctx = np.array(rec["context"], dtype=np.float32)
        pred = np.array(rec["prediction"], dtype=np.float32)
        full = np.concatenate([ctx, pred])

        if self.multi_horizon and len(full) > len(ctx) + self.min_horizon:
            max_h = len(pred)
            h = np.random.randint(self.min_horizon, max_h + 1)
            ctx_out = full[:len(ctx)]
            y_out = full[len(ctx):len(ctx) + h]
        else:
            ctx_out = ctx
            y_out = pred
            h = len(pred)

        if self.augment and np.random.rand() < self.augment_prob:
            ctx_out, y_out = self._reverso_augment(ctx_out, y_out)

        x = torch.tensor(ctx_out, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y_out, dtype=torch.float32).unsqueeze(0)
        freq_id = torch.tensor(rec["freq_id"], dtype=torch.long)
        covariates = torch.tensor(
            rec["context_covariates"], dtype=torch.float32
        )

        return {
            "x": x,
            "y": y,
            "freq_id": freq_id,
            "covariates": covariates,
            "horizon": h,
        }

    def _reverso_augment(
        self, ctx: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reverso/TSAug-style augmentation: jitter, scale, shift, mask, reverse."""
        if np.random.rand() > 0.3:
            std = max(np.std(ctx), 1e-6)
            ctx = ctx + np.random.randn(*ctx.shape).astype(np.float32) * std * 0.05
            y = y + np.random.randn(*y.shape).astype(np.float32) * std * 0.05

        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            ctx = ctx * scale
            y = y * scale

        if np.random.rand() > 0.5:
            shift = np.random.uniform(-1.0, 1.0)
            ctx = ctx + shift
            y = y + shift

        if np.random.rand() > 0.7:
            L = len(ctx)
            mask_len = int(np.random.uniform(0.05, 0.15) * L)
            start = np.random.randint(0, L - mask_len)
            ctx = ctx.copy()
            ctx[start:start + mask_len] = 0.0

        if np.random.rand() > 0.8:
            ctx = ctx[::-1].copy()

        return ctx, y


class ResolutionBatchSampler(Sampler):
    """Resolution-Aware Batch Sampler. Groups indices by frequency ID."""
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
    multi_horizon: bool = False,
    min_horizon: int = 12,
) -> torch.utils.data.DataLoader:
    """Helper function to wrap dataset in a DataLoader."""
    dataset = TimeSeriesDataset(
        records,
        augment=augment,
        multi_horizon=multi_horizon,
        min_horizon=min_horizon,
    )
    freq_ids = [rec["freq_id"] for rec in records]

    sampler = ResolutionBatchSampler(
        freq_ids,
        batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        min_batch_size=min_batch_size,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    return loader
