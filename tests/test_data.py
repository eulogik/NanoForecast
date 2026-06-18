import numpy as np
import torch

from nanoforecast.data.generator import SyntheticTimeSeriesGenerator
from nanoforecast.data.pipeline import TimeSeriesDataset, ResolutionBatchSampler, create_dataloader

def test_synthetic_generator():
    generator = SyntheticTimeSeriesGenerator(seed=42)
    length = 100
    
    for freq_id in range(5):
        series, covariates = generator.generate_single_series(length, freq_id)
        assert series.shape == (length,)
        assert covariates.shape == (4, length)
        assert not np.isnan(series).any()
        assert not np.isnan(covariates).any()


def test_dataset_item_and_augmentation():
    generator = SyntheticTimeSeriesGenerator(seed=101)
    records = generator.generate_dataset(num_series=10, context_len=64, prediction_len=16)
    
    dataset = TimeSeriesDataset(records, augment=True, augment_prob=1.0)
    
    assert len(dataset) == 10
    item = dataset[0]
    
    assert "x" in item
    assert "y" in item
    assert "freq_id" in item
    assert "covariates" in item
    
    assert item["x"].shape == (1, 64)
    assert item["y"].shape == (1, 16)
    assert item["covariates"].shape == (4, 64)


def test_resolution_batch_sampler():
    freq_ids = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]  # 4 of 0, 4 of 1, 4 of 2
    batch_size = 2

    # default: keep remainder batches (no silent drop)
    sampler = ResolutionBatchSampler(freq_ids, batch_size=batch_size, shuffle=False)
    batches = list(sampler)
    assert len(batches) == 6
    for batch in batches:
        assert len(batch) == batch_size
        assert freq_ids[batch[0]] == freq_ids[batch[1]]

    # drop_last=True yields only full batches
    sampler_drop = ResolutionBatchSampler(freq_ids, batch_size=batch_size, shuffle=False, drop_last=True)
    assert len(list(sampler_drop)) == 6

    # With uneven split and drop_last=False, remainder batches are emitted
    uneven = [0, 0, 0, 1, 1]  # freq 0 has 3 (one partial), freq 1 has 2
    sampler_uneven = ResolutionBatchSampler(uneven, batch_size=2, shuffle=False, drop_last=False)
    batches = list(sampler_uneven)
    assert len(batches) == 3  # [0,1], [2], [3,4]
    sizes = sorted(len(b) for b in batches)
    assert sizes == [1, 2, 2]

    # With drop_last=True, the partial [2] is dropped
    sampler_drop_uneven = ResolutionBatchSampler(uneven, batch_size=2, shuffle=False, drop_last=True)
    assert len(list(sampler_drop_uneven)) == 2


def test_dataloader_integration():
    generator = SyntheticTimeSeriesGenerator(seed=202)
    records = generator.generate_dataset(num_series=20, context_len=64, prediction_len=16)

    # Use drop_last=True so every batch is full and easy to assert on shape
    loader = create_dataloader(records, batch_size=4, shuffle=False, drop_last=True)

    for batch in loader:
        x = batch["x"]
        y = batch["y"]
        freq_ids = batch["freq_id"]
        covariates = batch["covariates"]

        assert x.shape == (4, 1, 64)
        assert y.shape == (4, 1, 16)
        assert freq_ids.shape == (4,)
        assert covariates.shape == (4, 4, 64)

        first_freq = freq_ids[0].item()
        assert torch.all(freq_ids == first_freq).item()
