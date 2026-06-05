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
    freq_ids = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2] # 4 of 0, 4 of 1, 4 of 2
    batch_size = 2
    
    sampler = ResolutionBatchSampler(freq_ids, batch_size=batch_size, shuffle=False)
    
    batches = list(sampler)
    
    # We should have 6 batches
    assert len(batches) == 6
    
    # Check that each batch has length batch_size and all elements share same frequency
    for batch in batches:
        assert len(batch) == batch_size
        idx1, idx2 = batch[0], batch[1]
        assert freq_ids[idx1] == freq_ids[idx2]


def test_dataloader_integration():
    generator = SyntheticTimeSeriesGenerator(seed=202)
    records = generator.generate_dataset(num_series=20, context_len=64, prediction_len=16)
    
    loader = create_dataloader(records, batch_size=4, shuffle=False)
    
    # Check that each batch has the correct shapes
    for batch in loader:
        x = batch["x"]
        y = batch["y"]
        freq_ids = batch["freq_id"]
        covariates = batch["covariates"]
        
        assert x.shape == (4, 1, 64)
        assert y.shape == (4, 1, 16)
        assert freq_ids.shape == (4,)
        assert covariates.shape == (4, 4, 64)
        
        # Verify resolution-aware batching: all freq_ids in the batch must be identical
        first_freq = freq_ids[0].item()
        assert torch.all(freq_ids == first_freq).item()
