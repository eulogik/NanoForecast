import numpy as np
from typing import Dict, Tuple, List

class SyntheticTimeSeriesGenerator:
    """
    High-speed, vectorized generator for realistic synthetic time series.
    Generates various patterns: linear/exponential trends, multi-seasonal cycles,
    outlier spikes, level shifts, and non-Gaussian noise.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_single_series(
        self, 
        length: int, 
        freq_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a single synthetic time series vector along with a covariate matrix.
        Args:
            length: Number of time steps to generate
            freq_id: Integer mapping to a specific frequency (0-9)
        Returns:
            series: Array of shape [length]
            covariates: Array of shape [cov_dim, length] (numeric indicators like holiday, time-of-day)
        """
        t = np.arange(length)
        
        # 1. Base Trend
        trend_type = self.rng.choice(["linear", "exponential", "piecewise", "flat"])
        if trend_type == "linear":
            slope = self.rng.uniform(-0.05, 0.05)
            trend = slope * t
        elif trend_type == "exponential":
            growth = self.rng.uniform(0.001, 0.005)
            trend = 0.1 * np.exp(growth * t)
        elif trend_type == "piecewise":
            changepoint = length // 2
            slope1 = self.rng.uniform(-0.05, 0.05)
            slope2 = self.rng.uniform(-0.05, 0.05)
            trend = np.zeros(length)
            trend[:changepoint] = slope1 * t[:changepoint]
            trend[changepoint:] = trend[changepoint-1] + slope2 * (t[changepoint:] - changepoint)
        else:
            trend = np.zeros(length)
            
        # Add random base level shift
        trend += self.rng.uniform(-10.0, 10.0)

        # 2. Seasonality (Frequency-dependent)
        season = np.zeros(length)
        if freq_id == 0:  # 5-minutely (period e.g., 288 steps/day)
            periods = [288]
        elif freq_id == 1:  # Hourly (period 24 steps/day, 168 steps/week)
            periods = [24, 168]
        elif freq_id == 2:  # Daily (period 7 steps/week)
            periods = [7]
        elif freq_id == 3:  # Weekly (period 52 steps/year)
            periods = [52]
        elif freq_id == 4:  # Monthly (period 12 steps/year)
            periods = [12]
        else:  # Other/Unknown
            periods = [self.rng.choice([10, 20, 30])]

        for period in periods:
            amplitude = self.rng.uniform(0.5, 3.0)
            phase = self.rng.uniform(0, 2 * np.pi)
            season += amplitude * np.sin(2 * np.pi * t / period + phase)
            
            # Add harmonic
            if self.rng.random() > 0.5:
                season += (amplitude * 0.3) * np.sin(4 * np.pi * t / period + phase)

        # 3. Level Shifts / Step Changes
        shift = np.zeros(length)
        if self.rng.random() > 0.7:
            shift_idx = self.rng.integers(int(length * 0.2), int(length * 0.8))
            shift_magnitude = self.rng.uniform(-5.0, 5.0)
            shift[shift_idx:] = shift_magnitude

        # 4. Combine Signal (Deterministic)
        signal = trend + season + shift

        # 5. Outliers / Anomalies
        outliers = np.zeros(length)
        if self.rng.random() > 0.8:
            num_outliers = self.rng.integers(1, 4)
            for _ in range(num_outliers):
                idx = self.rng.integers(0, length)
                outliers[idx] = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(5.0, 15.0)

        # 6. Noise
        noise_type = self.rng.choice(["gaussian", "student-t", "heteroscedastic"])
        if noise_type == "gaussian":
            noise = self.rng.normal(0, 0.5, size=length)
        elif noise_type == "student-t":
            noise = self.rng.standard_t(df=3, size=length) * 0.3
        else:  # Variance increases over time
            scale = 0.1 + 0.9 * (t / length)
            noise = self.rng.normal(0, scale)

        # Final Time Series
        series = signal + outliers + noise
        
        # Scale to random physical range
        series_scale = np.exp(self.rng.uniform(-2.0, 5.0))
        series = series * series_scale
        
        # 7. Covariates (Generate 4 channels of exogenous signals)
        # Cov 0: Sine wave capturing cyclic time-of-day/week
        # Cov 1: Binary holiday/promotion indicator (random spikes)
        # Cov 2: Binary weekend indicator
        # Cov 3: Linear step indicator
        covariates = np.zeros((4, length))
        
        # Periodic covariate
        covariates[0, :] = np.sin(2 * np.pi * t / 24)
        
        # Binary event indicator (promotions/holidays)
        event_indices = self.rng.choice(length, size=int(length * 0.05), replace=False)
        covariates[1, event_indices] = 1.0
        
        # Weekend indicator (for daily-ish series)
        if freq_id in [1, 2]:
            weekend_mask = (t % 7 >= 5).astype(float)
            covariates[2, :] = weekend_mask
            
        # Generic trend covariate
        covariates[3, :] = t / length
        
        return series.astype(np.float32), covariates.astype(np.float32)

    def generate_dataset(
        self, 
        num_series: int, 
        context_len: int, 
        prediction_len: int
    ) -> List[Dict]:
        """
        Generates a list of dictionaries representing a dataset.
        """
        dataset = []
        total_len = context_len + prediction_len
        
        for i in range(num_series):
            # Select random frequency
            freq_id = int(self.rng.integers(0, 5))
            series, covariates = self.generate_single_series(total_len, freq_id)
            
            context_target = series[:context_len]
            prediction_target = series[context_len:]
            
            context_covariates = covariates[:, :context_len]
            prediction_covariates = covariates[:, context_len:]
            
            dataset.append({
                "context": context_target,
                "prediction": prediction_target,
                "freq_id": freq_id,
                "context_covariates": context_covariates,
                "prediction_covariates": prediction_covariates
            })
            
        return dataset
