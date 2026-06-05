from dataclasses import dataclass, field
from typing import List

@dataclass
class NanoForecastConfig:
    # Context and prediction windows
    context_length: int = 512
    prediction_length: int = 96
    
    # Architecture dimensions
    d_model: int = 32
    num_layers: int = 4
    patch_size: int = 8
    dropout: float = 0.1
    expansion_factor: int = 2
    
    # Model capabilities
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    num_frequencies: int = 10  # Learned embeddings for hourly, daily, weekly, etc.
    num_channels: int = 16      # Maximum multivariate channels supported natively
    covariate_dim: int = 4     # Dimensionality of exogenous covariates
    
    # Gating and features
    use_gated_router: bool = True
    
    @classmethod
    def nano_200k(cls) -> "NanoForecastConfig":
        return cls(
            d_model=32,
            num_layers=4,
            patch_size=8,
            dropout=0.1,
            expansion_factor=2
        )
        
    @classmethod
    def nano_500k(cls) -> "NanoForecastConfig":
        return cls(
            d_model=64,
            num_layers=8,
            patch_size=8,
            dropout=0.1,
            expansion_factor=2
        )
