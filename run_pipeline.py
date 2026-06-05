import os
import torch
import numpy as np

from nanoforecast.config import NanoForecastConfig
from nanoforecast.model.core import NanoForecast
from nanoforecast.data.generator import SyntheticTimeSeriesGenerator
from nanoforecast.data.pipeline import create_dataloader
from nanoforecast.train.loss import MultiTaskLoss
from nanoforecast.train.trainer import NanoForecastTrainer
from nanoforecast.evaluation.benchmark import TimeSeriesEvaluator
from nanoforecast.export.onnx_export import export_to_onnx

def main():
    print("=" * 60)
    print("NANOFORECAST PIPELINE: SYNTHETIC PRETRAINING & BENCHMARKING")
    print("=" * 60)
    
    # 1. Configuration Setup (Nano-200K Profile)
    config = NanoForecastConfig(
        context_length=256,       # Shorter context for faster dry-run execution
        prediction_length=48,     # Horizon steps
        d_model=32,               # Hidden dim
        num_layers=4,             # Depth
        patch_size=8,             # Patch size
        covariate_dim=4           # 4 exogenous covariates
    )
    print(f"Loaded Configuration: {config}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 2. Generate Synthetic Dataset
    print("\n[Step 1/5] Generating synthetic time series data...")
    generator = SyntheticTimeSeriesGenerator(seed=42)
    
    # Generate 400 series for training, 100 for validation
    train_records = generator.generate_dataset(num_series=400, context_len=256, prediction_len=48)
    val_records = generator.generate_dataset(num_series=100, context_len=256, prediction_len=48)
    
    # Build data loaders using resolution-aware batch sampler
    train_loader = create_dataloader(train_records, batch_size=16, augment=True, shuffle=True)
    val_loader = create_dataloader(val_records, batch_size=16, augment=False, shuffle=False)
    print(f"--> Train batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
    
    # 3. Model & Loss Initialization
    print("\n[Step 2/5] Initializing NanoForecast model & MultiTask Loss...")
    model = NanoForecast(config)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--> Total Trainable Parameters: {total_params / 1e3:.2f}K parameters")
    
    loss_fn = MultiTaskLoss(quantiles=config.quantiles)
    
    # 4. Training Loop
    print("\n[Step 3/5] Starting model training (2 epochs for validation)...")
    trainer = NanoForecastTrainer(
        model=model,
        loss_fn=loss_fn,
        lr=1e-3,
        checkpoint_dir="checkpoints"
    )
    
    trainer.fit(train_loader, val_loader, epochs=2)
    
    # 5. Model Evaluation
    print("\n[Step 4/5] Evaluating best model checkpoint against validation set...")
    # Load best model checkpoint
    checkpoint_path = "checkpoints/best_model.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"--> Successfully loaded best checkpoint from epoch {checkpoint['epoch']}")
        
    model.eval()
    evaluator = TimeSeriesEvaluator()
    
    contexts = []
    targets = []
    forecasts = []
    quantiles_list = []
    
    # Predict on validation data to compute exact benchmarks
    with torch.no_grad():
        for record in val_records:
            # Prepare tensor shape: [B=1, C=1, L]
            x_tensor = torch.tensor(record["context"], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(trainer.device)
            freq_id_tensor = torch.tensor([record["freq_id"]], dtype=torch.long).to(trainer.device)
            cov_tensor = torch.tensor(record["context_covariates"], dtype=torch.float32).unsqueeze(0).to(trainer.device)
            
            outputs = model(x_tensor, freq_id_tensor, cov_tensor)
            
            # Squeeze batch & channel dimensions: [prediction_length]
            forecast = outputs["forecast"].squeeze(0).squeeze(0).cpu().numpy()
            # Shape: [num_quantiles, prediction_length]
            quantiles = outputs["quantiles"].squeeze(0).squeeze(0).cpu().numpy()
            
            contexts.append(record["context"])
            targets.append(record["prediction"])
            forecasts.append(forecast)
            quantiles_list.append(quantiles)
            
    metrics = evaluator.evaluate_batch(
        contexts=contexts,
        targets=targets,
        forecasts=forecasts,
        quantiles=quantiles_list,
        quantile_levels=config.quantiles
    )
    
    print("\n" + "=" * 40)
    print("VAL METRIC BENCHMARKS:")
    print("-" * 40)
    print(f"MASE:  {metrics['mase']:.4f}  (Baseline target < 1.0)")
    print(f"sMAPE: {metrics['smape']:.2f}%")
    print(f"MSE:   {metrics['mse']:.4f}")
    print(f"MAE:   {metrics['mae']:.4f}")
    print("\nQuantile Coverage Calibration:")
    for q in config.quantiles:
        print(f"  Target p{int(q*100):02d} coverage: {q:.2f} | Empirical coverage: {metrics[f'coverage_{q:.2f}']:.3f}")
    print("=" * 40)
    
    # 6. Export to ONNX
    print("\n[Step 5/5] Exporting best model to ONNX format...")
    onnx_fp = "checkpoints/nanoforecast.onnx"
    quant_onnx_fp = export_to_onnx(model, onnx_fp, quantize=True)
    
    # Report sizes
    if os.path.exists(onnx_fp):
        size_fp32 = os.path.getsize(onnx_fp) / 1024
        print(f"--> Exported FP32 ONNX Model Size: {size_fp32:.2f} KB")
    if os.path.exists(quant_onnx_fp):
        size_int8 = os.path.getsize(quant_onnx_fp) / 1024
        print(f"--> Exported INT8 Quantized Model Size: {size_int8:.2f} KB")
        
    print("\n" + "=" * 60)
    print("NANOFORECAST PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
