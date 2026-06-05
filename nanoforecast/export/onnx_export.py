import os
import torch
import argparse
from typing import Optional

from nanoforecast.model.core import NanoForecast
from nanoforecast.config import NanoForecastConfig

def export_to_onnx(
    model: NanoForecast,
    export_path: str,
    quantize: bool = True
) -> str:
    """
    Exports the NanoForecast PyTorch model to ONNX format and optionally
    quantizes it to INT8 using onnxruntime-quantization.
    """
    model.eval()
    config = model.config
    
    # 1. Prepare dummy inputs matching model signature
    # Shape: [Batch, Channels, Context_Length]
    dummy_x = torch.randn(1, 1, config.context_length)
    dummy_freq = torch.zeros(1, dtype=torch.long)
    
    dummy_cov = None
    input_names = ["context", "freq_ids"]
    inputs = (dummy_x, dummy_freq)
    
    if config.covariate_dim > 0:
        dummy_cov = torch.randn(1, config.covariate_dim, config.context_length)
        input_names.append("covariates")
        inputs = (dummy_x, dummy_freq, dummy_cov)
        
    output_names = ["forecast", "quantiles", "reconstructed", "trend", "seasonal", "residual"]
    
    # Define dynamic axes to allow batch inference scaling
    dynamic_axes = {
        "context": {0: "batch_size"},
        "freq_ids": {0: "batch_size"},
        "forecast": {0: "batch_size"},
        "quantiles": {0: "batch_size"},
        "reconstructed": {0: "batch_size"},
        "trend": {0: "batch_size"},
        "seasonal": {0: "batch_size"},
        "residual": {0: "batch_size"},
    }
    
    if config.covariate_dim > 0:
        dynamic_axes["covariates"] = {0: "batch_size"}

    # 2. PyTorch ONNX export (Opset 17 supports DFT operations)
    print(f"Exporting PyTorch model to ONNX at {export_path}...")
    torch.onnx.export(
        model,
        inputs,
        export_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print("--> Export complete!")
    
    # 3. Dynamic INT8 Quantization if requested
    if quantize:
        quant_path = export_path.replace(".onnx", "_int8.onnx")
        print(f"Quantizing ONNX model to INT8 at {quant_path}...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                model_input=export_path,
                model_output=quant_path,
                weight_type=QuantType.QUInt8
            )
            print("--> Quantization complete!")
            return quant_path
        except ImportError:
            print("WARNING: 'onnxruntime' or 'onnxruntime-quantization' not installed.")
            print("Skipping INT8 quantization. To quantize, run: pip install onnxruntime-quantization")
            
    return export_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export NanoForecast model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt checkpoint")
    parser.add_argument("--output", type=str, default="nanoforecast.onnx", help="Output ONNX filename")
    parser.add_argument("--no-quantize", action="store_true", help="Disable INT8 quantization")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist.")
        exit(1)
        
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    
    model = NanoForecast(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    export_to_onnx(model, args.output, quantize=not args.no_quantize)
