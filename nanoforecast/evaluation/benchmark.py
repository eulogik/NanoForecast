import numpy as np
from typing import Dict, List, Tuple

class TimeSeriesEvaluator:
    """
    Computes time series forecasting metrics including MASE, sMAPE, MSE, MAE,
    and quantile coverage calibration metrics.
    """
    @staticmethod
    def smape(target: np.ndarray, forecast: np.ndarray) -> float:
        """
        Symmetric Mean Absolute Percentage Error.
        Args:
            target: Shape [H]
            forecast: Shape [H]
        """
        denominator = (np.abs(target) + np.abs(forecast)) / 2.0
        # Avoid division by zero
        non_zero = denominator > 1e-5
        if not np.any(non_zero):
            return 0.0
            
        diff = np.abs(target[non_zero] - forecast[non_zero]) / denominator[non_zero]
        return float(100.0 * np.mean(diff))

    @staticmethod
    def mase(
        context: np.ndarray, 
        target: np.ndarray, 
        forecast: np.ndarray, 
        seasonality: int = 1
    ) -> float:
        """
        Mean Absolute Scaled Error.
        Compares forecast MAE to in-sample naive 1-step baseline MAE.
        Args:
            context: Context window history of shape [L]
            target: Ground truth target of shape [H]
            forecast: Predicted point forecast of shape [H]
            seasonality: Period of seasonality (default 1 for naive persistence)
        """
        # In-sample naive baseline MAE
        n = len(context)
        if n <= seasonality:
            # Context too short, fall back to simple denominator
            scale = np.mean(np.abs(context))
        else:
            scale = np.mean(np.abs(context[seasonality:] - context[:-seasonality]))
            
        if scale < 1e-5:
            scale = 1e-5 # Avoid division by zero
            
        mae = np.mean(np.abs(target - forecast))
        return float(mae / scale)

    @staticmethod
    def quantile_coverage(
        target: np.ndarray, 
        quantiles: np.ndarray, 
        quantile_levels: List[float]
    ) -> Dict[float, float]:
        """
        Calculates empirical coverage of quantiles to check calibration.
        Args:
            target: Shape [H]
            quantiles: Shape [num_quantiles, H]
            quantile_levels: List of quantile levels corresponding to the rows of quantiles
        Returns:
            Dict mapping quantile level to empirical coverage fraction
        """
        coverage = {}
        H = len(target)
        for i, q in enumerate(quantile_levels):
            # Fraction of values below predicted quantile bound
            q_bound = quantiles[i, :]
            cov_fraction = np.sum(target <= q_bound) / H
            coverage[q] = float(cov_fraction)
        return coverage

    def evaluate_batch(
        self,
        contexts: List[np.ndarray],
        targets: List[np.ndarray],
        forecasts: List[np.ndarray],
        quantiles: List[np.ndarray], # List of [num_quantiles, H]
        quantile_levels: List[float]
    ) -> Dict[str, float]:
        """
        Averages metrics over multiple series evaluation.
        """
        mases = []
        smapes = []
        mses = []
        maes = []
        coverages = {q: [] for q in quantile_levels}
        
        for ctx, tgt, fcast, quant in zip(contexts, targets, forecasts, quantiles):
            # Compute basic metrics
            mse = np.mean((tgt - fcast) ** 2)
            mae = np.mean(np.abs(tgt - fcast))
            
            mses.append(mse)
            maes.append(mae)
            
            smapes.append(self.smape(tgt, fcast))
            mases.append(self.mase(ctx, tgt, fcast))
            
            cov = self.quantile_coverage(tgt, quant, quantile_levels)
            for q in quantile_levels:
                coverages[q].append(cov[q])
                
        metrics = {
            "mase": float(np.mean(mases)),
            "smape": float(np.mean(smapes)),
            "mse": float(np.mean(mses)),
            "mae": float(np.mean(maes)),
        }
        
        # Average coverages
        for q in quantile_levels:
            metrics[f"coverage_{q:.2f}"] = float(np.mean(coverages[q]))
            
        return metrics
