"""
TimesFM wrapper for easy forecasting.

TimesFM is a foundation model for time-series forecasting developed by Google Research.
It provides zero-shot forecasting - no training needed, just pass your data.

Paper: https://arxiv.org/abs/2310.10688
Model: https://huggingface.co/google/timesfm-2.5-200m-pytorch
"""
import os
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

# Suppress noisy HuggingFace Hub warnings before importing anything else
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF Hub.*")


@dataclass
class ForecastResult:
    """Container for forecast results."""
    timestamps: pd.DatetimeIndex
    point_forecast: np.ndarray
    quantile_forecast: Optional[np.ndarray]  # Shape: (horizon, num_quantiles)
    quantile_levels: Optional[list[float]]   # e.g., [0.1, 0.2, ..., 0.9]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with point forecast and quantile columns."""
        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "forecast": self.point_forecast,
        })
        
        if self.quantile_forecast is not None and self.quantile_levels is not None:
            for i, q in enumerate(self.quantile_levels):
                df[f"q{int(q*100):02d}"] = self.quantile_forecast[:, i]
        
        return df


class TimesFMForecaster:
    """
    Wrapper around Google's TimesFM model for easy forecasting.
    
    Usage:
        forecaster = TimesFMForecaster()
        forecaster.load_model()
        
        result = forecaster.forecast(
            series=df["volume"].values,
            timestamps=df["timestamp"],
            horizon=60
        )
    """
    
    def __init__(
        self,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        max_context: int = 512,
        max_horizon: int = 256,
        use_quantiles: bool = True,
    ):
        self.model_name = model_name
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.use_quantiles = use_quantiles
        self.model = None
        
    def load_model(self):
        """Load the TimesFM model from HuggingFace."""
        import warnings
        import logging
        import torch
        
        # Suppress HuggingFace warnings aggressively
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
        
        import timesfm
        
        print(f"Loading TimesFM model: {self.model_name}")
        torch.set_float32_matmul_precision("high")
        
        # Handle different TimesFM versions
        if hasattr(timesfm, 'TimesFM_2p5_200M_torch'):
            # Source install (recommended)
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_name)
        elif hasattr(timesfm, 'TimesFm'):
            # Older API
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu" if torch.cuda.is_available() else "cpu",
                    per_core_batch_size=32,
                    horizon_len=self.max_horizon,
                    context_len=self.max_context,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.model_name
                ),
            )
            print("✓ Model loaded (legacy API)")
            return
        else:
            raise RuntimeError(
                "TimesFM not properly installed. Please install from source:\n"
                "  git clone https://github.com/google-research/timesfm.git\n"
                "  cd timesfm && pip install -e '.[torch]'"
            )
        
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=self.use_quantiles,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        print("✓ Model loaded and compiled")
        
    def forecast(
        self,
        series: np.ndarray,
        timestamps: pd.DatetimeIndex,
        horizon: int,
        freq: str = "1min",
    ) -> ForecastResult:
        """
        Generate forecasts for a time series.
        
        Args:
            series: 1D numpy array of historical values
            timestamps: DatetimeIndex corresponding to series
            horizon: Number of steps to forecast
            freq: Frequency for generating future timestamps
            
        Returns:
            ForecastResult with point and quantile forecasts
        """
        if self.model is None:
            self.load_model()
        
        # Ensure we don't exceed max context
        if len(series) > self.max_context:
            series = series[-self.max_context:]
            timestamps = timestamps[-self.max_context:]
        
        # Run forecast
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=horizon,
            inputs=[series.astype(np.float32)],
        )
        
        # Generate future timestamps
        last_ts = timestamps[-1]
        future_timestamps = pd.date_range(
            start=last_ts + pd.Timedelta(freq),
            periods=horizon,
            freq=freq
        )
        
        # Extract results (model returns batch dimension)
        point = point_forecast[0]  # Shape: (horizon,)
        
        if quantile_forecast is not None:
            # Shape: (1, horizon, 10) -> (horizon, 10)
            # TimesFM 2.5 returns 10 columns: [mean, q10, q20, q30, q40, q50, q60, q70, q80, q90]
            # Drop the mean (column 0) so quantiles are aligned with their labels.
            quantiles = quantile_forecast[0][:, 1:]
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            quantiles = None
            quantile_levels = None
        
        return ForecastResult(
            timestamps=future_timestamps,
            point_forecast=point,
            quantile_forecast=quantiles,
            quantile_levels=quantile_levels,
        )
    
    def forecast_multiple(
        self,
        series_list: list[np.ndarray],
        horizon: int,
    ) -> list[np.ndarray]:
        """
        Forecast multiple time series efficiently in a single batch.
        
        Args:
            series_list: List of 1D numpy arrays
            horizon: Number of steps to forecast
            
        Returns:
            List of point forecasts (one per input series)
        """
        if self.model is None:
            self.load_model()
        
        # Truncate each series to max_context
        inputs = [
            s[-self.max_context:].astype(np.float32) 
            for s in series_list
        ]
        
        point_forecasts, _ = self.model.forecast(
            horizon=horizon,
            inputs=inputs,
        )
        
        return [point_forecasts[i] for i in range(len(inputs))]
