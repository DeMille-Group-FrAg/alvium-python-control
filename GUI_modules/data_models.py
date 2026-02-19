# ==========================================
# data_models.py
# ==========================================
"""Data structures shared across modules."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ImageData:
    """Container for image data from acquisition thread."""
    atom: np.ndarray
    probe: np.ndarray
    dark: np.ndarray
    sig_bkg: np.ndarray
    od: np.ndarray
    counter: int
    is_scan: bool
    scan_value: Optional[float] = None

@dataclass
class FitResult:
    """Container for Gaussian fit results."""
    amp: float
    x_mean: float
    y_mean: float
    x_width: float
    y_width: float
    offset: float
    peak: float