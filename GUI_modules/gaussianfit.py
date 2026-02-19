import logging
from typing import Optional
import numpy as np
import scipy.optimize as optimize
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable
from GUI_modules.data_models import FitResult

# ==========================================
# GAUSSIAN FIT MODEL
# ==========================================

def gaussian(amp, x_mean, y_mean, x_width, y_width, offset):
    """Return a 2D gaussian function."""
    x_width = float(x_width)
    y_width = float(y_width)
    return lambda x, y: amp*np.exp(-0.5*((x-x_mean)/x_width)**2-0.5*((y-y_mean)/y_width)**2) + offset

def gaussianfit(data) -> Optional[FitResult]:
    """
    Fit a 2D gaussian to the data.
    Returns FitResult object or None if fit fails.
    """
    try:
        # calculate moments for initial guess
        total = np.sum(data)
        if total == 0:
            return None
        
        X, Y = np.indices(data.shape)
        x_mean = np.sum(X*data)/total
        x_mean = np.clip(x_mean, 0, data.shape[0]-1)
        y_mean = np.sum(Y*data)/total
        y_mean = np.clip(y_mean, 0, data.shape[1]-1)
        
        col = data[:, int(y_mean)]
        if col.sum() == 0:
            x_width = 1
        else:
            x_width = np.sqrt(np.abs((np.arange(col.size)-x_mean)**2*col).sum()/col.sum())
        
        row = data[int(x_mean), :]
        if row.sum() == 0:
            y_width = 1
        else:
            y_width = np.sqrt(np.abs((np.arange(row.size)-y_mean)**2*row).sum()/row.sum())
        
        offset = (data[0, :].sum()+data[-1, :].sum()+data[:, 0].sum()+data[:, -1].sum())/np.sum(data.shape)/2
        amp = data.max() - offset

        # use optimize function to obtain 2D gaussian fit
        errorfunction = lambda p: np.ravel(gaussian(*p)(X, Y) - data)
        p, success = optimize.leastsq(errorfunction, (amp, x_mean, y_mean, x_width, y_width, offset), maxfev=100)

        return FitResult(
            amp=p[0],
            x_mean=p[1],
            y_mean=p[2],
            x_width=p[3],
            y_width=p[4],
            offset=p[5],
            peak=np.max(data)
        )
    except Exception as e:
        logging.warning(f"Gaussian fit error: {e}")
        return None


# ==========================================
# GAUSSIAN FIT WORKER (THREADING)
# ==========================================

class FitWorkerSignals(QObject):
    """Signals for the fit worker thread."""
    finished = pyqtSignal(object)  

class FitWorker(QRunnable):
    """Worker to perform Gaussian fit in a background thread."""
    
    def __init__(self, data, maxfev=100):
        super().__init__()
        self.data = data
        self.maxfev = maxfev
        self.signals = FitWorkerSignals()

    def run(self):
        """Run the fit in a background thread."""
        # Directly return FitResult from gaussianfit (no conversion needed)
        fit_result = gaussianfit(self.data)
        self.signals.finished.emit(fit_result)