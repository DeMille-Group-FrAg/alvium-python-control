import sys
import numpy as np
from scipy import optimize
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QCheckBox, QGridLayout, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal, QRunnable, QObject, pyqtSlot, QTime, QTimer
import pyqtgraph as pg
import time

# ==========================================
# 1. FIT FUNCTIONS (FIXED for 2D arrays)
# ==========================================

def gaussian(amp, x_mean, y_mean, x_width, y_width, offset):
    x_width = float(x_width)
    y_width = float(y_width)
    return lambda x, y: amp*np.exp(-0.5*((x-x_mean)/x_width)**2-0.5*((y-y_mean)/y_width)**2) + offset

def gaussianfit(data, max_iterations=100):
    """
    Fit a 2D Gaussian to the data.
    max_iterations controls how long the fit takes (for stress testing)
    """
    try:
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

        # CRITICAL: maxfev controls iterations - increase for stress test
        X, Y = np.indices(data.shape)
        errorfunction = lambda p: np.ravel(gaussian(*p)(X, Y) - data)
        p, success = optimize.leastsq(errorfunction, (amp, x_mean, y_mean, x_width, y_width, offset), maxfev=max_iterations)

        return {
            "amp": p[0], "x_mean": p[1], "y_mean": p[2],
            "x_width": p[3], "y_width": p[4], "offset": p[5]
        }
    except Exception as e:
        print(f"Fit error: {e}")
        return None

# ==========================================
# 2. THREADING WORKER
# ==========================================

class FitWorkerSignals(QObject):
    finished = pyqtSignal(dict)
    fit_time = pyqtSignal(float)

class FitWorker(QRunnable):
    def __init__(self, data, max_iterations=100):
        super().__init__()
        self.data = data
        self.max_iterations = max_iterations
        self.signals = FitWorkerSignals()

    def run(self):
        start_time = time.time()
        result = gaussianfit(self.data, max_iterations=self.max_iterations)
        fit_duration = time.time() - start_time

        if result:
            result['fit_time'] = fit_duration
            self.signals.finished.emit(result)
            self.signals.fit_time.emit(fit_duration)
        else:
            self.signals.finished.emit({})
            self.signals.fit_time.emit(fit_duration)

# ==========================================
# 3. SIMULATION THREAD (Fake Camera - 1000x1000)
# ==========================================

class SimCameraThread(QThread):
    new_image = pyqtSignal(np.ndarray)

    def __init__(self, image_size=1000):
        super().__init__()
        self.running = True
        self.counter = 0
        self.image_size = image_size

    def run(self):
        # Pre-create coordinate grids (expensive, do once)
        x = np.linspace(-5, 5, self.image_size)
        y = np.linspace(-5, 5, self.image_size)
        X, Y = np.meshgrid(x, y)

        while self.running:
            # Wobble the center slightly
            cx = np.sin(self.counter / 10.0) * 2
            cy = np.cos(self.counter / 10.0) * 2

            # Generate 1000x1000 Gaussian with noise
            Z = 100 * np.exp(-0.5*((X-cx)/1.5)**2 - 0.5*((Y-cy)/1.5)**2)
            noise = np.random.normal(0, 5, Z.shape)
            image = (Z + noise).astype(np.float32)

            self.new_image.emit(image)
            self.counter += 1
            self.msleep(100)  # 10 Hz = 100ms per frame

    def stop(self):
        self.running = False
        self.wait()

# ==========================================
# 4. MAIN GUI WITH FPS METER
# ==========================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaussian Fit Threading Stress Test (1000x1000)")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)

        # Image View
        self.img_view = pg.ImageView()
        layout.addWidget(self.img_view, 0, 0, 6, 1)

        # Results Labels
        self.lbl_amp = QLabel("Amp: 0")
        self.lbl_x = QLabel("X Mean: 0")
        self.lbl_y = QLabel("Y Mean: 0")
        self.lbl_fit_time = QLabel("Fit Time: 0 ms")

        # FPS Counter (measures GUI responsiveness)
        self.lbl_fps = QLabel("GUI FPS: 0")
        self.lbl_fps.setStyleSheet("font-weight: bold; font-size: 14pt;")

        # Lag Indicator (turns red when GUI is blocked)
        self.lbl_lag = QLabel("Status: Idle")
        self.lbl_lag.setStyleSheet("font-weight: bold; background-color: green; color: white; padding: 5px;")

        layout.addWidget(self.lbl_amp, 0, 1)
        layout.addWidget(self.lbl_x, 1, 1)
        layout.addWidget(self.lbl_y, 2, 1)
        layout.addWidget(self.lbl_fit_time, 3, 1)
        layout.addWidget(self.lbl_fps, 4, 1)
        layout.addWidget(self.lbl_lag, 5, 1)

        # Controls
        self.chk_thread = QCheckBox("Use Threading (Recommended)")
        self.chk_thread.setChecked(True)
        self.chk_thread.stateChanged.connect(self.on_thread_toggle)
        layout.addWidget(self.chk_thread, 6, 1)

        self.chk_stress = QCheckBox("Stress Mode (More Iterations)")
        self.chk_stress.setChecked(False)
        layout.addWidget(self.chk_stress, 7, 1)

        self.btn_start = QPushButton("Start Simulation")
        self.btn_start.clicked.connect(self.toggle_simulation)
        layout.addWidget(self.btn_start, 8, 1)

        # FPS Timer
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.gui_responsive = True

        # State
        self.cam_thread = None
        self.fit_start_time = 0

    def on_thread_toggle(self):
        """Reset lag indicator when toggling threading mode"""
        self.lbl_lag.setText("Status: Idle")
        self.lbl_lag.setStyleSheet("font-weight: bold; background-color: green; color: white; padding: 5px;")

    def toggle_simulation(self):
        if self.cam_thread is None or not self.cam_thread.isRunning():
            image_size = 1000 if self.chk_stress.isChecked() else 500
            self.cam_thread = SimCameraThread(image_size=image_size)
            self.cam_thread.new_image.connect(self.on_new_image)
            self.cam_thread.start()
            self.btn_start.setText("Stop Simulation")
            self.lbl_lag.setText("Status: Running")
            self.frame_count = 0
        else:
            self.cam_thread.stop()
            self.cam_thread = None
            self.btn_start.setText("Start Simulation")
            self.lbl_lag.setText("Status: Stopped")

    def on_new_image(self, image):
        """Called when new image arrives from camera thread"""
        self.frame_count += 1

        # Mark GUI as potentially blocked
        self.gui_responsive = False
        self.lbl_lag.setText("Status: Processing...")
        self.lbl_lag.setStyleSheet("font-weight: bold; background-color: orange; color: black; padding: 5px;")

        # Record when we started processing
        self.fit_start_time = time.time()

        # 1. Update Image (Fast)
        self.img_view.setImage(image, autoLevels=False)

        # 2. Trigger Fit
        max_iter = 200 if self.chk_stress.isChecked() else 100

        if self.chk_thread.isChecked():
            # --- THREADED MODE ---
            self.lbl_lag.setText("Status: Fitting (Thread)")
            self.lbl_lag.setStyleSheet("font-weight: bold; background-color: yellow; color: black; padding: 5px;")

            worker = FitWorker(image, max_iterations=max_iter)
            worker.signals.finished.connect(self.on_fit_complete)
            worker.signals.fit_time.connect(self.on_fit_time)
            pg.QtCore.QThreadPool.globalInstance().start(worker)

            # GUI remains responsive - mark as responsive immediately
            self.gui_responsive = True
            self.lbl_lag.setStyleSheet("font-weight: bold; background-color: green; color: white; padding: 5px;")
        else:
            # --- MAIN THREAD MODE (Will Lag) ---
            self.lbl_lag.setText("Status: Fitting (Main Thread)")
            self.lbl_lag.setStyleSheet("font-weight: bold; background-color: red; color: white; padding: 5px;")

            # Force GUI to process pending events (shows the lag)
            QApplication.processEvents()

            # This blocks the GUI
            result = gaussianfit(image, max_iterations=max_iter)

            fit_duration = time.time() - self.fit_start_time
            self.lbl_fit_time.setText(f"Fit Time: {fit_duration*1000:.1f} ms")
            self.update_labels(result)

            # GUI becomes responsive again
            self.gui_responsive = True
            self.lbl_lag.setStyleSheet("font-weight: bold; background-color: green; color: white; padding: 5px;")

    @pyqtSlot(dict)
    def on_fit_complete(self, result):
        """Called when fit completes in worker thread"""
        self.update_labels(result)

    @pyqtSlot(float)
    def on_fit_time(self, fit_duration):
        """Called with fit duration from worker thread"""
        self.lbl_fit_time.setText(f"Fit Time: {fit_duration*1000:.1f} ms")
        # GUI was responsive during fit
        self.lbl_lag.setStyleSheet("font-weight: bold; background-color: green; color: white; padding: 5px;")

    def update_labels(self, result):
        if result:
            self.lbl_amp.setText(f"Amp: {result['amp']:.2f}")
            self.lbl_x.setText(f"X Mean: {result['x_mean']:.2f}")
            self.lbl_y.setText(f"Y Mean: {result['y_mean']:.2f}")
        else:
            self.lbl_amp.setText("Amp: Fit Failed")

    def update_fps(self):
        """Update FPS counter - measures GUI responsiveness"""
        elapsed = time.time() - self.last_fps_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
            self.lbl_fps.setText(f"GUI FPS: {fps:.1f}")

            # If FPS drops below 5, GUI is significantly blocked
            if fps < 5 and self.cam_thread and self.cam_thread.isRunning():
                if not self.chk_thread.isChecked():
                    self.lbl_fps.setStyleSheet("font-weight: bold; font-size: 14pt; color: red;")
                else:
                    self.lbl_fps.setStyleSheet("font-weight: bold; font-size: 14pt; color: green;")
            else:
                self.lbl_fps.setStyleSheet("font-weight: bold; font-size: 14pt; color: white;")

        self.frame_count = 0
        self.last_fps_time = time.time()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())