# ==========================================
# GUI_modules/control.py
# ==========================================
"""Control panel class - handles user input and acquisition logic."""

import logging
import os
import time
import configparser
from collections import deque
from typing import Optional, Dict, List

import numpy as np
import PyQt5.QtWidgets as qt
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QRunnable, QObject, pyqtSlot, QThreadPool

from GUI_modules.data_models import ImageData, FitResult
from GUI_modules.gaussianfit import FitWorker
from widgets import NewSpinBox, NewDoubleSpinBox, NewComboBox, Scrollarea

from GUI_modules.QTcpServer_worker import ServerWorker
from GUI_modules.camera_wrapper import AlviumCameraWrapper
from GUI_modules.acquisition_thread import AcquisitionThread
import PyQt5

# TYPE_CHECKING prevents circular imports at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import CameraGUI
    from GUI_modules.image_window import ImageWin


# the class that places elements in UI and handles data processing
class Control(Scrollarea):
    server_stop = pyqtSignal()
    def __init__(self, parent: 'CameraGUI', image_win: 'ImageWin'):
        super().__init__(parent, label="", type="vbox")
        self.setMaximumWidth(400)
        self.frame.setContentsMargins(0,0,0,0)
        self.image_win = image_win

        # interpret data as fluorescence or optical density
        self.meas_mode = self.parent.defaults["measurement"].get("default")

        # number of pixels of the largest image we can do gaussian fit to in real time (i.e. updating in every experimental cycle)
        # it depends on CPU power and duration of experimental cycle
        self.cpu_limit = self.parent.defaults["gaussian_fit"].getint("cpu_limit")

        # number of images to take in each run
        self.num_img_to_take = self.parent.defaults["image_number"].getint("default")

        # image region of interest
        self.roi = {"xmin": self.parent.defaults["roi"].getint("xmin"),
                    "xmax": self.parent.defaults["roi"].getint("xmax"),
                    "ymin": self.parent.defaults["roi"].getint("ymin"),
                    "ymax": self.parent.defaults["roi"].getint("ymax")}

        # gaussian filter settings
        self.gaussian_fit = self.parent.defaults["gaussian_fit"].getboolean("default")
        self.gaussian_filter = self.parent.defaults["gaussian_filter"].getboolean("state")
        self.gaussian_filter_sigma = self.parent.defaults["gaussian_filter"].getfloat("sigma")

        self.img_save = self.parent.defaults["image_save"].getboolean("default")

        # boolean variable, turned on when the camera starts to take images
        self.active = False

        # control mode, can be "record" or "scan" in current implementation
        self.control_mode = None

        # save signal count
        self.signal_count_deque = deque([], maxlen=20)

        self.signal_count_scan_dict = {}

        # Gaussian fit thread pool
        self.fit_worker_pool = QThreadPool.globalInstance()
        self.fit_worker_pool.setMaxThreadCount(2)  # Limit concurrent fits

        # places GUI elements
        self.place_recording()
        self.place_image_control()
        self.place_cam_control()
        self.place_tcp_control()
        self.place_save_load()

    # place recording gui elements
    def place_recording(self):
        record_box = qt.QGroupBox("Recording")
        record_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        record_box.setMaximumHeight(270)
        record_frame = qt.QGridLayout()
        record_box.setLayout(record_frame)
        self.frame.addWidget(record_box)

        self.record_bt = qt.QPushButton("Record")
        self.record_bt.clicked[bool].connect(lambda val, mode="record": self.start(mode))
        record_frame.addWidget(self.record_bt, 0, 0)
        self.record_bt.setEnabled(True)

        self.scan_bt = qt.QPushButton("Scan")
        self.scan_bt.clicked[bool].connect(lambda val, mode="scan": self.start(mode))
        record_frame.addWidget(self.scan_bt, 0, 1)
        self.scan_bt.setEnabled(False)

        self.stop_bt = qt.QPushButton("Stop")
        self.stop_bt.clicked[bool].connect(lambda val: self.stop())
        record_frame.addWidget(self.stop_bt, 0, 2)
        self.stop_bt.setEnabled(False)

        record_frame.addWidget(qt.QLabel("Measurement:"), 1, 0, 1, 1)
        self.meas_rblist = []
        meas_bg = qt.QButtonGroup(self.parent)
        op = [x.strip() for x in self.parent.defaults["measurement"]["options"].split(',')]
        for j, i in enumerate(op):
            meas_rb = qt.QRadioButton(i)
            meas_rb.setFixedHeight(30)
            meas_rb.setChecked(True if i == self.meas_mode else False)
            meas_rb.toggled[bool].connect(lambda val, rb=meas_rb: self.set_meas_mode(rb.text(), val))
            self.meas_rblist.append(meas_rb)
            meas_bg.addButton(meas_rb)
            record_frame.addWidget(meas_rb, 1, 1+j, 1, 1)

        # display signal count in real time
        record_frame.addWidget(qt.QLabel("Signal count:"), 2, 0, 1, 1)
        self.signal_count = qt.QLabel()
        self.signal_count.setText("0")
        self.signal_count.setStyleSheet("QLabel{background-color: gray; font: 20pt}")
        self.signal_count.setToolTip("Singal after bkg subtraction or OD")
        record_frame.addWidget(self.signal_count, 2, 1, 1, 2)

        # display mean of signal count in real time in "record" mode
        record_frame.addWidget(qt.QLabel("Signal mean:"), 3, 0, 1, 1)
        self.signal_count_mean = qt.QLabel()
        self.signal_count_mean.setText("0")
        self.signal_count_mean.setStyleSheet("QLabel{background-color: gray; font: 20pt}")
        self.signal_count_mean.setToolTip("Signal after bkg subtraction or OD")
        record_frame.addWidget(self.signal_count_mean, 3, 1, 1, 2)

        # display error of mean of signal count in real time in "record" mode
        record_frame.addWidget(qt.QLabel("Signal error:"), 4, 0, 1, 1)
        self.signal_count_err_mean = qt.QLabel()
        self.signal_count_err_mean.setText("0")
        self.signal_count_err_mean.setStyleSheet("QLabel{background-color: gray; font: 20pt}")
        self.signal_count_err_mean.setToolTip("Signal after bkg subtraction or OD")
        record_frame.addWidget(self.signal_count_err_mean, 4, 1, 1, 2)

    # place image control gui elements
    def place_image_control(self):
        img_ctrl_box = qt.QGroupBox("Image Control")
        img_ctrl_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        img_ctrl_frame = qt.QFormLayout()
        img_ctrl_box.setLayout(img_ctrl_frame)
        self.frame.addWidget(img_ctrl_box)

        # a spinbox to set number of images to take in next run
        num_img_upperlimit = self.parent.defaults["image_number"].getint("max")
        self.num_img_to_take_sb = NewSpinBox(range=(1, num_img_upperlimit), suffix=None)
        self.num_img_to_take_sb.setValue(self.num_img_to_take)
        self.num_img_to_take_sb.valueChanged[int].connect(lambda val: self.set_num_img(val))
        img_ctrl_frame.addRow("Num of image to take:", self.num_img_to_take_sb)

        # spinboxes to set image region of interest in x
        self.x_min_sb = NewSpinBox(range=(0, self.roi["xmax"]-1), suffix=None)
        self.x_min_sb.setValue(self.roi["xmin"])
        self.x_max_sb = NewSpinBox(range=(self.roi["xmin"]+1, self.parent.device.image_shape["xmax"]), suffix=None)
        self.x_max_sb.setValue(self.roi["xmax"])
        self.x_min_sb.valueChanged[int].connect(lambda val, text='xmin', sb=self.x_max_sb:
                                                self.set_roi(text, val, sb))
        self.x_max_sb.valueChanged[int].connect(lambda val, text='xmax', sb=self.x_min_sb:
                                                self.set_roi(text, val, sb))

        x_range_box = qt.QWidget()
        x_range_layout = qt.QHBoxLayout()
        x_range_layout.setContentsMargins(0,0,0,0)
        x_range_box.setLayout(x_range_layout)
        x_range_layout.addWidget(self.x_min_sb)
        x_range_layout.addWidget(self.x_max_sb)
        img_ctrl_frame.addRow("ROI X range:", x_range_box)

        # spinboxes to set image region of interest in y
        self.y_min_sb = NewSpinBox(range=(0, self.roi["ymax"]-1), suffix=None)
        self.y_min_sb.setValue(self.roi["ymin"])
        self.y_max_sb = NewSpinBox(range=(self.roi["ymin"]+1, self.parent.device.image_shape["ymax"]), suffix=None)
        self.y_max_sb.setValue(self.roi["ymax"])
        self.y_min_sb.valueChanged[int].connect(lambda val, text='ymin', sb=self.y_max_sb:
                                                self.set_roi(text, val, sb))
        self.y_max_sb.valueChanged[int].connect(lambda val, text='ymax', sb=self.y_min_sb:
                                                self.set_roi(text, val, sb))

        y_range_box = qt.QWidget()
        y_range_layout = qt.QHBoxLayout()
        y_range_layout.setContentsMargins(0,0,0,0)
        y_range_box.setLayout(y_range_layout)
        y_range_layout.addWidget(self.y_min_sb)
        y_range_layout.addWidget(self.y_max_sb)
        img_ctrl_frame.addRow("ROI Y range:", y_range_box)

        # display number of images that have been taken
        self.num_image = qt.QLabel()
        self.num_image.setText("0")
        self.num_image.setStyleSheet("background-color: gray;")
        img_ctrl_frame.addRow("Num of recorded images:", self.num_image)

        # set hdf group name and whether to save image to a hdf file
        self.run_name_le = qt.QLineEdit()
        default_run_name = self.parent.defaults["image_save"]["run_name"]
        self.run_name_le.setText(default_run_name)
        self.run_name_le.setToolTip("HDF group name/run name")
        self.img_save_chb = qt.QCheckBox()
        self.img_save_chb.setTristate(False)
        self.img_save_chb.setChecked(self.img_save)
        self.img_save_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.img_save_chb.stateChanged[int].connect(lambda state: self.set_img_save(state))
        img_save_box = qt.QWidget()
        img_save_layout = qt.QHBoxLayout()
        img_save_layout.setContentsMargins(0,0,0,0)
        img_save_box.setLayout(img_save_layout)
        img_save_layout.addWidget(self.run_name_le)
        img_save_layout.addWidget(self.img_save_chb)
        img_ctrl_frame.addRow("Image auto save:", img_save_box)

        img_ctrl_frame.addRow("------------------", qt.QWidget())

        # set whether to apply gaussian filter
        self.gauss_filter_chb = qt.QCheckBox()
        self.gauss_filter_chb.setTristate(False)
        self.gauss_filter_chb.setChecked(self.gaussian_filter)
        self.gauss_filter_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.gauss_filter_chb.stateChanged[int].connect(lambda val, param="state": self.set_gauss_filter(val, param))
        img_ctrl_frame.addRow("gaussian filter:", self.gauss_filter_chb)

        # spinboxes to set gaussian filter sigma
        self.gaussian_filter_sigma_dsb = NewDoubleSpinBox(range=(0.01, 10000), decimals=2, suffix=None)
        self.gaussian_filter_sigma_dsb.setValue(self.gaussian_filter_sigma)
        self.gaussian_filter_sigma_dsb.valueChanged[float].connect(lambda val, param="sigma": self.set_gauss_filter(val, param))
        img_ctrl_frame.addRow("gaussian filter sigma:", self.gaussian_filter_sigma_dsb)

        img_ctrl_frame.addRow("------------------", qt.QWidget())

        # set whether to do gaussian fit in real time
        self.gauss_fit_chb = qt.QCheckBox()
        self.gauss_fit_chb.setTristate(False)
        self.gauss_fit_chb.setChecked(self.gaussian_fit)
        self.gauss_fit_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.gauss_fit_chb.stateChanged[int].connect(lambda state: self.set_gauss_fit(state))
        self.gauss_fit_chb.setToolTip(f"Can only be enabled when image size less than {self.cpu_limit} pixels.")
        img_ctrl_frame.addRow("2D gaussian fit:", self.gauss_fit_chb)

        if (self.roi["xmax"]-self.roi["xmin"])*(self.roi["ymax"]-self.roi["ymin"]) > self.cpu_limit:
            # this line has to be after gauss_fit_chb's connect()
            self.gauss_fit_chb.setChecked(False)
            self.gauss_fit_chb.setEnabled(False)

        # display 2D gaussian fit results
        self.x_mean = qt.QLabel()
        self.x_mean.setMaximumWidth(90)
        self.x_mean.setText("0")
        self.x_mean.setStyleSheet("QWidget{background-color: gray;}")
        self.x_mean.setToolTip("x mean")
        self.x_stand_dev = qt.QLabel()
        self.x_stand_dev.setMaximumWidth(90)
        self.x_stand_dev.setText("0")
        self.x_stand_dev.setStyleSheet("QWidget{background-color: gray;}")
        self.x_stand_dev.setToolTip("x standard deviation")
        gauss_x_box = qt.QWidget()
        gauss_x_layout = qt.QHBoxLayout()
        gauss_x_layout.setContentsMargins(0,0,0,0)
        gauss_x_box.setLayout(gauss_x_layout)
        gauss_x_layout.addWidget(self.x_mean)
        gauss_x_layout.addWidget(self.x_stand_dev)
        img_ctrl_frame.addRow("2D gaussian fit (x):", gauss_x_box)

        self.y_mean = qt.QLabel()
        self.y_mean.setMaximumWidth(90)
        self.y_mean.setText("0")
        self.y_mean.setStyleSheet("QWidget{background-color: gray;}")
        self.y_mean.setToolTip("y mean")
        self.y_stand_dev = qt.QLabel()
        self.y_stand_dev.setMaximumWidth(90)
        self.y_stand_dev.setText("0")
        self.y_stand_dev.setStyleSheet("QWidget{background-color: gray;}")
        self.y_stand_dev.setToolTip("y standard deviation")
        gauss_y_box = qt.QWidget()
        gauss_y_layout = qt.QHBoxLayout()
        gauss_y_layout.setContentsMargins(0,0,0,0)
        gauss_y_box.setLayout(gauss_y_layout)
        gauss_y_layout.addWidget(self.y_mean)
        gauss_y_layout.addWidget(self.y_stand_dev)
        img_ctrl_frame.addRow("2D gaussian fit (y):", gauss_y_box)

        self.amp = qt.QLabel()
        self.amp.setText("0")
        self.amp.setStyleSheet("QWidget{background-color: gray;}")
        img_ctrl_frame.addRow("2D gaussian fit (amp.):", self.amp)

        self.offset = qt.QLabel()
        self.offset.setText("0")
        self.offset.setStyleSheet("QWidget{background-color: gray;}")
        img_ctrl_frame.addRow("2D gaussian fit (offset):", self.offset)
        
        self.peak = qt.QLabel()
        self.peak.setText("0")
        self.peak.setStyleSheet("QWidget{background-color: gray;}")
        img_ctrl_frame.addRow("Peak Signal:", self.peak)

    # place camera control gui elements
    def place_cam_control(self):
        self.cam_ctrl_box = qt.QGroupBox("Camera Control")
        self.cam_ctrl_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        cam_ctrl_frame = qt.QFormLayout()
        self.cam_ctrl_box.setLayout(cam_ctrl_frame)
        self.frame.addWidget(self.cam_ctrl_box)

        # set trigger mode
        self.trig_mode_rblist = []
        trig_bg = qt.QButtonGroup(self.parent)
        self.trig_box = qt.QWidget()
        self.trig_box.setMaximumWidth(200)
        trig_layout = qt.QHBoxLayout()
        trig_layout.setContentsMargins(0,0,0,0)
        self.trig_box.setLayout(trig_layout)
        op = [x.strip() for x in self.parent.defaults["trigger_mode"]["options"].split(',')]
        for i in op:
            trig_mode_rb = qt.QRadioButton(i)
            trig_mode_rb.setChecked(True if i == self.parent.device.trigger_mode else False)
            trig_mode_rb.toggled[bool].connect(lambda val, rb=trig_mode_rb: self.parent.device.set_trigger_mode(rb.text(), val))
            self.trig_mode_rblist.append(trig_mode_rb)
            trig_bg.addButton(trig_mode_rb)
            trig_layout.addWidget(trig_mode_rb)
        cam_ctrl_frame.addRow("Trigger mode:", self.trig_box)

        # set exposure time and unit
        expo_cf = self.parent.defaults["expo_time"]
        default_unit = self.parent.defaults["expo_unit"]["default"]
        default_unit_num = self.parent.defaults["expo_unit"].getfloat(default_unit)
        default_time = expo_cf.getfloat("default")/default_unit_num
        self.expo_dsb = NewDoubleSpinBox(range=(expo_cf.getfloat("min")/default_unit_num, expo_cf.getfloat("max")/default_unit_num), decimals=int(expo_cf.getint("decimals")+np.log10(default_unit_num)))
        self.expo_dsb.setValue(default_time)
        self.expo_unit_cb = NewComboBox()
        self.expo_unit_cb.setMaximumHeight(30)
        op = [x.strip() for x in self.parent.defaults["expo_unit"]["options"].split(',')]
        self.expo_unit_cb.addItems(op)
        self.expo_unit_cb.setCurrentText(default_unit)
        self.expo_dsb.valueChanged[float].connect(lambda val, cb=self.expo_unit_cb, type="time":
                                            self.set_expo_time(val, cb.currentText(), type))
        self.expo_unit_cb.currentTextChanged[str].connect(lambda val, dsb=self.expo_dsb, type="unit": self.set_expo_time(dsb.value(), val, type))
        expo_box = qt.QWidget()
        expo_box.setMaximumWidth(200)
        expo_layout = qt.QHBoxLayout()
        expo_layout.setContentsMargins(0,0,0,0)
        expo_box.setLayout(expo_layout)
        expo_layout.addWidget(self.expo_dsb)
        expo_layout.addWidget(self.expo_unit_cb)
        cam_ctrl_frame.addRow("Exposure time:", expo_box)

        # Set gain (it's a simple double spinbox in dB)
        gain_cf = self.parent.defaults["gain"]
        gain_min = gain_cf.getfloat("min")
        gain_max = gain_cf.getfloat("max")
        self.gain_dsb = NewDoubleSpinBox(range=(gain_min, gain_max), decimals=1)
        self.gain_dsb.valueChanged[float].connect(lambda val: self.parent.device.set_gain(val))
        cam_ctrl_frame.addRow("Gain (dB):", self.gain_dsb)

        # set binning
        self.bin_horizontal = NewSpinBox(range=self.parent.device.BIN_RANGE)
        self.bin_vertical = NewSpinBox(range=self.parent.device.BIN_RANGE)
        self.bin_horizontal.setValue(self.parent.device.binning["horizontal"])
        self.bin_vertical.setValue(self.parent.device.binning["vertical"])
        self.bin_horizontal.valueChanged[int].connect(lambda val, text="hori", cb=self.bin_vertical: self.set_binning(text, val, cb.value()))
        self.bin_vertical.valueChanged[int].connect(lambda val, text="vert", cb=self.bin_horizontal: self.set_binning(text, cb.value(), val))
        bin_box = qt.QWidget()
        bin_box.setMaximumWidth(200)
        bin_layout = qt.QHBoxLayout()
        bin_layout.setContentsMargins(0,0,0,0)
        bin_box.setLayout(bin_layout)
        bin_layout.addWidget(self.bin_horizontal)
        bin_layout.addWidget(self.bin_vertical)
        cam_ctrl_frame.addRow("Binning H x V:", bin_box)

    # place gui elements related to TCP connection
    def place_tcp_control(self):
        tcp_ctrl_box = qt.QGroupBox("TCP Control")
        tcp_ctrl_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        tcp_ctrl_frame = qt.QFormLayout()
        tcp_ctrl_box.setLayout(tcp_ctrl_frame)
        self.frame.addWidget(tcp_ctrl_box)

        server_host = self.parent.defaults["tcp_connection"]["host_addr"]
        server_port = self.parent.defaults["tcp_connection"]["port"]
        self.server_addr_la = qt.QLabel(server_host+" ("+server_port+")")
        self.server_addr_la.setStyleSheet("QLabel{background-color: gray;}")
        self.server_addr_la.setToolTip("server = this PC")
        tcp_ctrl_frame.addRow("Server/This PC address:", self.server_addr_la)

        self.server_status_Lbl = qt.QLabel("No connection")
        self.server_status_Lbl.setStyleSheet("QLabel{background-color: gray;}")
        tcp_ctrl_frame.addRow("Server status:", self.server_status_Lbl)

        self.start_server_Btn = qt.QPushButton("Start server")
        self.start_server_Btn.clicked.connect(self.start_server_worker)

        self.stop_server_Btn = qt.QPushButton("Stop server")
        self.stop_server_Btn.clicked.connect(self.stop_server_worker)
        
        tcp_ctrl_frame.addRow(self.start_server_Btn, self.stop_server_Btn)

    # place save/load program configuration gui elements
    def place_save_load(self):
        self.save_load_box = qt.QGroupBox("Save/Load Settings")
        self.save_load_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        save_load_frame = qt.QFormLayout()
        self.save_load_box.setLayout(save_load_frame)
        self.frame.addWidget(self.save_load_box)

        self.file_name_le = qt.QLineEdit()
        default_file_name = self.parent.defaults["setting_save"]["file_name"]
        self.file_name_le.setText(default_file_name)
        save_load_frame.addRow("File name to save:", self.file_name_le)

        self.date_time_chb = qt.QCheckBox()
        self.date_time_chb.setTristate(False)
        date = self.parent.defaults["setting_save"].getboolean("append_time")
        self.date_time_chb.setChecked(date)
        self.date_time_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        save_load_frame.addRow("Auto append time:", self.date_time_chb)

        self.save_settings_bt = qt.QPushButton("save settings")
        self.save_settings_bt.setMaximumWidth(200)
        self.save_settings_bt.clicked[bool].connect(lambda val: self.save_settings())
        save_load_frame.addRow("Save settings:", self.save_settings_bt)

        self.load_settings_bt = qt.QPushButton("load settings")
        self.load_settings_bt.setMaximumWidth(200)
        self.load_settings_bt.clicked[bool].connect(lambda val: self.load_settings())
        save_load_frame.addRow("Load settings:", self.load_settings_bt)

    # start to take images
    def start(self, seq_info):
        """Start image acquisition."""
        self.active = True
        
        # 1. Clear all displays (delegate to ImageWin)
        self.image_win.clear_all()
        
        # 2. Clear control panel labels
        self.signal_count.setText("0")
        self.signal_count_mean.setText("0")
        self.signal_count_err_mean.setText("0")
        self.num_image.setText("0")
        self.amp.setText("0")
        self.offset.setText("0")
        self.x_mean.setText("0")
        self.x_stand_dev.setText("0")
        self.y_mean.setText("0")
        self.y_stand_dev.setText("0")
        self.peak.setText("0")
        
        # 3. Store sequence info
        self.seq_info = seq_info
        self.num_img_to_take_sb.setValue(self.seq_info["general"].getint("element_number"))
        
        # 4. Determine scan mode
        self.scan_elem_name = self.seq_info["general"].get("scanned_devices_parameters", "")
        self.scan_elem_name = self.scan_elem_name.split(",")[0].strip() if self.scan_elem_name else ""
        
        if self.scan_elem_name:
            self.control_mode = "scan"
            self.signal_count_scan_dict = {}
            self.image_win.scan_plot_widget.setLabel("bottom", self.scan_elem_name)
            self.image_win.ave_scan_tab.setCurrentIndex(1)
        else:
            self.control_mode = "record"
        
        # 5. Disable controls during acquisition
        self.enable_widgets(False)
        
        # 6. Set image tab based on measurement mode
        if self.meas_mode == "fluorescence":
            self.image_win.img_tab.setCurrentIndex(2)
        elif self.meas_mode == "absorption":
            self.image_win.img_tab.setCurrentIndex(3)
        else:
            logging.warning("Measurement mode not supported.")
            return
        
        # 7. Start acquisition thread
        self.rec = AcquisitionThread(
            self, 
            self.parent.defaults, 
            "Triggered", 
            seq_info, 
            abstract_camera=AlviumCameraWrapper(self.parent.device)
        )
        self.rec.update_signal.connect(self.img_ctrl_update)
        self.rec.finished.connect(self.stop)
        self.rec.start()

    # force to stop image taking
    def stop(self):
        if self.active:
            self.active = False
            try:
                self.rec.wait() #  wait until thread closed
            except AttributeError:
                pass

            # don't reset control_mode to None, bcause img_ctrl_update function for the last image may be called after this function being called
            # self.control_mode = None

            self.enable_widgets(True)

    
    def img_ctrl_update(self, data: ImageData):
        """
        Update GUI with new image data from acquisition thread.
        Now receives ImageData object instead of dict.
        """
        # 1. Update raw images (delegate to ImageWin)
        self.image_win.update_raw_images(data.atom, data.probe, data.dark)
        
        # 2. Select and update target image based on measurement mode
        if self.meas_mode == "fluorescence":
            target_image = data.sig_bkg
            self.image_win.update_target_image(target_image, "Signal minus ave bkg")
        elif self.meas_mode == "absorption":
            target_image = data.od
            self.image_win.update_target_image(target_image, "Optical density")
        else:
            logging.warning(f"Measurement type '{self.meas_mode}' not supported")
            return
        
        # 3. Update projections
        self.image_win.update_projections(target_image)
        
        # 4. Extract ROI and update ROI plots
        img_roi = target_image[self.roi["xmin"]:self.roi["xmax"], 
                            self.roi["ymin"]:self.roi["ymax"]]
        self.image_win.update_roi_projections(img_roi)
        
        # 5. Calculate signal count
        signal_count = np.sum(img_roi)
        self.num_image.setText(str(data.counter))
        self.signal_count.setText(f"{signal_count:.4e}")
        self.signal_count_deque.append(signal_count)
        self.image_win.update_running_signal(self.signal_count_deque)
        
        # 6. Update scan plot if in scan mode
        if self.control_mode == "scan" and data.is_scan:
            if not hasattr(self, "signal_count_scan_dict"):
                self.signal_count_scan_dict = {}
            if data.scan_value not in self.signal_count_scan_dict:
                self.signal_count_scan_dict[data.scan_value] = []
            self.signal_count_scan_dict[data.scan_value].append(signal_count)
            self.image_win.update_scan_plot(self.signal_count_scan_dict)
        
        # 7. Trigger Gaussian fit (in background thread)
        if self.gaussian_fit:
            img_roi_contiguous = np.ascontiguousarray(img_roi)
            worker = FitWorker(img_roi_contiguous, maxfev=50)
            worker.signals.finished.connect(self.on_fit_complete)
            self.fit_worker_pool.start(worker)
        else:
            self.image_win.update_fit_curves(None, None)


    @pyqtSlot(object)
    def on_fit_complete(self, fit_result: Optional[FitResult]):
        """Receive fit results and update display."""
        if fit_result:
            # Update labels (direct attribute access)
            self.amp.setText("{:.2f}".format(fit_result.amp))
            self.offset.setText("{:.2f}".format(fit_result.offset))
            self.x_mean.setText("{:.2f}".format(fit_result.x_mean + self.roi["xmin"]))
            self.x_stand_dev.setText("{:.2f}".format(fit_result.x_width))
            self.y_mean.setText("{:.2f}".format(fit_result.y_mean + self.roi["ymin"]))
            self.y_stand_dev.setText("{:.2f}".format(fit_result.y_width))
            self.peak.setText("{:.2f}".format(fit_result.peak))
            
            # Update fit curves (delegate to ImageWin)
            roi_bounds = (self.roi["xmin"], self.roi["xmax"], 
                        self.roi["ymin"], self.roi["ymax"])
            self.image_win.update_fit_curves(fit_result, roi_bounds)
        else:
            self.image_win.update_fit_curves(None, None)
    


    def enable_widgets(self, arg):
        # enable/disable controls
        # self.stop_bt.setEnabled(not arg)
        # self.record_bt.setEnabled(arg)
        # self.scan_bt.setEnabled(arg)
        for rb in self.meas_rblist:
            rb.setEnabled(arg)

        self.num_img_to_take_sb.setEnabled(arg)
        # self.gauss_fit_chb.setEnabled(arg)
        self.img_save_chb.setEnabled(arg)
        self.run_name_le.setEnabled(arg)
        self.cam_ctrl_box.setEnabled(arg)
        self.save_load_box.setEnabled(arg)

        # enable/disable in image ROI selection
        # self.x_min_sb.setEnabled(arg)
        # self.x_max_sb.setEnabled(arg)
        # self.y_min_sb.setEnabled(arg)
        # self.y_max_sb.setEnabled(arg)
        # for key, roi in self.parent.image_win.img_roi_dict.items():
        #     roi.setEnabled(arg)
        # self.parent.image_win.x_plot_lr.setMovable(arg)
        # self.parent.image_win.y_plot_lr.setMovable(arg)

        # force GUI to respond now
        self.parent.app.processEvents()

    def set_num_img(self, val):
        self.num_img_to_take = val

    def set_roi(self, text, val, sb):
        if text == "xmin":
            sb.setMinimum(val+1)
        elif text == "xmax":
            sb.setMaximum(val-1)
        elif text == "ymin":
            sb.setMinimum(val+1)
        elif text == "ymax":
            sb.setMaximum(val-1)

        self.roi[text] = val

        # set in image ROI selection boxes position/size
        x_range = self.roi["xmax"]-self.roi["xmin"]
        y_range = self.roi["ymax"]-self.roi["ymin"]
        for key, roi in self.parent.image_win.img_roi_dict.items():
            roi.setPos(pos=(self.roi["xmin"], self.roi["ymin"]))
            roi.setSize(size=(x_range, y_range))
        self.parent.image_win.x_plot_lr.setRegion((self.roi["xmin"], self.roi["xmax"]))
        self.parent.image_win.y_plot_lr.setRegion((self.roi["ymin"], self.roi["ymax"]))

        # disable 2D gaussian fit if ROI is too larges
        if x_range*y_range > self.cpu_limit:
            if self.gauss_fit_chb.isEnabled():
                self.gauss_fit_chb.setChecked(False)
                self.gauss_fit_chb.setEnabled(False)
        else:
            if not self.gauss_fit_chb.isEnabled():
                self.gauss_fit_chb.setEnabled(True)

    def set_gauss_fit(self, state):
        self.gaussian_fit = bool(state)

    def set_gauss_filter(self, val, param):
        if param == "state":
            self.gaussian_filter = bool(val)
        elif param == "sigma":
            self.gaussian_filter_sigma = val
        else:
            logging.warning(f"Unsupported guassian filter setting: {param}.")

    def set_img_save(self, state):
        self.img_save = state

    def set_expo_time(self, time, unit, change_type):
        unit_num = self.parent.defaults["expo_unit"].getfloat(unit)
        minimum = self.parent.defaults["expo_time"].getfloat("min")
        maximum = self.parent.defaults["expo_time"].getfloat("max")
        d = self.parent.defaults["expo_time"].getint("decimals")
        if change_type == "unit":
            self.expo_dsb.setRange(minimum/unit_num, maximum/unit_num)
            self.expo_dsb.setDecimals(int(d+np.log10(unit_num)))
        elif change_type == "time":
            pass
        else:
            logging.warning("set_expo_time: invalid change_type")
            return

        t = time*unit_num
        t = max(t, minimum)
        t = min(t, maximum)
        self.parent.device.set_expo_time(t)

    def set_binning(self, text, bin_h, bin_v):
        self.parent.device.set_binning(bin_h, bin_v)

        # set bounds for ROI spinboxes
        if text == "hori":
            x_max = self.parent.device.image_shape["xmax"]
            self.x_max_sb.setMaximum(int(x_max))
        elif text == "vert":
            y_max = self.parent.device.image_shape["ymax"]
            self.y_max_sb.setMaximum(int(y_max))
        else:
            logging.warning("Binning type not supported.")

        # set boundaries for in image ROI selections
        for key, roi in self.parent.image_win.img_roi_dict.items():
            roi.setBounds(pos=[0,0], size=[self.parent.device.image_shape["xmax"], self.parent.device.image_shape["ymax"]])
        self.parent.image_win.x_plot_lr.setBounds([0, self.parent.device.image_shape["xmax"]])
        self.parent.image_win.y_plot_lr.setBounds([0, self.parent.device.image_shape["ymax"]])

    def start_server_worker(self):
        """Start the server in a background thread"""
        
        # Create thread and worker
        self.server_thread = QThread()
        self.server_worker = ServerWorker(self.parent)
        
        # Move worker to thread
        self.server_worker.moveToThread(self.server_thread)
        
        # Connect worker signals to slots
        self.server_worker.connection_status.connect(self.server_status_Lbl.setText)
        self.server_worker.data_received.connect(self.server_worker.process_received_objects)   # Important: this has to be done after moving the worker to the thread
        self.server_worker.start_signal.connect(self.start)
        self.server_worker.stop_signal.connect(self.stop)
        
        # Create timers in main thread for worker
        self.create_timers_for_server_worker()
        
        # Connect thread signals
        self.server_thread.started.connect(self.server_worker.start_server)
        self.server_worker.finished.connect(self.server_thread.quit)
        self.server_worker.finished.connect(self.server_worker.deleteLater)
        self.server_thread.finished.connect(self.server_thread.deleteLater)
        self.server_stop.connect(self.server_worker.stop_server)
        
        # Start the thread
        self.server_thread.start()
        self.cmd_timer.start()
        self.update_status_timer.start()

        self.start_server_Btn.setEnabled(False)
        self.stop_server_Btn.setEnabled(True)

    
    def create_timers_for_server_worker(self):
        """Create timers in main thread that call worker methods"""
        self.cmd_timer = QTimer()
        self.update_status_timer = QTimer()
        
        self.cmd_timer.setInterval(int(self.server_worker.cmd_interval * 1000))
        self.update_status_timer.setInterval(int(self.server_worker.update_status_interval * 1000))
        
        self.cmd_timer.timeout.connect(self.server_worker.process_commands)
        self.update_status_timer.timeout.connect(self.server_worker.update_server_status)
        
    
    def stop_server_worker(self):
        """Stop the server by sending command to worker"""
        if getattr(self, "server_worker", None) and is_qt_object_alive(self.server_worker):
            self.cmd_timer.stop()
            self.cmd_timer.timeout.disconnect()
            self.cmd_timer.deleteLater()

            self.update_status_timer.stop()
            self.update_status_timer.disconnect()
            self.update_status_timer.deleteLater()

            self.server_stop.emit()

            self.start_server_Btn.setEnabled(True) 
            self.stop_server_Btn.setEnabled(False)


    def save_settings(self, latest=False):
        if latest:
            file_name = "program_setting_latest.ini"
        else:
        # compile file name
            file_name = ""
            if self.file_name_le.text():
                file_name += self.file_name_le.text()
            if self.date_time_chb.isChecked():
                if file_name != "":
                    file_name += "_"
                file_name += time.strftime("%Y%m%d_%H%M%S")
            file_name += ".ini"
            file_name = r"saved_settings/"+file_name

            # check if the file name already exists
            if os.path.exists(file_name):
                overwrite = qt.QMessageBox.warning(self, 'File name exists',
                                                'File name already exists. Continue to overwrite it?',
                                                qt.QMessageBox.Yes | qt.QMessageBox.No,
                                                qt.QMessageBox.No)
                if overwrite == qt.QMessageBox.No:
                    return

        config = configparser.ConfigParser()
        config.optionxform = str

        config["record_control"] = {}
        config["record_control"]["meas_mode"] = self.meas_mode

        config["image_control"] = {}
        config["image_control"]["num_image"] = str(self.num_img_to_take_sb.value())
        config["image_control"]["xmin"] = str(self.x_min_sb.value())
        config["image_control"]["xmax"] = str(self.x_max_sb.value())
        config["image_control"]["ymin"] = str(self.y_min_sb.value())
        config["image_control"]["ymax"] = str(self.y_max_sb.value())
        config["image_control"]["2D_gaussian_fit"] = str(self.gaussian_fit)
        config["image_control"]["run_name"] = self.run_name_le.text()
        config["image_control"]["image_auto_save"] = str(self.img_save_chb.isChecked())
        config["image_control"]["gaussian_filter"] = str(self.gaussian_filter)
        config["image_control"]["gaussian_filter_sigma"] = str(self.gaussian_filter_sigma)
        for name in self.parent.image_win.imgs_name:
            config["image_control"][f"auto_scale_state_{name}"] = str(self.parent.image_win.auto_scale_state_dict[name])
        config["image_control"]["auto_scale_state_Average_image"] = str(self.parent.image_win.ave_img_auto_scale_state)
        
        config["camera_control"] = {}
        for i in self.trig_mode_rblist:
            if i.isChecked():
                t = i.text()
                break
        config["camera_control"]["trigger_mode"] = t
        config["camera_control"]["exposure_time"] = str(self.expo_dsb.value())
        config["camera_control"]["exposure_unit"] = self.expo_unit_cb.currentText()
        config["camera_control"]["gain"] = str(self.gain_dsb.value())
        config["camera_control"]["binning_horizontal"] = str(self.bin_horizontal.value())
        config["camera_control"]["binning_vertical"] = str(self.bin_vertical.value())

        config["tcp_control"] = self.parent.defaults["tcp_connection"]

        configfile = open(file_name, "w")
        config.write(configfile)
        configfile.close()

    def load_settings(self, latest=False):
        if latest:
            try:
                config = configparser.ConfigParser()
                config.read("program_setting_latest.ini")
            except KeyError:
                # could not find file
                return
        else:
            # open a file dialog to choose a configuration file to load
            file_name, _ = qt.QFileDialog.getOpenFileName(self, "Load settigns", "saved_settings/", "All Files (*);;INI File (*.ini)")
            if not file_name:
                return

            config = configparser.ConfigParser()
            config.read(file_name)

        for i in self.meas_rblist:
            if i.text() == config["record_control"]["meas_mode"]:
                i.setChecked(True)
                break

        self.num_img_to_take_sb.setValue(config["image_control"].getint("num_image"))
        # the spinbox emits 'valueChanged' signal, and its connected function will be called
        self.x_min_sb.setValue(config["image_control"].getint("xmin"))
        self.x_max_sb.setValue(config["image_control"].getint("xmax"))
        self.y_min_sb.setValue(config["image_control"].getint("ymin"))
        self.y_max_sb.setValue(config["image_control"].getint("ymax"))
        # make sure image range is updated BEFORE gauss_fit_chb
        self.gauss_fit_chb.setChecked(config["image_control"].getboolean("2d_gaussian_fit"))
        # the combobox emits 'stateChanged' signal, and its connected function will be called
        self.img_save_chb.setChecked(config["image_control"].getboolean("image_auto_save"))
        self.run_name_le.setText(config["image_control"].get("run_name"))

        self.gauss_filter_chb.setChecked(config["image_control"].getboolean("gaussian_filter"))
        self.gaussian_filter_sigma_dsb.setValue(config["image_control"].getfloat("gaussian_filter_sigma"))

        for name in self.parent.image_win.imgs_name:
            self.parent.image_win.auto_scale_chb_dict[name].setChecked(config["image_control"].getboolean(f"auto_scale_state_{name}"))
        self.parent.image_win.ave_img_auto_scale_chb.setChecked(config["image_control"].getboolean("auto_scale_state_Average_image"))

        for i in self.trig_mode_rblist:
            if i.text() == config["camera_control"]["trigger_mode"]:
                i.setChecked(True)
                break

        # make sure expo_unit_cb changes before time, because it changes expo_dsb range
        self.expo_unit_cb.setCurrentText(config["camera_control"]["exposure_unit"])
        self.expo_dsb.setValue(config["camera_control"].getfloat("exposure_time"))

        self.gain_dsb.setValue(config["camera_control"].getfloat("gain"))

        self.bin_horizontal.setValue(int(config["camera_control"].get("binning_horizontal")))
        self.bin_vertical.setValue(int(config["camera_control"].get("binning_vertical")))

        self.parent.defaults["tcp_connection"] = config["tcp_control"]
        server_host = self.parent.defaults["tcp_connection"]["host_addr"]
        server_port = self.parent.defaults["tcp_connection"]["port"]
        self.server_addr_la.setText(server_host+" ("+server_port+")")
        self.start_server_worker()

    def set_meas_mode(self, text, checked):
        if checked:
            self.meas_mode = text


def is_qt_object_alive(obj):
    """A reliable way to check if Qt object truly exists"""
    try:
        return not PyQt5.sip.isdeleted(obj) if obj is not None else False
    except Exception as e:
        logging.error(f"Error checking if Qt object is alive: {e}")
        return False