# ==========================================
# GUI_modules/image_window.py
# ==========================================
"""Image window class - handles all image display and plotting."""

import logging
from typing import Optional, Dict, List
from collections import deque

import numpy as np
import pyqtgraph as pg
import PyQt5.QtWidgets as qt

from widgets import Scrollarea, imageWidget
from GUI_modules.data_models import FitResult
from GUI_modules.gaussianfit import gaussian

# TYPE_CHECKING prevents circular imports at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import CameraGUI


# the class that places images and plots
class ImageWin(Scrollarea):
    def __init__(self, parent):
        super().__init__(parent, label="Images", type="grid")
        self.frame.setColumnStretch(0,7)
        self.frame.setColumnStretch(1,4)
        self.frame.setRowStretch(0,1)
        self.frame.setRowStretch(1,1)
        self.frame.setRowStretch(2,1)
        self.frame.setContentsMargins(0,0,0,0)
        self.imgs_dict = {}
        self.img_roi_dict = {}
        self.auto_scale_chb_dict = {}
        self.auto_scale_state_dict = {}
        self.imgs_name = ["Background", "Raw Signal", "Dark", "Signal minus ave bkg", "Optical density"]

        for name in self.imgs_name:
            self.auto_scale_state_dict[name] = self.parent.defaults.getboolean("image_auto_scale", name)
        self.ave_img_auto_scale_state = self.parent.defaults.getboolean("image_auto_scale", "Average image")

        # place images and plots
        self.place_sgn_imgs()
        self.place_axis_plots()

        self.ave_scan_tab = qt.QTabWidget()
        self.frame.addWidget(self.ave_scan_tab, 2, 0)
        self.place_ave_image()
        self.place_scan_plot()

        self.place_sc_plot()

    # place background and signal images
    def place_sgn_imgs(self):
        self.img_tab = qt.QTabWidget()
        self.frame.addWidget(self.img_tab, 0, 0, 2, 1)
        for i, name in enumerate(self.imgs_name):
            imgwidget = imageWidget(parent=self, name=name, include_ROI=True, colorname="viridis", 
                                    dummy_data_xmax=self.parent.device.image_shape["xmax"],
                                    dummy_data_ymax=self.parent.device.image_shape["ymax"],
                                    )

            # add the widget to the front panel
            self.img_tab.addTab(imgwidget.graphlayout, " "+name+" ")

            # config ROI
            imgwidget.img_roi.setPos(pos=(self.parent.defaults["roi"].getint("xmin"), self.parent.defaults["roi"].getint("ymin")))
            imgwidget.img_roi.setSize(size=(self.parent.defaults["roi"].getint("xmax")-self.parent.defaults["roi"].getint("xmin"),
                                            self.parent.defaults["roi"].getint("ymax")-self.parent.defaults["roi"].getint("ymin")))
            imgwidget.img_roi.sigRegionChanged.connect(lambda roi=imgwidget.img_roi: self.img_roi_update(roi))
            imgwidget.img_roi.setBounds(pos=[0,0], size=[self.parent.device.image_shape["xmax"], self.parent.device.image_shape["ymax"]])

            imgwidget.chb.setChecked(self.auto_scale_state_dict[name])
            imgwidget.chb.stateChanged[int].connect(lambda val, param=name: self.set_auto_scale(val, param))

            self.img_roi_dict[name] = imgwidget.img_roi
            self.imgs_dict[name] = imgwidget.img
            self.auto_scale_chb_dict[name] = imgwidget.chb
        
        self.starting_data = imgwidget.dummy_data

        self.img_tab.setCurrentIndex(2) # make tab #2 (count from 0) to show as default

    # place plots of signal_count along one axis
    def place_axis_plots(self):
        tickstyle = {"showValues": False}

        self.curve_tab = qt.QTabWidget()
        self.frame.addWidget(self.curve_tab, 0, 1, 2, 1)

        # place plot of signal_count along x axis
        x_data = np.sum(self.starting_data, axis=1)
        graphlayout = pg.GraphicsLayoutWidget(parent=self, border=True)
        self.curve_tab.addTab(graphlayout, " Full Frame Signal ")
        x_plot = graphlayout.addPlot(title="Signal count v.s. X")
        x_plot.showGrid(True, True)
        x_plot.setLabel("top")
        # x_plot.getAxis("top").setTicks([])
        x_plot.getAxis("top").setStyle(**tickstyle)
        x_plot.setLabel("right")
        # x_plot.getAxis("right").setTicks([])
        x_plot.getAxis("right").setStyle(**tickstyle)
        self.x_plot_curve = x_plot.plot(x_data)

        # add ROI selection
        self.x_plot_lr = pg.LinearRegionItem([self.parent.defaults["roi"].getint("xmin"),
                                                self.parent.defaults["roi"].getint("xmax")], swapMode="block")
        # no "snap" option for LinearRegion item?
        self.x_plot_lr.setBounds([0, self.parent.device.image_shape["xmax"]])
        x_plot.addItem(self.x_plot_lr)
        self.x_plot_lr.sigRegionChanged.connect(self.x_plot_lr_update)

        graphlayout.nextRow()

        # place plot of signal_count along y axis
        y_data = np.sum(self.starting_data, axis=0)
        y_plot = graphlayout.addPlot(title="Signal count v.s. Y")
        y_plot.showGrid(True, True)
        y_plot.setLabel("top")
        y_plot.getAxis("top").setStyle(**tickstyle)
        y_plot.setLabel("right")
        y_plot.getAxis("right").setStyle(**tickstyle)
        self.y_plot_curve = y_plot.plot(y_data)

        # add ROI selection
        self.y_plot_lr = pg.LinearRegionItem([self.parent.defaults["roi"].getint("ymin"),
                                                self.parent.defaults["roi"].getint("ymax")], swapMode="block")
        self.y_plot_lr.setBounds([0, self.parent.device.image_shape["ymax"]])
        y_plot.addItem(self.y_plot_lr)
        self.y_plot_lr.sigRegionChanged.connect(self.y_plot_lr_update)

        graphlayout = pg.GraphicsLayoutWidget(parent=self, border=True)
        self.curve_tab.addTab(graphlayout, " Signal in ROI ")

        x_plot = graphlayout.addPlot(title="Signal count v.s. X")
        x_plot.showGrid(True, True)
        x_plot.setLabel("top")
        # x_plot.getAxis("top").setTicks([])
        x_plot.getAxis("top").setStyle(**tickstyle)
        x_plot.setLabel("right")
        # x_plot.getAxis("right").setTicks([])
        x_plot.getAxis("right").setStyle(**tickstyle)
        data_roi = self.starting_data[self.parent.defaults["roi"].getint("xmin"):self.parent.defaults["roi"].getint("xmax"),
                                        self.parent.defaults["roi"].getint("ymin"):self.parent.defaults["roi"].getint("ymax")]
        x_data = np.sum(data_roi, axis=1)
        self.x_plot_roi_curve = x_plot.plot(x_data)
        self.x_plot_roi_fit_curve = x_plot.plot(np.array([]))

        graphlayout.nextRow()

        # place plot of signal_count along y axis
        y_plot = graphlayout.addPlot(title="Signal count v.s. Y")
        y_plot.showGrid(True, True)
        y_plot.setLabel("top")
        y_plot.getAxis("top").setStyle(**tickstyle)
        y_plot.setLabel("right")
        y_plot.getAxis("right").setStyle(**tickstyle)
        y_data = np.sum(data_roi, axis=0)
        self.y_plot_roi_curve = y_plot.plot(y_data)
        self.y_plot_roi_fit_curve = y_plot.plot(np.array([]))

    # place averaged image
    def place_ave_image(self):
        name = "Average image"
        imgwidget = imageWidget(parent=self, name=name, include_ROI=False, colorname="viridis", 
                                dummy_data_xmax=self.parent.device.image_shape["xmax"],
                                dummy_data_ymax=self.parent.device.image_shape["ymax"],
                                )

        self.ave_scan_tab.addTab(imgwidget.graphlayout, " "+name+" ")
        self.ave_img = imgwidget.img
        self.ave_img_auto_scale_chb = imgwidget.chb
        self.ave_img_auto_scale_chb.setChecked(self.ave_img_auto_scale_state)
        self.ave_img_auto_scale_chb.stateChanged[int].connect(lambda val, param="Average image": self.set_auto_scale(val, param))

    # place scan plots
    def place_scan_plot(self):
        tickstyle = {"showValues": False}

        self.scan_plot_widget = pg.PlotWidget(title="Signal count v.s. Scan param.")
        self.scan_plot_widget.showGrid(True, True)
        self.scan_plot_widget.setLabel("top")
        self.scan_plot_widget.getAxis("top").setStyle(**tickstyle)
        self.scan_plot_widget.setLabel("right")
        self.scan_plot_widget.getAxis("right").setStyle(**tickstyle)
        fontstyle = {"color": "#919191", "font-size": "11pt"}
        self.scan_plot_widget.setLabel("bottom", "Scan parameter", **fontstyle)
        self.scan_plot_widget.getAxis("bottom").enableAutoSIPrefix(False)
        self.scan_plot_curve = self.scan_plot_widget.plot()

        # place error bar
        self.scan_plot_errbar = pg.ErrorBarItem()
        self.scan_plot_widget.addItem(self.scan_plot_errbar)

        self.ave_scan_tab.addTab(self.scan_plot_widget, " Scan Plot ")

    # place a plot showing running signal count
    def place_sc_plot(self):
        tickstyle = {"showValues": False}

        self.sc_plot_widget = pg.PlotWidget(title="Signal count")
        self.sc_plot_widget.showGrid(True, True)
        self.sc_plot_widget.setLabel("top")
        self.sc_plot_widget.getAxis("top").setStyle(**tickstyle)
        self.sc_plot_widget.setLabel("right")
        self.sc_plot_widget.getAxis("right").setStyle(**tickstyle)
        self.sc_plot_curve = self.sc_plot_widget.plot()

        self.frame.addWidget(self.sc_plot_widget, 2, 1)

    # set ROI in background/signal imgaes
    def img_roi_update(self, roi):
        x_min = roi.pos()[0]
        y_min = roi.pos()[1]
        x_max = x_min + roi.size()[0]
        y_max = y_min + roi.size()[1]

        self.parent.control.x_min_sb.setValue(round(x_min))
        self.parent.control.x_max_sb.setValue(round(x_max))
        self.parent.control.y_min_sb.setValue(round(y_min))
        self.parent.control.y_max_sb.setValue(round(y_max))

    # set ROI in the plot of signal count along x-axis
    def x_plot_lr_update(self):
        x_min = self.x_plot_lr.getRegion()[0]
        x_max = self.x_plot_lr.getRegion()[1]

        self.parent.control.x_min_sb.setValue(round(x_min))
        self.parent.control.x_max_sb.setValue(round(x_max))

    # set ROI in the plot of signal count along y-axis
    def y_plot_lr_update(self):
        y_min = self.y_plot_lr.getRegion()[0]
        y_max = self.y_plot_lr.getRegion()[1]

        self.parent.control.y_min_sb.setValue(round(y_min))
        self.parent.control.y_max_sb.setValue(round(y_max))

    def set_auto_scale(self, val, param):
        # logging.info(str(val))
        if param == "Average image":
            self.ave_img_auto_scale_state = bool(val)
        elif param in self.imgs_name:
            self.auto_scale_state_dict[param] = bool(val)
        else:
            logging.warning(f"Unsupported auto scale param: {param}.")

    # ==========================================
    # Image Update Methods
    # ==========================================
    
    def update_raw_images(self, atom: np.ndarray, probe: np.ndarray, dark: np.ndarray):
        """Update the three raw camera images."""
        self.imgs_dict["Raw Signal"].setImage(atom, autoLevels=self.auto_scale_state_dict["Raw Signal"])
        self.imgs_dict["Background"].setImage(probe, autoLevels=self.auto_scale_state_dict["Background"])
        self.imgs_dict["Dark"].setImage(dark, autoLevels=self.auto_scale_state_dict["Dark"])
    
    def update_target_image(self, image: np.ndarray, name: str):
        """Update the processed target image (fluorescence or OD)."""
        self.imgs_dict[name].setImage(image, autoLevels=self.auto_scale_state_dict[name])
    
    def update_projections(self, image: np.ndarray):
        """Update full-frame X/Y projection plots."""
        self.x_plot_curve.setData(np.sum(image, axis=1))
        self.y_plot_curve.setData(np.sum(image, axis=0))
    
    def update_roi_projections(self, roi_image: np.ndarray):
        """Update ROI projection plots."""
        self.x_plot_roi_curve.setData(np.sum(roi_image, axis=1))
        self.y_plot_roi_curve.setData(np.sum(roi_image, axis=0))
    
    def update_fit_curves(self, fit_result: Optional[FitResult], roi_bounds: tuple):
        """Update Gaussian fit curves on ROI plots."""
        if fit_result:
            xmin, xmax, ymin, ymax = roi_bounds
            xy = np.indices((xmax - xmin, ymax - ymin))
            fit = gaussian(fit_result.amp, fit_result.x_mean, fit_result.y_mean,
                        fit_result.x_width, fit_result.y_width, fit_result.offset)(*xy)
            self.x_plot_roi_fit_curve.setData(np.sum(fit, axis=1), pen=pg.mkPen('r'))
            self.y_plot_roi_fit_curve.setData(np.sum(fit, axis=0), pen=pg.mkPen('r'))
        else:
            self.x_plot_roi_fit_curve.setData(np.array([]))
            self.y_plot_roi_fit_curve.setData(np.array([]))
    
    def update_running_signal(self, signal_deque: deque):
        """Update the running signal count plot."""
        self.sc_plot_curve.setData(np.array(signal_deque), symbol='o')
    
    def update_scan_plot(self, scan_data: Dict[float, List[float]]):
        """Update the scan parameter vs signal count plot."""
        if not scan_data:
            return
        
        x, y, err = [], [], []
        for param, sc_list in scan_data.items():
            x.append(float(param))
            y.append(np.mean(sc_list))
            err.append(np.std(sc_list) / np.sqrt(len(sc_list)))
        
        x, y, err = np.array(x), np.array(y), np.array(err)
        order = x.argsort()
        x, y, err = x[order], y[order], err[order]
        
        self.scan_plot_curve.setData(x, y, symbol='o')
        beam_width = (x[-1] - x[0]) / len(x) * 0.2 if len(x) > 1 else 0.1
        self.scan_plot_errbar.setData(x=x, y=y, top=err, bottom=err, 
                                      beam=beam_width, pen=pg.mkPen('w', width=1.2))
    
    def clear_all(self):
        """Clear all images and plots (called at start of acquisition)."""
        img = np.zeros((self.parent.device.image_shape["xmax"], 
                       self.parent.device.image_shape["ymax"]))
        for key, image_show in self.imgs_dict.items():
            image_show.setImage(img, autoLevels=self.auto_scale_state_dict[key])
        self.x_plot_curve.setData(np.sum(img, axis=1))
        self.y_plot_curve.setData(np.sum(img, axis=0))
        self.ave_img.setImage(img, autoLevels=self.ave_img_auto_scale_chb.isChecked())
        self.x_plot_roi_fit_curve.setData(np.array([]))
        self.y_plot_roi_fit_curve.setData(np.array([]))