from collections import deque
from contextlib import ExitStack
import logging
import traceback

import pco

import numpy as np
from vmbpy import PixelFormat, VmbSystem

class Alvium:
    """
    Interface to Allied Vision's Alvium cameras, using the Vimba X SDK.

    Due to the supremely irritating structure of the SDK, almost all calls to the camera must be executed inside two
    with statements, one for the SDK singleton and one for the camera itself. In particular, leaving the camera's
    context manager block stops image acquisition. This makes it almost impossible to abstract away this implementation
    detail. Additionally, acquiring the vmbpy singleton is quite slow. Thanks, Allied Vision!

    To deal with this limitation, code that wants to acquire images needs to use the class this way::

        cam = Alvium("Camera ID")
        with cam.start():
            image = cam.read_image()
    """

    # Horizontal and vertical binning range, px
    BIN_MIN = 1
    BIN_MAX = 8
    BIN_RANGE = (BIN_MIN, BIN_MAX)

    def __init__(self, camera_id):
        self.trigger_mode = "software"
        self.sensor_format = ""

        self.frame_queue = deque()

        with VmbSystem.get_instance() as vmb:
            self.cam = vmb.get_camera_by_id(camera_id)

            with self.cam:
                self.get_image_shape()
                self.binning = {"horizontal": self.cam.BinningHorizontal.get(), "vertical": self.cam.BinningVertical.get()}

                self.cam.AcquisitionMode.set("Continuous")
                self.cam.TriggerMode.set("On")
                self.cam.TriggerSelector.set("FrameStart")
                self.cam.TriggerSource.set("Software")

                self.cam.BinningHorizontalMode.set("Sum") # Setting horizontal binning mode also sets vertical binning mode

                self.cam.set_pixel_format(PixelFormat.Mono12) # Use full bit depth

    def start(self):
        # 1. Create an empty stack
        vmb_contexts = ExitStack()

        # 2. Force the VmbSystem and the Camera to open, and tell
        # the stack to hold the door open for us.
        vmb_contexts.enter_context(VmbSystem.get_instance())
        vmb_contexts.enter_context(self.cam)

        # 3. Start dumping images into the background queue
        self.cam.start_streaming(self.queue_frame)

        # 4. Tell the stack: "Hey, whenever you DO finally close,
        # please make sure to run stop_streaming() first."
        vmb_contexts.callback(self.cam.stop_streaming)

        # 5. Hand this open stack back to whoever called start()
        return vmb_contexts

    def queue_frame(self, cam, stream, frame):
        image = frame.as_numpy_ndarray().copy() # vmbpy will reuse the underlying frame object, so copy out the data
        image = np.squeeze(image) # Remove length-1 axes
        self.frame_queue.append(image)
        cam.queue_frame(frame)

    def set_trigger_mode(self, text, checked):
        if checked:
            with VmbSystem.get_instance(), self.cam:
                self.trigger_mode = text
                if text == "software":
                    self.cam.TriggerSource.set("Software")
                elif text == "external TTL":
                    self.cam.TriggerSource.set("Line0")

    def set_expo_time(self, expo_time):
        with VmbSystem.get_instance(), self.cam:
            self.cam.ExposureTime.set(expo_time * 1e6)

    def set_gain(self, gain):
        with VmbSystem.get_instance(), self.cam:
            if gain < 0 or gain > 47.9:
                logging.warning(f"Gain value {gain} is out of range (0-47.9 dB). Gain not set.")
            else:
                self.cam.Gain.set(gain)

    def get_image_shape(self):
        with VmbSystem.get_instance(), self.cam:
            self.image_shape = {"xmax": self.cam.Width.get(), "ymax": self.cam.Height.get()}

    def set_binning(self, bin_h, bin_v):
        if not bin_h in range(self.BIN_MIN, self.BIN_MAX + 1) and bin_v in range(self.BIN_MIN, self.BIN_MAX + 1):
            raise ValueError(f"Binning must be between 1 and 8, was ({bin_h}, {bin_v})")

        with VmbSystem.get_instance(), self.cam:
            self.cam.BinningHorizontal.set(bin_h)
            self.cam.BinningVertical.set(bin_v)

        self.get_image_shape()

    def num_images_available(self):
        return len(self.frame_queue)

    def software_trigger(self):
        with VmbSystem.get_instance(), self.cam:
            self.cam.TriggerSoftware.run()

    def stop(self):
        with VmbSystem.get_instance(), self.cam:
            self.cam.stop_streaming()

    def read_image(self):
        if len(self.frame_queue) == 0:
            raise RuntimeError("No images available")
        return self.frame_queue.popleft()

# the class that handles camera interface (except taking images) and configuration
class pixelfly:
    def __init__(self, parent):
        self.parent = parent

        try:
            # due to some unknow issues in computer IO and the way pco package is coded,
            # an explicit assignment to "interface" keyword is required
            self.cam = pco.Camera(interface='USB 2.0')
        except Exception as err:
            logging.error(traceback.format_exc())
            logging.error("Can't open camera")
            return

        # initialize camera
        self.set_sensor_format(self.parent.defaults["sensor_format"]["default"])
        self.set_clock_rate(self.parent.defaults["clock_rate"]["default"])
        self.set_conv_factor(self.parent.defaults["conv_factor"]["default"])
        self.set_trigger_mode(self.parent.defaults["trigger_mode"]["default"], True)
        self.set_expo_time(self.parent.defaults["expo_time"].getfloat("default"))
        self.set_binning(self.parent.defaults["binning"].getint("horizontal_default"),
                        self.parent.defaults["binning"].getint("vertical_default"))
        self.set_image_shape()
        self.set_record_mode()

    def set_sensor_format(self, arg):
        self.sensor_format = arg
        format_cam = self.parent.defaults["sensor_format"][arg]
        self.cam.sdk.set_sensor_format(format_cam)
        self.cam.sdk.arm_camera()
        # print(f"sensor format = {arg}")

    def set_clock_rate(self, arg):
        rate = self.parent.defaults["clock_rate"].getint(arg)
        self.cam.configuration = {"pixel rate": rate}
        # print(f"clock rate = {arg}")

    # conversion factor, which is 1/gain or number of electrons/count
    def set_conv_factor(self, arg):
        conv = self.parent.defaults["conv_factor"].getint(arg)
        self.cam.sdk.set_conversion_factor(conv)
        self.cam.sdk.arm_camera()
        # print(f"conversion factor = {arg}")

    def set_trigger_mode(self, text, checked):
        if checked:
            self.trigger_mode = text
            mode_cam = self.parent.defaults["trigger_mode"][text]
            self.cam.configuration = {"trigger": mode_cam}
            # print(f"trigger source = {arg}")

    def set_expo_time(self, expo_time):
        self.cam.configuration = {'exposure time': expo_time}
        # print(f"exposure time (in seconds) = {expo_time}")

    # 4*4 binning at most
    def set_binning(self, bin_h, bin_v):
        self.binning = {"horizontal": int(bin_h), "vertical": int(bin_v)}
        self.cam.configuration = {'binning': (self.binning["horizontal"], self.binning["vertical"])}
        # print(f"binning = {bin_h} (horizontal), {bin_v} (vertical)")

    # image size of camera returned image, depends on sensor format and binning
    def set_image_shape(self):
        format_str = self.sensor_format + " absolute_"
        self.image_shape = {"xmax": int(self.parent.defaults["sensor_format"].getint(format_str+"xmax")/self.binning["horizontal"]),
                            "ymax": int(self.parent.defaults["sensor_format"].getint(format_str+"ymax")/self.binning["vertical"])}

    def num_images_available(self):
        return self.cam.rec.get_status()["dwProcImgCount"]

    def software_trigger(self):
        self.cam.sdk.force_trigger()

    def set_record_mode(self):
        self.cam.record(number_of_images=4, mode="ring buffer") # number_of_images is buffer size in ring buffer mode, and has to be at least 4

    def stop(self):
        self.cam.stop()

    def read_latest_image(self):
        return self.cam.image(image_index=0xFFFFFFFF)

    def close(self):
        self.cam.close()


if __name__ == "__main__":
    import sys

    # 1. Safety Check: Close any existing camera session from a previous run
    #    If you run this script twice without closing, Vimba will throw "Device Busy"
    if 'cam_stack' in globals() and globals()['cam_stack'] is not None:
        print("Closing previous camera session...")
        globals()['cam_stack'].close()

    if 'cam' in globals() and globals()['cam'] is not None:
        globals()['cam'] = None

    # 2. Initialize Camera
    #    You can hardcode an ID here, or try to find the first available one
    camera_id = "DEV_1AB22C069010"  # Empty string usually finds the first available camera
    try:
        print(f"Initializing Alvium camera (ID: '{camera_id}')...")
        cam = Alvium(camera_id)
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        print("Make sure the camera is connected and Vimba drivers are installed.")
        sys.exit(1)

    # 3. Start Streaming
    #    This returns the ExitStack that keeps the camera context alive
    print("Starting image acquisition stream...")
    cam_stack = cam.start()

    # 4. Inject into Globals
    #    This is the magic trick for Pyzo/IDE shells. It ensures 'cam' and 'cam_stack'
    #    remain accessible in the console after the script finishes running.
    globals()['cam'] = cam
    globals()['cam_stack'] = cam_stack

    # 5. Print Interactive Help
    print("\n" + "="*50)
    print(" CAMERA READY FOR INTERACTIVE TESTING")
    print("="*50)
    print("The following variables are now available in your shell:")
    print("  - cam        : The Alvium interface object")
    print("  - cam_stack  : The context manager (keep this alive!)")
    print("\nQuick Commands to try:")
    print("  > cam.num_images_available()       # Check buffer")
    print("  > cam.software_trigger()           # Snap an image")
    print("  > img = cam.read_image()           # Get the image (numpy array)")
    print("  > cam.cam.Gain.get()               # Read current Gain")
    print("  > cam.cam.Gain.set(5.0)            # Set Gain to 5.0 dB")
    print("  > cam.set_expo_time(0.01)          # Set Exposure to 10ms")
    print("\nCleanup:")
    print("  > cam_stack.close()                # IMPORTANT: Run this before closing shell")
    print("="*50 + "\n")

