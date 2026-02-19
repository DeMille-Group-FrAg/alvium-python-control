import time
import numpy as np
import logging
# The wrappers are designed to provide a consistent interface for the acquisition thread, abstracting away the specific details of each camera type.

class AlviumCameraWrapper:
    def __init__(self, alvium_device):
        self.cam = alvium_device
        self._stack = None

    def prepare_sequence(self, total_images):
        # start() returns an ExitStack. 
        # We manually enter it and store it to keep the camera alive.
        self._stack = self.cam.start()
        self._stack.__enter__()

    def grab_frame(self, timeout=5.0):
        # 1. Trigger if necessary
        if self.cam.trigger_mode == "software":
            self.cam.software_trigger()
            time.sleep(0.5)

        # 2. Wait for image to hit the queue
        start_time = time.time()
        while self.cam.num_images_available() == 0:
            if (time.time() - start_time) > timeout:
                # Return a zero-array to prevent downstream math from crashing
                # The shape comes from the driver's cached image_shape property
                shape = (self.cam.image_shape["xmax"], self.cam.image_shape["ymax"])
                return np.zeros(shape, dtype=np.float32)
            time.sleep(0.001)

        # 3. Pop the image off the deque
        image = self.cam.read_image().T
        return image.astype(np.float32)

    def stop_sequence(self):
        # Manually exit the ExitStack. This kills the camera connection 
        # and automatically runs the stop_streaming callback.
        if self._stack is not None:
            self._stack.__exit__(None, None, None)
            self._stack = None




class AndorCameraWrapper:
    def __init__(self, andor_device):
        self.cam = andor_device
        self.image_size = int(self.cam.read_image_size() * 1000)

    def prepare_sequence(self, total_images):
        self.cam.start_acquisition(cycle_mode="Fixed", frame_count=total_images)

    def grab_frame(self, timeout=5.0):
        # Andor-specific acquisition logic
        try:
            image, _ = self.cam.read_buffer(circular_buffer=True, image_size=self.image_size, timeout=int(timeout*1000))
            if image is not None:
                return np.array(image, dtype=np.float32)
        except Exception as e:
            logging.error(f"Andor failed to grab frame: {e}")
        return np.zeros((1000, 1000), dtype=np.float32)

    def stop_sequence(self):
        self.cam.stop_acquisition()