import logging, os
import numpy as np
import PyQt5.QtCore
from scipy.ndimage import gaussian_filter
from astropy.io import fits  
from datetime import datetime
import h5py
import time
from GUI_modules.data_models import ImageData

class AcquisitionThread(PyQt5.QtCore.QThread):
    update_signal = PyQt5.QtCore.pyqtSignal(object)
    finished = PyQt5.QtCore.pyqtSignal()

    def __init__(self, parent, config, mode="Triggered", seq_info=None, abstract_camera=None):
        super().__init__()
        self.parent = parent
        self.config = config
        self.seq_info = seq_info
        self.abstract_camera = abstract_camera

        # Grab config preferences
        self.include_dark = self.config["measurement"].getboolean("include_dark")
        self.meas_mode = self.config["measurement"]["options"]
        self.images_per_run = 3 if self.include_dark else 2

        self.hdf_file = None
        self.run_group = None
        self.mode = mode

        self.full_file_path = None
        self.run_group_name = None
        
        # Tracking dictionaries
        self.group_counters = {}  # Maps unique_id -> current sequence count (int)

    def run(self):
        if self.mode == "Live":
            self.run_live()
        else:
            self.run_exp_controlled()

    def run_live(self):
        # ... (Same as before) ...
        live_planned_frames = self.parent.control_gui.live_planned_frames_Spb.value()
        self.abstract_camera = self.parent.control_gui.camera
        image_size = int(self.abstract_camera.read_image_size() * 1000)

        self.abstract_camera.start_acquisition(cycle_mode="Fixed", frame_count=live_planned_frames)
        frame_counter = 0
        while self.parent.active and frame_counter < live_planned_frames:
            image = self.acquire_image(image_size)
            self.update_signal.emit({"image": {"Atom": image}, "counter": frame_counter})
            frame_counter += 1

        self.abstract_camera.stop_acquisition()
        self.finished.emit()

    def run_exp_controlled(self):
        """Main experimental loop. Refactored for clarity."""
        
        # 1. Setup Parameters and Device
        self.sample_num = int(self.seq_info["general"]["sample_number"])
        self.seq_num_per_sample = int(self.seq_info["general"]["image_number"])
        seq_info_timestamp = self.seq_info["general"]["timestamp"]
        
        # Parse scan parameters
        scan_params_raw = self.seq_info["general"].get("scanned_devices_parameters", "")
        self.scanned_param_names = [p.strip() for p in scan_params_raw.split(',') if p.strip()]
        is_scan = len(self.scanned_param_names) > 0
        primary_scan_key = self.scanned_param_names[0] if is_scan else None

        total_sequences = self.sample_num * self.seq_num_per_sample
        total_images = total_sequences * 3 # atom, probe, dark


        # 2. Setup HDF5
        self.setup_hdf_initial(seq_info_timestamp, is_scan, scan_params_raw)
        # Reset tracking
        self.group_counters = {}
        image_counter = 0

        # ==========================================
        # HARDWARE ABSTRACTION: Setup
        # ========================================== 
        self.abstract_camera.prepare_sequence(total_images) # Universal call

        print(f"- - - - Run {seq_info_timestamp} started. - - - -")

        # 3. Unified Acquisition Loop
        for global_seq_index in range(total_sequences):
            if not self.parent.active:
                break
            

            # --- PHASE A: ACQUIRE (HDF File is CLOSED here) ---
            # Analyzing code can safely read the file during these seconds/milliseconds
            atom_image = self.abstract_camera.grab_frame(timeout=5.0)
            probe_image = self.abstract_camera.grab_frame(timeout=5.0)
            
            if self.include_dark:
                dark_image = self.abstract_camera.grab_frame(timeout=5.0)
            else:
                dark_image = np.zeros_like(atom_image)

            SigMinusBkg_image = (atom_image - dark_image) - (probe_image - dark_image)
            SigMinusBkg_image = - SigMinusBkg_image  # Invert for visualization in the case of absorption
            OD_image = calculate_od(atom_image, probe_image, dark_image)


            # --- PHASE B: SAVE (File is OPENED briefly here) ---
            # This is a discrete transaction. Open -> Write -> Close.
            # This method determines the correct subgroup to save the images into.
            subgroup_name, seq_attributes, primary_val = self._determine_subgroup_info(global_seq_index, is_scan, primary_scan_key)
            self.save_sequence_snapshot(
                subgroup_name, seq_attributes, global_seq_index, atom_image, probe_image, dark_image
            )

            image_data = ImageData(
                atom=atom_image,
                probe=probe_image,
                dark=dark_image,
                sig_bkg=SigMinusBkg_image,
                od=OD_image,
                counter=image_counter,
                is_scan=is_scan,
                scan_value=primary_val
            )
            self.update_signal.emit(image_data)

            # Increment by 3 or 2 depending on how many actual hardware frames were pulled
            image_counter += self.images_per_run
            

        # ==========================================
        # HARDWARE ABSTRACTION: Teardown
        # ==========================================
        self.abstract_camera.stop_sequence()
        print(f"- - - - Run Ended. Total images: {image_counter} - - - - -")
        self.finished.emit()


    def setup_hdf_initial(self, seq_info_timestamp, is_scan, scan_params_str):
        """Creates the file (if missing) and the top-level run group."""
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        hdf_filename = f"images_{date_str}.hdf"
        save_folder = self.config["image_save"]["save_folder"]
        self.full_file_path = f"{save_folder}\\{hdf_filename}"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.full_file_path), exist_ok=True)
        
        # Store the Run Group Name string so we don't regenerate it
        self.run_group_name = f"run_{seq_info_timestamp}"

        # OPEN -> CREATE STRUCT -> CLOSE
        # mode="a" creates file if it doesn't exist, opens if it does.
        with h5py.File(self.full_file_path, "a") as f:
            if self.run_group_name not in f:
                run_group = f.create_group(self.run_group_name)
                run_group.attrs["acquisition_start"] = now.isoformat()
                run_group.attrs["total_sequences"] = self.sample_num * self.seq_num_per_sample
                run_group.attrs["is_scan"] = is_scan
                run_group.attrs["scanned_parameters"] = scan_params_str if is_scan else "None"
                run_group.attrs["include_dark"] = self.include_dark

                cam_name = getattr(self.abstract_camera, "device_name", "Unknown_Camera")
                # pix_size = getattr(self.parent, "pixel_size_mm", 0.0)

                run_group.attrs["camera_device_name"] = cam_name
                # run_group.attrs["pixel_size_mm"] = pix_size       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Let me add this later when I have the actual value coming from the camera wrapper

                print(f"Initialized HDF5 group: {self.run_group_name}")
            else:
                print(f"HDF5 group {self.run_group_name} already exists.")
        
    def save_sequence_snapshot(self, subgroup_name, seq_attributes, global_seq_index, atom, probe, dark):
        """Opens file, resolves context, writes 3 images, closes file."""
        
        if not subgroup_name:
            print(f"[Error] Could not determine group for Sequence {global_seq_index}")
            return # Error in info lookup
        
        local_idx = 0 # Placeholder to store the index for printing later

        # --- CRITICAL SECTION: FILE LOCK ---
        try:
            with h5py.File(self.full_file_path, "a") as f:
                
                # Get the Run Group
                if self.run_group_name in f:
                    run_group = f[self.run_group_name]
                else:
                    # Fallback safety in case file was deleted manually
                    run_group = f.create_group(self.run_group_name)

                # Get or Create the Specific Scan Group
                if subgroup_name in run_group:
                    current_group = run_group[subgroup_name]
                else:
                    current_group = run_group.create_group(subgroup_name)
                    # Save scan parameters (attributes)
                    for key, val in seq_attributes.items():
                        if key != "image_type":
                            try:
                                current_group.attrs[key] = float(val)
                            except (ValueError, TypeError):
                                current_group.attrs[key] = str(val)

                # Manage Local Counter for this group
                if subgroup_name not in self.group_counters:
                    self.group_counters[subgroup_name] = 0
                
                local_idx = self.group_counters[subgroup_name]

                # Save the images
                self.save_image_with_attributes(current_group, f"seq_{local_idx}_atom", atom)
                self.save_image_with_attributes(current_group, f"seq_{local_idx}_probe", probe)
                self.save_image_with_attributes(current_group, f"seq_{local_idx}_dark", dark)
                
                # Increment counter
                self.group_counters[subgroup_name] += 1
                
        except Exception as e:
            logging.error(f"Failed to save sequence {global_seq_index}: {e}")
            self.parent.active = False # Immediately stop the hardware loop
            raise RuntimeError(f"CRITICAL FILE ERROR: {e}") # Crash loudly

        # 3. Logging (Done AFTER file is closed to save time)
        # This confirms to you exactly where the data went
        print(f"Saved Global Seq {global_seq_index} -> Group '{subgroup_name}' (Local Index: {local_idx})")
            

    def _determine_subgroup_info(self, global_seq_index, is_scan, primary_scan_key):
        """
        Pure logic helper. Returns (subgroup_name_string, attributes_dict).
        Does NOT touch the HDF5 file.
        """
        if not is_scan:
            return "sample_0", {}, 0
        
        seq_section_name = f"scan_value_{global_seq_index}"
        if seq_section_name not in self.seq_info:
            logging.error(f"Missing sequence info for index {global_seq_index}")
            return None, None

        seq_data = self.seq_info[seq_section_name]
        if primary_scan_key not in seq_data:
            raise KeyError(f"Sequence data is missing the scanned parameter: {primary_scan_key}")
        primary_val = float(seq_data[primary_scan_key])
        
        subgroup_name = f"{primary_scan_key}_{primary_val}"
        return subgroup_name, seq_data, primary_val


    def save_image_with_attributes(self, group, name, image_data):
        dset = group.create_dataset(name, data=image_data.astype(np.float32))
        dset.attrs["CLASS"] = np.bytes_("IMAGE")
        dset.attrs["IMAGE_VERSION"] = np.bytes_("1.2")
        dset.attrs["IMAGE_SUBCLASS"] = np.bytes_("IMAGE_GRAYSCALE")
        dset.attrs["IMAGE_WHITE_IS_ZERO"] = 0
        return dset



def calculate_od(atom_image, probe_image, dark_image):
    atom_image = np.array(atom_image, dtype=np.float32)
    probe_image = np.array(probe_image, dtype=np.float32)
    dark_image = np.array(dark_image, dtype=np.float32)

    atom_minus_dark = np.clip(atom_image - dark_image, 1, 2**16-1)
    probe_minus_dark = np.clip(probe_image - dark_image, 1, 2**16-1)

    ratio = np.divide(atom_minus_dark, probe_minus_dark)
    od = np.clip(-np.log(ratio),-0.1, 5.0)
    return od