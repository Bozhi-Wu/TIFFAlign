import os
import sys
import scipy
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPalette, QColor
from scipy.ndimage import shift, rotate, zoom
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from tifffile import TiffWriter, memmap, TiffFile, imread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QComboBox, QFileDialog, QProgressBar, QSizePolicy,
    QLineEdit, QCheckBox, QGroupBox, QGridLayout
)

# --- Helper functions for loading SBX ---
def loadmat(matfile):
    info = scipy.io.loadmat(matfile)['info']
    return info

def sbx_to_frames(fpath):
    info = loadmat(fpath + '.mat')
    height, width = info['sz'][0][0][0].astype(np.uint32)
    framesize = height * width
    maxint = np.iinfo(np.uint16).max

    fid = np.memmap(fpath + '.sbx', dtype=np.uint16, mode='r')
    n_frames = fid.shape[0] // framesize
    vid = np.reshape(fid[:n_frames * framesize], (n_frames, height, width))

    return (maxint - vid).astype(np.uint16)

def sbx_to_frames_optimized(fpath, max_frames=100):
    """Optimized SBX loading that only loads the first max_frames for mean calculation"""
    info = loadmat(fpath + '.mat')
    height, width = info['sz'][0][0][0].astype(np.uint32)
    framesize = height * width
    maxint = np.iinfo(np.uint16).max

    fid = np.memmap(fpath + '.sbx', dtype=np.uint16, mode='r')
    n_frames = fid.shape[0] // framesize
    n_frames_to_load = min(n_frames, max_frames)

    vid = np.reshape(fid[:n_frames_to_load * framesize], (n_frames_to_load, height, width))
    return (maxint - vid).astype(np.uint16), n_frames

def load_tiff_optimized(filepath, max_frames=100):
    """Optimized TIFF loading that only loads the first max_frames for mean calculation"""
    with TiffFile(filepath) as tif:
        n_frames = len(tif.pages)
        n_frames_to_load = min(n_frames, max_frames)

        first_frame = tif.pages[0].asarray()
        dtype = first_frame.dtype

        frames = []
        for i in range(n_frames_to_load):
            frames.append(tif.pages[i].asarray())

        return np.array(frames), n_frames, dtype

def correct_blocked_lines(data, threshold=1000, n_max_window=2):
    n_frames, height, _ = data.shape
    corrected = data.copy()

    blocked_mask = np.zeros((n_frames, height), dtype=bool)
    for i in range(n_frames):
        row_means = data[i].mean(axis=1)
        blocked_mask[i, :] = row_means < threshold

    for i in range(n_frames):
        blocked_rows = np.where(blocked_mask[i])[0]
        if len(blocked_rows) == 0:
            continue

        for r in blocked_rows:
            clean_before = None
            clean_after = None

            for k in range(1, n_max_window + 1):
                if i - k >= 0 and not blocked_mask[i - k, r]:
                    clean_before = data[i - k, r, :]
                    break
            for k in range(1, n_max_window + 1):
                if i + k < n_frames and not blocked_mask[i + k, r]:
                    clean_after = data[i + k, r, :]
                    break

            if clean_before is not None and clean_after is not None:
                corrected_row = np.stack([clean_before, clean_after]).mean(axis=0).astype(np.uint16)
            elif clean_before is not None:
                corrected_row = clean_before.copy()
            elif clean_after is not None:
                corrected_row = clean_after.copy()
            else:
                corrected_row = np.zeros_like(data[i, r, :], dtype=np.uint16)
            corrected[i, r, :] = corrected_row

    return corrected


# --- Background Loading Thread ---
class DataLoaderThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    loading_finished = pyqtSignal(dict)

    def __init__(self, folderpath, exten, n_frames_averaged=100):
        super().__init__()
        self.folderpath = folderpath
        self.exten = exten
        self.n_frames_averaged = n_frames_averaged
        self.detected_dtype = None

    def run(self):
        try:
            self.status_updated.emit("Finding files...")
            if self.exten == "*.tiff":
                files = sorted(list(self.folderpath.rglob("*.tiff")) +
                               list(self.folderpath.rglob("*.tif")))
            else:
                files = sorted(self.folderpath.rglob(self.exten))

            if not files:
                self.status_updated.emit(f"No files found with extension {self.exten}")
                return

            self.status_updated.emit(f"Processing {len(files)} files...")
            mean_frames = []
            n_sessions = 0
            n_total_frames = 0

            for i, file in enumerate(files):
                self.status_updated.emit(f"Processing file {i+1}/{len(files)}: {file.name}")

                if self.exten == "*.sbx":
                    fpath = file.as_posix().split('.')[0]
                    frames, total_frames = sbx_to_frames_optimized(fpath, self.n_frames_averaged)
                    if self.detected_dtype is None:
                        self.detected_dtype = np.uint16
                elif self.exten == "*.tiff":
                    frames, total_frames, dtype = load_tiff_optimized(file.as_posix(), self.n_frames_averaged)
                    if self.detected_dtype is None:
                        self.detected_dtype = dtype

                n_total_frames += total_frames
                n_sessions += 1

                mean_frame = np.mean(frames, axis=0)
                mean_frames.append(mean_frame.astype(np.uint16))

                progress = int((i + 1) / len(files) * 100)
                self.progress_updated.emit(progress)

            result = {
                'mean_frames': mean_frames,
                'n_sessions': n_sessions,
                'n_total_frames': n_total_frames,
                'detected_dtype': self.detected_dtype
            }

            self.loading_finished.emit(result)

        except Exception as e:
            self.status_updated.emit(f"Error during loading: {str(e)}")
            self.loading_finished.emit(None)


# --- Background Saving Thread ---
class SaveThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    saving_finished = pyqtSignal(bool, str)

    def __init__(self, folderpath, exten, params_all, ref_idx, savepath,
                 detected_dtype, blocked_sessions=None, blocked_threshold=1000,
                 blocked_window=2, crop_params=None):
        super().__init__()
        self.folderpath = folderpath
        self.exten = exten
        self.params_all = params_all
        self.ref_idx = ref_idx
        self.savepath = savepath
        self.detected_dtype = detected_dtype
        self.blocked_sessions = set(blocked_sessions or [])
        self.blocked_threshold = int(blocked_threshold)
        self.blocked_window = int(blocked_window)
        self.crop_params = crop_params  # (top, bottom, left, right) or None

    def run(self):
        try:
            if os.path.exists(self.savepath):
                try:
                    os.remove(self.savepath)
                except Exception as e:
                    self.saving_finished.emit(False, f"Error deleting existing file: {e}")
                    return

            if self.exten == "*.tiff":
                files = sorted(list(self.folderpath.rglob("*.tiff")) +
                               list(self.folderpath.rglob("*.tif")))
            else:
                files = sorted(self.folderpath.rglob(self.exten))

            if not files:
                self.saving_finished.emit(False, f"No files found with extension {self.exten}")
                return

            total_frames = 0
            per_file_counts = []
            for file in files:
                if self.exten == "*.tiff":
                    with TiffFile(file.as_posix()) as tif:
                        n = len(tif.pages)
                else:
                    info = loadmat(file.as_posix().split('.')[0] + '.mat')
                    height, width = info['sz'][0][0][0].astype(np.uint32)
                    framesize = height * width
                    fid = np.memmap(file.as_posix().split('.')[0] + '.sbx', dtype=np.uint16, mode='r')
                    n = fid.shape[0] // framesize
                per_file_counts.append(n)
                total_frames += n

            frames_done = 0

            self.status_updated.emit("Writing aligned TIFF...")
            with TiffWriter(self.savepath, bigtiff=True) as tiff_writer:
                for nn, file in enumerate(files):
                    self.status_updated.emit(f"Processing file {nn+1}/{len(files)}: {file.name}")

                    if self.exten == "*.sbx":
                        fpath = file.as_posix().split('.')[0]
                        frames = sbx_to_frames(fpath)
                    else:
                        try:
                            frames = memmap(file.as_posix())
                        except Exception:
                            self.status_updated.emit("Cannot use memory mapping, reading directly to memory instead.")
                            frames = imread(file.as_posix())

                    params = self.params_all['sessions'].get(
                        nn, {'x_shift': 0, 'y_shift': 0, 'rotation': 0, 'scale': 1.0}
                    )

                    chunk_size = 1000
                    apply_correction = nn in self.blocked_sessions
                    for i in range(0, frames.shape[0], chunk_size):
                        end_idx = min(i + chunk_size, frames.shape[0])

                        if apply_correction:
                            win_start = max(0, i - self.blocked_window)
                            win_end = min(frames.shape[0], end_idx + self.blocked_window)
                            window = frames[win_start:win_end]
                            try:
                                window_corr = correct_blocked_lines(
                                    window,
                                    threshold=self.blocked_threshold,
                                    n_max_window=self.blocked_window
                                )
                            except Exception as e:
                                self.status_updated.emit(
                                    f"Blocked-lines correction error (session {nn}): {e}. Skipping."
                                )
                                window_corr = window
                            core_start = i - win_start
                            core_end = core_start + (end_idx - i)
                            chunk_iter = window_corr[core_start:core_end]
                        else:
                            chunk_iter = frames[i:end_idx]

                        for frame in chunk_iter:
                            if nn != self.ref_idx:
                                if params['scale'] != 1.0:
                                    scaled = zoom(frame, params['scale'], order=0)
                                    if params['scale'] > 1.0:
                                        h, w = frame.shape
                                        sh, sw = scaled.shape
                                        start_h = (sh - h) // 2
                                        start_w = (sw - w) // 2
                                        scaled = scaled[start_h:start_h+h, start_w:start_w+w]
                                    else:
                                        h, w = frame.shape
                                        sh, sw = scaled.shape
                                        pad_h = (h - sh) // 2
                                        pad_w = (w - sw) // 2
                                        padded = np.zeros_like(frame)
                                        padded[pad_h:pad_h+sh, pad_w:pad_w+sw] = scaled
                                        scaled = padded
                                    frame = scaled
                                rotated = rotate(frame, params['rotation'], reshape=False, order=0)
                                frame = shift(rotated, [params['y_shift'], params['x_shift']], order=0)

                            # Apply crop if enabled
                            if self.crop_params is not None:
                                ct, cb, cl, cr = self.crop_params
                                fh, fw = frame.shape
                                row_end = fh - cb if cb > 0 else fh
                                col_end = fw - cr if cr > 0 else fw
                                frame = frame[ct:row_end, cl:col_end]

                            tiff_writer.write(frame.astype(self.detected_dtype))

                            frames_done += 1
                            if total_frames > 0 and frames_done % 50 == 0:
                                progress = int(frames_done / total_frames * 100)
                                self.progress_updated.emit(progress)

                    progress = int(frames_done / total_frames * 100) if total_frames > 0 else 100
                    self.progress_updated.emit(progress)

            self.progress_updated.emit(100)
            self.saving_finished.emit(True, "Saved aligned TIFF!")
        except Exception as e:
            self.saving_finished.emit(False, f"Error during saving: {str(e)}")


# --- Background Auto Alignment Thread ---
class AutoAlignThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished_with_params = pyqtSignal(dict)

    def __init__(self, ref_img, mov_img, rot_min=-10.0, rot_max=10.0, rot_step=0.1,
                 scale_min=0.95, scale_max=1.05, scale_step=0.01,
                 crop_frac=0.6, trans_max=30, verbose=True):
        super().__init__()
        self.ref_img = ref_img
        self.mov_img = mov_img
        self.rot_min = rot_min
        self.rot_max = rot_max
        self.rot_step = rot_step
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_step = scale_step
        self.crop_frac = crop_frac
        self.trans_max = trans_max
        self.verbose = verbose

    def run(self):
        try:
            if self.verbose:
                self.status_updated.emit("Auto align: preparing...")
            ref = self._normalize(self.ref_img.astype(np.float32))
            mov = self._normalize(self.mov_img.astype(np.float32))

            rotations = np.arange(self.rot_min, self.rot_max + 1e-6, self.rot_step)
            scales = np.arange(self.scale_min, self.scale_max + 1e-9, self.scale_step)
            num_steps = max(1, len(rotations) * len(scales))
            best = {'rotation': 0.0, 'scale': 1.0, 'x_shift': 0, 'y_shift': 0, 'score': -np.inf}

            ref_c = self._center_crop(ref, self.crop_frac)

            step_counter = 0
            for s in scales:
                scaled_full = self._apply_scale_to_size(mov, s)
                for rot in rotations:
                    if self.verbose:
                        self.status_updated.emit(f"Auto align: scale {s:.3f}, rot {rot:.1f}°")
                    rotated = rotate(scaled_full, rot, reshape=False, order=0)
                    rot_c = self._center_crop(rotated, self.crop_frac)

                    if self.trans_max > 0:
                        dy, dx = self._estimate_translation_fft(ref_c, rot_c)
                        dy = max(-self.trans_max, min(self.trans_max, dy))
                        dx = max(-self.trans_max, min(self.trans_max, dx))
                        shifted = shift(rotated, (dy, dx), order=0)
                    else:
                        dy, dx = 0, 0
                        shifted = rotated

                    shifted_c = self._center_crop(shifted, self.crop_frac)
                    score = self._pearson(ref_c, shifted_c)

                    if score > best['score']:
                        best = {
                            'rotation': float(rot),
                            'scale': float(s),
                            'x_shift': int(round(dx)),
                            'y_shift': int(round(dy)),
                            'score': float(score)
                        }

                    step_counter += 1
                    if num_steps > 1:
                        self.progress_updated.emit(int(step_counter / num_steps * 100))

            if self.verbose:
                self.status_updated.emit(f"Auto align: best corr {best['score']:.4f}")
            self.finished_with_params.emit({
                'rotation': best['rotation'],
                'scale': best['scale'],
                'x_shift': best['x_shift'],
                'y_shift': best['y_shift']
            })
        except Exception as e:
            self.status_updated.emit(f"Auto align error: {str(e)}")
            self.finished_with_params.emit({'rotation': 0.0, 'scale': 1.0, 'x_shift': 0, 'y_shift': 0})

    def _center_crop(self, img, frac):
        h, w = img.shape
        ch = max(1, int(h * frac))
        cw = max(1, int(w * frac))
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        return img[y0:y0+ch, x0:x0+cw]

    def _normalize(self, img):
        m = np.mean(img)
        s = np.std(img)
        if s == 0:
            return img * 0.0
        return (img - m) / s

    def _pearson(self, a, b):
        a_flat = a.ravel()
        b_flat = b.ravel()
        am = a_flat - a_flat.mean()
        bm = b_flat - b_flat.mean()
        denom = np.linalg.norm(am) * np.linalg.norm(bm)
        if denom == 0:
            return -1.0
        return float(np.dot(am, bm) / denom)

    def _estimate_translation_fft(self, ref, img):
        f0 = np.fft.fft2(ref)
        f1 = np.fft.fft2(img)
        eps = 1e-9
        cross_power = f0 * np.conj(f1)
        cross_power /= np.maximum(np.abs(cross_power), eps)
        c = np.fft.ifft2(cross_power)
        c = np.abs(c)
        maxpos = np.unravel_index(np.argmax(c), c.shape)
        shifts = np.array(maxpos, dtype=np.int64)
        h, w = ref.shape
        if shifts[0] > h // 2:
            shifts[0] -= h
        if shifts[1] > w // 2:
            shifts[1] -= w
        return int(shifts[0]), int(shifts[1])

    def _apply_scale_to_size(self, img, scale):
        if abs(scale - 1.0) < 1e-6:
            return img.copy()
        scaled = zoom(img, scale, order=0)
        h, w = img.shape
        sh, sw = scaled.shape
        if scale > 1.0:
            start_h = (sh - h) // 2
            start_w = (sw - w) // 2
            return scaled[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = max(0, (h - sh) // 2)
            pad_w = max(0, (w - sw) // 2)
            out = np.zeros_like(img)
            out[pad_h:pad_h+sh, pad_w:pad_w+sw] = scaled
            return out


# --- GUI Class ---
class AlignGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('TIFFAlign')
        self.center_window()
        self.session_idx = 0
        self.ref_idx = 0
        self.alpha = 0.5
        self.exten = "*.sbx"
        self.mean_frames = None
        self.n_sessions = 0
        self.n_total_frames = 0
        self.detected_dtype = np.uint16
        self.params_all = {}
        self.loading_thread = None
        self.save_thread = None

        self.set_dark_theme()

        # --- Layouts ---
        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        selectors_layout = QVBoxLayout()
        sliders_layout = QVBoxLayout()

        # --- Selectors layout ---
        def add_row(label_text, widget):
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            row.addWidget(widget)
            selectors_layout.addLayout(row)

        # Extension selector
        self.exten_selector = QComboBox()
        self.exten_selector.addItems(['sbx', 'tiff'])
        self.exten_selector.currentIndexChanged.connect(self.change_exten)
        add_row("File Extension:", self.exten_selector)

        # Folder selection
        self.folder_button = QPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_folder)
        add_row("", self.folder_button)

        # Reference selector
        self.ref_selector = QComboBox()
        add_row("Reference Session:", self.ref_selector)

        # Moving session selector
        self.session_selector = QComboBox()
        add_row("Moving Session:", self.session_selector)

        # --- Auto Alignment Parameters ---
        param_group = QGroupBox("Auto Alignment Parameters")
        param_group.setStyleSheet(self._groupbox_style())
        param_layout = QGridLayout()

        self.trans_checkbox = QCheckBox("Translation")
        self.trans_checkbox.setChecked(True)
        param_layout.addWidget(self.trans_checkbox, 0, 0)
        param_layout.addWidget(QLabel("Max Shift:"), 0, 1)
        self.trans_max_input = QLineEdit("20")
        self.trans_max_input.setMaximumWidth(60)
        param_layout.addWidget(self.trans_max_input, 0, 2)
        param_layout.addWidget(QLabel("Crop:"), 0, 5)
        self.crop_input = QLineEdit("0.6")
        self.crop_input.setMaximumWidth(60)
        param_layout.addWidget(self.crop_input, 0, 6)

        self.rot_checkbox = QCheckBox("Rotation")
        self.rot_checkbox.setChecked(True)
        param_layout.addWidget(self.rot_checkbox, 1, 0)
        param_layout.addWidget(QLabel("Min:"), 1, 1)
        self.rot_min_input = QLineEdit("-5.0")
        self.rot_min_input.setMaximumWidth(60)
        param_layout.addWidget(self.rot_min_input, 1, 2)
        param_layout.addWidget(QLabel("Max:"), 1, 3)
        self.rot_max_input = QLineEdit("5.0")
        self.rot_max_input.setMaximumWidth(60)
        param_layout.addWidget(self.rot_max_input, 1, 4)
        param_layout.addWidget(QLabel("Step:"), 1, 5)
        self.rot_step_input = QLineEdit("0.1")
        self.rot_step_input.setMaximumWidth(60)
        param_layout.addWidget(self.rot_step_input, 1, 6)

        self.scale_checkbox = QCheckBox("Scale")
        self.scale_checkbox.setChecked(False)
        param_layout.addWidget(self.scale_checkbox, 2, 0)
        param_layout.addWidget(QLabel("Min:"), 2, 1)
        self.scale_min_input = QLineEdit("0.95")
        self.scale_min_input.setMaximumWidth(60)
        param_layout.addWidget(self.scale_min_input, 2, 2)
        param_layout.addWidget(QLabel("Max:"), 2, 3)
        self.scale_max_input = QLineEdit("1.05")
        self.scale_max_input.setMaximumWidth(60)
        param_layout.addWidget(self.scale_max_input, 2, 4)
        param_layout.addWidget(QLabel("Step:"), 2, 5)
        self.scale_step_input = QLineEdit("0.01")
        self.scale_step_input.setMaximumWidth(60)
        param_layout.addWidget(self.scale_step_input, 2, 6)

        param_group.setLayout(param_layout)
        selectors_layout.addWidget(param_group)

        # --- Blocked-Frames Correction ---
        self.correction_group = QGroupBox("Stimulation Blocked Frames Correction")
        self.correction_group.setStyleSheet(self._groupbox_style())
        self.correction_layout = QGridLayout()
        self.correction_layout.addWidget(QLabel("Threshold:"), 0, 0)
        self.correction_threshold_input = QLineEdit("1000")
        self.correction_threshold_input.setMaximumWidth(80)
        self.correction_layout.addWidget(self.correction_threshold_input, 0, 1)
        self.correction_layout.addWidget(QLabel("Window:"), 0, 2)
        self.correction_window_input = QLineEdit("2")
        self.correction_window_input.setMaximumWidth(60)
        self.correction_layout.addWidget(self.correction_window_input, 0, 3)
        self.correction_enable_checkbox = QCheckBox("Interpolate Blocked Frames")
        self.correction_enable_checkbox.setChecked(False)
        self.correction_layout.addWidget(self.correction_enable_checkbox, 1, 0, 1, 4)
        self.correction_group.setLayout(self.correction_layout)
        selectors_layout.addWidget(self.correction_group)

        # --- Crop Settings ---
        self.crop_group = QGroupBox("Crop Settings")
        self.crop_group.setStyleSheet(self._groupbox_style())
        crop_layout = QGridLayout()

        self.crop_enable_checkbox = QCheckBox("Enable Cropping")
        self.crop_enable_checkbox.setChecked(False)
        self.crop_enable_checkbox.stateChanged.connect(self.update_image)
        crop_layout.addWidget(self.crop_enable_checkbox, 0, 0, 1, 4)

        for row, (label_a, attr_a, label_b, attr_b) in enumerate([
            ("Top:",  "crop_top_input",    "Bottom:", "crop_bottom_input"),
            ("Left:", "crop_left_input",   "Right:",  "crop_right_input"),
        ], start=1):
            crop_layout.addWidget(QLabel(label_a), row, 0)
            inp_a = QLineEdit("0")
            inp_a.setMaximumWidth(60)
            inp_a.editingFinished.connect(self.update_image)
            setattr(self, attr_a, inp_a)
            crop_layout.addWidget(inp_a, row, 1)
            crop_layout.addWidget(QLabel(label_b), row, 2)
            inp_b = QLineEdit("0")
            inp_b.setMaximumWidth(60)
            inp_b.editingFinished.connect(self.update_image)
            setattr(self, attr_b, inp_b)
            crop_layout.addWidget(inp_b, row, 3)

        self.auto_crop_button = QPushButton("Auto Detect Crop")
        self.auto_crop_button.clicked.connect(self.auto_detect_crop)
        crop_layout.addWidget(self.auto_crop_button, 3, 0, 1, 4)

        self.crop_size_label = QLabel("Output size: —")
        self.crop_size_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        crop_layout.addWidget(self.crop_size_label, 4, 0, 1, 4)

        self.crop_group.setLayout(crop_layout)
        selectors_layout.addWidget(self.crop_group)

        # --- Sliders ---
        self.x_label, self.x_slider, self.x_input = self.create_slider("X Shift", -50, 50, 0, sliders_layout)
        self.y_label, self.y_slider, self.y_input = self.create_slider("Y Shift", -50, 50, 0, sliders_layout)
        self.rot_label, self.rot_slider, self.rot_input = self.create_slider("Rotation", -100, 100, 0, sliders_layout)
        self.scale_label, self.scale_slider, self.scale_input = self.create_slider("Scale", 50, 150, 100, sliders_layout)
        self.alpha_label, self.alpha_slider, self.alpha_input = self.create_slider("Alpha", 0, 100, 50, sliders_layout)

        # Auto alignment buttons
        self.auto_align_button = QPushButton("Auto Align Current Session")
        self.auto_align_button.clicked.connect(self.auto_align)
        sliders_layout.addWidget(self.auto_align_button)
        self.auto_align_all_button = QPushButton("Auto Align All Sessions")
        self.auto_align_all_button.clicked.connect(self.auto_align_all)
        sliders_layout.addWidget(self.auto_align_all_button)

        # Save / Load buttons
        self.save_params_button = QPushButton("Save Alignment Parameters")
        self.save_params_button.clicked.connect(self.save_params)
        sliders_layout.addWidget(self.save_params_button)

        self.load_params_button = QPushButton("Load Alignment Parameters")
        self.load_params_button.clicked.connect(self.load_params)
        sliders_layout.addWidget(self.load_params_button)

        self.save_button = QPushButton("Save Aligned TIFF")
        self.save_button.clicked.connect(self.apply_and_save)
        sliders_layout.addWidget(self.save_button)

        # --- Progress / status ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #aaaaaa; font-size: 14px;")

        self.enable_controls(False)

        # --- Combine layouts ---
        control_layout.addLayout(selectors_layout, stretch=1)
        control_layout.addLayout(sliders_layout, stretch=2)

        # --- Matplotlib canvas ---
        self.fig, self.ax = plt.subplots(figsize=(14, 14))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.axis('off')
        self.canvas = FigureCanvas(self.fig)

        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.canvas)

        footer_layout = QHBoxLayout()
        footer_layout.addWidget(self.status_label, 0)
        footer_layout.addWidget(self.progress_bar, 1)
        main_layout.addLayout(footer_layout)
        self.setLayout(main_layout)

        if self.mean_frames is not None:
            self.update_image()

    # ------------------- Style Helper -------------------
    def _groupbox_style(self):
        return """
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """

    # ------------------- Helper Methods -------------------
    def center_window(self):
        screen = QApplication.desktop().screenGeometry()
        window_width, window_height = 1250, 1400
        x = (screen.width() - window_width) // 2
        y = (screen.height() - window_height) // 2
        self.setGeometry(x, y, window_width, window_height)

    def set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        self.setPalette(palette)

    def create_slider(self, name, min_val, max_val, init_val, layout):
        label = QLabel(f"{name}: {init_val}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        slider.setStyleSheet("""
            QSlider::handle:horizontal {
                background: #aaaaaa;
                border: 2px solid #5c5c5c;
                width: 10px;
                height: 10px;
                margin: -7px 0;
                border-radius: 7px;
            }
        """)

        text_input = QLineEdit()
        text_input.setMaximumWidth(80)
        text_input.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px 5px;
                color: white;
            }
            QLineEdit:focus {
                border: 1px solid #0078d4;
            }
        """)

        if name == "Rotation":
            text_input.setText(f"{init_val / 10:.1f}")
        elif name == "Alpha":
            text_input.setText(f"{init_val / 100:.2f}")
        elif name == "Scale":
            text_input.setText(f"{init_val / 100:.2f}")
        else:
            text_input.setText(str(init_val))

        slider.valueChanged.connect(
            lambda val, l=label, n=name, t=text_input: self.slider_changed(val, l, n, t)
        )
        text_input.returnPressed.connect(
            lambda t=text_input, s=slider, n=name, mn=min_val, mx=max_val:
                self.text_input_changed(t, s, n, mn, mx)
        )
        text_input.editingFinished.connect(
            lambda t=text_input, s=slider, n=name, mn=min_val, mx=max_val:
                self.text_input_changed(t, s, n, mn, mx)
        )

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(slider, 1)
        slider_layout.addWidget(text_input, 0)

        layout.addWidget(label)
        layout.addLayout(slider_layout)
        return label, slider, text_input

    def slider_changed(self, value, label, name, text_input=None):
        if name == "Rotation":
            label.setText(f"{name}: {value / 10:.1f}")
            if text_input:
                text_input.setText(f"{value / 10:.1f}")
        elif name == "Alpha":
            label.setText(f"{name}: {value / 100:.2f}")
            if text_input:
                text_input.setText(f"{value / 100:.2f}")
        elif name == "Scale":
            label.setText(f"{name}: {value / 100:.2f}x")
            if text_input:
                text_input.setText(f"{value / 100:.2f}")
        else:
            label.setText(f"{name}: {value}")
            if text_input:
                text_input.setText(str(value))
        if self.mean_frames is not None:
            self.update_image()

    def text_input_changed(self, text_input, slider, name, min_val, max_val):
        try:
            value = float(text_input.text())
            if name == "Rotation":
                slider_value = int(value * 10)
            elif name in ("Alpha", "Scale"):
                slider_value = int(value * 100)
            else:
                slider_value = int(value)
            slider_value = max(min_val, min(max_val, slider_value))
            slider.setValue(slider_value)
            if name == "Rotation":
                text_input.setText(f"{slider_value / 10:.1f}")
            elif name == "Alpha":
                text_input.setText(f"{slider_value / 100:.2f}")
            elif name == "Scale":
                text_input.setText(f"{slider_value / 100:.2f}")
            else:
                text_input.setText(str(slider_value))
        except ValueError:
            if name == "Rotation":
                text_input.setText(f"{slider.value() / 10:.1f}")
            elif name == "Alpha":
                text_input.setText(f"{slider.value() / 100:.2f}")
            elif name == "Scale":
                text_input.setText(f"{slider.value() / 100:.2f}")
            else:
                text_input.setText(str(slider.value()))

    def enable_controls(self, enable=True):
        for widget in (
            self.x_slider, self.y_slider, self.rot_slider, self.scale_slider, self.alpha_slider,
            self.x_input, self.y_input, self.rot_input, self.scale_input, self.alpha_input,
            self.save_button, self.save_params_button, self.load_params_button,
            self.ref_selector, self.session_selector,
        ):
            widget.setEnabled(enable)
        for attr in (
            'auto_align_button', 'auto_align_all_button',
            'rot_checkbox', 'rot_min_input', 'rot_max_input', 'rot_step_input',
            'scale_checkbox', 'scale_min_input', 'scale_max_input', 'scale_step_input',
            'trans_checkbox', 'trans_max_input', 'crop_input',
            'crop_enable_checkbox', 'crop_top_input', 'crop_bottom_input',
            'crop_left_input', 'crop_right_input', 'auto_crop_button',
        ):
            if hasattr(self, attr):
                getattr(self, attr).setEnabled(enable)

    # ------------------- Crop Helpers -------------------
    def get_crop_params(self):
        """Return (top, bottom, left, right) crop pixel amounts, clamped to >= 0."""
        try:
            top    = max(0, int(self.crop_top_input.text()))
            bottom = max(0, int(self.crop_bottom_input.text()))
            left   = max(0, int(self.crop_left_input.text()))
            right  = max(0, int(self.crop_right_input.text()))
        except ValueError:
            top = bottom = left = right = 0
        return top, bottom, left, right

    def auto_detect_crop(self):
        """
        Apply each session's stored transform to its mean frame, then find
        the tightest crop that removes all near-zero (black) border pixels
        across every session.
        """
        if self.mean_frames is None:
            return

        ref_img = self.mean_frames[self.ref_idx]
        h, w = ref_img.shape

        # Threshold: 0.2 % of uint16 max (~131) — values below this are "black"
        threshold = max(1, np.iinfo(np.uint16).max * 0.002)

        global_top = global_bottom = global_left = global_right = 0

        for i in range(self.n_sessions):
            params = self.params_all['sessions'].get(
                i, {'x_shift': 0, 'y_shift': 0, 'rotation': 0, 'scale': 1.0}
            )
            img = self.mean_frames[i].astype(np.float32)

            # Replicate the same transform used in update_image / SaveThread
            sc = params['scale']
            if abs(sc - 1.0) > 1e-4:
                scaled = zoom(img, sc, order=0)
                sh, sw = scaled.shape
                if sc > 1.0:
                    y0 = (sh - h) // 2
                    x0 = (sw - w) // 2
                    scaled = scaled[y0:y0+h, x0:x0+w]
                else:
                    pad_h = (h - sh) // 2
                    pad_w = (w - sw) // 2
                    padded = np.zeros((h, w), dtype=np.float32)
                    padded[pad_h:pad_h+sh, pad_w:pad_w+sw] = scaled
                    scaled = padded
                img = scaled

            rotated = rotate(img, params['rotation'], reshape=False, order=0)
            aligned = shift(rotated, (params['y_shift'], params['x_shift']), order=0)

            # Max-projection along each axis for fast boundary detection
            row_max = aligned.max(axis=1)
            col_max = aligned.max(axis=0)

            top    = next((r for r in range(h)        if row_max[r] > threshold), 0)
            bottom = next((r for r in range(h-1,-1,-1) if row_max[r] > threshold), h - 1)
            left   = next((c for c in range(w)        if col_max[c] > threshold), 0)
            right  = next((c for c in range(w-1,-1,-1) if col_max[c] > threshold), w - 1)

            global_top    = max(global_top,    top)
            global_bottom = max(global_bottom, h - 1 - bottom)
            global_left   = max(global_left,   left)
            global_right  = max(global_right,  w - 1 - right)

        self.crop_top_input.setText(str(global_top))
        self.crop_bottom_input.setText(str(global_bottom))
        self.crop_left_input.setText(str(global_left))
        self.crop_right_input.setText(str(global_right))
        self.crop_enable_checkbox.setChecked(True)
        self.update_image()

    # ------------------- Folder / Data Loading -------------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folderpath = Path(folder)
            print(f"Folder selected: {self.folderpath}")
            self.load_data()
            self.load_params()

    def load_data(self):
        self.savepath = self.folderpath / 'tiff_manual_aligned.tiff'
        self.n_frames_averaged = 100
        self.pickle_path = self.folderpath / 'mean_frames.pkl'
        self.params_path = self.folderpath / 'params_all.pkl'

        if self.pickle_path.exists():
            print("Loading mean frames from pickle ---")
            self.status_label.setText("Loading cached data...")
            try:
                with open(self.pickle_path, 'rb') as f:
                    data = pickle.load(f)
                    self.mean_frames = data['mean_frames']
                    self.n_sessions = data['n_sessions']
                    self.n_total_frames = data['n_total_frames']
                    self.detected_dtype = data.get('detected_dtype', np.uint16)
                self.finish_data_loading()
            except Exception as e:
                print(f"Error loading pickle: {e}")
                self.status_label.setText("Error loading cached data, recomputing...")
                self.start_background_loading()
        else:
            print("Computing and saving mean frames ---")
            self.start_background_loading()

    def start_background_loading(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting data loading...")
        self.enable_controls(False)

        self.loading_thread = DataLoaderThread(self.folderpath, self.exten, self.n_frames_averaged)
        self.loading_thread.progress_updated.connect(self.progress_bar.setValue)
        self.loading_thread.status_updated.connect(self.status_label.setText)
        self.loading_thread.loading_finished.connect(self.on_loading_finished)
        self.loading_thread.start()

    def on_loading_finished(self, result):
        self.progress_bar.setVisible(False)
        if result is None:
            self.status_label.setText("Loading failed")
            return

        self.mean_frames = result['mean_frames']
        self.n_sessions = result['n_sessions']
        self.n_total_frames = result['n_total_frames']
        self.detected_dtype = result['detected_dtype']

        data = {
            'mean_frames': self.mean_frames,
            'n_sessions': self.n_sessions,
            'n_total_frames': self.n_total_frames,
            'detected_dtype': self.detected_dtype
        }
        try:
            with open(self.pickle_path, 'wb') as f:
                pickle.dump(data, f)
            print("Mean frames saved to pickle")
        except Exception as e:
            print(f"Error saving pickle: {e}")

        self.finish_data_loading()

    def finish_data_loading(self):
        self.params_all = {
            'reference_session': 0,
            'sessions': {i: {'x_shift': 0, 'y_shift': 0, 'rotation': 0, 'scale': 1.0}
                         for i in range(self.n_sessions)}
        }

        self.session_selector.clear()
        self.session_selector.addItems([f"Session {i}" for i in range(self.n_sessions)])
        self.ref_selector.clear()
        self.ref_selector.addItems([f"Session {i}" for i in range(self.n_sessions)])

        self.session_selector.currentIndexChanged.connect(self.change_session)
        self.ref_selector.currentIndexChanged.connect(self.change_reference)

        self.session_idx = 0
        self.ref_idx = 0
        self.ref_selector.setCurrentIndex(self.ref_idx)
        self.update_moving_session_selector()

        self.enable_controls(True)
        self.load_params()
        self.status_label.setText(
            f"Ready - {self.n_sessions} sessions loaded (data type: {self.detected_dtype})"
        )
        self.update_image()

    # ------------------- Session Changes -------------------
    def change_session(self, selector_idx):
        available_sessions = [i for i in range(self.n_sessions) if i != self.ref_idx]
        if selector_idx < len(available_sessions):
            self.session_idx = available_sessions[selector_idx]
        else:
            self.session_idx = 0

        params = self.params_all['sessions'].get(
            self.session_idx, {'x_shift': 0, 'y_shift': 0, 'rotation': 0, 'scale': 1.0}
        )
        self.x_slider.setValue(params['x_shift'])
        self.y_slider.setValue(params['y_shift'])
        self.rot_slider.setValue(int(params['rotation'] * 10))
        self.scale_slider.setValue(int(params['scale'] * 100))
        self.slider_changed(self.x_slider.value(), self.x_label, "X Shift", self.x_input)
        self.slider_changed(self.y_slider.value(), self.y_label, "Y Shift", self.y_input)
        self.slider_changed(self.rot_slider.value(), self.rot_label, "Rotation", self.rot_input)
        self.slider_changed(self.scale_slider.value(), self.scale_label, "Scale", self.scale_input)
        self.slider_changed(self.alpha_slider.value(), self.alpha_label, "Alpha", self.alpha_input)
        self.update_image()

    def update_moving_session_selector(self):
        if not hasattr(self, 'session_selector') or not hasattr(self, 'ref_selector'):
            return

        current_selection = self.session_selector.currentIndex()
        self.session_selector.clear()
        available_sessions = []
        for i in range(self.n_sessions):
            if i != self.ref_idx:
                available_sessions.append(i)
                self.session_selector.addItem(f"Session {i}")

        if current_selection == self.ref_idx or current_selection not in available_sessions:
            if available_sessions:
                self.session_idx = available_sessions[0]
                self.session_selector.setCurrentIndex(0)
            else:
                self.session_idx = 0
        else:
            try:
                new_index = available_sessions.index(current_selection)
                self.session_idx = current_selection
                self.session_selector.setCurrentIndex(new_index)
            except ValueError:
                self.session_idx = available_sessions[0]
                self.session_selector.setCurrentIndex(0)

    def change_reference(self, idx):
        self.ref_idx = idx
        self.params_all['reference_session'] = idx
        self.update_moving_session_selector()
        if self.mean_frames is not None:
            self.update_image()

    def change_exten(self):
        ext = self.exten_selector.currentText()
        self.exten = f"*.{ext}"

    # ------------------- Image Update -------------------
    def update_image(self):
        if self.mean_frames is None:
            return

        x_shift  = self.x_slider.value()
        y_shift  = self.y_slider.value()
        rotation = self.rot_slider.value() / 10
        scale    = self.scale_slider.value() / 100.0
        alpha    = self.alpha_slider.value() / 100.0

        self.params_all['sessions'][self.session_idx] = {
            'x_shift': x_shift,
            'y_shift': y_shift,
            'rotation': rotation,
            'scale': scale
        }

        ref_img    = self.mean_frames[self.ref_idx]
        target_img = self.mean_frames[self.session_idx]

        # Apply scale
        if scale != 1.0:
            scaled = zoom(target_img, scale, order=0)
            if scale > 1.0:
                h, w = target_img.shape
                sh, sw = scaled.shape
                start_h = (sh - h) // 2
                start_w = (sw - w) // 2
                scaled = scaled[start_h:start_h+h, start_w:start_w+w]
            else:
                h, w = target_img.shape
                sh, sw = scaled.shape
                pad_h = (h - sh) // 2
                pad_w = (w - sw) // 2
                padded = np.zeros_like(target_img)
                padded[pad_h:pad_h+sh, pad_w:pad_w+sw] = scaled
                scaled = padded
        else:
            scaled = target_img.copy()

        rotated = rotate(scaled, rotation, reshape=False, order=0)
        aligned = shift(rotated, (y_shift, x_shift), order=0)

        self.ax.clear()
        self.ax.imshow(ref_img, cmap='gray', alpha=1)
        self.ax.imshow(aligned, cmap='inferno', alpha=alpha)

        # Draw crop overlay
        if self.crop_enable_checkbox.isChecked():
            top, bottom, left, right = self.get_crop_params()
            h_img, w_img = ref_img.shape
            crop_w = w_img - left - right
            crop_h = h_img - top - bottom

            if crop_w > 0 and crop_h > 0:
                # Semi-transparent dark bands over the discarded regions
                band_color = 'black'
                band_alpha = 0.6
                for (bx, by, bw, bh) in [
                    (-0.5,               -0.5,               w_img,  top),            # top
                    (-0.5,               h_img - bottom - 0.5, w_img, bottom + 0.5),  # bottom
                    (-0.5,               top - 0.5,          left,   crop_h + 1),     # left
                    (w_img - right - 0.5, top - 0.5,         right + 0.5, crop_h + 1), # right
                ]:
                    self.ax.add_patch(Rectangle(
                        (bx, by), bw, bh,
                        linewidth=0, facecolor=band_color, alpha=band_alpha
                    ))
                # Cyan dashed border around the kept region
                self.ax.add_patch(Rectangle(
                    (left - 0.5, top - 0.5), crop_w, crop_h,
                    linewidth=2, edgecolor='#00e5ff', facecolor='none', linestyle='--'
                ))
                self.crop_size_label.setText(f"Output size: {crop_w} × {crop_h} px")
            else:
                self.crop_size_label.setText("Output size: invalid crop")
        else:
            if self.mean_frames is not None:
                h_img, w_img = self.mean_frames[self.ref_idx].shape
                self.crop_size_label.setText(f"Output size: {w_img} × {h_img} px (no crop)")

        self.canvas.draw_idle()

    # ------------------- Save / Load -------------------
    def apply_and_save(self):
        print("Saving to:", self.savepath)
        self.status_label.setText("Saving aligned TIFF...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.enable_controls(False)

        # Blocked-frames correction
        if self.correction_enable_checkbox.isChecked():
            selected_sessions = list(range(self.n_sessions))
        else:
            selected_sessions = []
        try:
            corr_threshold = int(float(self.correction_threshold_input.text()))
        except Exception:
            corr_threshold = 1000
        try:
            corr_window = int(float(self.correction_window_input.text()))
        except Exception:
            corr_window = 2

        # Crop params
        crop_params = None
        if self.crop_enable_checkbox.isChecked():
            crop_params = self.get_crop_params()

        self.save_thread = SaveThread(
            self.folderpath,
            self.exten,
            self.params_all,
            self.ref_idx,
            self.savepath.as_posix(),
            self.detected_dtype,
            blocked_sessions=selected_sessions,
            blocked_threshold=corr_threshold,
            blocked_window=corr_window,
            crop_params=crop_params,
        )
        self.save_thread.progress_updated.connect(self.progress_bar.setValue)
        self.save_thread.status_updated.connect(self.status_label.setText)
        self.save_thread.saving_finished.connect(self.on_saving_finished)
        self.save_thread.start()

    def on_saving_finished(self, success, message):
        self.progress_bar.setVisible(False)
        self.enable_controls(True)
        if success:
            self.status_label.setText(f"{message} ({self.n_sessions} sessions)")
            print(message)
        else:
            self.status_label.setText(message)
            print(message)

    def save_params(self):
        if hasattr(self, 'params_path'):
            if self.params_path.exists():
                os.remove(self.params_path)
            self.params_all['reference_session'] = self.ref_idx
            with open(self.params_path, 'wb') as f:
                pickle.dump(self.params_all, f)
            print("Alignment parameters saved.")

    def load_params(self):
        if hasattr(self, 'params_path') and self.params_path.exists():
            with open(self.params_path, 'rb') as f:
                self.params_all = pickle.load(f)
            self.ref_idx = self.params_all['reference_session']
            print("Alignment parameters loaded.")
            if hasattr(self, 'ref_selector') and hasattr(self, 'session_selector'):
                self.ref_selector.setCurrentIndex(self.ref_idx)
                self.update_moving_session_selector()
                self.change_session(0)
        else:
            print("No saved parameters found.")

    # ------------------- Auto Alignment Parameter Parsing -------------------
    def get_auto_align_params(self):
        try:
            params = {}
            if self.rot_checkbox.isChecked():
                params['rot_min']  = float(self.rot_min_input.text())
                params['rot_max']  = float(self.rot_max_input.text())
                params['rot_step'] = float(self.rot_step_input.text())
            else:
                params['rot_min'] = params['rot_max'] = 0.0
                params['rot_step'] = 1.0

            if self.scale_checkbox.isChecked():
                params['scale_min']  = float(self.scale_min_input.text())
                params['scale_max']  = float(self.scale_max_input.text())
                params['scale_step'] = float(self.scale_step_input.text())
            else:
                params['scale_min'] = params['scale_max'] = 1.0
                params['scale_step'] = 1.0

            params['trans_max'] = int(float(self.trans_max_input.text())) \
                if self.trans_checkbox.isChecked() else 0

            params['crop_frac'] = float(self.crop_input.text())

            if params['rot_min'] > params['rot_max']:
                params['rot_min'], params['rot_max'] = params['rot_max'], params['rot_min']
            if params['scale_min'] > params['scale_max']:
                params['scale_min'], params['scale_max'] = params['scale_max'], params['scale_min']
            if not (0 < params['crop_frac'] <= 1):
                params['crop_frac'] = 0.6

            return params
        except (ValueError, AttributeError):
            return {
                'rot_min': -10.0, 'rot_max': 10.0, 'rot_step': 0.1,
                'scale_min': 0.95, 'scale_max': 1.05, 'scale_step': 0.01,
                'trans_max': 50, 'crop_frac': 0.6
            }

    # ------------------- Auto Alignment -------------------
    def auto_align(self):
        if self.mean_frames is None or self.n_sessions < 2:
            return
        if self.session_idx == self.ref_idx:
            self.status_label.setText("Select a moving session different from the reference.")
            return

        ref_img = self.mean_frames[self.ref_idx]
        mov_img = self.mean_frames[self.session_idx]

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Aligning Session {self.session_idx}")
        self.enable_controls(False)

        params = self.get_auto_align_params()
        self.auto_thread = AutoAlignThread(
            ref_img=ref_img, mov_img=mov_img,
            rot_min=params['rot_min'], rot_max=params['rot_max'], rot_step=params['rot_step'],
            scale_min=params['scale_min'], scale_max=params['scale_max'], scale_step=params['scale_step'],
            crop_frac=params['crop_frac'], trans_max=params['trans_max'], verbose=False
        )
        self.auto_thread.progress_updated.connect(self.progress_bar.setValue)
        self.auto_thread.finished_with_params.connect(self.on_auto_align_finished)
        self.auto_thread.start()

    def on_auto_align_finished(self, best_params):
        try:
            x_shift  = int(best_params.get('x_shift', 0))
            y_shift  = int(best_params.get('y_shift', 0))
            rotation = float(best_params.get('rotation', 0.0))
            scale    = float(best_params.get('scale', 1.0))
        except Exception:
            x_shift = y_shift = 0; rotation = 0.0; scale = 1.0

        self.x_slider.setValue(x_shift)
        self.y_slider.setValue(y_shift)
        self.rot_slider.setValue(int(round(rotation * 10)))
        self.scale_slider.setValue(int(round(scale * 100)))
        self.slider_changed(self.x_slider.value(), self.x_label, "X Shift", self.x_input)
        self.slider_changed(self.y_slider.value(), self.y_label, "Y Shift", self.y_input)
        self.slider_changed(self.rot_slider.value(), self.rot_label, "Rotation", self.rot_input)
        self.slider_changed(self.scale_slider.value(), self.scale_label, "Scale", self.scale_input)

        self.update_image()
        self.progress_bar.setVisible(False)
        self.enable_controls(True)
        self.status_label.setText("Auto alignment complete.")

    def auto_align_all(self):
        if self.mean_frames is None or self.n_sessions < 2:
            return
        self.enable_controls(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._sessions_to_align = [i for i in range(self.n_sessions) if i != self.ref_idx]
        self._align_all_index = 0
        self._run_next_auto_align_for_all()

    def _run_next_auto_align_for_all(self):
        if self._align_all_index >= len(self._sessions_to_align):
            self.progress_bar.setVisible(False)
            self.enable_controls(True)
            self.status_label.setText("Auto alignment for all sessions complete.")
            self.update_image()
            return

        cur_session = self._sessions_to_align[self._align_all_index]
        self.status_label.setText(f"Aligning Session {cur_session}")
        ref_img = self.mean_frames[self.ref_idx]
        mov_img = self.mean_frames[cur_session]
        self._aligning_session_idx = cur_session

        params = self.get_auto_align_params()
        self.auto_thread = AutoAlignThread(
            ref_img=ref_img, mov_img=mov_img,
            rot_min=params['rot_min'], rot_max=params['rot_max'], rot_step=params['rot_step'],
            scale_min=params['scale_min'], scale_max=params['scale_max'], scale_step=params['scale_step'],
            crop_frac=params['crop_frac'], trans_max=params['trans_max'], verbose=False
        )
        self.auto_thread.progress_updated.connect(self._on_all_progress)
        self.auto_thread.finished_with_params.connect(self._on_one_session_auto_finished)
        self.auto_thread.start()

    def _on_all_progress(self, p):
        total = len(self._sessions_to_align)
        idx = self._align_all_index
        overall = int(((idx + p / 100.0) / max(1, total)) * 100)
        self.progress_bar.setValue(overall)

    def _on_one_session_auto_finished(self, best_params):
        session_idx = getattr(self, '_aligning_session_idx', self.session_idx)
        try:
            x_shift  = int(best_params.get('x_shift', 0))
            y_shift  = int(best_params.get('y_shift', 0))
            rotation = float(best_params.get('rotation', 0.0))
            scale    = float(best_params.get('scale', 1.0))
        except Exception:
            x_shift = y_shift = 0; rotation = 0.0; scale = 1.0

        self.params_all['sessions'][session_idx] = {
            'x_shift': x_shift, 'y_shift': y_shift,
            'rotation': rotation, 'scale': scale
        }

        if session_idx == self.session_idx:
            self.x_slider.setValue(x_shift)
            self.y_slider.setValue(y_shift)
            self.rot_slider.setValue(int(round(rotation * 10)))
            self.scale_slider.setValue(int(round(scale * 100)))
            self.slider_changed(self.x_slider.value(), self.x_label, "X Shift", self.x_input)
            self.slider_changed(self.y_slider.value(), self.y_label, "Y Shift", self.y_input)
            self.slider_changed(self.rot_slider.value(), self.rot_label, "Rotation", self.rot_input)
            self.slider_changed(self.scale_slider.value(), self.scale_label, "Scale", self.scale_input)
            self.update_image()

        self._align_all_index += 1
        self._run_next_auto_align_for_all()


# ------------------- Run -------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AlignGUI()
    window.show()
    sys.exit(app.exec_())