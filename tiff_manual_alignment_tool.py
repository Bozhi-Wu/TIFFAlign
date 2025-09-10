import os
import sys
import scipy
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import shift, rotate
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from tifffile import TiffWriter, memmap, TiffFile
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QFileDialog, QProgressBar, QSizePolicy

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
    
    # Only load the first max_frames for mean calculation
    vid = np.reshape(fid[:n_frames_to_load * framesize], (n_frames_to_load, height, width))
    return (maxint - vid).astype(np.uint16), n_frames

def load_tiff_optimized(filepath, max_frames=100):
    """Optimized TIFF loading that only loads the first max_frames for mean calculation"""
    with TiffFile(filepath) as tif:
        n_frames = len(tif.pages)
        n_frames_to_load = min(n_frames, max_frames)
        
        # Load only the first max_frames
        frames = []
        for i in range(n_frames_to_load):
            frames.append(tif.pages[i].asarray())
        
        return np.array(frames), n_frames

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
            n_sections = 0
            n_total_frames = 0
            
            for i, file in enumerate(files):
                self.status_updated.emit(f"Processing file {i+1}/{len(files)}: {file.name}")
                
                if self.exten == "*.sbx":
                    fpath = file.as_posix().split('.')[0]
                    frames, total_frames = sbx_to_frames_optimized(fpath, self.n_frames_averaged)
                elif self.exten == "*.tiff":
                    frames, total_frames = load_tiff_optimized(file.as_posix(), self.n_frames_averaged)
                
                n_total_frames += total_frames
                n_sections += 1
                
                # Calculate mean frame
                mean_frame = np.mean(frames, axis=0)
                mean_frames.append(mean_frame.astype(np.uint16))
                
                # Update progress
                progress = int((i + 1) / len(files) * 100)
                self.progress_updated.emit(progress)
            
            result = {
                'mean_frames': mean_frames,
                'n_sections': n_sections,
                'n_total_frames': n_total_frames
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

    def __init__(self, folderpath, exten, params_all, ref_idx, savepath):
        super().__init__()
        self.folderpath = folderpath
        self.exten = exten
        self.params_all = params_all
        self.ref_idx = ref_idx
        self.savepath = savepath

    def run(self):
        try:
            # Remove existing output if present
            if os.path.exists(self.savepath):
                try:
                    os.remove(self.savepath)
                except Exception as e:
                    self.saving_finished.emit(False, f"Error deleting existing file: {e}")
                    return

            # Discover files to process
            if self.exten == "*.tiff":
                files = sorted(list(self.folderpath.rglob("*.tiff")) +
                               list(self.folderpath.rglob("*.tif")))
            else:
                files = sorted(self.folderpath.rglob(self.exten))

            if not files:
                self.saving_finished.emit(False, f"No files found with extension {self.exten}")
                return

            # Pre-compute total frames for progress
            total_frames = 0
            per_file_counts = []
            for file in files:
                if self.exten == "*.tiff":
                    with TiffFile(file.as_posix()) as tif:
                        n = len(tif.pages)
                else:  # *.sbx
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
                    else:  # *.tiff
                        frames = memmap(file.as_posix())

                    params = self.params_all.get(nn, {'x_shift': 0, 'y_shift': 0, 'rotation': 0})

                    chunk_size = 1000
                    for i in range(0, frames.shape[0], chunk_size):
                        end_idx = min(i + chunk_size, frames.shape[0])
                        chunk = frames[i:end_idx]

                        for frame in chunk:
                            if nn != self.ref_idx:
                                rotated = rotate(frame, params['rotation'], reshape=False, order=0)
                                frame = shift(rotated, [params['y_shift'], params['x_shift']], order=0)
                            tiff_writer.write(frame.astype(np.uint16))

                            # Update progress per frame
                            frames_done += 1
                            # Avoid excessive signal emissions by batching a bit
                            if total_frames > 0 and frames_done % 50 == 0:
                                progress = int(frames_done / total_frames * 100)
                                self.progress_updated.emit(progress)

                    # Ensure progress updates at file boundaries
                    progress = int(frames_done / total_frames * 100) if total_frames > 0 else 100
                    self.progress_updated.emit(progress)

            self.progress_updated.emit(100)
            self.saving_finished.emit(True, "Saved aligned TIFF!")
        except Exception as e:
            self.saving_finished.emit(False, f"Error during saving: {str(e)}")

# --- GUI Class ---
class AlignGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Manual Alignment Tool')
        self.setGeometry(300, 150, 900, 1200)
        self.section_idx = 0
        self.ref_idx = 0
        self.alpha = 0.5
        self.exten = "*.sbx"
        self.mean_frames = None
        self.n_sections = 0
        self.n_total_frames = 0
        self.params_all = {}
        self.loading_thread = None
        self.save_thread = None

        self.set_dark_theme()

        # --- Layouts ---
        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()      # left: selectors, right: sliders
        selectors_layout = QVBoxLayout()    # stacked vertically
        sliders_layout = QVBoxLayout()
        
        # --- Selectors layout ---
        selectors_layout = QVBoxLayout()

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

        # Folder selection button (use empty label to align with others)
        self.folder_button = QPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_folder)
        add_row("", self.folder_button)

        # Reference selector
        self.ref_selector = QComboBox()
        add_row("Reference Section:", self.ref_selector)

        # Moving section selector
        self.section_selector = QComboBox()
        add_row("Moving Section:", self.section_selector)

        # --- Sliders ---
        self.x_label, self.x_slider = self.create_slider("X Shift", -50, 50, 0, sliders_layout)
        self.y_label, self.y_slider = self.create_slider("Y Shift", -50, 50, 0, sliders_layout)
        self.rot_label, self.rot_slider = self.create_slider("Rotation", -100, 100, 0, sliders_layout)
        self.alpha_label, self.alpha_slider = self.create_slider("Alpha", 0, 100, 50, sliders_layout)

        # --- Save buttons ---
        self.save_params_button = QPushButton("Save Alignment Parameters")
        self.save_params_button.clicked.connect(self.save_params)
        sliders_layout.addWidget(self.save_params_button)

        self.load_params_button = QPushButton("Load Alignment Parameters")
        self.load_params_button.clicked.connect(self.load_params)
        sliders_layout.addWidget(self.load_params_button)
        
        self.save_button = QPushButton("Save Aligned TIFF")
        self.save_button.clicked.connect(self.apply_and_save)
        sliders_layout.addWidget(self.save_button)
        
        # --- Progress indicators (footer) ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #aaaaaa; font-size: 14px;")

        # Disable sliders/buttons initially
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

        # Footer layout at the bottom spanning full width
        footer_layout = QHBoxLayout()
        footer_layout.addWidget(self.status_label, 0)
        footer_layout.addWidget(self.progress_bar, 1)
        main_layout.addLayout(footer_layout)
        self.setLayout(main_layout)
        
        if self.mean_frames is not None:
            self.update_image()

    # ------------------- Helper Methods -------------------
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
            QSlider::handle:horizontal{
                background: #aaaaaa;
                border: 2px solid #5c5c5c;
                width: 10px;
                height: 10px;
                margin: -7px 0;
                border-radius: 7px;
            }""")
        slider.valueChanged.connect(lambda val, l=label, n=name: self.slider_changed(val, l, n))
        layout.addWidget(label)
        layout.addWidget(slider)
        return label, slider

    def slider_changed(self, value, label, name):
        if name == "Rotation":
            label.setText(f"{name}: {value / 10:.1f}")
        elif name == "Alpha":
            label.setText(f"{name}: {value / 100:.2f}")
        else:
            label.setText(f"{name}: {value}")
        if self.mean_frames is not None:
            self.update_image()

    def enable_controls(self, enable=True):
        self.x_slider.setEnabled(enable)
        self.y_slider.setEnabled(enable)
        self.rot_slider.setEnabled(enable)
        self.alpha_slider.setEnabled(enable)
        self.save_button.setEnabled(enable)
        self.save_params_button.setEnabled(enable)
        self.load_params_button.setEnabled(enable)
        self.ref_selector.setEnabled(enable)
        self.section_selector.setEnabled(enable)

    # ------------------- Folder / SBX Loading -------------------
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

        # Load or compute mean frames
        if self.pickle_path.exists():
            print("Loading mean frames from pickle ---")
            self.status_label.setText("Loading cached data...")
            try:
                with open(self.pickle_path, 'rb') as f:
                    data = pickle.load(f)
                    self.mean_frames = data['mean_frames']
                    self.n_sections = data['n_sections']
                    self.n_total_frames = data['n_total_frames']
                self.finish_data_loading()
            except Exception as e:
                print(f"Error loading pickle: {e}")
                self.status_label.setText("Error loading cached data, recomputing...")
                self.start_background_loading()
        else:
            print("Computing and saving mean frames ---")
            self.start_background_loading()
    
    def start_background_loading(self):
        """Start background loading with progress indication"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting data loading...")
        
        # Disable controls during loading
        self.enable_controls(False)
        
        # Start background thread
        self.loading_thread = DataLoaderThread(self.folderpath, self.exten, self.n_frames_averaged)
        self.loading_thread.progress_updated.connect(self.progress_bar.setValue)
        self.loading_thread.status_updated.connect(self.status_label.setText)
        self.loading_thread.loading_finished.connect(self.on_loading_finished)
        self.loading_thread.start()
    
    def on_loading_finished(self, result):
        """Handle completion of background loading"""
        self.progress_bar.setVisible(False)
        
        if result is None:
            self.status_label.setText("Loading failed")
            return
            
        # Store the loaded data
        self.mean_frames = result['mean_frames']
        self.n_sections = result['n_sections']
        self.n_total_frames = result['n_total_frames']
        
        # Save to pickle for future use
        data = {
            'mean_frames': self.mean_frames,
            'n_sections': self.n_sections,
            'n_total_frames': self.n_total_frames
        }
        
        try:
            with open(self.pickle_path, 'wb') as f:
                pickle.dump(data, f)
            print("Mean frames saved to pickle")
        except Exception as e:
            print(f"Error saving pickle: {e}")
        
        self.finish_data_loading()
    
    def finish_data_loading(self):
        """Complete the data loading process"""
        # Prepare params_all
        self.params_all = {i: {'x_shift': 0, 'y_shift': 0, 'rotation': 0} for i in range(self.n_sections)}

        # Update selectors
        self.section_selector.clear()
        self.section_selector.addItems([f"Section {i}" for i in range(self.n_sections)])
        self.ref_selector.clear()
        self.ref_selector.addItems([f"Section {i}" for i in range(self.n_sections)])

        # Reconnect signals
        self.section_selector.currentIndexChanged.connect(self.change_section)
        self.ref_selector.currentIndexChanged.connect(self.change_reference)

        # Set defaults
        self.section_idx = 0
        self.ref_idx = 0
        self.section_selector.setCurrentIndex(self.section_idx)
        self.ref_selector.setCurrentIndex(self.ref_idx)

        # Enable sliders and buttons
        self.enable_controls(True)
        self.status_label.setText(f"Ready - {self.n_sections} sections loaded")
        self.update_image()

    # ------------------- Section Changes -------------------
    def change_section(self, idx):
        self.section_idx = idx
        # Get parameters for this section, default to zeros
        params = self.params_all.get(self.section_idx, {'x_shift': 0, 'y_shift': 0, 'rotation': 0})
        # Update sliders to match stored parameters
        self.x_slider.setValue(params['x_shift'])
        self.y_slider.setValue(params['y_shift'])
        self.rot_slider.setValue(int(params['rotation'] * 10))  # slider stores 10x rotation
        # Update slider labels
        self.slider_changed(self.x_slider.value(), self.x_label, "X Shift")
        self.slider_changed(self.y_slider.value(), self.y_label, "Y Shift")
        self.slider_changed(self.rot_slider.value(), self.rot_label, "Rotation")
        self.slider_changed(self.alpha_slider.value(), self.alpha_label, "Alpha")
        # Refresh image
        self.update_image()

    def change_reference(self, idx):
        self.ref_idx = idx
        if self.mean_frames is not None:
            self.update_image()
            
    def change_exten(self):
        ext = self.exten_selector.currentText()
        self.exten = f"*.{ext}"

    # ------------------- Image Update -------------------
    def update_image(self):
        if self.mean_frames is None:
            return

        # Read current slider values
        x_shift = self.x_slider.value()
        y_shift = self.y_slider.value()
        rotation = self.rot_slider.value() / 10
        alpha = self.alpha_slider.value() / 100.0

        # Update params_all for current section
        self.params_all[self.section_idx] = {
            'x_shift': x_shift,
            'y_shift': y_shift,
            'rotation': rotation
        }

        # Get images
        ref_img = self.mean_frames[self.ref_idx]
        target_img = self.mean_frames[self.section_idx]

        # Apply rotation & shift
        rotated = rotate(target_img, rotation, reshape=False, order=0)
        aligned = shift(rotated, (y_shift, x_shift), order=0)

        # Display
        self.ax.clear()
        self.ax.imshow(ref_img, cmap='gray', alpha=1)
        self.ax.imshow(aligned, cmap='inferno', alpha=alpha)
        self.canvas.draw_idle()

    # ------------------- Save / Load -------------------
    def apply_and_save(self):
        print("Saving to:", self.savepath)
        self.status_label.setText("Saving aligned TIFF...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.enable_controls(False)

        # Start background save thread
        self.save_thread = SaveThread(self.folderpath, self.exten, self.params_all, self.ref_idx, self.savepath.as_posix())
        self.save_thread.progress_updated.connect(self.progress_bar.setValue)
        self.save_thread.status_updated.connect(self.status_label.setText)
        self.save_thread.saving_finished.connect(self.on_saving_finished)
        self.save_thread.start()

    def on_saving_finished(self, success, message):
        self.progress_bar.setVisible(False)
        self.enable_controls(True)
        if success:
            self.status_label.setText(f"{message} ({self.n_sections} sections)")
            print(message)
        else:
            self.status_label.setText(message)
            print(message)

    def save_params(self):
        if hasattr(self, 'params_path'):
            if self.params_path.exists():
                os.remove(self.params_path)
            with open(self.params_path, 'wb') as f:
                pickle.dump(self.params_all, f)
            print(f"Alignment parameters saved to {self.params_path}")

    def load_params(self):
        if hasattr(self, 'params_path') and self.params_path.exists():
            with open(self.params_path, 'rb') as f:
                self.params_all = pickle.load(f)
            print(f"Loaded alignment parameters from {self.params_path}")
            # Update sliders and image for current section
            self.change_section(self.section_idx)
        else:
            print("No saved parameters found.")


# ------------------- Run -------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AlignGUI()
    window.show()
    sys.exit(app.exec_())
