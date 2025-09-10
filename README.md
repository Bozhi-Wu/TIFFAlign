# Manual Alignment Tool for Imaging Sessions

This repository provides a **PyQt5-based GUI** for manually aligning one-photon / two-photon imaging sessions (`.sbx` or `.tiff` files). It allows interactive adjustment of **x/y shifts, rotation, and transparency**, and can save aligned data into a single TIFF stack.

Although many excellent automated motion correction algorithms exist, they are generally optimized for **within-session motion artifacts**. When working with **multiple sessions concatenated together**, shifts or rotations often occur between sessions. In these cases, we found that a manual **pre-alignment** step produces better results before applying automated motion correction. This little GUI was developed to streamline that process.

![GUI Screenshot](GUI_sample.png)

---

## Features

- Supports both `.sbx` and `.tiff` files
- Compute **mean frames** for the first 100 frames of each session (cached with pickle for faster reloads)
- Select any session as the **reference** session
- Interactive controls for:
  - X shift  
  - Y shift  
  - Rotation  
  - Transparency blending
- Save / load **alignment parameters**
- Export aligned sessions as a concatenated tiff file.
- Dark theme interface with **live overlay preview**


## Requirements and Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/Bozhi-Wu/tiff_manual_alignment_tool.git
```
- Python 3.8+
- PyQt5
- numpy
- scipy
- matplotlib
- tifffile
- tqdm

You can install them directly:

```bash
pip install pyqt5 numpy scipy matplotlib tifffile tqdm
```

---

## Usage

Run the GUI:

```bash
python tiff_manual_alignment_tool.py
```
1. Choose the **File Extension** (`sbx` or `tiff`).
2. Select a **Folder** containing `.sbx` or `.tif`/`.tiff` files. 
   - The files can exist in subfolders as `rglob` is used to find the files.
   - For the sessions to be concatenated in the correct order, please name the files so that they can be easily sorted with the python `sorted` function.
   - For `.sbx` files, the corresponding `.mat` metadata files are also required.
   - For the very first time, it will read and compute and **mean frames** for the first 100 frames of each session. These will be cached into `mean_frames.pkl` for future reload.
   - If there is an existing alignment parameter file `params_all.pkl` in the folder, it will be automatically loaded.
3. Pick a **reference session** and a **moving session**.
   - After picking the reference session, please only align the other sessions to the reference, without adjust the reference session itself. During the saving process, no adjustment will be made to the reference session. 
4. Adjust alignment using the **sliders**:
   - X Shift
   - Y Shift
   - Rotation
   - Alpha (transparency)
5. Save alignment parameters for later use.
6. Export the aligned file with `Save Aligned TIFF`.

---

## Outputs

- `params_all.pkl` – Stored alignment parameters.  
- `mean_frames.pkl` – Cached mean frames for faster reload.  
- `tiff_manual_aligned.tiff` – Exported aligned tiff file. 
