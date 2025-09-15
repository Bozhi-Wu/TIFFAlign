# TIFFAlign: Auto Alignment Across Imaging Sessions

This repository provides a **PyQt5-based GUI** for both automatic and manual alignment of one-photon / two-photon imaging sessions (`.sbx` or `.tiff` files). It allows interactive adjustment of **x/y shifts, rotation, scaling, and transparency**, and can save aligned data into a single TIFF stack.

Although many excellent automated motion correction algorithms exist, they are generally optimized for **within-session motion artifacts**. When working with **multiple sessions concatenated together**, shifts or rotations often occur between sessions. In these cases, we found that a manual **pre-alignment** step produces better results before applying automated motion correction. This little GUI was developed to streamline that process.

![GUI Screenshot](preview/GUI_sample.png)

---

## Features

- Supports both `.sbx` and `.tiff` files
- Compute **mean frames** for the first 100 frames of each session (cached with pickle for faster reloads)
- Select any session as the **reference session**
- Interactive controls for:
  - X shift  
  - Y shift  
  - Rotation  
  - Scaling
  - Alpha (Transparency)
- **Auto alignment** with grid search
- Save / load alignment parameters
- Export aligned sessions as a concatenated tiff file.
- Dark theme interface with **live overlay preview**


## Requirements and Installation

### 1. Clone this repository
```bash
git clone https://github.com/Bozhi-Wu/TIFFAlign.git
cd TIFFAlign
```

### 2. Create and activate a conda environment
```bash
conda create -n tiffalign python=3.11
conda activate tiffalign
```

### 3. Install dependencies
```bash
conda install pyqt=5 numpy=1.24 scipy=1.14 tifffile=2023 matplotlib -c conda-forge
```

---

## Usage

Run the GUI:

```bash
python TIFFAlign.py
```
1. Choose the **File Extension** (`sbx` or `tiff`).
2. Select a **Folder** containing `.sbx` or `.tif`/`.tiff` files. 
   - The files can exist in subfolders as `rglob` is used to find the files.
   - For the sessions to be concatenated in the correct order, please name the files so that they can be easily sorted with the python `sorted` function (e.g., by adding leading numbers like 000, 001, etc.)
   - For `.sbx` files, the corresponding `.mat` metadata files are also required.
   - For the very first time, it will read and compute and mean frames for the first 100 frames of each session. These will be cached into `mean_frames.pkl` for future reload.
   - If there is an existing alignment parameter file `params_all.pkl` in the folder, it will be automatically loaded.
3. Pick a **Reference Session** and a **Moving Session**.
   - The reference session is shown in grayscale, while the moving session is overlaid using the *inferno* colormap.  
   - Use the alpha slider to adjust transparency and better visualize the alignment.  
   - When saving, the reference session remains unchanged; only the moving sessions will be adjusted.  
4.	Adjust parameters in the **“Auto Alignment Parameters”** section:
	- Use the checkboxes to select which parameters will be included in the search.
	- On the right panel, set the **Max**, **Min**, and **Step** for each parameter.
	- Since the edges of the imaging field can vary more between sessions and may not contain neurons, you can adjust the **Crop** setting so that only the central region of the frame is used for auto-alignment evaluation.
5.	Run auto alignment using either **“Auto Align Current Session”** or **“Auto Align All Sessions”**.
6. (Optional) Adjust alignment manually using the **Sliders**:
   - X Shift
   - Y Shift
   - Rotation
   - Scaling
   - Alpha (Transparency)
7. Save alignment parameters for later use.
8. Export the aligned file with **Save Aligned TIFF**.  
   - Files are read using `memmap` by default for efficiency.  
   - The output TIFF is written using the detected data type of the original files (for `.sbx` files, `uint16` is used).  
   - If `memmap` fails, the tool automatically falls back to `imread`, which loads the entire file into memory.  
   - ⚠️ Make sure your computer has enough available RAM when working with large files.  

---

## Outputs

- `params_all.pkl` – Stored alignment parameters.  
- `mean_frames.pkl` – Cached mean frames for faster reload.  
- `tiff_manual_aligned.tiff` – Exported aligned tiff file. 
