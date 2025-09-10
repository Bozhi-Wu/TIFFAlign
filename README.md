# Manual Alignment Tool for Imaging Sessions

This repository provides a **PyQt5-based GUI** for manually aligning two-photon imaging sessions (`.sbx` or `.tiff` files).  
It allows interactive adjustment of **x/y shifts, rotation, and transparency**, and can save aligned data into a single TIFF stack.

---

## ✨ Features

- Load and preview `.sbx` or `.tiff` files
- Compute **mean frames** for each session (cached with pickle for faster reloads)
- Select any session as the **reference**
- Interactive controls for:
  - X shift  
  - Y shift  
  - Rotation  
  - Transparency blending
- Save / load alignment parameters
- Export aligned movies as a **multi-page TIFF**
- Dark theme interface with live overlay preview

---

## 📦 Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Requirements

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

## 🚀 Usage

Run the GUI:

```bash
python align_gui.py
```

### Workflow
1. Select a folder containing `.sbx` or `.tiff` files.
2. Choose the **file type** (`sbx` or `tiff`).
3. Pick a **reference section** and a **moving section**.
4. Adjust alignment using the **sliders**:
   - X Shift
   - Y Shift
   - Rotation
   - Alpha (transparency)
5. Save alignment parameters for later use.
6. Export aligned data as a **multi-page TIFF**.

---

## 🖼️ Screenshot

*(Add a screenshot of the GUI here, e.g. `GUI_sample.png`)*

![GUI Screenshot](GUI_sample.png)

---

## 💾 Outputs

- `params_all.pkl` – stored alignment parameters  
- `mean_frames.pkl` – cached mean frames for faster reload  
- `tiff_manual_aligned.tiff` – exported aligned movie  

---

## ⚠️ Notes

- `.sbx` files require corresponding `.mat` metadata files.  
- Only the first **100 frames** are loaded initially for mean frame calculation (configurable).  

---

## 📜 License

MIT License (modify as needed).  
