import os
import sys
import copy
import pickle
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io
from scipy.ndimage import shift, rotate, zoom
from tifffile import TiffWriter, TiffFile, memmap

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtGui import QPalette, QColor, QIntValidator
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QComboBox, QFileDialog, QProgressBar, QSizePolicy, QMessageBox,
    QLineEdit, QCheckBox, QGroupBox, QGridLayout, QFormLayout, QScrollArea, QFrame,
    QToolButton, QAbstractSlider, QStyleFactory
)

OUTPUT_NAME = 'tiff_aligned.tiff'
# Never treat our own output as an input session.
EXCLUDED_NAMES = {OUTPUT_NAME}
DEFAULT_PARAMS = {'x_shift': 0, 'y_shift': 0, 'rotation': 0.0, 'scale': 1.0}
# Saving pipeline. The chunk size only needs to be large enough to amortise per-chunk overhead
# (benchmarks: disk throughput is flat from ~50 frames up) and small enough to bound memory:
# roughly (SAVE_PREFETCH + 2) chunks are in RAM at once. 256 MB/chunk ≈ 200 frames of 800x796.
DEFAULT_N_AVG = 100       # default number of frames averaged into each mean frame
SAVE_CHUNK_BYTES = 256 * 1024 ** 2
SAVE_CHUNK_MIN, SAVE_CHUNK_MAX = 16, 2000
SAVE_PREFETCH = 2          # chunks read ahead in the background while the current one is processed
SAVE_THREADS = max(1, min(8, (os.cpu_count() or 2) - 2))   # worker threads for the alignment gather


def auto_chunk_size(frame_shape, dtype):
    frame_bytes = int(np.prod(frame_shape)) * np.dtype(dtype).itemsize
    return int(np.clip(SAVE_CHUNK_BYTES // max(1, frame_bytes), SAVE_CHUNK_MIN, SAVE_CHUNK_MAX))
SBX_MAX = np.uint16(np.iinfo(np.uint16).max)


def format_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def estimate_output_bytes(n_frames, h, w, dtype):
    """Pixel data + per-page BigTIFF overhead (~1 KB/page, generous) + 1% safety margin."""
    pixels = n_frames * h * w * np.dtype(dtype).itemsize
    return int((pixels + n_frames * 1024) * 1.01) + 16 * 1024 ** 2


def format_duration(seconds):
    """Human-readable duration, e.g. 45s, 3m 05s, 1h 02m."""
    s = int(round(max(0.0, seconds)))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def default_params():
    return dict(DEFAULT_PARAMS)


# =====================================================================
# File helpers
# =====================================================================
def find_files(folder, exten):
    """Sorted session files in `folder` (recursive), excluding our own output file."""
    folder = Path(folder)
    if exten == "*.tiff":
        files = list(folder.rglob("*.tiff")) + list(folder.rglob("*.tif"))
    else:
        files = list(folder.rglob(exten))
    return sorted(f for f in files if f.name not in EXCLUDED_NAMES)


def loadmat(matfile):
    return scipy.io.loadmat(matfile)['info']


def sbx_memmap(fpath):
    """Raw (non-inverted) SBX data as a read-only (n_frames, h, w) memmap. `fpath` has no extension."""
    info = loadmat(fpath + '.mat')
    height, width = (int(v) for v in info['sz'][0][0][0])
    framesize = height * width
    fid = np.memmap(fpath + '.sbx', dtype=np.uint16, mode='r')
    n_frames = fid.shape[0] // framesize
    return fid[:n_frames * framesize].reshape(n_frames, height, width)


class StackReader:
    """Chunked, read-only frame access for one .sbx or .tif(f) session.

    Reads only the frames requested, so whole sessions never need to fit in RAM.
    """

    def __init__(self, file, exten):
        file = Path(file)
        self._arr = None
        self._tif = None
        self.invert = False

        if exten == "*.sbx":
            self._arr = sbx_memmap(str(file.with_suffix('')))
            self.invert = True
            self.dtype = np.dtype(np.uint16)
            self.n_frames = self._arr.shape[0]
            self.frame_shape = self._arr.shape[1:]
            return

        self._tif = TiffFile(str(file))
        page0 = self._tif.pages[0]
        self.dtype = np.dtype(page0.dtype)
        self.frame_shape = tuple(page0.shape[-2:])
        n_pages = len(self._tif.pages)
        h, w = self.frame_shape
        try:
            arr = memmap(str(file), mode='r')
            n_mm = arr.size // (h * w)
            # Use the memmap only if it covers the whole file (it maps series[0] only).
            if arr.size == n_mm * h * w and n_mm >= n_pages:
                self._arr = arr.reshape(n_mm, h, w)
        except Exception:
            self._arr = None
        self.n_frames = self._arr.shape[0] if self._arr is not None else n_pages

    def read(self, start, stop):
        if self._arr is not None:
            data = np.array(self._arr[start:stop])
        else:
            data = np.stack([self._tif.pages[i].asarray() for i in range(start, stop)])
        if self.invert:
            np.subtract(SBX_MAX, data, out=data)
        return data

    def close(self):
        self._arr = None
        if self._tif is not None:
            self._tif.close()
            self._tif = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# =====================================================================
# Image processing
# =====================================================================
def correct_blocked_lines(data, threshold=1000, n_max_window=2):
    """Replace rows whose mean < threshold by the average of the nearest clean rows in neighbouring frames."""
    n_frames = data.shape[0]
    blocked_mask = data.mean(axis=2) < threshold          # (n_frames, height)
    if not blocked_mask.any():
        return data

    corrected = data.copy()
    for i, r in zip(*np.nonzero(blocked_mask)):
        clean_before = clean_after = None
        for k in range(1, n_max_window + 1):
            if i - k >= 0 and not blocked_mask[i - k, r]:
                clean_before = data[i - k, r, :]
                break
        for k in range(1, n_max_window + 1):
            if i + k < n_frames and not blocked_mask[i + k, r]:
                clean_after = data[i + k, r, :]
                break

        if clean_before is not None and clean_after is not None:
            corrected[i, r, :] = (clean_before.astype(np.float64) + clean_after) / 2
        elif clean_before is not None:
            corrected[i, r, :] = clean_before
        elif clean_after is not None:
            corrected[i, r, :] = clean_after
        else:
            corrected[i, r, :] = 0
    return corrected


def scale_to_size(img, scale):
    """Zoom about the centre (nearest neighbour), then crop / zero-pad back to the original size."""
    if abs(scale - 1.0) < 1e-9:
        return img
    scaled = zoom(img, scale, order=0)
    h, w = img.shape
    sh, sw = scaled.shape
    if scale > 1.0:
        y0, x0 = (sh - h) // 2, (sw - w) // 2
        return scaled[y0:y0 + h, x0:x0 + w]
    out = np.zeros_like(img)
    y0, x0 = max(0, (h - sh) // 2), max(0, (w - sw) // 2)
    out[y0:y0 + sh, x0:x0 + sw] = scaled
    return out


def transform_image(img, params):
    """Reference implementation: scale -> rotate -> shift (all nearest neighbour)."""
    img = scale_to_size(img, params['scale'])
    img = rotate(img, params['rotation'], reshape=False, order=0)
    return shift(img, (params['y_shift'], params['x_shift']), order=0)


def is_identity(params):
    return (params['x_shift'] == 0 and params['y_shift'] == 0
            and params['rotation'] == 0 and abs(params['scale'] - 1.0) < 1e-9)


@lru_cache(maxsize=32)
def _index_map(h, w, x_shift, y_shift, rotation, scale):
    # Because every step is nearest-neighbour, transforming an image of pixel indices
    # yields, for each output pixel, exactly which input pixel it copies (0 = outside).
    # Applying that lookup table to a frame is bit-identical to transform_image(frame),
    # but is a single gather that can be applied to many frames at once.
    idx = np.arange(1, h * w + 1, dtype=np.float64).reshape(h, w)
    params = {'x_shift': x_shift, 'y_shift': y_shift, 'rotation': rotation, 'scale': scale}
    m = np.rint(transform_image(idx, params)).astype(np.int64)
    valid = m > 0
    src = np.where(valid, m - 1, 0)
    src.setflags(write=False)
    valid.setflags(write=False)
    return src, valid


def imaged_rect(frame, frac=0.1):
    """Imaged region of a raw mean frame, ignoring dark blanked borders.

    Scanner blanking appears as whole dark columns / rows at the edges whose level is the
    detector offset (not zero; e.g. ~500 counts on 2P sbx data). A column/row counts as imaged
    when its mean is above dark + frac * (tissue - dark), where dark is the 1st percentile and
    tissue the median pixel value. Returns (row0, row1, col0, col1), exclusive ends.
    """
    f = frame.astype(np.float64)
    dark, tissue = np.percentile(f, 1), np.median(f)
    thr = dark + frac * (tissue - dark)
    rows = np.flatnonzero(f.mean(axis=1) > thr)
    cols = np.flatnonzero(f.mean(axis=0) > thr)
    if rows.size == 0 or cols.size == 0:
        return 0, f.shape[0], 0, f.shape[1]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def largest_rectangle(mask):
    """Largest-area axis-aligned rectangle containing only True pixels.

    Returns (row0, row1, col0, col1) with exclusive ends. O(h*w) histogram/stack method.
    """
    h, w = mask.shape
    heights = np.zeros(w + 1, dtype=np.int64)   # trailing 0 flushes the stack each row
    best = (0, 0, 0, 0, 0)                      # area, row0, row1, col0, col1
    for r in range(h):
        heights[:w] = np.where(mask[r], heights[:w] + 1, 0)
        stack = []                              # column indices with increasing heights
        for c in range(w + 1):
            start = c
            while stack and heights[stack[-1]] >= heights[c]:
                start = stack.pop()
                ht = heights[start]
                left = stack[-1] + 1 if stack else 0
                area = ht * (c - left)
                if area > best[0]:
                    best = (area, r - ht + 1, r + 1, left, c)
            stack.append(c)
    return tuple(int(v) for v in best[1:])


def crop_slices(h, w, crop):
    if crop is None:
        return slice(0, h), slice(0, w)
    top, bottom, left, right = crop
    return slice(top, h - bottom), slice(left, w - right)


def apply_transform(frames, params, crop=None):
    """Apply alignment (and optional crop) to a frame (h, w) or a stack (n, h, w)."""
    h, w = frames.shape[-2:]
    rs, cs = crop_slices(h, w, crop)
    if is_identity(params):
        return frames[..., rs, cs]
    src, valid = _index_map(h, w, int(params['x_shift']), int(params['y_shift']),
                            float(params['rotation']), float(params['scale']))
    src, valid = src[rs, cs], valid[rs, cs]
    out = frames.reshape(frames.shape[:-2] + (h * w,))[..., src]
    if not valid.all():
        out[..., ~valid] = 0
    return out


@lru_cache(maxsize=32)
def _flat_index_map(h, w, x_shift, y_shift, rotation, scale, crop):
    """Flattened lookup table for transform_stack: (source index per output pixel, indices to zero)."""
    src, valid = _index_map(h, w, x_shift, y_shift, rotation, scale)
    rs, cs = crop_slices(h, w, crop)
    src_flat = np.ascontiguousarray(src[rs, cs]).ravel().astype(np.intp)
    zero_idx = np.flatnonzero(~valid[rs, cs])
    return src_flat, zero_idx


def transform_stack(frames, params, crop=None, pool=None, n_threads=1):
    """Fast version of apply_transform for a stack (n, h, w); returns a C-contiguous array.

    Each frame is gathered with np.take into a preallocated contiguous buffer, split across
    `n_threads` worker threads (np.take releases the GIL). Bit-identical to apply_transform.
    """
    k, h, w = frames.shape
    rs, cs = crop_slices(h, w, crop)
    if is_identity(params):
        return np.ascontiguousarray(frames[:, rs, cs])
    src_flat, zero_idx = _flat_index_map(
        h, w, int(params['x_shift']), int(params['y_shift']),
        float(params['rotation']), float(params['scale']),
        None if crop is None else tuple(int(c) for c in crop))
    flat = np.ascontiguousarray(frames).reshape(k, h * w)
    out = np.empty((k, src_flat.size), dtype=frames.dtype)

    def work(j0):
        for j in range(j0, k, n_threads):
            np.take(flat[j], src_flat, out=out[j])
            if zero_idx.size:
                out[j, zero_idx] = 0

    if pool is None or n_threads <= 1:
        n_threads = 1
        work(0)
    else:
        list(pool.map(work, range(n_threads)))
    return out.reshape(k, rs.stop - rs.start, cs.stop - cs.start)


# =====================================================================
# Background loading thread
# =====================================================================
class DataLoaderThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    loading_finished = pyqtSignal(object)   # dict or None

    def __init__(self, folderpath, exten, n_frames_averaged=100):
        super().__init__()
        self.folderpath = folderpath
        self.exten = exten
        self.n_frames_averaged = n_frames_averaged

    def run(self):
        try:
            self.status_updated.emit("Finding files...")
            files = find_files(self.folderpath, self.exten)
            if not files:
                self.status_updated.emit(f"No files found with extension {self.exten}")
                self.loading_finished.emit(None)
                return

            mean_frames, n_total_frames, detected_dtype = [], 0, None
            for i, file in enumerate(files):
                with StackReader(file, self.exten) as reader:
                    if detected_dtype is None:
                        detected_dtype = reader.dtype
                    n_total_frames += reader.n_frames
                    k = max(1, min(reader.n_frames, self.n_frames_averaged))
                    self.status_updated.emit(
                        f"Averaging {k:,} frames of file {i+1}/{len(files)}: {file.name}")
                    # Accumulate in chunks so averaging many frames never needs them all in RAM.
                    chunk = auto_chunk_size(reader.frame_shape, reader.dtype)
                    acc = np.zeros(reader.frame_shape, dtype=np.float64)
                    for s in range(0, k, chunk):
                        if self.isInterruptionRequested():
                            self.loading_finished.emit(None)
                            return
                        e = min(s + chunk, k)
                        acc += reader.read(s, e).sum(axis=0, dtype=np.float64)
                        self.progress_updated.emit(int((i + e / k) / len(files) * 100))
                mean_frames.append((acc / k).astype(np.uint16))
                self.progress_updated.emit(int((i + 1) / len(files) * 100))

            self.loading_finished.emit({
                'mean_frames': mean_frames,
                'n_sessions': len(files),
                'n_total_frames': n_total_frames,
                'detected_dtype': detected_dtype,
                'exten': self.exten,
                'files': [f.relative_to(self.folderpath).as_posix() for f in files],
                'n_frames_averaged': self.n_frames_averaged,
            })
        except Exception as e:
            self.status_updated.emit(f"Error during loading: {e}")
            self.loading_finished.emit(None)


# =====================================================================
# Background saving thread
# =====================================================================
class _Cancelled(Exception):
    pass


class SaveThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    eta_updated = pyqtSignal(str)
    saving_finished = pyqtSignal(bool, str)

    def __init__(self, folderpath, exten, params_all, ref_idx, savepath,
                 detected_dtype, blocked_sessions=None, blocked_threshold=1000,
                 blocked_window=2, crop_params=None, chunk_size=None):
        super().__init__()
        self.folderpath = folderpath
        self.exten = exten
        self.params_all = copy.deepcopy(params_all)
        self.ref_idx = ref_idx
        self.savepath = str(savepath)
        self.detected_dtype = np.dtype(detected_dtype)
        self.blocked_sessions = set(blocked_sessions or [])
        self.blocked_threshold = int(blocked_threshold)
        self.blocked_window = int(blocked_window)
        self.crop_params = crop_params  # (top, bottom, left, right) or None
        self.chunk_size = chunk_size    # None = auto from frame size (auto_chunk_size)

    def _session_params(self, nn):
        if nn == self.ref_idx:
            return default_params()
        return self.params_all['sessions'].get(nn, default_params())

    def _read_chunks(self, files, out_q, stop):
        """Producer (background thread): read raw chunks from disk ahead of processing.

        Puts (session index, file, chunk start, chunk end, window start, data) items, then None.
        With blocked-frame correction the data includes `blocked_window` extra frames each side.
        """
        def put(item):
            while not stop.is_set():
                try:
                    out_q.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    pass
            return False

        try:
            for nn, file in enumerate(files):
                correct = nn in self.blocked_sessions
                with StackReader(file, self.exten) as reader:
                    n = reader.n_frames
                    for i in range(0, n, self.chunk_size):
                        end = min(i + self.chunk_size, n)
                        if correct:
                            ws, we = max(0, i - self.blocked_window), min(n, end + self.blocked_window)
                        else:
                            ws, we = i, end
                        if stop.is_set() or not put((nn, file, i, end, ws, reader.read(ws, we))):
                            return
            put(None)
        except BaseException as e:   # hand errors to the consumer
            put(e)

    def _frames(self, files, total_frames, pool):
        """Yield aligned, corrected, cropped frames one at a time.

        Pipeline: a background thread reads the next chunks from disk while this thread
        corrects + aligns (multi-threaded) the current chunk and tifffile writes it.
        """
        frames_done = 0
        t_start = time.monotonic()
        q = queue.Queue(maxsize=SAVE_PREFETCH)
        stop = threading.Event()
        reader = threading.Thread(target=self._read_chunks, args=(files, q, stop), daemon=True)
        reader.start()
        current_nn, file_label = None, ""
        try:
            while True:
                if self.isInterruptionRequested():
                    raise _Cancelled()
                try:
                    item = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                nn, file, i, end, ws, data = item
                if nn != current_nn:
                    current_nn = nn
                    file_label = f"Saving file {nn+1}/{len(files)}: {file.name}"
                    self.status_updated.emit(file_label)
                    params = self._session_params(nn)

                if nn in self.blocked_sessions:
                    try:
                        data = correct_blocked_lines(data, self.blocked_threshold, self.blocked_window)
                    except Exception as e:
                        self.status_updated.emit(
                            f"Blocked-lines correction error (session {nn}): {e}. Skipping.")
                    data = data[i - ws:i - ws + (end - i)]

                out = transform_stack(data, params, self.crop_params, pool, SAVE_THREADS)
                if out.dtype != self.detected_dtype:
                    out = out.astype(self.detected_dtype)
                yield from out

                frames_done += end - i
                self.progress_updated.emit(int(frames_done / total_frames * 100))

                # Estimated time left from the average throughput so far
                # (includes reading, correction, alignment and writing).
                elapsed = time.monotonic() - t_start
                if elapsed > 0:
                    fps = frames_done / elapsed
                    eta = format_duration((total_frames - frames_done) / fps)
                    self.eta_updated.emit(f"~{eta} left")
                    self.status_updated.emit(
                        f"{file_label} — {frames_done:,}/{total_frames:,} frames, "
                        f"{fps:,.0f} frames/s, ~{eta} left "
                        f"(elapsed {format_duration(elapsed)})")
        finally:
            stop.set()
            while True:   # unblock the reader if it is waiting on a full queue
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            reader.join(timeout=10)

    def run(self):
        tmp_path = self.savepath + '.part'
        try:
            files = find_files(self.folderpath, self.exten)
            if not files:
                self.saving_finished.emit(False, f"No files found with extension {self.exten}")
                return

            # Count frames and check all sessions share the same frame size.
            counts, shapes = [], set()
            for file in files:
                with StackReader(file, self.exten) as reader:
                    counts.append(reader.n_frames)
                    shapes.add(tuple(reader.frame_shape))
            if len(shapes) != 1:
                self.saving_finished.emit(False, f"Sessions have different frame sizes: {sorted(shapes)}")
                return
            h, w = shapes.pop()
            rs, cs = crop_slices(h, w, self.crop_params)
            out_h, out_w = rs.stop - rs.start, cs.stop - cs.start
            if out_h <= 0 or out_w <= 0:
                self.saving_finished.emit(False, "Invalid crop: output would be empty.")
                return
            total_frames = sum(counts)
            if self.chunk_size is None:
                self.chunk_size = auto_chunk_size((h, w), self.detected_dtype)

            # Fail fast instead of hours into a huge save. The new file is written next to
            # any existing output (which is only replaced at the end), so that is not counted.
            needed = estimate_output_bytes(total_frames, out_h, out_w, self.detected_dtype)
            free = shutil.disk_usage(os.path.dirname(os.path.abspath(self.savepath))).free
            if needed > free:
                self.saving_finished.emit(
                    False, f"Not enough free space on the output disk: need ~{format_bytes(needed)}, "
                           f"only {format_bytes(free)} free. Choose another output folder.")
                return

            self.status_updated.emit(
                f"Writing aligned TIFF (~{format_bytes(needed)}, {total_frames:,} frames)...")
            with ThreadPoolExecutor(max_workers=SAVE_THREADS) as pool:
                frames = self._frames(files, total_frames, pool)
                try:
                    with TiffWriter(tmp_path, bigtiff=True) as tiff_writer:
                        tiff_writer.write(frames, shape=(total_frames, out_h, out_w),
                                          dtype=self.detected_dtype, photometric='minisblack')
                finally:
                    frames.close()   # stops the background reader even if writing failed
            os.replace(tmp_path, self.savepath)

            self.progress_updated.emit(100)
            self.saving_finished.emit(True, "Saved aligned TIFF!")
        except _Cancelled:
            # The TiffWriter has been closed by the `with` block, so the partial file can go.
            self.saving_finished.emit(False, "Saving cancelled. " + self._remove(tmp_path))
        except Exception as e:
            self.saving_finished.emit(False, f"Error during saving: {e}. " + self._remove(tmp_path))

    @staticmethod
    def _remove(path):
        """Delete the partial output file; returns a message describing what happened."""
        if not os.path.exists(path):
            return "No partial file was left."
        try:
            os.remove(path)
            return "Partial file deleted."
        except OSError as e:
            return f"Could not delete partial file {path}: {e}"


# =====================================================================
# Background auto-alignment thread
# =====================================================================
class AutoAlignThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished_with_params = pyqtSignal(dict)   # empty dict on error / cancel

    def __init__(self, ref_img, mov_img, rot_min=-10.0, rot_max=10.0, rot_step=0.1,
                 scale_min=0.95, scale_max=1.05, scale_step=0.01,
                 crop_frac=0.6, trans_max=30, verbose=True):
        super().__init__()
        self.ref_img = ref_img
        self.mov_img = mov_img
        self.rot_min, self.rot_max, self.rot_step = rot_min, rot_max, rot_step
        self.scale_min, self.scale_max, self.scale_step = scale_min, scale_max, scale_step
        self.crop_frac = crop_frac
        self.trans_max = int(trans_max)
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

            # Everything about the reference is constant: compute it once.
            h, w = mov.shape
            ref_c = self._center_crop(ref, self.crop_frac)
            ch, cw = ref_c.shape
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            ref_cm = (ref_c - ref_c.mean()).ravel()
            ref_norm = np.linalg.norm(ref_cm)
            P = self.trans_max
            F_ref = np.fft.fft2(ref_c) if P > 0 else None

            best = {'rotation': 0.0, 'scale': 1.0, 'x_shift': 0, 'y_shift': 0, 'score': -np.inf}
            step_counter = 0
            for s in scales:
                scaled_full = scale_to_size(mov, s)
                for rot in rotations:
                    if self.isInterruptionRequested():
                        self.finished_with_params.emit({})
                        return
                    if self.verbose:
                        self.status_updated.emit(f"Auto align: scale {s:.3f}, rot {rot:.1f}°")
                    rotated = rotate(scaled_full, rot, reshape=False, order=0)

                    if P > 0:
                        rot_c = rotated[y0:y0 + ch, x0:x0 + cw]
                        dy, dx = self._estimate_translation_fft(F_ref, rot_c)
                        dy = max(-P, min(P, dy))
                        dx = max(-P, min(P, dx))
                        # Integer shift + crop == slice of a zero-padded image (no full-image shift needed)
                        padded = np.pad(rotated, P)
                        ys, xs = y0 - dy + P, x0 - dx + P
                        cand = padded[ys:ys + ch, xs:xs + cw]
                    else:
                        dy = dx = 0
                        cand = rotated[y0:y0 + ch, x0:x0 + cw]

                    cm = cand.ravel() - cand.mean()
                    denom = ref_norm * np.linalg.norm(cm)
                    score = float(np.dot(ref_cm, cm) / denom) if denom != 0 else -1.0

                    if score > best['score']:
                        best = {'rotation': float(rot), 'scale': float(s),
                                'x_shift': int(dx), 'y_shift': int(dy), 'score': score}

                    step_counter += 1
                    self.progress_updated.emit(int(step_counter / num_steps * 100))

            if self.verbose:
                self.status_updated.emit(f"Auto align: best corr {best['score']:.4f}")
            best.pop('score')
            self.finished_with_params.emit(best)
        except Exception as e:
            self.status_updated.emit(f"Auto align error: {e}")
            self.finished_with_params.emit({})

    @staticmethod
    def _center_crop(img, frac):
        h, w = img.shape
        ch, cw = max(1, int(h * frac)), max(1, int(w * frac))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return img[y0:y0 + ch, x0:x0 + cw]

    @staticmethod
    def _normalize(img):
        s = np.std(img)
        if s == 0:
            return img * 0.0
        return (img - np.mean(img)) / s

    @staticmethod
    def _estimate_translation_fft(F_ref, img):
        cross_power = F_ref * np.conj(np.fft.fft2(img))
        cross_power /= np.maximum(np.abs(cross_power), 1e-9)
        c = np.abs(np.fft.ifft2(cross_power))
        dy, dx = np.unravel_index(np.argmax(c), c.shape)
        h, w = img.shape
        if dy > h // 2:
            dy -= h
        if dx > w // 2:
            dx -= w
        return int(dy), int(dx)


# =====================================================================
# GUI
# =====================================================================
# key: (label, slider min, slider max, initial slider value, slider units per displayed unit)
SLIDER_SPECS = {
    'x':     ("X Shift",      -100, 100,   0,   1),
    'y':     ("Y Shift",      -100, 100,   0,   1),
    'rot':   ("Rotation (°)", -200, 200,   0,  10),
    'scale': ("Scale",          50, 150, 100, 100),
    'alpha': ("Alpha",           0, 100,  50, 100),
}

CROP_SIDES = ('top', 'bottom', 'left', 'right')    # same order as crop_params tuples
CROP_OPPOSITE = {'top': 'bottom', 'bottom': 'top', 'left': 'right', 'right': 'left'}

PANEL_WIDTH = 440


# ---------------------------------------------------------------------
# Theme: identical look on macOS, Linux and Windows.
# The platform's native style (macOS Aqua, GTK/Breeze on Ubuntu, ...) interprets a custom
# palette differently, so the cross-platform "Fusion" style is used everywhere together with
# a complete dark palette (including disabled colours) and one application-wide stylesheet.
# ---------------------------------------------------------------------
ACCENT = "#3d8ee8"


def dark_palette():
    p = QPalette()
    roles = {
        QPalette.Window: (30, 30, 30), QPalette.WindowText: (235, 235, 235),
        QPalette.Base: (43, 43, 43), QPalette.AlternateBase: (53, 53, 53),
        QPalette.ToolTipBase: (58, 58, 58), QPalette.ToolTipText: (235, 235, 235),
        QPalette.Text: (235, 235, 235), QPalette.PlaceholderText: (130, 130, 130),
        QPalette.Button: (58, 58, 58), QPalette.ButtonText: (235, 235, 235),
        QPalette.BrightText: (255, 80, 80), QPalette.Link: (61, 142, 232),
        QPalette.Highlight: (61, 142, 232), QPalette.HighlightedText: (255, 255, 255),
        QPalette.Light: (75, 75, 75), QPalette.Midlight: (62, 62, 62),
        QPalette.Mid: (45, 45, 45), QPalette.Dark: (20, 20, 20), QPalette.Shadow: (0, 0, 0),
    }
    for role, rgb in roles.items():
        p.setColor(role, QColor(*rgb))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(120, 120, 120))
    p.setColor(QPalette.Disabled, QPalette.Base, QColor(36, 36, 36))
    p.setColor(QPalette.Disabled, QPalette.Button, QColor(42, 42, 42))
    p.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    return p


APP_STYLESHEET = f"""
QGroupBox {{
    font-weight: bold;
    border: 2px solid #555555;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
}}
QLineEdit {{
    background-color: #2b2b2b;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 2px 5px;
    color: #ebebeb;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{ border: 1px solid {ACCENT}; }}
QLineEdit:disabled {{ background-color: #242424; border-color: #3a3a3a; color: #6e6e6e; }}
QPushButton {{
    background-color: #3a3a3a;
    border: 1px solid #5a5a5a;
    border-radius: 4px;
    padding: 4px 10px;
    color: #ebebeb;
}}
QPushButton:hover {{ background-color: #464646; border-color: #6a6a6a; }}
QPushButton:pressed {{ background-color: #2c2c2c; }}
QPushButton:disabled {{ background-color: #2c2c2c; border-color: #3a3a3a; color: #6e6e6e; }}
QToolButton {{
    background-color: #3a3a3a;
    border: 1px solid #5a5a5a;
    border-radius: 3px;
    color: #ebebeb;
    padding: 0px;
}}
QToolButton:hover {{ background-color: #464646; }}
QToolButton:pressed {{ background-color: #2c2c2c; }}
QToolButton:disabled {{ background-color: #2c2c2c; border-color: #3a3a3a; color: #6e6e6e; }}
QSlider::groove:horizontal {{ height: 4px; background: #4a4a4a; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::add-page:horizontal {{ background: #4a4a4a; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: #d6d6d6;
    border: 1px solid #5c5c5c;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider::sub-page:horizontal:disabled {{ background: #555555; }}
QSlider::handle:horizontal:disabled {{ background: #6e6e6e; border-color: #444444; }}
QToolTip {{ color: #ebebeb; background-color: #3a3a3a; border: 1px solid #5a5a5a; }}
"""


def apply_theme(app):
    """Apply the same Fusion + dark theme on every platform (idempotent)."""
    if app is None or app.property("tiffalign_theme"):
        return
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setPalette(dark_palette())
    app.setStyleSheet(APP_STYLESHEET)
    app.setProperty("tiffalign_theme", True)



class NoWheelSlider(QSlider):
    """Slider that ignores the mouse wheel / trackpad scroll and resets on double-click.

    Wheel events are passed on to the parent, so scrolling over a slider scrolls the control
    panel instead of changing a value by accident. Double-clicking returns the slider to
    `default_value`.
    """

    def __init__(self, orientation, default_value=0):
        super().__init__(orientation)
        self.default_value = default_value
        # Take keyboard focus on click (macOS defaults to Tab-only), so the
        # Left/Right arrow keys step the value after a slider is clicked.
        self.setFocusPolicy(Qt.StrongFocus)
        self.setToolTip("Double-click to reset")

    def wheelEvent(self, event):
        event.ignore()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setValue(self.default_value)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


class AlignGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('TIFFAlign')
        self.center_window()
        self.session_idx = 0
        self.ref_idx = 0
        self.exten = "*.sbx"
        self.folderpath = None
        self.mean_frames = None
        self.n_sessions = 0
        self.n_total_frames = 0
        self.detected_dtype = np.uint16
        self.params_all = {}
        self._moving_sessions = []
        self.loading_thread = None
        self.save_thread = None
        self.auto_thread = None
        self._ref_artist = None
        self._mov_artist = None
        self._crop_patches = []
        self._imaged_masks = None
        self._crop_is_auto = False      # True while the crop comes from Auto Detect Crop
        self._kept_state = None         # alignment kept while mean frames are recomputed
        self._loaded_n_avg = None       # averaging depth of the currently loaded mean frames
        self._auto_crop_timer = QTimer(self)
        self._auto_crop_timer.setSingleShot(True)
        self._auto_crop_timer.setInterval(250)
        self._auto_crop_timer.timeout.connect(lambda: self.auto_detect_crop(quiet=True))

        self.set_dark_theme()
        self._build_ui()
        self.set_busy(False)

    # ------------------- UI construction -------------------
    def _build_ui(self):
        # Right-hand control panel ------------------------------------------------
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(4, 4, 8, 4)

        # Data
        data_group = self._group("Data")
        data_form = QFormLayout()
        self.exten_selector = QComboBox()
        self.exten_selector.addItems(['sbx', 'tiff'])
        self.exten_selector.currentIndexChanged.connect(self.change_exten)
        data_form.addRow("File extension:", self.exten_selector)
        self.folder_button = QPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_folder)
        data_form.addRow(self.folder_button)
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        data_form.addRow(self.folder_label)
        self.n_avg_input = QLineEdit(str(DEFAULT_N_AVG))
        self.n_avg_input.setValidator(QIntValidator(1, 10_000_000, self))   # whole numbers >= 1
        self.n_avg_input.setToolTip(
            "Number of frames from the start of each session averaged into the mean frame used "
            "for display, alignment and auto-crop (1 = first frame only).\n"
            "Press Enter or leave the box to apply; the mean frames are recomputed and the "
            "current alignment is kept.")
        self.n_avg_input.editingFinished.connect(self.change_n_averaged)
        data_form.addRow("Frames to average:", self.n_avg_input)
        self.ref_selector = QComboBox()
        self.ref_selector.currentIndexChanged.connect(self.change_reference)
        data_form.addRow("Reference session:", self.ref_selector)
        self.session_selector = QComboBox()
        self.session_selector.currentIndexChanged.connect(self.change_session)
        data_form.addRow("Moving session:", self.session_selector)
        data_group.setLayout(data_form)
        panel_layout.addWidget(data_group)

        # Manual alignment
        self.manual_group = self._group("Manual Alignment")
        manual_grid = QGridLayout()
        self.sliders = {}
        for row, (key, (name, mn, mx, init, factor)) in enumerate(SLIDER_SPECS.items()):
            slider = NoWheelSlider(Qt.Horizontal, default_value=init)
            slider.setRange(mn, mx)
            slider.setValue(init)
            edit = QLineEdit(self._fmt(key, init))
            edit.setFixedWidth(64)
            slider.valueChanged.connect(lambda v, k=key: self.slider_changed(k, v))
            edit.editingFinished.connect(lambda k=key: self.text_input_changed(k))
            manual_grid.addWidget(QLabel(name), row, 0)
            manual_grid.addLayout(self._with_step_buttons(slider), row, 1)
            manual_grid.addWidget(edit, row, 2)
            self.sliders[key] = (slider, edit, factor)
        self.manual_group.setLayout(manual_grid)
        panel_layout.addWidget(self.manual_group)

        # Auto alignment
        self.auto_group = self._group("Auto Alignment")
        ag = QGridLayout()
        for col, text in enumerate(["Min", "Max", "Step"], start=1):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            ag.addWidget(lbl, 0, col)

        def range_row(row, label, checked, vals):
            cb = QCheckBox(label)
            cb.setChecked(checked)
            ag.addWidget(cb, row, 0)
            edits = []
            for col, v in enumerate(vals, start=1):
                e = QLineEdit(v)
                e.setFixedWidth(64)
                ag.addWidget(e, row, col)
                edits.append(e)
            return (cb, *edits)

        (self.rot_checkbox, self.rot_min_input, self.rot_max_input,
         self.rot_step_input) = range_row(1, "Rotation", True, ["-5.0", "5.0", "0.1"])
        (self.scale_checkbox, self.scale_min_input, self.scale_max_input,
         self.scale_step_input) = range_row(2, "Scale", False, ["0.95", "1.05", "0.01"])

        self.trans_checkbox = QCheckBox("Translation")
        self.trans_checkbox.setChecked(True)
        ag.addWidget(self.trans_checkbox, 3, 0)
        ag.addWidget(QLabel("Max shift:"), 3, 1, 1, 2, Qt.AlignRight)
        self.trans_max_input = QLineEdit("20")
        self.trans_max_input.setFixedWidth(64)
        ag.addWidget(self.trans_max_input, 3, 3)

        ag.addWidget(QLabel("Centre-crop fraction:"), 4, 0, 1, 3)
        self.crop_input = QLineEdit("0.6")
        self.crop_input.setFixedWidth(64)
        ag.addWidget(self.crop_input, 4, 3)

        self.auto_align_button = QPushButton("Align Current")
        self.auto_align_button.clicked.connect(self.auto_align)
        self.auto_align_all_button = QPushButton("Align All Sessions")
        self.auto_align_all_button.clicked.connect(self.auto_align_all)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.auto_align_button)
        btn_row.addWidget(self.auto_align_all_button)
        ag.addLayout(btn_row, 5, 0, 1, 4)
        self.auto_group.setLayout(ag)
        panel_layout.addWidget(self.auto_group)

        # Blocked-frames correction
        self.correction_group = self._group("Stimulation Blocked Frames Correction")
        cg = QGridLayout()
        self.correction_enable_checkbox = QCheckBox("Interpolate blocked frames")
        cg.addWidget(self.correction_enable_checkbox, 0, 0, 1, 4)
        cg.addWidget(QLabel("Threshold:"), 1, 0)
        self.correction_threshold_input = QLineEdit("1000")
        self.correction_threshold_input.setFixedWidth(72)
        cg.addWidget(self.correction_threshold_input, 1, 1)
        cg.addWidget(QLabel("Window:"), 1, 2)
        self.correction_window_input = QLineEdit("2")
        self.correction_window_input.setFixedWidth(64)
        cg.addWidget(self.correction_window_input, 1, 3)
        self.correction_group.setLayout(cg)
        panel_layout.addWidget(self.correction_group)

        # Crop
        self.crop_group = self._group("Crop Settings")
        crop_layout = QGridLayout()
        self.crop_enable_checkbox = QCheckBox("Enable cropping")
        self.crop_enable_checkbox.stateChanged.connect(self.update_image)
        crop_layout.addWidget(self.crop_enable_checkbox, 0, 0, 1, 3)
        self.crop_sliders = {}
        for row, key in enumerate(CROP_SIDES, start=1):
            slider = NoWheelSlider(Qt.Horizontal, default_value=0)
            slider.setRange(0, 0)          # real range is set once data is loaded
            edit = QLineEdit("0")
            edit.setFixedWidth(64)
            slider.valueChanged.connect(lambda v, k=key: self.crop_slider_changed(k, v))
            edit.editingFinished.connect(lambda k=key: self.crop_text_changed(k))
            crop_layout.addWidget(QLabel(key.capitalize()), row, 0)
            crop_layout.addLayout(self._with_step_buttons(slider), row, 1)
            crop_layout.addWidget(edit, row, 2)
            self.crop_sliders[key] = (slider, edit)
            setattr(self, f"crop_{key}_input", edit)
        self.auto_crop_button = QPushButton("Auto Detect Crop")
        self.auto_crop_button.setToolTip(
            "Largest rectangle containing only pixels imaged in every aligned session "
            "(no dark borders or rotation wedges).\nKeeps updating as the alignment changes "
            "until a crop slider is moved by hand.")
        self.auto_crop_button.clicked.connect(lambda: self.auto_detect_crop())
        crop_layout.addWidget(self.auto_crop_button, 6, 0, 1, 3)
        self.crop_size_label = QLabel("Output size: —")
        self.crop_size_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        crop_layout.addWidget(self.crop_size_label, 7, 0, 1, 3)
        self.crop_group.setLayout(crop_layout)
        panel_layout.addWidget(self.crop_group)

        # Save / load
        self.save_group = self._group("Parameters && Output")
        sg = QVBoxLayout()
        row = QHBoxLayout()
        self.save_params_button = QPushButton("Save Parameters")
        self.save_params_button.clicked.connect(self.save_params)
        self.load_params_button = QPushButton("Load Parameters")
        self.load_params_button.clicked.connect(self.load_params)
        row.addWidget(self.save_params_button)
        row.addWidget(self.load_params_button)
        sg.addLayout(row)
        out_row = QHBoxLayout()
        self.output_label = QLabel("Output folder: (data folder)")
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self.output_button = QPushButton("Change...")
        self.output_button.setToolTip(
            "Where tiff_aligned.tiff is written. A different (fast, internal) disk than the one "
            "holding the recordings avoids read/write contention and can roughly halve save time.")
        self.output_button.clicked.connect(self.select_output_folder)
        out_row.addWidget(self.output_label, 1)
        out_row.addWidget(self.output_button, 0)
        sg.addLayout(out_row)
        self.save_button = QPushButton("Save Aligned TIFF")
        self.save_button.setMinimumHeight(34)
        self.save_button.clicked.connect(self.apply_and_save)
        sg.addWidget(self.save_button)
        self.save_group.setLayout(sg)
        panel_layout.addWidget(self.save_group)
        panel_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(PANEL_WIDTH)
        scroll.viewport().setBackgroundRole(QPalette.Window)

        # Left-hand display -------------------------------------------------------
        self.fig = Figure(facecolor='black')
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_facecolor('black')
        self.ax.axis('off')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Footer ------------------------------------------------------------------
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #aaaaaa; font-size: 14px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedWidth(PANEL_WIDTH)

        body = QHBoxLayout()
        body.addWidget(self.canvas, 1)
        body.addWidget(scroll, 0)
        footer = QHBoxLayout()
        footer.addWidget(self.status_label, 1)
        footer.addWidget(self.progress_bar, 0)
        self.cancel_save_button = QPushButton("Cancel Saving")
        self.cancel_save_button.setToolTip("Stop writing the aligned TIFF and delete the partial file")
        self.cancel_save_button.clicked.connect(self.cancel_saving)
        self.cancel_save_button.setVisible(False)
        footer.addWidget(self.cancel_save_button, 0)
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(body, 1)
        main_layout.addLayout(footer)

    # ------------------- Style helpers -------------------
    def _group(self, title):
        return QGroupBox(title)   # styled by APP_STYLESHEET

    @staticmethod
    def _with_step_buttons(slider):
        """[◀] slider [▶] — each click moves one step (hold to repeat)."""
        row = QHBoxLayout()
        row.setSpacing(2)
        for text, action, tip in (("◀", QAbstractSlider.SliderSingleStepSub, "Decrease by one step"),
                                  ("▶", QAbstractSlider.SliderSingleStepAdd, "Increase by one step")):
            btn = QToolButton()
            btn.setText(text)
            btn.setToolTip(tip)
            btn.setFixedSize(22, 22)
            btn.setAutoRepeat(True)
            btn.setAutoRepeatDelay(400)
            btn.setAutoRepeatInterval(60)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(lambda _=False, a=action: slider.triggerAction(a))
            if action == QAbstractSlider.SliderSingleStepSub:
                row.addWidget(btn)
                row.addWidget(slider, 1)
            else:
                row.addWidget(btn)
        return row


    def center_window(self):
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(1500, int(screen.width() * 0.95))
        h = min(950, int(screen.height() * 0.92))
        self.setGeometry(screen.x() + (screen.width() - w) // 2,
                         screen.y() + (screen.height() - h) // 2, w, h)

    def set_dark_theme(self):
        apply_theme(QApplication.instance())
        self.setPalette(dark_palette())

    # ------------------- Slider handling -------------------
    @staticmethod
    def _fmt(key, slider_value):
        factor = SLIDER_SPECS[key][4]
        if factor == 1:
            return str(slider_value)
        return f"{slider_value / factor:.{1 if factor == 10 else 2}f}"

    def _value(self, key):
        slider, _, factor = self.sliders[key]
        return slider.value() / factor if factor != 1 else slider.value()

    def slider_changed(self, key, value):
        self.sliders[key][1].setText(self._fmt(key, value))
        if self.mean_frames is None:
            return
        if key == 'alpha':
            if self._mov_artist is not None:
                self._mov_artist.set_alpha(value / 100.0)
                self.canvas.draw_idle()
            return
        self._store_current_params()
        self.update_image()

    def text_input_changed(self, key):
        slider, edit, factor = self.sliders[key]
        try:
            v = int(round(float(edit.text()) * factor))
            slider.setValue(max(slider.minimum(), min(slider.maximum(), v)))
        except ValueError:
            pass
        edit.setText(self._fmt(key, slider.value()))

    def _set_sliders(self, params):
        """Show `params` on the sliders without triggering one redraw per slider."""
        values = {'x': int(params['x_shift']), 'y': int(params['y_shift']),
                  'rot': int(round(params['rotation'] * 10)),
                  'scale': int(round(params['scale'] * 100))}
        for key, v in values.items():
            slider, edit, _ = self.sliders[key]
            slider.blockSignals(True)
            slider.setValue(v)
            slider.blockSignals(False)
            edit.setText(self._fmt(key, slider.value()))
        self._store_current_params()   # stores values clamped to slider ranges

    def _store_current_params(self):
        if not self.params_all:
            return
        self.params_all['sessions'][self.session_idx] = {
            'x_shift': self._value('x'),
            'y_shift': self._value('y'),
            'rotation': self._value('rot'),
            'scale': self._value('scale'),
        }
        self._alignment_changed()

    def set_busy(self, busy):
        """Disable inputs while a background job runs (and data-dependent inputs until data is loaded)."""
        self.folder_button.setEnabled(not busy)
        self.exten_selector.setEnabled(not busy)
        self.n_avg_input.setEnabled(not busy)
        has_data = self.mean_frames is not None and not busy
        for w in (self.ref_selector, self.session_selector, self.manual_group, self.auto_group,
                  self.correction_group, self.crop_group, self.save_group):
            w.setEnabled(has_data)

    def _start_progress(self, text):
        self.status_label.setText(text)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.set_busy(True)

    def _end_progress(self, text):
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        self.set_busy(False)
        self.status_label.setText(text)

    # ------------------- Crop helpers -------------------
    def get_crop_params(self):
        """Return (top, bottom, left, right) crop pixel amounts."""
        return tuple(self.crop_sliders[k][0].value() for k in CROP_SIDES)

    def _reset_crop_ranges(self):
        """Fit the crop slider ranges to the loaded frame size."""
        h, w = self.mean_frames[self.ref_idx].shape
        for key in CROP_SIDES:
            slider, edit = self.crop_sliders[key]
            slider.blockSignals(True)
            slider.setRange(0, (h if key in ('top', 'bottom') else w) - 1)
            slider.blockSignals(False)
        for key in CROP_SIDES:   # re-clamp so opposite sides never meet
            self._clamp_crop(key)
            self.crop_sliders[key][1].setText(str(self.crop_sliders[key][0].value()))

    def _clamp_crop(self, key):
        """Keep at least 1 px between `key` and the opposite side. Returns True if it changed."""
        if self.mean_frames is None:
            return False
        h, w = self.mean_frames[self.ref_idx].shape
        size = h if key in ('top', 'bottom') else w
        slider = self.crop_sliders[key][0]
        limit = size - 1 - self.crop_sliders[CROP_OPPOSITE[key]][0].value()
        if slider.value() > limit:
            slider.setValue(max(0, limit))
            return True
        return False

    def crop_slider_changed(self, key, value):
        if self._clamp_crop(key):
            return   # setValue re-entered this handler with the clamped value
        self.crop_sliders[key][1].setText(str(value))
        self._crop_is_auto = False   # a manual crop is no longer updated automatically
        if value > 0 and not self.crop_enable_checkbox.isChecked():
            self.crop_enable_checkbox.setChecked(True)   # triggers update_image
        else:
            self.update_image()

    def crop_text_changed(self, key):
        slider, edit = self.crop_sliders[key]
        try:
            slider.setValue(int(round(float(edit.text()))))
        except ValueError:
            pass
        edit.setText(str(slider.value()))

    def _set_crop(self, top, bottom, left, right):
        for key, v in zip(CROP_SIDES, (top, bottom, left, right)):
            slider, edit = self.crop_sliders[key]
            slider.blockSignals(True)
            slider.setValue(int(v))
            slider.blockSignals(False)
            edit.setText(str(slider.value()))

    def _imaged_mask(self, i):
        """Rectangle of session i's *raw* mean frame that was actually imaged (see imaged_rect).

        Independent of the alignment, so cached per session.
        """
        if self._imaged_masks is None:
            self._imaged_masks = {}
        if i not in self._imaged_masks:
            frame = self.mean_frames[i]
            r0, r1, c0, c1 = imaged_rect(frame)
            mask = np.zeros(frame.shape, dtype=bool)
            mask[r0:r1, c0:c1] = True
            self._imaged_masks[i] = mask
        return self._imaged_masks[i]

    def overlap_mask(self):
        """Pixels covered by imaged data in every session after alignment.

        Each session's imaged mask is moved with the exact same transform as its frames, so the
        empty wedges created by rotation / shift / scale are excluded geometrically, independent
        of image brightness.
        """
        h, w = self.mean_frames[self.ref_idx].shape
        mask = np.ones((h, w), dtype=bool)
        for i in range(self.n_sessions):
            params = default_params() if i == self.ref_idx else \
                self.params_all['sessions'].get(i, default_params())
            mask &= apply_transform(self._imaged_mask(i), params)
        return mask

    def auto_detect_crop(self, quiet=False):
        """Crop to the largest rectangle containing only pixels imaged in every aligned session.

        The crop then keeps following the alignment (re-computed whenever a session's
        parameters change) until a crop slider is moved by hand.
        """
        if self.mean_frames is None:
            return
        h, w = self.mean_frames[self.ref_idx].shape
        mask = self.overlap_mask()
        if not mask.any():
            self.status_label.setText("Auto crop: the aligned sessions do not overlap.")
            return
        r0, r1, c0, c1 = largest_rectangle(mask)
        self._set_crop(r0, h - r1, c0, w - c1)
        self._crop_is_auto = True
        self.crop_enable_checkbox.blockSignals(True)
        self.crop_enable_checkbox.setChecked(True)
        self.crop_enable_checkbox.blockSignals(False)
        if not quiet:
            self.status_label.setText(
                f"Auto crop: kept {c1 - c0} × {r1 - r0} px "
                f"({100 * (r1 - r0) * (c1 - c0) / (h * w):.0f}% of the frame); "
                f"it will follow further alignment changes.")
        self.update_image()

    def _alignment_changed(self):
        """Called whenever any session's alignment or the reference changes."""
        if self._crop_is_auto and self.crop_enable_checkbox.isChecked():
            self._auto_crop_timer.start()   # debounced: recompute once dragging pauses

    # ------------------- Folder / data loading -------------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folderpath = Path(folder)
            self.folder_label.setText(str(self.folderpath))
            print(f"Folder selected: {self.folderpath}")
            self.load_data()

    def select_output_folder(self):
        start = str(self.savepath.parent) if getattr(self, 'savepath', None) else ""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", start)
        if folder:
            self._set_output_dir(Path(folder))

    def _set_output_dir(self, folder):
        self.savepath = Path(folder) / OUTPUT_NAME
        try:
            free = format_bytes(shutil.disk_usage(folder).free) + " free"
        except OSError:
            free = "free space unknown"
        same = " (data folder)" if Path(folder) == self.folderpath else ""
        self.output_label.setText(f"Output folder{same}: {folder}  [{free}]")

    def load_data(self):
        self._set_output_dir(self.folderpath)
        self.n_frames_averaged = self._n_avg_value()
        self.pickle_path = self.folderpath / 'mean_frames.pkl'
        self.params_path = self.folderpath / 'params_all.pkl'

        if self.pickle_path.exists():
            self.status_label.setText("Loading cached data...")
            try:
                with open(self.pickle_path, 'rb') as f:
                    data = pickle.load(f)
                if self._cache_is_valid(data):
                    print("Loading mean frames from pickle ---")
                    self.mean_frames = data['mean_frames']
                    self.n_sessions = data['n_sessions']
                    self.n_total_frames = data['n_total_frames']
                    self.detected_dtype = data.get('detected_dtype', np.uint16)
                    self.finish_data_loading()
                    return
                print("Cached mean frames do not match the folder contents or averaging "
                      "settings, recomputing ---")
            except Exception as e:
                print(f"Error loading pickle: {e}")
        print("Computing and saving mean frames ---")
        self.start_background_loading()

    def _n_avg_value(self):
        """Frames-to-average from the text box (>= 1); invalid text falls back to the last value."""
        try:
            n = max(1, int(self.n_avg_input.text()))
        except ValueError:
            n = self._loaded_n_avg or DEFAULT_N_AVG
        self.n_avg_input.setText(str(n))
        return n

    def change_n_averaged(self):
        """Recompute the mean frames when the averaging depth changes (alignment is kept)."""
        n = self._n_avg_value()
        if self.mean_frames is None or n == self._loaded_n_avg:
            return
        self._kept_state = {
            'n_sessions': self.n_sessions, 'mean_frames': self.mean_frames,
            'n_avg': self._loaded_n_avg, 'params_all': copy.deepcopy(self.params_all),
            'ref_idx': self.ref_idx, 'session_idx': self.session_idx,
            'crop': self.get_crop_params(), 'crop_is_auto': self._crop_is_auto,
        }
        self.n_frames_averaged = n
        self.start_background_loading()

    def _cache_is_valid(self, data):
        if data.get('exten', self.exten) != self.exten:
            return False
        if data.get('n_frames_averaged', 100) != self.n_frames_averaged:   # old caches used 100
            return False
        files = find_files(self.folderpath, self.exten)
        if 'files' in data:
            return data['files'] == [f.relative_to(self.folderpath).as_posix() for f in files]
        return data.get('n_sessions') == len(files)   # caches from older versions

    def start_background_loading(self):
        self.mean_frames = None
        self._start_progress("Starting data loading...")
        self.loading_thread = DataLoaderThread(self.folderpath, self.exten, self.n_frames_averaged)
        self.loading_thread.progress_updated.connect(self.progress_bar.setValue)
        self.loading_thread.status_updated.connect(self.status_label.setText)
        self.loading_thread.loading_finished.connect(self.on_loading_finished)
        self.loading_thread.start()

    def on_loading_finished(self, result):
        if result is None:
            kept = self._kept_state
            if kept is not None:
                # Recomputing failed or was interrupted: go back to the previous mean frames.
                msg = self.status_label.text()
                self.progress_bar.setVisible(False)
                self.mean_frames = kept['mean_frames']
                self.n_frames_averaged = kept['n_avg']
                self.n_avg_input.setText(str(kept['n_avg']))
                self.finish_data_loading()
                self.status_label.setText(f"{msg} (kept previous mean frames)")
                return
            self._end_progress(self.status_label.text())   # keep the thread's error message
            return
        self.progress_bar.setVisible(False)

        self.mean_frames = result['mean_frames']
        self.n_sessions = result['n_sessions']
        self.n_total_frames = result['n_total_frames']
        self.detected_dtype = result['detected_dtype']
        try:
            with open(self.pickle_path, 'wb') as f:
                pickle.dump(result, f)
            print("Mean frames saved to pickle")
        except Exception as e:
            print(f"Error saving pickle: {e}")
        self.finish_data_loading()

    def finish_data_loading(self):
        self._loaded_n_avg = self.n_frames_averaged
        kept = self._kept_state
        self._kept_state = None
        if kept is not None and kept['n_sessions'] == self.n_sessions:
            # Mean frames were only recomputed with a different averaging depth:
            # keep the current (possibly unsaved) alignment, reference, session and crop.
            self.params_all = kept['params_all']
            self.ref_idx = kept['ref_idx']
            self._imaged_masks = None
            self._ref_artist = None
            self.set_busy(False)
            self.ref_selector.blockSignals(True)
            self.ref_selector.setCurrentIndex(self.ref_idx)
            self.ref_selector.blockSignals(False)
            crop_auto = kept['crop_is_auto']
            self._populate_moving_selector(keep=kept['session_idx'])
            self._set_crop(*kept['crop'])
            self._crop_is_auto = crop_auto
            if crop_auto and self.crop_enable_checkbox.isChecked():
                self.auto_detect_crop(quiet=True)   # dark-border detection uses the new means
            self.update_image()
            self.status_label.setText(
                f"Mean frames recomputed from the first {self.n_frames_averaged:,} frame(s) of "
                f"each session; alignment kept.")
            return

        self.params_all = {
            'reference_session': 0,
            'sessions': {i: default_params() for i in range(self.n_sessions)},
        }
        self.ref_idx = 0
        self.session_idx = 0
        self._ref_artist = None   # force the canvas to be rebuilt for the new data
        self._imaged_masks = None
        self._crop_is_auto = False

        self.ref_selector.blockSignals(True)
        self.ref_selector.clear()
        self.ref_selector.addItems([f"Session {i}" for i in range(self.n_sessions)])
        self.ref_selector.setCurrentIndex(0)
        self.ref_selector.blockSignals(False)

        self._reset_crop_ranges()
        self.set_busy(False)
        if not self.load_params(quiet=True):
            self._populate_moving_selector()
        self.status_label.setText(
            f"Ready - {self.n_sessions} sessions, {self.n_total_frames} frames "
            f"(data type: {np.dtype(self.detected_dtype).name})")

    # ------------------- Session changes -------------------
    def _populate_moving_selector(self, keep=None):
        self._moving_sessions = [i for i in range(self.n_sessions) if i != self.ref_idx]
        self.session_selector.blockSignals(True)
        self.session_selector.clear()
        self.session_selector.addItems([f"Session {i}" for i in self._moving_sessions])
        if keep in self._moving_sessions:
            target = keep
        else:
            target = self._moving_sessions[0] if self._moving_sessions else self.ref_idx
        if self._moving_sessions:
            self.session_selector.setCurrentIndex(self._moving_sessions.index(target))
        self.session_selector.blockSignals(False)
        self._show_session(target)

    def _show_session(self, idx):
        self.session_idx = idx
        self._set_sliders(self.params_all['sessions'].setdefault(idx, default_params()))
        self.update_image()

    def change_session(self, combo_idx):
        if 0 <= combo_idx < len(self._moving_sessions):
            self._show_session(self._moving_sessions[combo_idx])

    def change_reference(self, idx):
        if idx < 0 or not self.params_all:
            return
        self.ref_idx = idx
        self.params_all['reference_session'] = idx
        self._populate_moving_selector(keep=self.session_idx)

    def change_exten(self):
        self.exten = f"*.{self.exten_selector.currentText()}"

    # ------------------- Display -------------------
    def _init_canvas(self, ref_img, aligned, alpha):
        self.ax.clear()
        self.ax.axis('off')
        self._ref_artist = self.ax.imshow(ref_img, cmap='gray', interpolation='nearest')
        self._mov_artist = self.ax.imshow(aligned, cmap='inferno', alpha=alpha,
                                          interpolation='nearest')
        self._crop_patches = []

    def update_image(self, *_):
        if self.mean_frames is None or not self.params_all:
            return

        ref_img = self.mean_frames[self.ref_idx]
        params = self.params_all['sessions'].get(self.session_idx, default_params())
        aligned = apply_transform(self.mean_frames[self.session_idx], params)
        alpha = self._value('alpha')

        if self._ref_artist is None or self._ref_artist.get_array().shape != ref_img.shape:
            self._init_canvas(ref_img, aligned, alpha)
        else:
            # Re-use the existing artists instead of clearing / re-creating the axes.
            self._ref_artist.set_data(ref_img)
            self._mov_artist.set_data(aligned)
            self._mov_artist.set_alpha(alpha)
        self._ref_artist.set_clim(ref_img.min(), ref_img.max())
        self._mov_artist.set_clim(aligned.min(), aligned.max())

        for p in self._crop_patches:
            p.remove()
        self._crop_patches = []
        h_img, w_img = ref_img.shape
        if self.crop_enable_checkbox.isChecked():
            top, bottom, left, right = self.get_crop_params()
            crop_w = w_img - left - right
            crop_h = h_img - top - bottom
            if crop_w > 0 and crop_h > 0:
                for (bx, by, bw, bh) in [
                    (-0.5,                -0.5,                 w_img,       top),
                    (-0.5,                h_img - bottom - 0.5, w_img,       bottom + 0.5),
                    (-0.5,                top - 0.5,            left,        crop_h + 1),
                    (w_img - right - 0.5, top - 0.5,            right + 0.5, crop_h + 1),
                ]:
                    self._crop_patches.append(self.ax.add_patch(Rectangle(
                        (bx, by), bw, bh, linewidth=0, facecolor='black', alpha=0.6)))
                self._crop_patches.append(self.ax.add_patch(Rectangle(
                    (left - 0.5, top - 0.5), crop_w, crop_h,
                    linewidth=2, edgecolor='#00e5ff', facecolor='none', linestyle='--')))
                self.crop_size_label.setText(f"Output size: {crop_w} × {crop_h} px")
            else:
                self.crop_size_label.setText("Output size: invalid crop")
        else:
            self.crop_size_label.setText(f"Output size: {w_img} × {h_img} px (no crop)")

        self.canvas.draw_idle()

    # ------------------- Save / load -------------------
    def apply_and_save(self):
        if self.mean_frames is None:
            return
        crop_params = None
        if self.crop_enable_checkbox.isChecked():
            crop_params = self.get_crop_params()
            h, w = self.mean_frames[self.ref_idx].shape
            if crop_params[0] + crop_params[1] >= h or crop_params[2] + crop_params[3] >= w:
                self.status_label.setText("Invalid crop: output would be empty.")
                return

        selected_sessions = list(range(self.n_sessions)) \
            if self.correction_enable_checkbox.isChecked() else []
        try:
            corr_threshold = int(float(self.correction_threshold_input.text()))
        except ValueError:
            corr_threshold = 1000
        try:
            corr_window = max(1, int(float(self.correction_window_input.text())))
        except ValueError:
            corr_window = 2

        print("Saving to:", self.savepath)
        self._start_progress("Saving aligned TIFF...")
        self.save_thread = SaveThread(
            self.folderpath, self.exten, self.params_all, self.ref_idx,
            self.savepath, self.detected_dtype,
            blocked_sessions=selected_sessions,
            blocked_threshold=corr_threshold,
            blocked_window=corr_window,
            crop_params=crop_params,
        )
        self.save_thread.progress_updated.connect(self.progress_bar.setValue)
        self.save_thread.status_updated.connect(self.status_label.setText)
        self.save_thread.eta_updated.connect(self._on_save_eta)
        self.save_thread.saving_finished.connect(self.on_saving_finished)
        self._save_t0 = time.monotonic()
        self.save_thread.start()
        self.cancel_save_button.setText("Cancel Saving")
        self.cancel_save_button.setEnabled(True)
        self.cancel_save_button.setVisible(True)

    def _on_save_eta(self, eta):
        if self.cancel_save_button.isEnabled():   # don't overwrite "cancelling..."
            self.progress_bar.setFormat(f"%p%  ·  {eta}")

    def cancel_saving(self):
        if self.save_thread is None or not self.save_thread.isRunning():
            return
        answer = QMessageBox.question(
            self, "Cancel saving",
            "Stop saving the aligned TIFF?\n\nThe partially written file will be deleted. "
            "Any previously completed output file is left untouched.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes or not self.save_thread.isRunning():
            return
        self.save_thread.requestInterruption()   # stops at the next chunk boundary
        self.cancel_save_button.setEnabled(False)
        self.cancel_save_button.setText("Cancelling...")
        self.progress_bar.setFormat("%p%  ·  cancelling...")

    def on_saving_finished(self, success, message):
        self.cancel_save_button.setVisible(False)
        if success:
            took = format_duration(time.monotonic() - getattr(self, '_save_t0', time.monotonic()))
            message = f"{message} ({self.n_sessions} sessions, took {took}) -> {self.savepath}"
        print(message)
        self._end_progress(message)

    def save_params(self):
        if not hasattr(self, 'params_path') or not self.params_all:
            return
        self.params_all['reference_session'] = self.ref_idx
        with open(self.params_path, 'wb') as f:
            pickle.dump(self.params_all, f)
        self.status_label.setText(f"Alignment parameters saved to {self.params_path.name}")
        print("Alignment parameters saved.")

    def load_params(self, quiet=False):
        """Load params_all.pkl from the folder. Returns True if loaded."""
        if not (hasattr(self, 'params_path') and self.params_path.exists()) or self.mean_frames is None:
            if not quiet:
                self.status_label.setText("No saved parameters found.")
            print("No saved parameters found.")
            return False
        try:
            with open(self.params_path, 'rb') as f:
                loaded = pickle.load(f)
            sessions = {i: default_params() for i in range(self.n_sessions)}
            for k, v in loaded.get('sessions', {}).items():
                if 0 <= int(k) < self.n_sessions:
                    sessions[int(k)] = {**default_params(), **v}
            ref = int(loaded.get('reference_session', 0))
            if not 0 <= ref < self.n_sessions:
                ref = 0
        except Exception as e:
            self.status_label.setText(f"Could not load parameters: {e}")
            return False

        self.params_all = {'reference_session': ref, 'sessions': sessions}
        self.ref_idx = ref
        self.ref_selector.blockSignals(True)
        self.ref_selector.setCurrentIndex(ref)
        self.ref_selector.blockSignals(False)
        self._populate_moving_selector(keep=self.session_idx)
        if not quiet:
            self.status_label.setText("Alignment parameters loaded.")
        print("Alignment parameters loaded.")
        return True

    # ------------------- Auto alignment -------------------
    def get_auto_align_params(self):
        """Parse the auto-align inputs. Returns None (and shows a message) if they are invalid."""
        try:
            p = {}
            if self.rot_checkbox.isChecked():
                p['rot_min'] = float(self.rot_min_input.text())
                p['rot_max'] = float(self.rot_max_input.text())
                p['rot_step'] = float(self.rot_step_input.text())
            else:
                p['rot_min'] = p['rot_max'] = 0.0
                p['rot_step'] = 1.0
            if self.scale_checkbox.isChecked():
                p['scale_min'] = float(self.scale_min_input.text())
                p['scale_max'] = float(self.scale_max_input.text())
                p['scale_step'] = float(self.scale_step_input.text())
            else:
                p['scale_min'] = p['scale_max'] = 1.0
                p['scale_step'] = 1.0
            p['trans_max'] = max(0, int(float(self.trans_max_input.text()))) \
                if self.trans_checkbox.isChecked() else 0
            p['crop_frac'] = float(self.crop_input.text())
        except ValueError:
            self.status_label.setText("Auto align: invalid number in parameters.")
            return None

        if p['rot_step'] <= 0 or p['scale_step'] <= 0:
            self.status_label.setText("Auto align: step sizes must be > 0.")
            return None
        if p['rot_min'] > p['rot_max']:
            p['rot_min'], p['rot_max'] = p['rot_max'], p['rot_min']
        if p['scale_min'] > p['scale_max']:
            p['scale_min'], p['scale_max'] = p['scale_max'], p['scale_min']
        if not (0 < p['crop_frac'] <= 1):
            p['crop_frac'] = 0.6
            self.crop_input.setText("0.6")
        return p

    def _start_auto_thread(self, mov_idx, params, on_progress, on_finished):
        self.auto_thread = AutoAlignThread(
            ref_img=self.mean_frames[self.ref_idx], mov_img=self.mean_frames[mov_idx],
            verbose=False, **params)
        self.auto_thread.progress_updated.connect(on_progress)
        self.auto_thread.status_updated.connect(self.status_label.setText)
        self.auto_thread.finished_with_params.connect(on_finished)
        self.auto_thread.start()

    def _apply_auto_result(self, session_idx, best):
        params = {
            'x_shift': int(best['x_shift']), 'y_shift': int(best['y_shift']),
            'rotation': float(best['rotation']), 'scale': float(best['scale']),
        }
        self.params_all['sessions'][session_idx] = params
        if session_idx == self.session_idx:
            self._set_sliders(params)
            self.update_image()
        else:
            self._alignment_changed()

    def auto_align(self):
        if self.mean_frames is None or self.n_sessions < 2:
            return
        if self.session_idx == self.ref_idx:
            self.status_label.setText("Select a moving session different from the reference.")
            return
        params = self.get_auto_align_params()
        if params is None:
            return
        self._start_progress(f"Aligning Session {self.session_idx}")
        session = self.session_idx
        self._start_auto_thread(session, params, self.progress_bar.setValue,
                                lambda best, s=session: self.on_auto_align_finished(s, best))

    def on_auto_align_finished(self, session_idx, best):
        if best:
            self._apply_auto_result(session_idx, best)
            self._end_progress("Auto alignment complete.")
        else:
            self._end_progress(self.status_label.text() or "Auto alignment failed.")

    def auto_align_all(self):
        if self.mean_frames is None or self.n_sessions < 2:
            return
        params = self.get_auto_align_params()
        if params is None:
            return
        self._align_all_params = params
        self._sessions_to_align = [i for i in range(self.n_sessions) if i != self.ref_idx]
        self._align_all_index = 0
        self._align_all_failed = []
        self._start_progress("Aligning all sessions...")
        self._run_next_auto_align_for_all()

    def _run_next_auto_align_for_all(self):
        if self._align_all_index >= len(self._sessions_to_align):
            msg = "Auto alignment for all sessions complete."
            if self._align_all_failed:
                msg += f" Failed: sessions {self._align_all_failed}"
            self._end_progress(msg)
            self.update_image()
            return
        cur = self._sessions_to_align[self._align_all_index]
        self.status_label.setText(
            f"Aligning Session {cur} ({self._align_all_index + 1}/{len(self._sessions_to_align)})")
        self._start_auto_thread(cur, self._align_all_params, self._on_all_progress,
                                lambda best, s=cur: self._on_one_session_auto_finished(s, best))

    def _on_all_progress(self, p):
        total = max(1, len(self._sessions_to_align))
        self.progress_bar.setValue(int((self._align_all_index + p / 100.0) / total * 100))

    def _on_one_session_auto_finished(self, session_idx, best):
        if best:
            self._apply_auto_result(session_idx, best)
        else:
            self._align_all_failed.append(session_idx)
        self._align_all_index += 1
        self._run_next_auto_align_for_all()

    # ------------------- Shutdown -------------------
    def closeEvent(self, event):
        for t in (self.loading_thread, self.save_thread, self.auto_thread):
            if t is not None and t.isRunning():
                t.requestInterruption()
                t.wait()
        super().closeEvent(event)


# ------------------- Run -------------------
if __name__ == '__main__':
    # Consistent sizing on high-DPI Linux/Windows displays (automatic on macOS).
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    apply_theme(app)
    window = AlignGUI()
    window.show()
    sys.exit(app.exec_())
