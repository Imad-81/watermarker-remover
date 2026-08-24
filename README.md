# Gemini Watermark Remover

A high-performance Python tool that completely removes the Gemini AI sparkle watermark from videos using **Precise Star Geometry Inpainting** and **FFmpeg**.

---

## 🚀 How It Works

1. **Exact 4-Pointed Star Geometry:**
   - Instead of generic square or astroid approximations, the model uses the **exact mathematical 4-pointed circular-arc geometry** of the Gemini logo.
   - Applies a slight dilation (2px default) to guarantee 100% coverage of the anti-aliased perimeter, completely eliminating outline rings and boundary artifacts.

2. **Fast Marching / Navier-Stokes Inpainting:**
   - Seamlessly propagates surrounding texture, lines, gradients, and lighting across the watermark area.
   - Leaves zero outline or ghosting artifacts.

3. **High-Speed Streaming:**
   - Streams raw video frames through FFmpeg standard pipes directly in Python.
   - Processes at **300–500+ FPS**.

4. **Dynamic Resolution Scaling:**
   - Probes resolution and computes exact bounding boxes for any video aspect ratio (720p, 1080p, 4K, 9:16 vertical shorts, etc.).

---

## 🛠️ Prerequisites

- **FFmpeg & FFprobe**: Must be installed and available in your `PATH`.
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: `winget install Gyan.FFmpeg`
- **Python 3.8+**
- **OpenCV & NumPy**: `pip install opencv-python numpy`

---

## 📖 Usage

### 1. Basic Removal (Single Video)
```bash
python remove_gemini_watermark.py input.mp4
# Cleaned video saved as: input_cleaned.mp4
```

### 2. Specify Custom Output File
```bash
python remove_gemini_watermark.py input.mp4 -o clean_video.mp4
```

### 3. Preview Removal on a Single Frame
Generates a test frame with the watermark area marked and cleaned:
```bash
python remove_gemini_watermark.py input.mp4 --preview
```

### 4. Batch Process an Entire Directory
```bash
python remove_gemini_watermark.py ./videos/ --batch
# Outputs saved to: ./videos/cleaned_videos/
```

### 5. Advanced Options
- `--dilation 2`: Adjust mask expansion in pixels (default: 2px, ensures no outline is left).
- `--radius 3`: Inpainting neighborhood radius (default: 3).
- `--method ns`: Use Navier-Stokes inpainting for high-frequency gradients.
- `--crf 18`: H.264 quality factor (0-51, default: 18 for visually lossless output).
