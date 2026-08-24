# Gemini Watermark Remover

A high-performance Python tool that completely removes the Gemini AI sparkle watermark from videos, frame image sequences, and standalone images using **Precise Geometric Star Inpainting** and **FFmpeg**.

---

## 🚀 Key Features

1. **Folder / Image Sequence Processing:**
   - Cleans all image frames within a directory in parallel using multi-threading.
   - Outputs to another folder keeping the **exact same frame filenames** (ideal for VFX / render pipelines / animation frame sequences).
2. **Exact 4-Pointed Star Geometry Inpainting:**
   - Uses the **exact mathematical 4-pointed circular-arc geometry** of the Gemini logo.
   - Built-in boundary dilation (2px default) to guarantee 100% anti-aliased edge coverage with zero outline rings or halo artifacts.
3. **Seamless Texture Propagation (Telea / Navier-Stokes):**
   - Seamlessly propagates surrounding texture, gradients, and lighting across the watermark area.
4. **Blazing Fast Performance:**
   - Multi-threaded frame processing for image sequences (hundreds of frames per second).
   - High-speed raw video stream processing at **300–500+ FPS**.
5. **Dynamic Resolution Scaling:**
   - Automatically computes exact watermark coordinates and scale for any resolution and aspect ratio (720p, 1080p, 4K, 9:16 vertical shorts, etc.).

---

## 🛠️ Prerequisites

- **Python 3.8+**
- **OpenCV & NumPy**:
  ```bash
  pip install opencv-python numpy
  ```
- **FFmpeg & FFprobe** *(required for video processing)*:
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: `winget install Gyan.FFmpeg`

---

## 📖 Usage

### 1. Process a Folder of Frame Images (Retaining Same Filenames)
Point the script to a folder containing frame images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`). It outputs cleaned frames into another folder with identical filenames:

```bash
# Clean folder of frames into a custom output folder
python remove_gemini_watermark.py ./frames_folder/ -o ./cleaned_frames/

# Default output folder will be named '<folder_name>_cleaned'
python remove_gemini_watermark.py ./frames_folder/
```

### 2. Process a Single Image Frame
```bash
python remove_gemini_watermark.py frame_0001.png -o frame_cleaned.png
```

### 3. Process a Single Video
```bash
# Automatic output naming (input_cleaned.mp4)
python remove_gemini_watermark.py input.mp4

# Custom output file
python remove_gemini_watermark.py input.mp4 -o clean_video.mp4
```

### 4. Preview Removal on a Single Frame / Video
Generates a test frame with the watermark area marked with a green box and cleaned:
```bash
python remove_gemini_watermark.py input.mp4 --preview
python remove_gemini_watermark.py frame.png --preview
```

### 5. Batch Process Video Files in a Folder
```bash
python remove_gemini_watermark.py ./videos/ --batch
# Outputs saved to: ./videos/cleaned_videos/
```

---

## ⚙️ Advanced CLI Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `-o`, `--output` | Auto | Output file or destination directory |
| `--workers` | CPU count | Number of parallel worker threads for frame folder processing |
| `--dilation` | `2` | Mask expansion in pixels (ensures no outline or halo is left) |
| `--radius` | `3` | Inpainting neighborhood radius |
| `--method` | `inpaint` | Inpainting method: `inpaint` (Telea) or `ns` (Navier-Stokes) |
| `--crf` | `18` | H.264 CRF quality factor for video (0-51, lower is higher quality) |
| `--preset` | `fast` | x264 video encoding preset (`ultrafast`, `fast`, `medium`, `slow`) |
| `--x`, `--y`, `-W`, `-H` | Auto | Override watermark position and bounding box dimensions |
